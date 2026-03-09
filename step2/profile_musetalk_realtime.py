"""
MuseTalk 真实推理链路 Profiling

与 demo_video / run_full_pipeline 完全一致的推理流程：
  latent(预计算) + audio_feat → pe() → UNet → VAE decode → blend

无 VAE encode（latent 来自预处理缓存）。
精确测量各模块耗时，分析推理性能瓶颈。

用法：
  cd $MUSE_ROOT
  PYTHONPATH=$MUSE_ROOT python $REPO/step2/profile_musetalk_realtime.py \
      --avatar_id avator_1 \
      --audio data/audio/avator_1.wav \
      --num_frames 100
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--avatar_id", type=str, default="avator_1")
parser.add_argument("--audio", type=str, default="data/audio/avator_1.wav")
parser.add_argument("--audio_feat", type=str, default="", help="预计算音频特征路径，覆盖默认 dataset/distill/audio_feats/{avatar_id}.pt")
parser.add_argument("--num_frames", type=int, default=100)
parser.add_argument("--version", type=str, default="v15")
parser.add_argument("--output_dir", type=str, default="profile_results")
parser.add_argument("--no_blend", action="store_true", help="跳过 blending，仅测量 pe+unet+vae_decode")
args = parser.parse_args()

os.chdir(MUSE_ROOT)
os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ==================== 加载预处理数据 ====================
avatar_path = f"results/{args.version}/avatars/{args.avatar_id}"
if not os.path.exists(avatar_path):
    print(f"✗ 预处理数据不存在: {avatar_path}")
    print("  请先运行 realtime_inference 预处理，或使用 prepare_distill_data + 预处理")
    sys.exit(1)

print(f"[加载数据] {avatar_path}")
input_latent_list = torch.load(os.path.join(avatar_path, "latents.pt"), map_location=device)
cycle_len = len(input_latent_list)

# 音频特征：优先预计算，否则实时提取
audio_feat_path = args.audio_feat or os.path.join("dataset/distill/audio_feats", f"{args.avatar_id}.pt")
if os.path.exists(audio_feat_path):
    audio_chunks = torch.load(audio_feat_path, map_location=device)
    print(f"  音频特征: 预计算 {audio_feat_path}")
else:
    print("  预计算音频特征不存在，使用 Whisper 实时提取...")
    from musetalk.utils.audio_processor import AudioProcessor
    from transformers import WhisperModel
    audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
    whisper = WhisperModel.from_pretrained("models/whisper").to(device).eval()
    wf, lib_len = audio_processor.get_audio_feature(args.audio, torch.float32)
    audio_chunks = audio_processor.get_whisper_chunk(
        wf, device, torch.float32, whisper, lib_len, fps=25,
        audio_padding_length_left=2, audio_padding_length_right=2,
    )

num_frames = min(args.num_frames, len(audio_chunks))
audio_chunks = audio_chunks[:num_frames]
print(f"  帧数: {num_frames}  latent 周期: {cycle_len}")

# ==================== 加载模型 ====================
print("\n[加载模型]")
from musetalk.utils.utils import load_all_model

vae, unet_wrapper, pe_module = load_all_model(
    unet_model_path="models/musetalkV15/unet.pth",
    unet_config="models/musetalkV15/musetalk.json",
)
unet = unet_wrapper.model.to(device)
vae.vae = vae.vae.to(device)
pe = pe_module.to(device)

# 使用 FP16 与 demo 一致
weight_dtype = torch.float16
unet = unet.to(weight_dtype)
vae.vae = vae.vae.to(weight_dtype)
unet.eval()
vae.vae.eval()

timesteps = torch.tensor([0], device=device, dtype=torch.long)
print("  模型加载完成 (FP16)")

# ==================== 预热 ====================
print("\n[预热 5 帧]")
with torch.no_grad():
    for i in range(5):
        lat = input_latent_list[i % cycle_len].to(device, dtype=weight_dtype)
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)
        af = audio_chunks[i]
        if isinstance(af, torch.Tensor):
            af = af.unsqueeze(0).to(device)
        else:
            af = torch.from_numpy(af).unsqueeze(0).to(device)
        af = pe(af)
        pred = unet(lat, timesteps, encoder_hidden_states=af, return_dict=False)[0]
        _ = vae.decode_latents(pred)
sync()
print("  预热完成")

# ==================== 逐帧 Profiling ====================
do_blend = not args.no_blend
if do_blend:
    with open(os.path.join(avatar_path, "coords.pkl"), "rb") as f:
        coords = pickle.load(f)
    with open(os.path.join(avatar_path, "mask_coords.pkl"), "rb") as f:
        mask_coords = pickle.load(f)
    from musetalk.utils.preprocessing import read_imgs
    from musetalk.utils.blending import get_image_blending
    img_list = sorted(
        glob.glob(os.path.join(avatar_path, "full_imgs", "*.[jpJP][pnPN]*[gG]")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )
    mask_list = sorted(
        glob.glob(os.path.join(avatar_path, "mask", "*.[jpJP][pnPN]*[gG]")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )
    frames = read_imgs(img_list)
    masks = read_imgs(mask_list)
else:
    coords = mask_coords = frames = masks = None

print(f"\n[Profiling 推理链路，{num_frames} 帧] blend={'on' if do_blend else 'off'}")

t_pe, t_unet, t_vae_dec, t_blend, t_total = [], [], [], [], []

with torch.no_grad():
    for i in range(num_frames):
        lat = input_latent_list[i % cycle_len].to(device, dtype=weight_dtype)
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)
        af = audio_chunks[i]
        if isinstance(af, torch.Tensor):
            af = af.unsqueeze(0).to(device)
        else:
            af = torch.from_numpy(af).unsqueeze(0).to(device)

        sync()
        t0 = time.time()

        # 1. 音频位置编码
        t_pe0 = time.time()
        af = pe(af)
        sync()
        t_pe.append((time.time() - t_pe0) * 1000)

        # 2. UNet 前向
        t_unet0 = time.time()
        pred = unet(lat, timesteps, encoder_hidden_states=af, return_dict=False)[0]
        sync()
        t_unet.append((time.time() - t_unet0) * 1000)

        # 3. VAE Decode
        t_dec0 = time.time()
        recon = vae.decode_latents(pred)
        sync()
        t_vae_dec.append((time.time() - t_dec0) * 1000)

        # 4. Blending（CPU，含 resize + get_image_blending）
        if do_blend:
            t_blend0 = time.time()
            face = recon[0]
            if hasattr(face, "permute"):
                face = (face.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            bbox = coords[i % cycle_len]
            ori = frames[i % cycle_len].copy()
            x1, y1, x2, y2 = bbox
            res = cv2.resize(face, (x2 - x1, y2 - y1))
            mask = masks[i % cycle_len]
            mcbox = mask_coords[i % cycle_len]
            _ = get_image_blending(ori, res, bbox, mask, mcbox)
            t_blend.append((time.time() - t_blend0) * 1000)
        else:
            t_blend.append(0.0)

        sync()
        t_total.append((time.time() - t0) * 1000)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{num_frames} 帧...")

# ==================== 统计输出 ====================
def stats(arr):
    arr = np.array(arr)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }

results = {
    "audio_pe": stats(t_pe),
    "unet_forward": stats(t_unet),
    "vae_decode": stats(t_vae_dec),
    "blending": stats(t_blend),
    "total_per_frame": stats(t_total),
}

total_mean = results["total_per_frame"]["mean_ms"]
fps = 1000 / total_mean

# 占比
for k in ["audio_pe", "unet_forward", "vae_decode", "blending"]:
    denom = total_mean if total_mean > 0 else 1
    results[k]["pct"] = results[k]["mean_ms"] / denom * 100

print("\n" + "=" * 65)
print("  MuseTalk 真实推理链路 Profiling 结果")
print("=" * 65)
print(f"\n  {'模块':<20} {'平均(ms)':>10} {'占比':>8} {'说明'}")
print("  " + "-" * 55)
print(f"  {'audio_pe':<20} {results['audio_pe']['mean_ms']:>10.2f} {results['audio_pe']['pct']:>6.1f}%  音频位置编码")
print(f"  {'unet_forward':<20} {results['unet_forward']['mean_ms']:>10.2f} {results['unet_forward']['pct']:>6.1f}%  UNet 前向（主要瓶颈）")
print(f"  {'vae_decode':<20} {results['vae_decode']['mean_ms']:>10.2f} {results['vae_decode']['pct']:>6.1f}%  VAE 解码")
print(f"  {'blending':<20} {results['blending']['mean_ms']:>10.2f} {results['blending']['pct']:>6.1f}%  图像融合")
print("  " + "-" * 55)
print(f"  {'单帧总计':<20} {total_mean:>10.2f}  100.0%")
print(f"  {'理论 FPS':<20} {fps:>10.1f}")

# 瓶颈排序
order = sorted(
    [("audio_pe", results["audio_pe"]), ("unet_forward", results["unet_forward"]),
     ("vae_decode", results["vae_decode"]), ("blending", results["blending"])],
    key=lambda x: -x[1]["mean_ms"]
)
bottleneck = order[0][0]
print(f"\n  主要瓶颈: {bottleneck} ({order[0][1]['mean_ms']:.1f}ms, {order[0][1]['pct']:.1f}%)")

# 保存
out_path = os.path.join(args.output_dir, "profile_musetalk_realtime.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "config": {"avatar_id": args.avatar_id, "num_frames": num_frames, "device": device},
        "timing_ms": {k: v["mean_ms"] for k, v in results.items() if k != "total_per_frame"},
        "total_ms": total_mean,
        "fps": fps,
        "bottleneck": bottleneck,
    }, f, ensure_ascii=False, indent=2)
print(f"\n  结果已保存: {out_path}")
