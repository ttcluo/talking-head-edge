"""
VAE 替换效果对比 Demo：SD VAE vs TAESD

同一 UNet 输出，分别用 SD VAE 与 TAESD 解码，生成左右对比视频。
仅替换 VAE，其余流程（UNet、blending）完全一致。

用法（在 MuseTalk 目录下执行）：
  cd $MUSE_ROOT
  PYTHONPATH=$PWD python ../step2/demo_vae_taesd.py \
      --avatar_id avator_1 \
      --audio data/audio/avator_1.wav \
      --num_frames 200 \
      --taesd_dir models/taesd_cache \
      --out profile_results/vae_taesd_demo.mp4
"""

import argparse
import copy
import glob
import os
import pickle
import subprocess
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
parser.add_argument("--audio_feat", type=str, default="")
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--out", type=str, default="profile_results/vae_taesd_demo.mp4")
parser.add_argument("--version", type=str, default="v15")
parser.add_argument("--taesd_dir", type=str, default="",
                    help="TAESD 本地目录，空则尝试从 HuggingFace 加载")
args = parser.parse_args()

os.chdir(MUSE_ROOT)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("  VAE 替换效果对比：SD VAE vs TAESD")
print("=" * 65)

# ==================== 加载预处理数据 ====================
avatar_path = f"results/{args.version}/avatars/{args.avatar_id}"
if not os.path.exists(avatar_path):
    print(f"\n  ✗ 预处理数据不存在: {avatar_path}")
    sys.exit(1)

print(f"\n[加载预处理数据] {avatar_path}")
input_latent_list_cycle = torch.load(os.path.join(avatar_path, "latents.pt"))
with open(os.path.join(avatar_path, "coords.pkl"), "rb") as f:
    coord_list_cycle = pickle.load(f)
with open(os.path.join(avatar_path, "mask_coords.pkl"), "rb") as f:
    mask_coords_list_cycle = pickle.load(f)

img_list = sorted(
    glob.glob(os.path.join(avatar_path, "full_imgs", "*.[jpJP][pnPN]*[gG]")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)
mask_list_files = sorted(
    glob.glob(os.path.join(avatar_path, "mask", "*.[jpJP][pnPN]*[gG]")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending

frame_list_cycle = read_imgs(img_list)
mask_list_cycle = read_imgs(mask_list_files)
cycle_len = len(frame_list_cycle)
print(f"  ✓ 预处理帧数: {cycle_len}")

# ==================== 加载模型 ====================
print("\n[加载模型]")
from musetalk.utils.utils import load_all_model

vae_sd, unet_wrapper, pe_module = load_all_model(
    unet_model_path="models/musetalkV15/unet.pth",
    unet_config="models/musetalkV15/musetalk.json",
)
unet = unet_wrapper.model.to(device)
vae_sd.vae = vae_sd.vae.to(device)
pe = pe_module.to(device)
weight_dtype = unet.dtype
unet.eval()
vae_sd.vae.eval()
timesteps = torch.tensor([0], device=device)
sd_scaling = getattr(vae_sd.vae.config, "scaling_factor", 0.18215)

# TAESD
try:
    from diffusers import AutoencoderTiny
    if args.taesd_dir and os.path.isdir(args.taesd_dir):
        taesd = AutoencoderTiny.from_pretrained(args.taesd_dir, local_files_only=True).to(device, dtype=weight_dtype)
    else:
        taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device, dtype=weight_dtype)
    taesd.eval()
    TAESD_OK = True
    print(f"  ✓ TAESD 加载成功")
except Exception as e:
    TAESD_OK = False
    print(f"  ✗ TAESD 加载失败: {e}")
    sys.exit(1)

# ==================== 音频特征 ====================
audio_feat_path = args.audio_feat or os.path.join("dataset/distill/audio_feats", f"{args.avatar_id}.pt")
if os.path.exists(audio_feat_path):
    whisper_chunks = torch.load(audio_feat_path, map_location=device)
    print(f"\n[音频特征] 预计算 {audio_feat_path}")
else:
    print(f"\n[提取音频特征] {args.audio}")
    from musetalk.utils.audio_processor import AudioProcessor
    from transformers import WhisperModel
    audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
    whisper = WhisperModel.from_pretrained("models/whisper").to(device).eval()
    wf, lib_len = audio_processor.get_audio_feature(args.audio, torch.float32)
    whisper_chunks = audio_processor.get_whisper_chunk(
        wf, device, torch.float32, whisper, lib_len, fps=args.fps,
        audio_padding_length_left=2, audio_padding_length_right=2,
    )

total_frames = len(whisper_chunks)
if args.num_frames > 0:
    total_frames = min(total_frames, args.num_frames)
whisper_chunks = whisper_chunks[:total_frames]
print(f"  ✓ 帧数: {total_frames}")

# ==================== 工具函数 ====================
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def face_to_bgr(face):
    """统一转为 (H,W,3) BGR uint8"""
    if hasattr(face, "permute"):
        arr = (face.permute(1, 2, 0).float().cpu().numpy().clip(-1, 1) + 1) * 127.5
        arr = arr.clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return np.asarray(face, dtype=np.uint8)


def compose_frame(idx, res_face):
    bbox = coord_list_cycle[idx % cycle_len]
    ori = copy.deepcopy(frame_list_cycle[idx % cycle_len])
    x1, y1, x2, y2 = bbox
    res_face = face_to_bgr(res_face)
    try:
        res = cv2.resize(res_face, (x2 - x1, y2 - y1))
    except Exception:
        return ori
    mask = mask_list_cycle[idx % cycle_len]
    mcbox = mask_coords_list_cycle[idx % cycle_len]
    return get_image_blending(ori, res, bbox, mask, mcbox)


def add_overlay(frame, label, fps_val, color):
    out = frame.copy()
    h, w = out.shape[:2]
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"{fps_val:.1f} FPS", (w - 80, 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)
    return out


# ==================== 推理：SD VAE + TAESD ====================
print(f"\n[推理：SD VAE vs TAESD，{total_frames} 帧]")
sd_out, taesd_out = [], []
sd_t, taesd_t = [], []

with torch.no_grad():
    for i in range(0, total_frames, args.batch_size):
        bs = min(args.batch_size, total_frames - i)
        chunk_slice = whisper_chunks[i:i+bs]
        if isinstance(chunk_slice[0], torch.Tensor):
            w_batch = torch.stack([c.cpu() for c in chunk_slice]).to(device=device, dtype=weight_dtype)
        else:
            w_batch = torch.from_numpy(np.stack(chunk_slice)).to(device=device, dtype=weight_dtype)
        lat_batch = torch.cat(
            [input_latent_list_cycle[j % cycle_len] for j in range(i, i+bs)], 0
        ).to(device=device, dtype=weight_dtype)

        audio_feat = pe(w_batch)
        pred = unet(lat_batch, timesteps, encoder_hidden_states=audio_feat, return_dict=False)[0]
        pred = pred.to(dtype=weight_dtype)

        # SD VAE decode
        sync()
        t0 = time.time()
        pred_sd = pred.to(dtype=vae_sd.vae.dtype)
        z_sd = pred_sd / sd_scaling
        recon_sd = vae_sd.vae.decode(z_sd).sample
        sync()
        sd_t.extend([(time.time() - t0) / bs] * bs)

        # TAESD decode
        sync()
        t0 = time.time()
        recon_taesd = taesd.decode(pred).sample
        sync()
        taesd_t.extend([(time.time() - t0) / bs] * bs)

        for k in range(bs):
            idx = i + k
            sd_out.append(compose_frame(idx, recon_sd[k]))
            taesd_out.append(compose_frame(idx, recon_taesd[k]))

        if (i + bs) % 50 < args.batch_size or (i + bs) >= total_frames:
            print(f"  [{i+bs}/{total_frames}]")

fps_sd = total_frames / sum(sd_t)
fps_taesd = total_frames / sum(taesd_t)
speedup = fps_taesd / fps_sd
print(f"  SD VAE:   {fps_sd:.1f} FPS")
print(f"  TAESD:   {fps_taesd:.1f} FPS  加速 {speedup:.2f}×")

# ==================== SSIM / PSNR ====================
def _ssim_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 1e-10 else 100.0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = g1.mean(), g2.mean()
    sig1, sig2 = g1.std(), g2.std()
    sig12 = ((g1 - mu1) * (g2 - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sig1**2 + sig2**2 + C2))
    return float(ssim), float(psnr)

ssim_vals, psnr_vals = [], []
for s, t in zip(sd_out, taesd_out):
    ss, pp = _ssim_psnr(s, t)
    ssim_vals.append(ss)
    psnr_vals.append(pp)
mean_ssim = float(np.mean(ssim_vals))
mean_psnr = float(np.mean(psnr_vals))
print(f"\n[质量] SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.2f} dB (SD vs TAESD)")

# ==================== 合成对比视频 ====================
print(f"\n[合成对比视频]")
target_h = min(sd_out[0].shape[0], 540)


def resize_h(f, h):
    oh, ow = f.shape[:2]
    nw = int(ow * h / oh)
    return cv2.resize(f, (nw, h))


comparison_frames = []
for i in range(total_frames):
    s = resize_h(sd_out[i], target_h)
    t = resize_h(taesd_out[i], target_h)
    s = add_overlay(s, "SD VAE (original)", fps_sd, (200, 200, 255))
    t = add_overlay(t, "TAESD (9× faster)", fps_taesd, (200, 255, 200))
    div = np.zeros((target_h, 4, 3), dtype=np.uint8)
    div[:] = (80, 80, 80)
    row = np.concatenate([s, div, t], axis=1)
    bw = row.shape[1]
    stat = np.zeros((30, bw, 3), dtype=np.uint8)
    info = (f"Frame {i+1}/{total_frames}   "
            f"SD VAE {fps_sd:.1f} FPS   TAESD {fps_taesd:.1f} FPS   "
            f"Speedup {speedup:.2f}×   SSIM {mean_ssim:.4f}")
    cv2.putText(stat, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA)
    row = np.concatenate([row, stat], axis=0)
    comparison_frames.append(row)


def write_video_with_audio(frames, audio_path, out_path, fps):
    tmp = out_path.replace(".mp4", "_tmp.mp4")
    h_o, w_o = frames[0].shape[:2]
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_o, h_o))
    for fr in frames:
        wr.write(fr)
    wr.release()
    if audio_path and os.path.exists(audio_path):
        cmd = (f"ffmpeg -loglevel error -nostdin -y "
               f"-i {tmp} -i {audio_path} "
               f"-c:v libx264 -crf 20 -preset fast "
               f"-c:a aac -b:a 128k -shortest {out_path}")
    else:
        cmd = f"ffmpeg -loglevel error -nostdin -y -i {tmp} -c:v libx264 -crf 20 -preset fast {out_path}"
    ret = subprocess.call(cmd, shell=True)
    if ret == 0:
        os.remove(tmp)
    else:
        os.rename(tmp, out_path)


write_video_with_audio(comparison_frames, args.audio, args.out, args.fps)
print(f"  ✓ 对比视频: {args.out}")

base_dir = os.path.dirname(args.out)
sd_path = os.path.join(base_dir, "vae_sd.mp4")
taesd_path = os.path.join(base_dir, "vae_taesd.mp4")
write_video_with_audio(sd_out, args.audio, sd_path, args.fps)
print(f"  ✓ 单独 SD VAE: {sd_path}")
write_video_with_audio(taesd_out, args.audio, taesd_path, args.fps)
print(f"  ✓ 单独 TAESD: {taesd_path}")

print(f"""
======================================================================
  VAE 替换 Demo 汇总
======================================================================
  SD VAE:   {fps_sd:.1f} FPS
  TAESD:    {fps_taesd:.1f} FPS  (加速 {speedup:.2f}×)
  SSIM:     {mean_ssim:.4f}  PSNR: {mean_psnr:.2f} dB
  输出:     {args.out}
""")
