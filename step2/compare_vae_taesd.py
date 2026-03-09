"""
TAESD vs SD VAE 质量对比

同一 UNet 输出，分别用 SD VAE 与 TAESD 解码，比较 SSIM/PSNR 及 VAE 耗时。
FP32 不变，评估 TAESD 替换的可行性。

用法（在 MuseTalk 目录下执行）：
  cd $MUSE_ROOT
  PYTHONPATH=$MUSE_ROOT python ../step2/compare_vae_taesd.py \
      --avatar_id avator_1 \
      --audio data/audio/avator_1.wav \
      --num_frames 50 \
      --out_dir profile_results/vae_taesd_compare
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
parser.add_argument("--audio_feat", type=str, default="")
parser.add_argument("--num_frames", type=int, default=50)
parser.add_argument("--version", type=str, default="v15")
parser.add_argument("--out_dir", type=str, default="profile_results/vae_taesd_compare")
parser.add_argument("--save_frames", action="store_true", help="保存左右对比帧")
args = parser.parse_args()

os.chdir(MUSE_ROOT)
os.makedirs(args.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) float [-1,1] → (H, W, 3) uint8"""
    img = t[0].permute(1, 2, 0).float().cpu().numpy()
    img = ((img.clip(-1, 1) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return img


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if SKIMAGE_OK:
        return float(sk_ssim(a, b, channel_axis=2, data_range=255))
    m = np.mean([np.corrcoef(a[:, :, c].ravel(), b[:, :, c].ravel())[0, 1] for c in range(3)])
    return float(m) if not np.isnan(m) else 0.0


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    if SKIMAGE_OK:
        return float(sk_psnr(a, b, data_range=255))
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


# ==================== 加载数据 ====================
avatar_path = f"results/{args.version}/avatars/{args.avatar_id}"
if not os.path.exists(avatar_path):
    print(f"✗ 预处理数据不存在: {avatar_path}")
    sys.exit(1)

print(f"[加载数据] {avatar_path}")
input_latent_list = torch.load(os.path.join(avatar_path, "latents.pt"), map_location=device)
cycle_len = len(input_latent_list)

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
print(f"  帧数: {num_frames}")

# ==================== 加载 UNet + PE ====================
print("\n[加载 UNet + PE]")
from musetalk.utils.utils import load_all_model

vae_sd, unet_wrapper, pe_module = load_all_model(
    unet_model_path="models/musetalkV15/unet.pth",
    unet_config="models/musetalkV15/musetalk.json",
)
unet = unet_wrapper.model.to(device)
pe = pe_module.to(device)
weight_dtype = unet.dtype
unet.eval()
timesteps = torch.tensor([0], device=device, dtype=torch.long)

# SD VAE（原始）
vae_sd.vae = vae_sd.vae.to(device)
vae_sd.vae.eval()
sd_scaling = getattr(vae_sd.vae.config, "scaling_factor", 0.18215)

# TAESD
try:
    from diffusers import AutoencoderTiny
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device, dtype=weight_dtype)
    taesd.eval()
    TAESD_OK = True
    print(f"  ✓ TAESD 加载成功 (madebyollin/taesd)")
except Exception as e:
    TAESD_OK = False
    print(f"  ✗ TAESD 加载失败: {e}")
    print("    请安装: pip install diffusers>=0.25 并确保可访问 HuggingFace")

# ==================== 收集 UNet 输出 ====================
print(f"\n[收集 UNet 输出，{num_frames} 帧]")
pred_latents = []
with torch.no_grad():
    for i in range(num_frames):
        lat = input_latent_list[i % cycle_len].to(device, dtype=weight_dtype)
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)
        af = audio_chunks[i]
        if isinstance(af, torch.Tensor):
            af = af.unsqueeze(0).to(device=device, dtype=weight_dtype)
        else:
            af = torch.from_numpy(af).unsqueeze(0).to(device=device, dtype=weight_dtype)
        af = pe(af)
        pred = unet(lat, timesteps, encoder_hidden_states=af, return_dict=False)[0]
        pred_latents.append(pred.detach())
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{num_frames}")

# ==================== SD VAE 解码 ====================
print("\n[SD VAE 解码]")
frames_sd = []
t_sd = []
with torch.no_grad():
    for pred in pred_latents:
        pred = pred.to(dtype=vae_sd.vae.dtype)
        sync()
        t0 = time.time()
        z = pred / sd_scaling
        out = vae_sd.vae.decode(z).sample
        sync()
        t_sd.append((time.time() - t0) * 1000)
        frames_sd.append(tensor_to_uint8(out))
ms_sd = np.mean(t_sd)
print(f"  SD VAE 平均: {ms_sd:.2f} ms/帧")

# ==================== TAESD 解码 ====================
if TAESD_OK:
    print("\n[TAESD 解码]")
    frames_taesd = []
    t_taesd = []
    with torch.no_grad():
        for pred in pred_latents:
            pred = pred.to(device=device, dtype=weight_dtype)
            sync()
            t0 = time.time()
            # TAESD 使用 scaling_factor=1.0，直接接收 diffusion 输出
            out = taesd.decode(pred).sample
            sync()
            t_taesd.append((time.time() - t0) * 1000)
            frames_taesd.append(tensor_to_uint8(out))
    ms_taesd = np.mean(t_taesd)
    print(f"  TAESD 平均: {ms_taesd:.2f} ms/帧  加速比: {ms_sd/ms_taesd:.2f}×")

# ==================== 质量对比 ====================
print("\n[质量对比]")
ssims, psnrs = [], []
if TAESD_OK:
    for i in range(num_frames):
        ssims.append(compute_ssim(frames_sd[i], frames_taesd[i]))
        psnrs.append(compute_psnr(frames_sd[i], frames_taesd[i]))
    ssim_mean = float(np.mean(ssims))
    psnr_mean = float(np.mean(psnrs))
    print(f"  SSIM (SD vs TAESD): {ssim_mean:.4f}")
    print(f"  PSNR (SD vs TAESD): {psnr_mean:.2f} dB")

# ==================== 保存结果 ====================
results = {
    "config": {"avatar_id": args.avatar_id, "num_frames": num_frames, "device": device},
    "vae_sd_ms": ms_sd,
    "vae_taesd_ms": ms_taesd if TAESD_OK else None,
    "speedup": ms_sd / ms_taesd if TAESD_OK else None,
    "ssim": ssim_mean if TAESD_OK else None,
    "psnr_db": psnr_mean if TAESD_OK else None,
}
with open(os.path.join(args.out_dir, "compare_result.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n  结果已保存: {args.out_dir}/compare_result.json")

if args.save_frames and TAESD_OK:
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(min(20, num_frames)):
        side = np.hstack([frames_sd[i], frames_taesd[i]])
        cv2.putText(side, "SD VAE", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(side, "TAESD", (264, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(frames_dir, f"{i:04d}.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    print(f"  对比帧已保存: {frames_dir}/")

# ==================== 总结 ====================
print("\n" + "=" * 55)
print("  TAESD vs SD VAE 质量对比总结")
print("=" * 55)
print(f"  SD VAE:    {ms_sd:.2f} ms/帧")
if TAESD_OK:
    print(f"  TAESD:    {ms_taesd:.2f} ms/帧  加速 {ms_sd/ms_taesd:.2f}×")
    print(f"  SSIM:     {ssim_mean:.4f}  (1.0=完全相同)")
    print(f"  PSNR:     {psnr_mean:.2f} dB")
    if ssim_mean >= 0.95:
        print("\n  ✓ SSIM≥0.95，TAESD 替换可行")
    elif ssim_mean >= 0.90:
        print("\n  ~ SSIM 0.90~0.95，可接受，建议主观评估")
    else:
        print("\n  ✗ SSIM<0.90，质量损失较大，谨慎替换")
print("=" * 55)
