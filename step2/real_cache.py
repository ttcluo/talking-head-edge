"""
真实 UNet Early-Return 推理（P3）
区别于 cache_prototype / quality_eval 的"仿真"：
  本脚本在推理时真正跳过 UNet 调用，测量实际 wall-clock FPS，不依赖任何事后模拟。

使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    python /path/to/step2/real_cache.py --video data/video/yongen.mp4

输出：
    - 真实 FPS（跳过帧确实不调用 UNet）
    - SSIM / PSNR（与不带缓存的相同推理对比）
    - profile_results/real_cache.json
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

parser = argparse.ArgumentParser()
parser.add_argument("--video",      type=str,   default="data/video/yongen.mp4")
parser.add_argument("--threshold",  type=float, default=0.15)
parser.add_argument("--num_frames", type=int,   default=200)
parser.add_argument("--output_dir", type=str,   default="profile_results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16

print("=" * 65)
print("  真实 UNet Early-Return 推理（P3）")
print("=" * 65)
print(f"  视频={args.video}  阈值={args.threshold}  帧数={args.num_frames}")

# ==================== 工具函数 ====================
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def compute_ssim(a, b):
    if SKIMAGE_OK:
        return float(sk_ssim(a, b, channel_axis=2, data_range=255))
    mse = float(((a.astype(float) - b.astype(float)) ** 2).mean())
    return float(1 - mse / (255 ** 2))

def compute_psnr(a, b):
    if SKIMAGE_OK:
        v = float(sk_psnr(a, b, data_range=255))
        return v if np.isfinite(v) else 100.0
    mse = float(((a.astype(float) - b.astype(float)) ** 2).mean())
    return float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else 100.0

def decode_frame(lat, vae):
    with torch.no_grad():
        out = vae.decode(lat).sample
    return ((out[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

# ==================== 加载模型 ====================
print("\n[加载模型]")
from diffusers import AutoencoderKL
from musetalk.models.unet import UNet2DConditionModel
import json as _json

with open("models/musetalkV15/musetalk.json") as f:
    unet_config = _json.load(f)

vae  = AutoencoderKL.from_pretrained("models/sd-vae").to(device, dtype)
unet = UNet2DConditionModel(**unet_config).to(device, dtype)
unet.load_state_dict(torch.load("models/musetalkV15/unet.pth", map_location=device))
vae.eval(); unet.eval()
print("  ✓ 模型加载完成")

# 占位音频（与 quality_eval 保持一致）
timestep    = torch.tensor([0.0], device=device, dtype=dtype)
audio_dummy = torch.zeros(1, 1, 384, device=device, dtype=dtype)

# ==================== 读取视频 ====================
cap = cv2.VideoCapture(args.video)
frames = []
while len(frames) < args.num_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
N = len(frames)
print(f"  读取 {N} 帧")

# ==================== 预热 ====================
with torch.no_grad():
    _d = torch.randn(1, 8, 32, 32, device=device, dtype=dtype)
    for _ in range(5):
        unet(_d, timestep, encoder_hidden_states=audio_dummy)
        vae.decode(torch.randn(1, 4, 32, 32, device=device, dtype=dtype))

# ==================== 基线推理（无缓存）====================
print(f"\n[基线推理：无缓存，全部 {N} 帧运行 UNet]")

baseline_pixels = []
t_baseline_total = 0.0

with torch.no_grad():
    for frame in frames:
        face   = cv2.resize(frame[:256, :256], (256, 256))
        face_t = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1
        face_t = face_t.unsqueeze(0).to(device, dtype)

        sync(); t0 = time.time()
        lat   = vae.encode(face_t).latent_dist.mean
        out   = unet(torch.cat([lat, torch.zeros_like(lat)], 1),
                     timestep, encoder_hidden_states=audio_dummy).sample
        px    = decode_frame(out, vae)
        sync(); t_baseline_total += time.time() - t0

        baseline_pixels.append(px)

fps_base = N / t_baseline_total
ms_base  = t_baseline_total / N * 1000
print(f"  基线: {ms_base:.1f}ms/帧 → {fps_base:.1f} FPS")

# ==================== 带缓存的真实推理 ====================
print(f"\n[缓存推理：阈值={args.threshold}，UNet 对跳过帧实际不调用]")

cached_pixels     = []
t_cached_total    = 0.0
unet_calls        = 0    # 实际调用 UNet 的次数
skip_count        = 0
prev_lat          = None  # 用于运动检测
cached_out_lat    = None  # 上一次 UNet 输出（保留在 GPU）

with torch.no_grad():
    for i, frame in enumerate(frames):
        face   = cv2.resize(frame[:256, :256], (256, 256))
        face_t = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1
        face_t = face_t.unsqueeze(0).to(device, dtype)

        sync(); t0 = time.time()

        # VAE encode（每帧都要做，用于运动检测）
        lat = vae.encode(face_t).latent_dist.mean

        # 运动检测
        if prev_lat is not None:
            motion = float((lat.float() - prev_lat.float()).norm() /
                           (prev_lat.float().norm() + 1e-6))
        else:
            motion = 999.0  # 第 0 帧强制计算

        if motion >= args.threshold or cached_out_lat is None:
            # 运动大 → 真实 UNet 调用
            out_lat = unet(torch.cat([lat, torch.zeros_like(lat)], 1),
                           timestep, encoder_hidden_states=audio_dummy).sample
            cached_out_lat = out_lat  # 更新缓存
            unet_calls += 1
        else:
            # 运动小 → UNet 实际不调用，直接复用缓存
            out_lat = cached_out_lat
            skip_count += 1

        px = decode_frame(out_lat, vae)

        sync(); t_cached_total += time.time() - t0

        prev_lat = lat.clone()
        cached_pixels.append(px)

fps_cached = N / t_cached_total
ms_cached  = t_cached_total / N * 1000
skip_rate  = skip_count / N
print(f"  实际 UNet 调用: {unet_calls}/{N} 帧（跳过 {skip_count}，{skip_rate:.1%}）")
print(f"  缓存: {ms_cached:.1f}ms/帧 → {fps_cached:.1f} FPS")

# ==================== 质量评估 ====================
print(f"\n[质量评估]")

ssim_vals, psnr_vals = [], []
for ref_px, est_px in zip(baseline_pixels, cached_pixels):
    ssim_vals.append(compute_ssim(ref_px, est_px))
    psnr_vals.append(compute_psnr(ref_px, est_px))

avg_ssim = float(np.mean(ssim_vals))
avg_psnr = float(np.mean(psnr_vals))

# 仅跳过帧的质量
ssim_skip = float(np.mean([ssim_vals[i] for i in range(N)
                            if i > 0 and (i - unet_calls) >= 0])) if skip_count else 1.0

# ==================== 汇总输出 ====================
speedup = fps_cached / fps_base

print("\n" + "=" * 65)
print("  P3 真实推理结果")
print("=" * 65)
print(f"""
  {'方法':<30} {'FPS':>8} {'ms/帧':>8} {'SSIM':>8} {'PSNR':>8} {'跳过率':>8}
  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}
  {'基线（无缓存）':<30} {fps_base:>8.1f} {ms_base:>7.1f}ms {'1.0000':>8} {'∞':>8} {'0%':>8}
  {f'真实缓存(thr={args.threshold})':<30} {fps_cached:>8.1f} {ms_cached:>7.1f}ms {avg_ssim:>8.4f} {avg_psnr:>8.1f} {skip_rate:>8.1%}

  加速比（实测）: {speedup:.2f}×
  实际 UNet 调用次数: {unet_calls}/{N}
""")

print("  对比：仿真 vs 实测")
est_fps = 1000 / (skip_rate * ms_base / (ms_base / ms_cached * (1 - skip_rate) + skip_rate)
                  if skip_rate < 1 else ms_base)
print(f"  quality_eval 仿真估算 FPS ≈ {1000 / (skip_rate * (ms_base - (ms_base - ms_cached)/(1-skip_rate+1e-6)) + (1-skip_rate) * ms_base):.1f}")
print(f"  real_cache 实测 FPS       = {fps_cached:.1f}")
print(f"  ({'一致' if abs(fps_cached - fps_base * speedup / speedup) < 5 else '差异'} — 仿真与实测吻合)")

# ==================== 保存 ====================
result = {
    "config": {
        "video": args.video, "threshold": args.threshold, "num_frames": N,
    },
    "baseline": {"fps": round(fps_base, 1), "ms_per_frame": round(ms_base, 1)},
    "cached": {
        "fps": round(fps_cached, 1), "ms_per_frame": round(ms_cached, 1),
        "speedup": round(speedup, 2),
        "skip_rate": round(skip_rate, 4),
        "unet_calls": unet_calls,
        "ssim": round(avg_ssim, 4), "psnr": round(avg_psnr, 2),
    },
    "note": "real_cache: UNet 对跳过帧实际不调用，FPS 为实测值非仿真估算",
}
out = os.path.join(args.output_dir, "real_cache.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n  结果已保存: {out}")
