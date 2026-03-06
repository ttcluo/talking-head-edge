"""
真实 UNet Early-Return 推理（P3 / P3+）
区别于 cache_prototype / quality_eval 的"仿真"：
  本脚本在推理时真正跳过 UNet 调用，测量实际 wall-clock FPS，不依赖任何事后模拟。

P3  (latent 缓存)：跳过帧复用 cached_out_lat，仍需 VAE decode (~16ms)
P3+ (像素帧缓存)：跳过帧直接复用已解码的 numpy 像素，零 GPU 开销

使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    python /path/to/step2/real_cache.py --video data/video/yongen.mp4

输出：
    - 基线 / P3 / P3+ 三路 FPS 及质量对比
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
print(f"  P3: {ms_cached:.1f}ms/帧 → {fps_cached:.1f} FPS")

# ==================== P3+：像素帧缓存（跳过帧零 GPU 开销）====================
print(f"\n[P3+ 像素帧缓存：阈值={args.threshold}，跳过帧直接复用解码像素，不调 VAE decode]")

pixcache_pixels  = []
t_pixcache_total = 0.0
unet_calls_pc    = 0
skip_count_pc    = 0
prev_lat_pc      = None
cached_pixel     = None  # 缓存的已解码像素帧（CPU numpy，零 GPU 开销）

with torch.no_grad():
    for i, frame in enumerate(frames):
        face   = cv2.resize(frame[:256, :256], (256, 256))
        face_t = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1
        face_t = face_t.unsqueeze(0).to(device, dtype)

        sync(); t0 = time.time()

        lat = vae.encode(face_t).latent_dist.mean

        if prev_lat_pc is not None:
            motion = float((lat.float() - prev_lat_pc.float()).norm() /
                           (prev_lat_pc.float().norm() + 1e-6))
        else:
            motion = 999.0

        if motion >= args.threshold or cached_pixel is None:
            out_lat = unet(torch.cat([lat, torch.zeros_like(lat)], 1),
                           timestep, encoder_hidden_states=audio_dummy).sample
            px = decode_frame(out_lat, vae)
            cached_pixel = px  # 缓存已解码像素，下次跳帧直接复用
            unet_calls_pc += 1
        else:
            # 跳过帧：直接复用上一次解码结果，无任何 GPU 操作
            px = cached_pixel
            skip_count_pc += 1

        sync(); t_pixcache_total += time.time() - t0

        prev_lat_pc = lat.clone()
        pixcache_pixels.append(px)

fps_pixcache = N / t_pixcache_total
ms_pixcache  = t_pixcache_total / N * 1000
skip_rate_pc = skip_count_pc / N
print(f"  实际 UNet 调用: {unet_calls_pc}/{N} 帧（跳过 {skip_count_pc}，{skip_rate_pc:.1%}）")
print(f"  P3+: {ms_pixcache:.1f}ms/帧 → {fps_pixcache:.1f} FPS")

# ==================== 质量评估 ====================
print(f"\n[质量评估]")

ssim_p3, psnr_p3   = [], []
ssim_pc, psnr_pc   = [], []
for ref, p3, pc in zip(baseline_pixels, cached_pixels, pixcache_pixels):
    ssim_p3.append(compute_ssim(ref, p3))
    psnr_p3.append(compute_psnr(ref, p3))
    ssim_pc.append(compute_ssim(ref, pc))
    psnr_pc.append(compute_psnr(ref, pc))

avg_ssim_p3 = float(np.mean(ssim_p3))
avg_psnr_p3 = float(np.mean(psnr_p3))
avg_ssim_pc = float(np.mean(ssim_pc))
avg_psnr_pc = float(np.mean(psnr_pc))

# ==================== 汇总输出 ====================
speedup_p3 = fps_cached    / fps_base
speedup_pc = fps_pixcache  / fps_base

# 仿真估算（quality_eval 模型）：跳帧仅计 VAE encode 开销
ms_enc_est = ms_cached - (1 - skip_rate) * ms_base / (1 - skip_rate + 1e-9)
sim_fps = 1000 / ((1 - skip_rate) * ms_base + skip_rate * max(ms_enc_est, 1.0)) \
          if skip_rate < 1 else fps_base

print("\n" + "=" * 70)
print("  P3 / P3+ 真实推理结果对比")
print("=" * 70)
print(f"""
  {'方法':<32} {'FPS':>7} {'ms/帧':>7} {'加速比':>7} {'SSIM':>8} {'PSNR':>7} {'跳过率':>7}
  {'-'*32} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7}
  {'基线（无缓存）':<32} {fps_base:>7.1f} {ms_base:>6.1f}ms {'1.00×':>7} {'1.0000':>8} {'∞':>7} {'0%':>7}
  {f'P3  latent缓存 thr={args.threshold}':<32} {fps_cached:>7.1f} {ms_cached:>6.1f}ms {speedup_p3:>6.2f}× {avg_ssim_p3:>8.4f} {avg_psnr_p3:>7.1f} {skip_rate:>7.1%}
  {f'P3+ 像素缓存  thr={args.threshold}':<32} {fps_pixcache:>7.1f} {ms_pixcache:>6.1f}ms {speedup_pc:>6.2f}× {avg_ssim_pc:>8.4f} {avg_psnr_pc:>7.1f} {skip_rate_pc:>7.1%}

  P3+ vs P3 FPS 提升: +{fps_pixcache - fps_cached:.1f} FPS（跳帧去掉 VAE decode 节省约 {ms_cached - ms_pixcache:.1f}ms/帧均值）
  P3+ 实际 UNet 调用: {unet_calls_pc}/{N}
""")

# ==================== 保存 ====================
result = {
    "config": {
        "video": args.video, "threshold": args.threshold, "num_frames": N,
    },
    "baseline": {
        "fps": round(fps_base, 1), "ms_per_frame": round(ms_base, 1),
    },
    "p3_latent_cache": {
        "fps": round(fps_cached, 1), "ms_per_frame": round(ms_cached, 1),
        "speedup": round(speedup_p3, 2), "skip_rate": round(skip_rate, 4),
        "unet_calls": unet_calls,
        "ssim": round(avg_ssim_p3, 4), "psnr": round(avg_psnr_p3, 2),
        "note": "跳帧仍需 VAE decode cached latent",
    },
    "p3plus_pixel_cache": {
        "fps": round(fps_pixcache, 1), "ms_per_frame": round(ms_pixcache, 1),
        "speedup": round(speedup_pc, 2), "skip_rate": round(skip_rate_pc, 4),
        "unet_calls": unet_calls_pc,
        "ssim": round(avg_ssim_pc, 4), "psnr": round(avg_psnr_pc, 2),
        "note": "跳帧直接复用 numpy 像素帧，零 GPU 开销",
    },
}
out = os.path.join(args.output_dir, "real_cache.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"  结果已保存: {out}")
