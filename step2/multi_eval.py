"""
多视频批量评估脚本（P1）
用途：在 ≥5 个不同说话人视频上验证帧跳过策略的泛化性，
      输出 SSIM / PSNR / FPS / 跳过率汇总表（论文 Table 使用）
使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    # 方式 A：指定目录（自动搜索 *.mp4）
    python /path/to/step2/multi_eval.py --video_dir data/video/
    # 方式 B：指定文件列表
    python /path/to/step2/multi_eval.py --videos data/video/a.mp4 data/video/b.mp4 ...

输出：
    - 每个视频的 skip_rate / SSIM / lip_SSIM / PSNR / FPS
    - 均值 ± 标准差汇总行
    - profile_results/multi_eval.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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
parser.add_argument("--videos",     nargs="+", default=[])
parser.add_argument("--video_dir",  type=str,  default="")
parser.add_argument("--threshold",  type=float, default=0.15,
                    help="帧跳过阈值（推荐 0.15）")
parser.add_argument("--num_frames", type=int,  default=100,
                    help="每个视频取前 N 帧（越多越准，建议 ≥100）")
parser.add_argument("--output_dir", type=str,  default="profile_results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16

# ==================== 收集视频列表 ====================
video_list = list(args.videos)
if args.video_dir:
    video_list += sorted(str(p) for p in Path(args.video_dir).glob("*.mp4"))
if not video_list:
    parser.error("请通过 --videos 或 --video_dir 提供视频文件")
print(f"共找到 {len(video_list)} 个视频: {[os.path.basename(v) for v in video_list]}")

# ==================== 加载模型（只加载一次）====================
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
print("  ✓ VAE + UNet 加载完成")

# 零音频占位（multi_eval 只评估视觉质量，音频不影响 SSIM/PSNR 相对比较）
audio_dummy = torch.zeros(1, 1, 384, device=device, dtype=dtype)
timestep    = torch.tensor([0.0], device=device, dtype=dtype)

# ==================== 工具函数 ====================
def tensor_to_uint8(t):
    arr = ((t[0].permute(1, 2, 0).float().numpy() + 1) * 127.5).clip(0, 255)
    return arr.astype(np.uint8)

def compute_ssim(a, b):
    if SKIMAGE_OK:
        return float(sk_ssim(a, b, channel_axis=2, data_range=255))
    diff = a.astype(float) - b.astype(float)
    return float(1 - (diff ** 2).mean() / (255 ** 2))

def compute_psnr(a, b):
    if SKIMAGE_OK:
        v = float(sk_psnr(a, b, data_range=255))
        return v if np.isfinite(v) else 100.0
    mse = ((a.astype(float) - b.astype(float)) ** 2).mean()
    return float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else 100.0

def make_lip_mask(h=256, w=256):
    mask = np.zeros((h, w), dtype=bool)
    mask[int(h * 0.70):, int(w * 0.225):int(w * 0.775)] = True
    return mask

LIP_MASK = make_lip_mask()

# ==================== 单视频评估 ====================
def eval_video(video_path: str, threshold: float, num_frames: int) -> dict | None:
    name = os.path.basename(video_path)

    # 读帧
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    N = len(frames)
    if N < 5:
        print(f"  ⚠ {name}: 帧数不足（{N}），跳过")
        return None
    print(f"\n  [{name}] {N} 帧")

    # 基线推理
    input_latents, output_latents, decoded_frames = [], [], []
    t_enc, t_unet, t_dec = [], [], []

    with torch.no_grad():
        for frame in frames:
            face   = cv2.resize(frame[:256, :256], (256, 256))
            face_t = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1
            face_t = face_t.unsqueeze(0).to(device, dtype)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            lat = vae.encode(face_t).latent_dist.mean
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_enc.append(time.time() - t0)

            unet_in = torch.cat([lat, torch.zeros_like(lat)], dim=1)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            out = unet(unet_in, timestep, encoder_hidden_states=audio_dummy).sample
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_unet.append(time.time() - t0)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            dec = vae.decode(out).sample
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_dec.append(time.time() - t0)

            input_latents.append(lat.detach().cpu().float())
            output_latents.append(out.detach().cpu())
            decoded_frames.append(tensor_to_uint8(dec.cpu()))

    ms_enc  = np.mean(t_enc)  * 1000
    ms_unet = np.mean(t_unet) * 1000
    ms_dec  = np.mean(t_dec)  * 1000
    fps_base = 1000 / (ms_enc + ms_unet + ms_dec)

    # 运动量（与 quality_eval.py 一致）
    motions = [
        float((input_latents[i] - input_latents[i-1]).norm() /
              (input_latents[i-1].norm() + 1e-6))
        for i in range(1, N)
    ]

    # 模拟帧跳过
    skip_flags = [False]
    for i in range(1, N):
        skip_flags.append(motions[i-1] < threshold)
    skip_rate = sum(skip_flags) / N

    # FPS 估算（跳过帧只需 VAE encode for motion check）
    ms_skip_frame = ms_enc
    ms_full_frame = ms_enc + ms_unet + ms_dec
    fps_cached = 1000 / (
        skip_rate * ms_skip_frame + (1 - skip_rate) * ms_full_frame
    )

    # 像素空间质量
    last_out = output_latents[0]
    ssim_vals, psnr_vals, lip_ssim_vals = [], [], []
    with torch.no_grad():
        for i in range(N):
            if not skip_flags[i]:
                last_out = output_latents[i]
            est = vae.decode(last_out.to(device, dtype)).sample
            est_px = tensor_to_uint8(est.cpu())
            ref_px = decoded_frames[i]

            ssim_vals.append(compute_ssim(ref_px, est_px))
            psnr_vals.append(compute_psnr(ref_px, est_px))
            r_lip = ref_px[LIP_MASK].reshape(-1, 3)
            e_lip = est_px[LIP_MASK].reshape(-1, 3)
            # 嘴唇区域 SSIM 用小 patch 计算
            rh, rw = int(256 * 0.30), int(256 * 0.55)
            r_crop = ref_px[int(256*0.70):, int(256*0.225):int(256*0.775)]
            e_crop = est_px[int(256*0.70):, int(256*0.225):int(256*0.775)]
            lip_ssim_vals.append(compute_ssim(r_crop, e_crop))

    result = {
        "video":        name,
        "num_frames":   N,
        "skip_rate":    round(skip_rate, 4),
        "fps_base":     round(fps_base, 1),
        "fps_cached":   round(fps_cached, 1),
        "speedup":      round(fps_cached / fps_base, 2),
        "ssim":         round(float(np.mean(ssim_vals)), 4),
        "ssim_lip":     round(float(np.mean(lip_ssim_vals)), 4),
        "psnr":         round(float(np.mean(psnr_vals)), 2),
        "motion_mean":  round(float(np.mean(motions)), 4),
        "motion_median":round(float(np.median(motions)), 4),
    }
    print(f"    跳过率={result['skip_rate']:.1%}  SSIM={result['ssim']:.4f}  "
          f"PSNR={result['psnr']:.1f}dB  FPS {result['fps_base']:.1f}→{result['fps_cached']:.1f} "
          f"({result['speedup']:.2f}×)")
    return result

# ==================== 批量运行 ====================
print("\n" + "=" * 70)
print("  多视频帧跳过评估（阈值 = %.2f）" % args.threshold)
print("=" * 70)

# 预热
with torch.no_grad():
    _d = torch.randn(1, 8, 32, 32, device=device, dtype=dtype)
    for _ in range(3):
        unet(_d, timestep, encoder_hidden_states=audio_dummy)
        vae.decode(torch.randn(1, 4, 32, 32, device=device, dtype=dtype))

all_results = []
for vp in video_list:
    r = eval_video(vp, args.threshold, args.num_frames)
    if r is not None:
        all_results.append(r)

if not all_results:
    print("没有有效结果，退出")
    sys.exit(1)

# ==================== 汇总表 ====================
print("\n" + "=" * 70)
print("  汇总结果（论文 Table 用）")
print("=" * 70)

hdr = f"  {'视频':<22} {'跳过率':>7} {'SSIM':>7} {'唇SSIM':>8} {'PSNR':>8} {'基线FPS':>8} {'缓存FPS':>8} {'加速':>6}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in all_results:
    print(f"  {r['video']:<22} {r['skip_rate']:>7.1%} {r['ssim']:>7.4f} "
          f"{r['ssim_lip']:>8.4f} {r['psnr']:>8.1f} "
          f"{r['fps_base']:>8.1f} {r['fps_cached']:>8.1f} {r['speedup']:>5.2f}×")

# 均值 ± 标准差
keys = ["skip_rate", "ssim", "ssim_lip", "psnr", "fps_base", "fps_cached", "speedup"]
means = {k: float(np.mean([r[k] for r in all_results])) for k in keys}
stds  = {k: float(np.std( [r[k] for r in all_results])) for k in keys}

print("  " + "-" * (len(hdr) - 2))
print(f"  {'均值':<22} {means['skip_rate']:>7.1%} {means['ssim']:>7.4f} "
      f"{means['ssim_lip']:>8.4f} {means['psnr']:>8.1f} "
      f"{means['fps_base']:>8.1f} {means['fps_cached']:>8.1f} {means['speedup']:>5.2f}×")
print(f"  {'标准差':<22} {stds['skip_rate']:>7.1%} {stds['ssim']:>7.4f} "
      f"{stds['ssim_lip']:>8.4f} {stds['psnr']:>8.1f} "
      f"{stds['fps_base']:>8.1f} {stds['fps_cached']:>8.1f} {stds['speedup']:>5.2f}×")

print(f"\n  视频数量: {len(all_results)}")
print(f"  阈值:     {args.threshold}")
print(f"\n  论文结论: 平均加速 {means['speedup']:.2f}×，SSIM={means['ssim']:.4f}，"
      f"PSNR={means['psnr']:.1f}dB，跳过率={means['skip_rate']:.1%}")

# ==================== 保存结果 ====================
out = {
    "threshold":  args.threshold,
    "num_videos": len(all_results),
    "per_video":  all_results,
    "mean":       {k: round(means[k], 4) for k in keys},
    "std":        {k: round(stds[k],  4) for k in keys},
}
out_path = os.path.join(args.output_dir, "multi_eval.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n  结果已保存: {out_path}")
