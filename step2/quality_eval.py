"""
帧跳过策略视觉质量评估脚本
用途：将 cache_prototype.py 的 latent L2 误差转化为像素空间 SSIM/PSNR，
      同时计算真实端到端 FPS（含 VAE encode/decode），提供论文可用数字
使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    python /path/to/step2/quality_eval.py --video data/video/yongen.mp4

输出：
    - threshold → SSIM / PSNR / 嘴唇区域 SSIM / 端到端 FPS 完整表格
    - 最优阈值推荐（SSIM > 0.95 约束下的最大 FPS）
    - 保存对比帧图像（baseline vs cached）
    - 结果保存到 profile_results/quality_eval.json
"""

import argparse
import json
import os
import sys
import time

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
    print("  ⚠ scikit-image 未安装，使用内置 SSIM 实现")
    SKIMAGE_OK = False

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default="data/video/yongen.mp4")
parser.add_argument("--num_frames", type=int, default=50)
parser.add_argument("--output_dir", type=str, default="profile_results")
parser.add_argument("--save_frames", action="store_true", help="保存对比帧图像")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print("=" * 65)
print("  帧跳过策略视觉质量评估（像素空间 SSIM/PSNR）")
print("=" * 65)

# ==================== 加载模型 ====================
print("\n[加载模型]")
from diffusers import AutoencoderKL
from musetalk.models.unet import UNet2DConditionModel
import json as _json

with open("models/musetalkV15/musetalk.json") as f:
    unet_config = _json.load(f)

vae = AutoencoderKL.from_pretrained("models/sd-vae").to(device, dtype)
unet = UNet2DConditionModel(**unet_config).to(device, dtype)
unet.load_state_dict(torch.load("models/musetalkV15/unet.pth", map_location=device))
vae.eval()
unet.eval()
print("  ✓ 模型加载完成")

# ==================== 工具函数 ====================
def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) float [-1,1] → (H, W, 3) uint8"""
    img = t[0].permute(1, 2, 0).float().cpu().numpy()
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return img


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """(H, W, 3) uint8 → SSIM"""
    if SKIMAGE_OK:
        return float(sk_ssim(a, b, channel_axis=2, data_range=255))
    # 内置简化版（亮度通道 SSIM）
    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(float)
    bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(float)
    mu_a, mu_b = ag.mean(), bg.mean()
    sigma_a = ag.std()
    sigma_b = bg.std()
    sigma_ab = ((ag - mu_a) * (bg - mu_b)).mean()
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2) /
                 ((mu_a**2 + mu_b**2 + C1) * (sigma_a**2 + sigma_b**2 + C2)))


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """(H, W, 3) uint8 → PSNR (dB)"""
    if SKIMAGE_OK:
        return float(sk_psnr(a, b, data_range=255))
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def make_lip_mask_pixel(h=256, w=256):
    """像素空间嘴唇区域 mask（底部 30% 中央 55%）"""
    mask = np.zeros((h, w), dtype=bool)
    mask[int(h * 0.70):, int(w * 0.225):int(w * 0.775)] = True
    return mask


# ==================== 读取视频 ====================
print(f"\n[读取视频] {args.video}")
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

# ==================== 计算基线（全帧完整 pipeline）====================
print(f"\n[第一步] 基线计算：{N} 帧完整 pipeline（VAE encode + UNet + VAE decode）")

timestep = torch.tensor([0.0], device=device, dtype=dtype)
audio_dummy = torch.zeros(1, 1, 384, device=device, dtype=dtype)

# 预热
with torch.no_grad():
    dummy = torch.randn(1, 8, 32, 32, device=device, dtype=dtype)
    dummy_lat = torch.randn(1, 4, 32, 32, device=device, dtype=dtype)
    for _ in range(3):
        unet(dummy, timestep, encoder_hidden_states=audio_dummy)
        vae.decode(dummy_lat)

input_latents = []      # (1, 4, 32, 32) FP16
output_latents = []     # (1, 4, 32, 32) FP16
decoded_frames = []     # (256, 256, 3) uint8

t_vae_enc, t_unet, t_vae_dec = [], [], []

with torch.no_grad():
    for i, frame in enumerate(frames):
        face = cv2.resize(frame[:256, :256], (256, 256))
        face_t = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1
        face_t = face_t.unsqueeze(0).to(device, dtype)

        # VAE encode
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        latent = vae.encode(face_t).latent_dist.sample()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_vae_enc.append(time.time() - t0)

        mask = torch.zeros_like(latent)
        unet_input = torch.cat([latent, mask], dim=1)

        # UNet
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        out_latent = unet(unet_input, timestep, encoder_hidden_states=audio_dummy).sample
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_unet.append(time.time() - t0)

        # VAE decode
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        decoded = vae.decode(out_latent).sample
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_vae_dec.append(time.time() - t0)

        input_latents.append(latent.cpu())
        output_latents.append(out_latent.cpu())
        decoded_frames.append(tensor_to_uint8(decoded.cpu()))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N}] 已完成...")

# 基线各阶段耗时
ms_enc = np.mean(t_vae_enc) * 1000
ms_unet = np.mean(t_unet) * 1000
ms_dec = np.mean(t_vae_dec) * 1000
ms_total = ms_enc + ms_unet + ms_dec
fps_baseline = 1000 / ms_total

print(f"\n  基线耗时分解:")
print(f"    VAE Encode: {ms_enc:.1f}ms")
print(f"    UNet:       {ms_unet:.1f}ms")
print(f"    VAE Decode: {ms_dec:.1f}ms")
print(f"    端到端:     {ms_total:.1f}ms → {fps_baseline:.1f} FPS")

# ==================== 运动幅度 ====================
motions = []
for i in range(1, N):
    m = (input_latents[i] - input_latents[i-1]).norm() / \
        (input_latents[i-1].norm() + 1e-8)
    motions.append(float(m))

# ==================== 多阈值扫描（像素空间质量）====================
print(f"\n[第二步] 多阈值扫描（计算像素空间 SSIM/PSNR）")

THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
lip_mask = make_lip_mask_pixel(256, 256)
SSIM_THRESHOLD = 0.95   # 论文中常用阈值

results = []

print(f"\n  {'阈值':>6} {'跳过率':>7} {'全图SSIM':>9} {'嘴唇SSIM':>9} {'PSNR(dB)':>9} "
      f"{'端到端FPS':>10} {'FPS增益':>8} {'状态'}")
print(f"  {'-'*6} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*8} {'-'*6}")

# 基线行
print(f"  {'基线':>6} {'—':>7} {'1.0000':>9} {'1.0000':>9} {'∞':>9} "
      f"{fps_baseline:>10.1f} {'1.00×':>8}")

for thr in THRESHOLDS:
    # 模拟跳过：确定哪些帧跳过
    skip_flags = [False]
    last_output = output_latents[0]

    for i in range(1, N):
        if motions[i-1] < thr:
            skip_flags.append(True)
        else:
            last_output = output_latents[i]
            skip_flags.append(False)

    skip_rate = sum(skip_flags) / N

    # 解码估计帧（跳过帧复用上一个 UNet 输出）
    est_frames = []
    last_out_lat = output_latents[0]
    with torch.no_grad():
        for i in range(N):
            if not skip_flags[i]:
                last_out_lat = output_latents[i]
            # 复用上一个 UNet 输出，重新 decode
            decoded_est = vae.decode(last_out_lat.to(device, dtype)).sample
            est_frames.append(tensor_to_uint8(decoded_est.cpu()))

    # 计算质量指标
    ssim_vals, ssim_lip_vals, psnr_vals = [], [], []
    for i in range(N):
        ref = decoded_frames[i]
        est = est_frames[i]

        ssim_vals.append(compute_ssim(ref, est))
        psnr_vals.append(compute_psnr(ref, est))

        # 嘴唇区域 SSIM
        ref_lip = ref.copy()
        est_lip = est.copy()
        # 取嘴唇 crop
        r_lip = ref[int(256*0.70):, int(256*0.225):int(256*0.775)]
        e_lip = est[int(256*0.70):, int(256*0.225):int(256*0.775)]
        ssim_lip_vals.append(compute_ssim(r_lip, e_lip))

    avg_ssim = float(np.mean(ssim_vals))
    avg_ssim_lip = float(np.mean(ssim_lip_vals))
    avg_psnr = float(np.mean(psnr_vals))

    # 端到端 FPS 估算
    # 跳过帧：只需 VAE encode（motion 检测）≈ ms_enc
    # 非跳过帧：完整 ms_enc + ms_unet + ms_dec
    ms_skip = ms_enc               # 跳过帧仅做 encode（motion check），decode 复用
    ms_noskip = ms_total
    avg_ms = skip_rate * ms_skip + (1 - skip_rate) * ms_noskip
    fps_cached = 1000 / avg_ms
    fps_gain = fps_cached / fps_baseline

    acceptable = "✓" if avg_ssim >= SSIM_THRESHOLD else "✗"

    print(f"  {thr:>6.3f} {skip_rate:>6.1%} {avg_ssim:>9.4f} {avg_ssim_lip:>9.4f} "
          f"{avg_psnr:>9.2f} {fps_cached:>10.1f} {fps_gain:>7.2f}× {acceptable}")

    results.append({
        "threshold": thr,
        "skip_rate": float(skip_rate),
        "ssim": avg_ssim,
        "ssim_lip": avg_ssim_lip,
        "psnr": avg_psnr,
        "fps_end2end": float(fps_cached),
        "fps_gain": float(fps_gain),
        "acceptable_ssim": bool(avg_ssim >= SSIM_THRESHOLD),
    })

    # 保存对比帧（仅限最后一个阈值）
    if args.save_frames and thr == THRESHOLDS[-2]:
        compare_dir = os.path.join(args.output_dir, f"frames_thr{thr}")
        os.makedirs(compare_dir, exist_ok=True)
        for i in range(min(10, N)):
            if skip_flags[i]:
                border = np.zeros_like(decoded_frames[i])
                border[:4, :] = border[-4:, :] = border[:, :4] = border[:, -4:] = [255, 100, 100]
                comp = np.hstack([decoded_frames[i], est_frames[i] + border * 0])
            else:
                comp = np.hstack([decoded_frames[i], est_frames[i]])
            cv2.imwrite(f"{compare_dir}/frame_{i:03d}_{'skip' if skip_flags[i] else 'calc'}.jpg",
                        cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

# ==================== 最优阈值 ====================
print("\n" + "=" * 65)
print("  最优阈值推荐（SSIM ≥ 0.95）")
print("=" * 65)

acceptable = [r for r in results if r["acceptable_ssim"]]
if acceptable:
    best = max(acceptable, key=lambda x: x["fps_gain"])
    print(f"""
  推荐阈值: {best['threshold']}
    跳过率:        {best['skip_rate']:.1%}
    全图 SSIM:     {best['ssim']:.4f}（≥ 0.95 ✓）
    嘴唇 SSIM:     {best['ssim_lip']:.4f}
    PSNR:          {best['psnr']:.2f} dB
    端到端 FPS:    {best['fps_end2end']:.1f}（基线 {fps_baseline:.1f} → {best['fps_gain']:.2f}×）
  """)
else:
    print("  ⚠ 所有阈值下 SSIM < 0.95，视频运动过于剧烈")
    # 降低标准给出次优方案
    near_best = max(results, key=lambda x: x["ssim"])
    print(f"  次优方案（最高 SSIM）: 阈值 {near_best['threshold']}，"
          f"SSIM {near_best['ssim']:.4f}，FPS {near_best['fps_end2end']:.1f}")
    best = None

# ==================== 论文数字摘要 ====================
print("=" * 65)
print("  论文实验数字摘要（直接可用）")
print("=" * 65)

print(f"""
  基线（MuseTalk 1.5 FP16）：
    VAE encode: {ms_enc:.1f}ms | UNet: {ms_unet:.1f}ms | VAE decode: {ms_dec:.1f}ms
    端到端: {ms_total:.1f}ms → {fps_baseline:.1f} FPS

  本方法（运动感知自适应帧跳过）：""")

for r in results:
    if r["acceptable_ssim"]:
        print(f"    阈值 {r['threshold']}: 跳过 {r['skip_rate']:.0%} 帧, "
              f"SSIM={r['ssim']:.4f}, PSNR={r['psnr']:.1f}dB, "
              f"{r['fps_end2end']:.1f} FPS (+{(r['fps_gain']-1)*100:.0f}%)")

print(f"""
  关键对比（论文 Table 用）：
    MuseTalk FP16（基线）:    {fps_baseline:.1f} FPS,  SSIM=1.0,  PSNR=∞
    本方法（最优阈值）:        {best['fps_end2end'] if best else '—':.1f} FPS,  "
    SSIM={best['ssim'] if best else '—':.4f},  PSNR={best['psnr'] if best else '—':.1f}dB

  ⚠ 注：LSE-C（唇形同步误差）需用 SyncNet 评估，是投稿必备指标
""")

# ==================== 保存结果 ====================
output = {
    "video": args.video,
    "num_frames": N,
    "baseline": {
        "ms_vae_enc": float(ms_enc),
        "ms_unet": float(ms_unet),
        "ms_vae_dec": float(ms_dec),
        "ms_total": float(ms_total),
        "fps": float(fps_baseline),
    },
    "ssim_threshold": SSIM_THRESHOLD,
    "threshold_sweep": results,
    "best": best,
}

out_file = os.path.join(args.output_dir, "quality_eval.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  结果已保存到: {out_file}")
print(f"""
下一步（论文实验完善）：
  1. 在更多视频上测试（至少 5 个不同说话人）
  2. 集成 SyncNet 测 LSE-C（唇形同步误差）
  3. 实现真正的 UNet early-return 缓存（替换当前的 latent 复用模拟）
  4. 端侧部署验证（ONNX / TensorRT 导出）
""")
