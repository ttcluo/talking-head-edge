"""
运动感知自适应帧跳过原型验证脚本
核心假设：相邻帧输入 latent 变化幅度低时，UNet 输出可复用，跳过计算
使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    python /path/to/step2/cache_prototype.py --video data/video/yongen.mp4

输出：
    - 多阈值扫描：跳过率 vs 质量损失 vs FPS 提升
    - 最优阈值推荐（FPS 提升最大且质量可接受）
    - 结果保存到 profile_results/cache_prototype.json
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

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default="data/video/yongen.mp4")
parser.add_argument("--num_frames", type=int, default=50)
parser.add_argument("--output_dir", type=str, default="profile_results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print("=" * 65)
print("  运动感知自适应帧跳过原型验证")
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

# ==================== 读取视频帧 ====================
print(f"\n[读取视频] {args.video}")
cap = cv2.VideoCapture(args.video)
frames = []
while len(frames) < args.num_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"  读取 {len(frames)} 帧")
N = len(frames)

# ==================== 基线计算（全帧运行 UNet）====================
print(f"\n[第一步] 计算基线：全部 {N} 帧运行 UNet")
timestep = torch.tensor([0.0], device=device, dtype=dtype)
audio_dummy = torch.zeros(1, 1, 384, device=device, dtype=dtype)

input_latents = []     # VAE 编码的输入 latent
baseline_outputs = []  # UNet 输出 latent（基线真值）
baseline_times = []    # 每帧计时

with torch.no_grad():
    # 预热
    dummy = torch.randn(1, 8, 32, 32, device=device, dtype=dtype)
    for _ in range(3):
        unet(dummy, timestep, encoder_hidden_states=audio_dummy)

    for i, frame in enumerate(frames):
        face = cv2.resize(frame[:256, :256], (256, 256))
        face_t = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1
        face_t = face_t.unsqueeze(0).to(device, dtype)

        latent = vae.encode(face_t).latent_dist.sample()
        mask = torch.zeros_like(latent)
        unet_input = torch.cat([latent, mask], dim=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = unet(unet_input, timestep, encoder_hidden_states=audio_dummy).sample
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        baseline_times.append(time.time() - t0)

        input_latents.append(latent.cpu().float())
        baseline_outputs.append(out.cpu().float())

baseline_fps = N / sum(baseline_times)
baseline_ms = np.mean(baseline_times) * 1000
print(f"  基线：{baseline_ms:.1f}ms/帧 → {baseline_fps:.1f} FPS")

# ==================== 计算帧间运动幅度 ====================
print(f"\n[第二步] 计算逐帧运动幅度")
motions = []
for i in range(1, N):
    prev = input_latents[i - 1]
    curr = input_latents[i]
    # 相对 L2 差异作为运动幅度
    motion = (curr - prev).norm() / (prev.norm() + 1e-8)
    motions.append(float(motion))

print(f"  运动幅度统计:")
print(f"    最小: {min(motions):.4f}  最大: {max(motions):.4f}")
print(f"    均值: {np.mean(motions):.4f}  中位: {np.median(motions):.4f}")

# ==================== 自适应跳过策略模拟 ====================
print(f"\n[第三步] 多阈值扫描：跳过率 vs 质量损失")

# 扫描的阈值范围
THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

def simulate_skip(threshold, strategy="reuse"):
    """
    模拟跳过策略，返回各帧的"估计输出"。
    strategy:
        "reuse"  — 直接复用上一个计算帧的输出
        "interp" — 在两个关键帧之间线性插值
    """
    estimated = []
    skip_flags = []           # True = 跳过了该帧
    last_computed_idx = 0
    last_computed_out = baseline_outputs[0]

    # 第 0 帧必须计算
    estimated.append(baseline_outputs[0])
    skip_flags.append(False)

    for i in range(1, N):
        motion = motions[i - 1]

        if motion < threshold:
            # 跳过：复用或插值
            if strategy == "reuse":
                estimated.append(last_computed_out.clone())
            else:
                # 插值：需要知道下一个关键帧，这里用当前最近的已知帧（单向）
                estimated.append(last_computed_out.clone())
            skip_flags.append(True)
        else:
            # 不跳过：使用真实 UNet 输出（基线值）
            last_computed_out = baseline_outputs[i]
            last_computed_idx = i
            estimated.append(baseline_outputs[i])
            skip_flags.append(False)

    return estimated, skip_flags


def compute_quality(estimated_outputs):
    """计算与基线的平均相对 L2 误差（latent 空间）"""
    errors = []
    for est, ref in zip(estimated_outputs, baseline_outputs):
        err = (est - ref).norm() / (ref.norm() + 1e-8)
        errors.append(float(err))
    return np.mean(errors), np.max(errors)


results = []
print(f"\n  {'阈值':>8} {'跳过率':>8} {'理论FPS':>9} {'FPS增益':>8} {'平均L2误差':>11} {'最大L2误差':>11} {'状态'}")
print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*11} {'-'*11} {'-'*8}")

# 基线（无跳过）
print(f"  {'基线':>8} {'0.0%':>8} {baseline_fps:>9.1f} {'1.00×':>8} {'0.0000':>11} {'0.0000':>11}")

QUALITY_THRESHOLD = 0.05  # 可接受的最大平均误差

for thr in THRESHOLDS:
    estimated, skip_flags = simulate_skip(thr, strategy="reuse")
    avg_err, max_err = compute_quality(estimated)

    skip_rate = sum(skip_flags) / N
    # 理论 FPS：假设跳过帧只有 1% 的开销（motion 检测）
    effective_compute_ratio = 1 - skip_rate * 0.99
    theory_fps = baseline_fps / effective_compute_ratio
    fps_gain = theory_fps / baseline_fps

    acceptable = "✓" if avg_err < QUALITY_THRESHOLD else "✗"

    print(f"  {thr:>8.3f} {skip_rate:>7.1%} {theory_fps:>9.1f} {fps_gain:>7.2f}× "
          f"{avg_err:>11.4f} {max_err:>11.4f} {acceptable}")

    results.append({
        "threshold": thr,
        "skip_rate": float(skip_rate),
        "theory_fps": float(theory_fps),
        "fps_gain": float(fps_gain),
        "avg_l2_err": float(avg_err),
        "max_l2_err": float(max_err),
        "acceptable": bool(avg_err < QUALITY_THRESHOLD),
    })

# ==================== 最优阈值 ====================
print("\n" + "=" * 65)
print("  最优阈值推荐")
print("=" * 65)

acceptable_results = [r for r in results if r["acceptable"]]
if acceptable_results:
    best = max(acceptable_results, key=lambda x: x["fps_gain"])
    print(f"""
  推荐阈值: {best['threshold']}
    跳过率:    {best['skip_rate']:.1%}（每 10 帧约跳过 {best['skip_rate']*10:.0f} 帧）
    理论 FPS:  {best['theory_fps']:.1f}（基线 {baseline_fps:.1f} → 提升 {best['fps_gain']:.2f}×）
    平均误差:  {best['avg_l2_err']:.4f}（< {QUALITY_THRESHOLD} 阈值，质量可接受）
    最大误差:  {best['max_l2_err']:.4f}
  """)
else:
    print("  ⚠ 所有阈值下质量损失均超过阈值，视频运动较剧烈")
    best = None

# ==================== 运动分布分析 ====================
print("=" * 65)
print("  帧间运动分布（指导阈值选择）")
print("=" * 65)

motion_arr = np.array(motions)
percentiles = [10, 25, 50, 75, 90, 95]
print(f"\n  百分位分析（运动幅度）:")
for p in percentiles:
    v = np.percentile(motion_arr, p)
    skip_at_this = (motion_arr < v).mean()
    print(f"    P{p:>2}: {v:.4f}  →  跳过率 {skip_at_this:.1%}")

# 静止帧（运动 < 0.01）占比
quiet_ratio = (motion_arr < 0.01).mean()
print(f"\n  运动 < 0.01（近静止帧）占比: {quiet_ratio:.1%}")
print(f"  运动 < 0.05（低运动帧）占比:  {(motion_arr < 0.05).mean():.1%}")

# ==================== 核心结论 ====================
print("\n" + "=" * 65)
print("  核心结论")
print("=" * 65)

if best and best["fps_gain"] > 1.2:
    print(f"""
  ✅ 运动感知帧跳过可行，推荐阈值 {best['threshold']}：
     → FPS: {baseline_fps:.1f} → {best['theory_fps']:.1f}（+{(best['fps_gain']-1)*100:.0f}%）
     → 质量损失（latent L2）: {best['avg_l2_err']:.4f}（需进一步测 SSIM/PSNR）

  创新点 C 实现路线：
    1. 在 UNet forward 入口计算 latent 变化幅度
    2. 低于阈值 → 直接返回缓存 output，跳过整个 UNet
    3. 可扩展：低运动区域用插值，高运动区域正常计算（空间选择性跳过）

  论文实验设计（基于此原型）：
    - 对比实验：MuseTalk FP16 vs MuseTalk + 自适应跳过
    - 质量指标：SSIM / PSNR / LSE-C（唇形同步误差）
    - 速度指标：真实 FPS（含 motion detection 开销）
    - 展示：运动幅度热力图 + 跳过帧可视化
""")
elif best:
    print(f"""
  ⚠ 帧跳过可行但提升有限（{best['fps_gain']:.2f}×）：
     该视频运动剧烈，跳过空间较少
     建议使用运动更少的视频（如静止主播）重新测试
""")
else:
    print("""
  ✗ 该视频运动过于剧烈，帧跳过质量损失过大
    → 建议直接推进创新点 B（EchoMimic 步数蒸馏）
""")

# ==================== 保存结果 ====================
output = {
    "video": args.video,
    "num_frames": N,
    "baseline_fps": float(baseline_fps),
    "baseline_ms_per_frame": float(baseline_ms),
    "motion_stats": {
        "mean": float(motion_arr.mean()),
        "median": float(np.median(motion_arr)),
        "min": float(motion_arr.min()),
        "max": float(motion_arr.max()),
        "quiet_ratio_001": float(quiet_ratio),
        "quiet_ratio_005": float((motion_arr < 0.05).mean()),
    },
    "threshold_sweep": results,
    "best_threshold": best,
}

out_file = os.path.join(args.output_dir, "cache_prototype.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  结果已保存到: {out_file}")
print(f"\n下一步：")
print(f"  → 用 SSIM 替换 latent L2，更准确评估视觉质量损失")
print(f"  → 实现真正的缓存注入（在 UNet forward 中增加 early return）")
print(f"  → 在 EchoMimic 上实现步间特征缓存（diffusion 步骤间复用）\n")
