"""
UNet 中间层特征时序相似度分析脚本
用途：验证「背景区域特征在相邻帧间高度相似，可缓存复用」的假设（创新点 C 的实验依据）
使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    python /path/to/step2/feature_similarity.py --video data/video/yongen.mp4

输出：
    - 各层特征的帧间余弦相似度（全图 / 背景区域 / 嘴唇区域）
    - 相似度随时间的变化曲线（保存为图片）
    - 缓存潜力评估：哪些层/哪些区域最适合缓存
    - 结果保存到 profile_results/feature_similarity.json
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default="data/video/yongen.mp4")
parser.add_argument("--num_frames", type=int, default=50)
parser.add_argument("--output_dir", type=str, default="profile_results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print("=" * 65)
print("  UNet 中间层特征时序相似度分析")
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
print(f"  ✓ 模型加载完成")

# ==================== Hook 注册 ====================
# 在以下关键位置采样中间特征
HOOK_TARGETS = {
    "down_0_out":  "down_blocks.0",      # 浅层，结构信息
    "down_1_out":  "down_blocks.1",      # 中层
    "down_2_out":  "down_blocks.2",      # 中层
    "mid_out":     "mid_block",          # 瓶颈层，语义信息
    "up_1_out":    "up_blocks.1",        # 中层
    "up_2_out":    "up_blocks.2",        # 中层
    "up_3_out":    "up_blocks.3",        # 深层，细节信息（最敏感层所在）
}

feature_cache = {}
hooks = []

def make_hook(layer_name):
    def hook_fn(module, input, output):
        # output 可能是 tuple（如 down_blocks 返回 (hidden, res_samples)）
        feat = output[0] if isinstance(output, tuple) else output
        feature_cache[layer_name] = feat.detach().float()
    return hook_fn

for layer_name, module_path in HOOK_TARGETS.items():
    parts = module_path.split(".")
    module = unet
    for p in parts:
        module = getattr(module, p)
    hooks.append(module.register_forward_hook(make_hook(layer_name)))

print(f"  ✓ 注册 {len(hooks)} 个特征 Hook")

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

# ==================== 构建嘴唇 Mask ====================
def build_lip_mask(latent_h, latent_w):
    """
    在 latent 空间构造嘴唇区域 mask（底部 1/4 区域中央）
    与 profile_musetalk.py 保持一致：原图 [180:256, 60:196] → latent [23:32, 8:25]
    """
    mask = torch.zeros(latent_h, latent_w)
    lip_h_start = int(latent_h * 0.70)
    lip_w_start = int(latent_w * 0.23)
    lip_w_end   = int(latent_w * 0.77)
    mask[lip_h_start:, lip_w_start:lip_w_end] = 1.0
    return mask  # (H, W)

# ==================== 逐帧特征提取 ====================
print(f"\n[提取各帧 UNet 中间层特征]")
print(f"  (仅需 UNet 前向，无需完整推理)")

timestep = torch.tensor([0.0], device=device, dtype=dtype)
audio_dummy = torch.zeros(1, 1, 384, device=device, dtype=dtype)
in_channels = unet.conv_in.in_channels

all_features = {k: [] for k in HOOK_TARGETS}  # layer_name → [T, C, H, W]
all_latents = []

with torch.no_grad():
    for i, frame in enumerate(frames):
        face_crop = cv2.resize(frame[:256, :256], (256, 256))
        face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 127.5 - 1
        face_tensor = face_tensor.unsqueeze(0).to(device, dtype)

        latent = vae.encode(face_tensor).latent_dist.sample()         # (1, 4, 32, 32)
        mask = torch.zeros_like(latent)                                # (1, 4, 32, 32)
        unet_input = torch.cat([latent, mask], dim=1)                  # (1, 8, 32, 32)

        _ = unet(unet_input, timestep, encoder_hidden_states=audio_dummy).sample

        all_latents.append(latent.cpu().float())
        for k in HOOK_TARGETS:
            all_features[k].append(feature_cache[k].cpu())

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] 已提取...")

# 释放 Hook
for h in hooks:
    h.remove()

print(f"  ✓ 特征提取完成")

# 打印各层特征维度
print("\n  各层特征形状:")
for k in HOOK_TARGETS:
    shape = all_features[k][0].shape
    print(f"    {k:<15}: {tuple(shape)}")

# ==================== 相似度计算 ====================
print(f"\n[计算帧间余弦相似度]")


def cosine_sim_spatial(feat_a: torch.Tensor, feat_b: torch.Tensor,
                        mask: torch.Tensor = None) -> float:
    """
    计算两帧特征图的余弦相似度。
    feat: (C, H, W)，mask: (H, W) 0/1 mask，仅计算 mask=1 的 spatial 位置
    """
    a = feat_a.flatten(1)  # (C, H*W)
    b = feat_b.flatten(1)

    if mask is not None:
        idx = mask.flatten().bool()
        a = a[:, idx]
        b = b[:, idx]

    # 逐 spatial 位置计算余弦相似度，再平均
    a_norm = F.normalize(a, dim=0)
    b_norm = F.normalize(b, dim=0)
    sim = (a_norm * b_norm).sum(dim=0).mean().item()
    return sim


results = {}

for layer_name in HOOK_TARGETS:
    feats = all_features[layer_name]  # list of (1, C, H, W)
    C, H, W = feats[0].shape[1], feats[0].shape[2], feats[0].shape[3]

    lip_mask = build_lip_mask(H, W)        # (H, W)
    bg_mask = 1.0 - lip_mask               # (H, W)

    sims_full, sims_lip, sims_bg = [], [], []

    for i in range(len(feats) - 1):
        a = feats[i][0]    # (C, H, W)
        b = feats[i + 1][0]

        sims_full.append(cosine_sim_spatial(a, b))
        sims_lip.append(cosine_sim_spatial(a, b, lip_mask))
        sims_bg.append(cosine_sim_spatial(a, b, bg_mask))

    results[layer_name] = {
        "shape": [C, H, W],
        "sim_full_mean":   float(np.mean(sims_full)),
        "sim_full_std":    float(np.std(sims_full)),
        "sim_lip_mean":    float(np.mean(sims_lip)),
        "sim_lip_std":     float(np.std(sims_lip)),
        "sim_bg_mean":     float(np.mean(sims_bg)),
        "sim_bg_std":      float(np.std(sims_bg)),
        "bg_lip_gap":      float(np.mean(sims_bg) - np.mean(sims_lip)),
        "sims_full":       [float(x) for x in sims_full],
        "sims_bg":         [float(x) for x in sims_bg],
        "sims_lip":        [float(x) for x in sims_lip],
    }

# ==================== 输出结果表 ====================
print("\n" + "=" * 75)
print("  各层帧间余弦相似度（1.0 = 完全相同，可安全缓存）")
print("=" * 75)
print(f"\n  {'层名称':<16} {'全图':>8} {'背景':>8} {'嘴唇':>8} {'背景-嘴唇差':>12} {'尺寸'}")
print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*12}")

for k, r in results.items():
    gap = r["bg_lip_gap"]
    bar = "▲" * int(gap * 100)  # 差距越大说明背景和嘴唇分布越不同
    C, H, W = r["shape"]
    print(f"  {k:<16} {r['sim_full_mean']:>7.4f}  {r['sim_bg_mean']:>7.4f}  "
          f"{r['sim_lip_mean']:>7.4f}  {gap:>+11.4f}  {C}×{H}×{W}  {bar}")

# ==================== 缓存潜力评估 ====================
print("\n" + "=" * 75)
print("  缓存潜力评估")
print("=" * 75)

# 缓存判断标准：背景相似度 > 0.99 且 背景-嘴唇差 > 0.005
CACHE_SIM_THRESHOLD = 0.99
CACHE_GAP_THRESHOLD = 0.005

cacheable_layers = []
for k, r in results.items():
    if r["sim_bg_mean"] > CACHE_SIM_THRESHOLD and r["bg_lip_gap"] > CACHE_GAP_THRESHOLD:
        cacheable_layers.append(k)

print(f"\n  可缓存层（背景相似度 > {CACHE_SIM_THRESHOLD}，背景-嘴唇差 > {CACHE_GAP_THRESHOLD}）：")
if cacheable_layers:
    for k in cacheable_layers:
        r = results[k]
        print(f"    ✓ {k:<16}  背景: {r['sim_bg_mean']:.4f}  嘴唇: {r['sim_lip_mean']:.4f}  差距: {r['bg_lip_gap']:+.4f}")
else:
    print("    (无满足阈值的层，可放宽阈值再分析)")

# 背景-嘴唇差最大的层（最值得缓存）
top_gap = sorted(results.items(), key=lambda x: -x[1]["bg_lip_gap"])
print(f"\n  背景-嘴唇差 TOP-5（差距越大说明区域感知缓存收益越高）：")
for k, r in top_gap[:5]:
    print(f"    {k:<16}  差距: {r['bg_lip_gap']:+.4f}  "
          f"背景: {r['sim_bg_mean']:.4f}  嘴唇: {r['sim_lip_mean']:.4f}")

# ==================== 核心结论 ====================
print("\n" + "=" * 75)
print("  核心结论与创新点 C 支撑")
print("=" * 75)

avg_bg_sim = np.mean([r["sim_bg_mean"] for r in results.values()])
avg_lip_sim = np.mean([r["sim_lip_mean"] for r in results.values()])
avg_gap = avg_bg_sim - avg_lip_sim

print(f"""
  全层平均背景相似度: {avg_bg_sim:.4f}
  全层平均嘴唇相似度: {avg_lip_sim:.4f}
  平均差距:          {avg_gap:+.4f}
""")

if avg_gap > 0.01:
    print("  ✅ 假设验证通过：背景区域特征帧间相似度显著高于嘴唇区域")
    print("  → 支持创新点 C：时序感知特征缓存")
    print("  → 可在背景相似度高的层复用上帧特征，跳过计算")
    print(f"  → 理论计算节省：{len(cacheable_layers)}/{len(HOOK_TARGETS)} 层可缓存（背景区域）")
else:
    print("  ⚠ 背景-嘴唇差距较小，缓存收益可能有限")
    print("  → 建议聚焦创新点 B（步数蒸馏）")

# ==================== 保存结果 ====================
output = {
    "video": args.video,
    "num_frames": len(frames),
    "layers": results,
    "summary": {
        "avg_bg_sim": float(avg_bg_sim),
        "avg_lip_sim": float(avg_lip_sim),
        "avg_gap": float(avg_gap),
        "cacheable_layers": cacheable_layers,
        "hypothesis_c_supported": bool(avg_gap > 0.01),
    }
}

out_file = os.path.join(args.output_dir, "feature_similarity.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n  结果已保存到: {out_file}")
print(f"""
下一步：
  - 若假设通过 → 实现原型：在 UNet forward 中注入缓存机制，测实际加速
  - 若假设未通过 → 直接进入创新点 B（EchoMimic 步数蒸馏）
""")
