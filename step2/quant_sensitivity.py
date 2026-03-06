"""
MuseTalk UNet 量化敏感度分析脚本
用途：通过模拟量化（PTQ），识别哪些层对 INT8/INT4 量化最敏感，为区域感知混合精度量化提供实验依据
使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    python /path/to/step2/quant_sensitivity.py

输出：
    - 各层量化敏感度排名（L2 输出差异）
    - INT8 vs INT4 全模型速度/质量对比
    - 最优混合精度方案建议
    - 结果保存到 profile_results/quant_sensitivity.json
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("profile_results", exist_ok=True)

print("=" * 65)
print("  MuseTalk UNet 量化敏感度分析")
print("=" * 65)

# ==================== 加载模型 ====================
print("\n[加载模型]")
from diffusers import AutoencoderKL
from musetalk.models.unet import UNet2DConditionModel
import json as _json

with open("models/musetalkV15/musetalk.json") as f:
    unet_config = _json.load(f)

unet = UNet2DConditionModel(**unet_config).to(device, torch.float16)
state_dict = torch.load("models/musetalkV15/unet.pth", map_location=device)
unet.load_state_dict(state_dict)
unet.eval()
print(f"  ✓ UNet 加载完成 ({sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M 参数)")

# ==================== 量化工具 ====================
def fake_quant(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """对张量进行模拟 PTQ 量化（对称，per-tensor）"""
    q_max = 2 ** (bits - 1) - 1
    scale = tensor.abs().max() / q_max
    if scale == 0:
        return tensor
    return (tensor / scale).round().clamp(-q_max - 1, q_max) * scale


def apply_fake_quant_to_module(module: nn.Module, bits: int = 8):
    """将模块的所有权重做模拟量化（原地），返回原始权重备份"""
    original_weights = {}
    for name, param in module.named_parameters(recurse=False):
        original_weights[name] = param.data.clone()
        param.data = fake_quant(param.data, bits)
    return original_weights


def restore_module_weights(module: nn.Module, original_weights: dict):
    """恢复模块权重"""
    for name, data in original_weights.items():
        getattr(module, name).data = data


# ==================== 基线输出 ====================
print("\n[建立 FP16 基线]")

in_channels = unet.conv_in.in_channels
spatial = 32  # VAE 输出空间尺寸（256/8）
timestep = torch.tensor([0.0], device=device, dtype=torch.float16)
audio_dummy = torch.zeros(1, 1, 384, device=device, dtype=torch.float16)
dummy_input = torch.randn(1, in_channels, spatial, spatial, device=device, dtype=torch.float16)

with torch.no_grad():
    baseline_output = unet(dummy_input, timestep, encoder_hidden_states=audio_dummy).sample
    baseline_norm = baseline_output.norm().item()

print(f"  基线输出范数: {baseline_norm:.4f}")
print(f"  输入形状: {dummy_input.shape} | 输出形状: {baseline_output.shape}")

# ==================== 逐层敏感度分析 ====================
print("\n[逐层敏感度分析]  (模拟 INT8，对每个 Linear/Conv2d 单独量化)")
print("  这将遍历所有 Linear + Conv2d，共测量输出偏差...")

sensitivity_results = []

# 按模块类型分组分析
target_types = (nn.Linear, nn.Conv2d)

all_named_modules = [(name, m) for name, m in unet.named_modules()
                     if isinstance(m, target_types) and any(p.numel() > 0 for p in m.parameters(recurse=False))]

print(f"  共 {len(all_named_modules)} 个目标层\n")

with torch.no_grad():
    for idx, (name, module) in enumerate(all_named_modules):
        # 模拟 INT8 量化该层权重
        orig = apply_fake_quant_to_module(module, bits=8)
        out_int8 = unet(dummy_input, timestep, encoder_hidden_states=audio_dummy).sample
        restore_module_weights(module, orig)

        # 计算输出 L2 差异（相对误差）
        diff = (out_int8 - baseline_output).norm().item()
        rel_err = diff / (baseline_norm + 1e-8)

        sensitivity_results.append({
            "name": name,
            "type": type(module).__name__,
            "param_count": sum(p.numel() for p in module.parameters(recurse=False)),
            "int8_rel_err": rel_err,
        })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(all_named_modules)}] 已分析...")

# 按敏感度排序
sensitivity_results.sort(key=lambda x: -x["int8_rel_err"])

# ==================== 输出 TOP 敏感层 ====================
print("\n" + "=" * 65)
print("  TOP-20 最敏感层（INT8 量化后输出误差最大）")
print("=" * 65)
print(f"\n  {'排名':<5} {'层名称':<55} {'类型':<10} {'相对误差':>10}")
print(f"  {'-'*5} {'-'*55} {'-'*10} {'-'*10}")
for i, r in enumerate(sensitivity_results[:20]):
    bar = "█" * min(20, int(r["int8_rel_err"] * 200))
    name_short = r["name"][-54:] if len(r["name"]) > 54 else r["name"]
    print(f"  {i+1:<5} {name_short:<55} {r['type']:<10} {r['int8_rel_err']:>9.4f} {bar}")

# ==================== 全模型量化对比 ====================
print("\n" + "=" * 65)
print("  全模型量化 vs FP16 基线对比")
print("=" * 65)

def measure_full_quant(bits: int, n_runs: int = 20):
    """对 UNet 所有层应用模拟量化，测量速度和输出误差"""
    # 备份所有权重
    all_orig = {}
    for name, m in unet.named_modules():
        if isinstance(m, target_types):
            all_orig[name] = apply_fake_quant_to_module(m, bits)

    # 预热
    with torch.no_grad():
        for _ in range(3):
            unet(dummy_input, timestep, encoder_hidden_states=audio_dummy)

    # 计时
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            out = unet(dummy_input, timestep, encoder_hidden_states=audio_dummy).sample
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - t0)

    avg_ms = np.mean(times) * 1000
    diff = (out - baseline_output).norm().item()
    rel_err = diff / (baseline_norm + 1e-8)

    # 恢复权重
    for name, m in unet.named_modules():
        if isinstance(m, target_types) and name in all_orig:
            restore_module_weights(m, all_orig[name])

    return avg_ms, rel_err


# FP16 基线速度
times_fp16 = []
with torch.no_grad():
    for _ in range(3):
        unet(dummy_input, timestep, encoder_hidden_states=audio_dummy)
    for _ in range(20):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        unet(dummy_input, timestep, encoder_hidden_states=audio_dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_fp16.append(time.time() - t0)
fp16_ms = np.mean(times_fp16) * 1000

print(f"\n  FP16 基线: {fp16_ms:.1f}ms")

print(f"  测量全模型 INT8 量化...")
int8_ms, int8_err = measure_full_quant(8)
print(f"  全模型 INT8 (模拟): {int8_ms:.1f}ms  相对误差: {int8_err:.4f}  加速: {fp16_ms/int8_ms:.2f}×")

print(f"  测量全模型 INT4 量化...")
int4_ms, int4_err = measure_full_quant(4)
print(f"  全模型 INT4 (模拟): {int4_ms:.1f}ms  相对误差: {int4_err:.4f}  加速: {fp16_ms/int4_ms:.2f}×")

# ==================== 混合精度方案分析 ====================
print("\n" + "=" * 65)
print("  混合精度方案：Top-K 敏感层保留 FP16，其余 INT8")
print("=" * 65)

# 找敏感层名字集合，前 K 层保留 FP16
results_mixed = []
for k in [10, 20, 30, 50]:
    sensitive_names = {r["name"] for r in sensitivity_results[:k]}

    all_orig = {}
    for name, m in unet.named_modules():
        if isinstance(m, target_types) and name not in sensitive_names:
            all_orig[name] = apply_fake_quant_to_module(m, bits=8)

    with torch.no_grad():
        for _ in range(3):
            unet(dummy_input, timestep, encoder_hidden_states=audio_dummy)
        times_m = []
        for _ in range(20):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            out = unet(dummy_input, timestep, encoder_hidden_states=audio_dummy).sample
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times_m.append(time.time() - t0)

    mixed_ms = np.mean(times_m) * 1000
    mixed_err = (out - baseline_output).norm().item() / (baseline_norm + 1e-8)

    for name, m in unet.named_modules():
        if isinstance(m, target_types) and name in all_orig:
            restore_module_weights(m, all_orig[name])

    n_total = len(all_named_modules)
    n_quant = n_total - k
    quant_pct = n_quant / n_total * 100
    results_mixed.append({
        "top_k_fp16": k, "quant_pct": quant_pct,
        "ms": mixed_ms, "rel_err": mixed_err
    })
    print(f"  Top-{k:<3} FP16 ({quant_pct:.0f}% INT8): {mixed_ms:.1f}ms  误差: {mixed_err:.4f}  加速: {fp16_ms/mixed_ms:.2f}×")

# ==================== 输出最优方案建议 ====================
print("\n" + "=" * 65)
print("  最优混合精度方案建议")
print("=" * 65)

# 误差阈值：保持在 FP16 的 5% 以内
ACCEPTABLE_ERR = 0.05
best_plans = [r for r in results_mixed if r["rel_err"] < ACCEPTABLE_ERR]
if best_plans:
    best = min(best_plans, key=lambda x: x["ms"])
    print(f"""
  推荐方案：Top-{best['top_k_fp16']} 敏感层保持 FP16，其余 {best['quant_pct']:.0f}% 层使用 INT8
    速度: FP16 {fp16_ms:.1f}ms → 混合精度 {best['ms']:.1f}ms（加速 {fp16_ms/best['ms']:.2f}×）
    质量损失: 相对误差 {best['rel_err']:.4f}（<5% 阈值）
  """)
else:
    print("  ⚠ 所有方案误差超过 5% 阈值，建议仅量化非 Attention 层")

# 敏感层按模块块归因
print("  敏感层分布（Top-20 按模块位置）：")
block_counts = defaultdict(int)
for r in sensitivity_results[:20]:
    parts = r["name"].split(".")
    if len(parts) >= 2:
        block = parts[0] + "." + parts[1]
    else:
        block = parts[0]
    block_counts[block] += 1
for block, cnt in sorted(block_counts.items(), key=lambda x: -x[1]):
    print(f"    {block}: {cnt} 层")

print("""
  → 验证假设：Cross-Attention（音频条件）层是否集中在高敏感区？
    若是，则支持创新点 A「区域感知混合精度量化」的假设
    下一步：对照论文中嘴唇区域 PSNR 实验，量化此误差的视觉影响
""")

# ==================== 保存结果 ====================
result = {
    "model": "MuseTalk V1.5 UNet (FP16)",
    "device": device,
    "baseline_fp16_ms": fp16_ms,
    "full_int8_ms": int8_ms,
    "full_int8_rel_err": int8_err,
    "full_int4_ms": int4_ms,
    "full_int4_rel_err": int4_err,
    "mixed_precision": results_mixed,
    "top50_sensitive_layers": sensitivity_results[:50],
}

with open("profile_results/quant_sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("  结果已保存到: profile_results/quant_sensitivity.json")
