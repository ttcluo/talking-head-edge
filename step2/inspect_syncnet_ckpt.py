"""
检查 latentsync_syncnet.pt 的权重格式和 key 结构
帮助判断应该用哪个 SyncNet 模型类来加载
"""
import torch
import sys

ckpt_path = "models/syncnet/latentsync_syncnet.pt"

print(f"加载：{ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print(f"类型：{type(ckpt)}")

if isinstance(ckpt, dict):
    print(f"顶级 keys：{list(ckpt.keys())[:20]}")
    # 判断是否有 state_dict
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    print(f"\nstate_dict keys（前30）：")
    keys = list(state.keys())
    for k in keys[:30]:
        v = state[k]
        print(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else v}")
    print(f"\n总参数量：{sum(v.numel() for v in state.values() if hasattr(v, 'numel')):,}")
    print(f"总 keys 数：{len(keys)}")

    # 判断类型
    sample_keys = set(keys[:50])
    if any("visual_encoder" in k for k in sample_keys):
        print("\n  → 判断：LatentSync 完整 UNet（含视觉编码器）")
    elif any("lstm" in k.lower() for k in sample_keys):
        print("\n  → 判断：标准 Wav2Lip SyncNet（含 LSTM）")
    elif any("audio_encoder" in k for k in sample_keys):
        print("\n  → 判断：LatentSync 自定义 SyncNet")
    elif any("conv" in k.lower() for k in sample_keys):
        print("\n  → 判断：卷积为主的 SyncNet（可能是 Wav2Lip 架构）")
    else:
        print("\n  → 判断：未知架构，请人工检查")
else:
    print(f"  非字典格式，原始内容：{ckpt}")
