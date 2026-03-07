"""
Student UNet ONNX 导出 + INT8 量化。

Student（~554MB FP32）远小于 Teacher（3.4GB），不会触发外部数据 bug，
onnxruntime INT8 动态量化预计输出 ~138MB。

使用方式（在 $MUSE_ROOT 下）：
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/export_student_onnx.py \\
      --student_ckpt   exp_out/distill/distill_v1/student_unet-2000.pth \\
      --student_config $REPO/step3/distill/configs/student_musetalk.json \\
      --out_dir        models/student_onnx/
"""

import argparse
import json
import os
import sys
import time

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.getcwd())
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--student_ckpt",   required=True)
parser.add_argument("--student_config", required=True)
parser.add_argument("--out_dir",        default="models/student_onnx")
parser.add_argument("--opset",          type=int, default=17)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.chdir(MUSE_ROOT)

print("=" * 60)
print("  Student UNet ONNX 导出 + INT8 量化")
print("=" * 60)

# ==================== 加载 Student ====================
with open(args.student_config) as f:
    student_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

model = UNet2DConditionModel(**student_cfg)
ckpt  = torch.load(args.student_ckpt, map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.set_attn_processor(AttnProcessor())
model = model.float().cpu().eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Student: {n_params/1e6:.1f}M 参数  FP32≈{n_params*4/1e6:.0f}MB")

# ==================== UNet Wrapper ====================
import torch.nn as nn

class _UNetWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, latent: torch.Tensor,
                timestep: torch.Tensor,
                audio_feat: torch.Tensor) -> torch.Tensor:
        return self.m(latent, timestep,
                      encoder_hidden_states=audio_feat,
                      return_dict=False)[0]

wrapper = _UNetWrapper(model)

dummy_latent = torch.zeros(1, 8, 32, 32, dtype=torch.float32)
dummy_t      = torch.tensor([0], dtype=torch.long)
dummy_audio  = torch.zeros(1, 50, 384, dtype=torch.float32)

# ==================== 导出 FP32 ONNX ====================
fp32_path = os.path.join(args.out_dir, "unet_student_fp32.onnx")
print(f"\n[1/3] 导出 FP32 ONNX → {fp32_path}")

# 强制关闭 Flash Attention（ONNX opset 17 不支持 sdpa）
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

t0 = time.time()
with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (dummy_latent, dummy_t, dummy_audio),
        fp32_path,
        opset_version=args.opset,
        input_names=["latent", "timestep", "audio_feat"],
        output_names=["output"],
        dynamic_axes={
            "latent":     {0: "batch"},
            "timestep":   {0: "batch"},
            "audio_feat": {0: "batch"},
            "output":     {0: "batch"},
        },
        do_constant_folding=True,
    )
fp32_size = os.path.getsize(fp32_path) / 1e6
print(f"  ✓ FP32 ONNX: {fp32_size:.1f} MB  ({time.time()-t0:.1f}s)")

# ==================== 验证 FP32 ONNX ====================
print(f"\n[2/3] 验证 FP32 ONNX 推理")
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "latent":     dummy_latent.numpy(),
        "timestep":   dummy_t.numpy(),
        "audio_feat": dummy_audio.numpy(),
    })
    print(f"  ✓ 输出形状: {out[0].shape}")
except Exception as e:
    print(f"  ⚠ 验证失败: {e}")

# ==================== INT8 动态量化 ====================
int8_path = os.path.join(args.out_dir, "unet_student_int8.onnx")
print(f"\n[3/3] INT8 动态量化 → {int8_path}")
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    t0 = time.time()
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        per_channel=False,
        reduce_range=False,
    )
    int8_size = os.path.getsize(int8_path) / 1e6
    print(f"  ✓ INT8 ONNX: {int8_size:.1f} MB  ({time.time()-t0:.1f}s)")
    print(f"  压缩比: {fp32_size/int8_size:.1f}×  ({fp32_size:.0f}MB → {int8_size:.0f}MB)")

    # 验证 INT8 推理
    sess_q = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    out_q  = sess_q.run(None, {
        "latent":     dummy_latent.numpy(),
        "timestep":   dummy_t.numpy(),
        "audio_feat": dummy_audio.numpy(),
    })
    print(f"  ✓ INT8 推理验证通过，输出 {out_q[0].shape}")

except Exception as e:
    import traceback
    print(f"  ✗ INT8 量化失败:")
    traceback.print_exc()

print(f"""
==============================
  导出汇总
==============================
  FP32 ONNX: {fp32_size:.1f} MB  → {fp32_path}
  INT8 ONNX: {int8_size:.1f} MB  → {int8_path}  (目标 ~{n_params/1e6:.0f}MB)
  压缩比:    {fp32_size/int8_size:.1f}×
""")
