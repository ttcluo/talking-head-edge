"""
将 Student UNet 导出为 QDQ（Quantize-DequantizeLinear）格式 INT8 ONNX。

QDQ 格式是 NNAPI 真正支持的格式：
  - 标准 Conv/MatMul 算子不变
  - 在权重/激活前后插入 QuantizeLinear / DequantizeLinear 节点
  - Android NNAPI 可将整图加速（而 QOperator/ConvInteger 格式可能 fallback 到 CPU）

需要校准数据（静态量化）。

用法：
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/export_student_onnx_qdq.py \\
      --fp32_onnx    models/student_onnx/unet_student_fp32.onnx \\
      --avatar_list  dataset/distill/train_avatars.txt \\
      --audio_feat_dir dataset/distill/audio_feats \\
      --out_dir      models/student_onnx/ \\
      --n_calib      100
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.getcwd())
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from musetalk.models.unet import PositionalEncoding

parser = argparse.ArgumentParser()
parser.add_argument("--fp32_onnx",     required=True)
parser.add_argument("--avatar_list",   required=True)
parser.add_argument("--audio_feat_dir", required=True)
parser.add_argument("--avatar_base",   default="results/v15/avatars")
parser.add_argument("--out_dir",       default="models/student_onnx")
parser.add_argument("--n_calib",       type=int, default=100)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.chdir(MUSE_ROOT)

print("=" * 60)
print("  Student UNet QDQ ONNX 导出（NNAPI 静态 INT8）")
print("=" * 60)

# ==================== 收集校准样本 ====================
with open(args.avatar_list) as f:
    avatar_ids = [l.strip() for l in f if l.strip()]

pe = PositionalEncoding(d_model=384)

latents_all, audios_all = [], []
for aid in avatar_ids:
    lat_path   = os.path.join(args.avatar_base, aid, "latents.pt")
    audio_path = os.path.join(args.audio_feat_dir, f"{aid}.pt")
    if not os.path.exists(lat_path) or not os.path.exists(audio_path):
        continue
    latents = torch.load(lat_path, map_location="cpu")
    audios  = torch.load(audio_path, map_location="cpu")
    for i in range(min(len(latents), len(audios), 50)):
        latents_all.append(latents[i].float().unsqueeze(0))
        af = audios[i].float().unsqueeze(0)
        audios_all.append(pe(af))

latents_all = latents_all[:args.n_calib]
audios_all  = audios_all[:args.n_calib]
print(f"  校准样本: {len(latents_all)}")

# ==================== 构建校准数据集 ====================
import onnxruntime
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

class UNetCalibReader(CalibrationDataReader):
    def __init__(self, latents, audios):
        self.data = list(zip(latents, audios))
        self.idx  = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        lat, af = self.data[self.idx]
        self.idx += 1
        return {
            "latent":     lat.numpy(),
            "timestep":   np.array([0], dtype=np.int64),
            "audio_feat": af.numpy(),
        }

    def rewind(self):
        self.idx = 0

# ==================== 预处理（插入激活量化节点）====================
from onnxruntime.quantization.preprocess import quant_pre_process

fp32_path  = args.fp32_onnx
prep_path  = os.path.join(args.out_dir, "unet_student_prep.onnx")
qdq_path   = os.path.join(args.out_dir, "unet_student_qdq.onnx")

print(f"\n[1/2] 预处理 FP32 ONNX → {prep_path}")
quant_pre_process(fp32_path, prep_path, skip_symbolic_shape=True)

# ==================== 静态量化（QDQ 格式）====================
print(f"\n[2/2] 静态 QDQ 量化 → {qdq_path}")
reader = UNetCalibReader(latents_all, audios_all)

quantize_static(
    model_input    = prep_path,
    model_output   = qdq_path,
    calibration_data_reader = reader,
    quant_format   = QuantFormat.QDQ,          # NNAPI 要求 QDQ
    weight_type    = QuantType.QInt8,
    activation_type= QuantType.QInt8,
    per_channel    = True,                     # 每通道量化精度更好
    reduce_range   = False,
    op_types_to_quantize=["Conv", "MatMul", "Gemm"],
)

qdq_size  = os.path.getsize(qdq_path) / 1e6
fp32_size = os.path.getsize(fp32_path) / 1e6
print(f"  ✓ QDQ INT8 ONNX: {qdq_size:.1f} MB  (FP32={fp32_size:.1f}MB)")
print(f"  压缩比: {fp32_size/qdq_size:.1f}×")

# ==================== 验证 QDQ ====================
print("\n[验证 QDQ CPU 推理]")
try:
    sess = onnxruntime.InferenceSession(qdq_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "latent":     latents_all[0].numpy(),
        "timestep":   np.array([0], dtype=np.int64),
        "audio_feat": audios_all[0].numpy(),
    })
    print(f"  ✓ 输出形状: {out[0].shape}  （QDQ Conv 在 CPU 上正常运行）")
except Exception as e:
    print(f"  ⚠ QDQ CPU 验证: {e}")

print(f"""
==============================
  QDQ 导出完成
==============================
  QDQ INT8 ONNX: {qdq_size:.1f} MB → {qdq_path}

  NNAPI 部署（Android Java）：
    OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
    opts.addNnapi();     // 启用 NNAPI，QDQ 格式可被完整加速
    OrtSession session = env.createSession(modelPath, opts);

  adb push：
    adb push {qdq_path} /sdcard/Download/unet_student_qdq.onnx
""")
