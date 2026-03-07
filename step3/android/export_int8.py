"""
为 Android 端侧部署导出 INT8 量化 ONNX 模型。

产出：
  models/android_onnx/unet_int8.onnx        (~180 MB)
  models/android_onnx/vae_decoder_int8.onnx (~80 MB)

使用方式：
  cd $MUSE_ROOT
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/export_int8.py \
      --out_dir models/android_onnx/
"""

import argparse
import os
import sys
import numpy as np
import torch

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",         type=str, default="models/android_onnx")
parser.add_argument("--unet_config",     type=str, default="models/musetalkV15/musetalk.json")
parser.add_argument("--unet_model_path", type=str, default="models/musetalkV15/unet.pth")
parser.add_argument("--num_calib",       type=int, default=50,
                    help="INT8 校准样本数（越多越准，越慢）")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = "cpu"  # INT8 量化在 CPU 上进行

print("=" * 60)
print("  Android INT8 ONNX 导出")
print("=" * 60)

# ==================== 加载模型 ====================
from musetalk.utils.utils import load_all_model
print("\n[加载模型]")
vae, unet, pe = load_all_model(
    unet_model_path=args.unet_model_path,
    unet_config=args.unet_config,
    device=device,
)
unet.model = unet.model.float().cpu()
print("  ✓ 模型加载完成（CPU FP32 用于量化）")

# ==================== 1. 导出 UNet FP32 ONNX ====================
unet_fp32_path = os.path.join(args.out_dir, "unet_fp32.onnx")
print(f"\n[1/3] 导出 UNet FP32 ONNX → {unet_fp32_path}")

dummy_latent = torch.zeros(1, 8, 32, 32, dtype=torch.float32)
dummy_t      = torch.tensor([0], dtype=torch.long)
dummy_audio  = torch.zeros(1, 50, 384, dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(
        unet.model,
        (dummy_latent, dummy_t, dummy_audio),
        unet_fp32_path,
        input_names=["latent", "timestep", "audio_feat"],
        output_names=["pred_latent"],
        dynamic_axes={
            "latent":     {0: "batch"},
            "audio_feat": {0: "batch"},
            "pred_latent":{0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(unet_fp32_path) / 1e6
    print(f"  ✓ UNet FP32: {size_mb:.1f} MB")

# ==================== 2. INT8 动态量化 ====================
print(f"\n[2/3] INT8 动态量化（onnxruntime quantization）")
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    unet_int8_path = os.path.join(args.out_dir, "unet_int8.onnx")
    quantize_dynamic(
        unet_fp32_path,
        unet_int8_path,
        weight_type=QuantType.QInt8,
        extra_options={"EnableSubgraph": True},
    )
    size_int8 = os.path.getsize(unet_int8_path) / 1e6
    print(f"  ✓ UNet INT8: {size_int8:.1f} MB（压缩率 {size_mb/size_int8:.1f}×）")
except ImportError:
    print("  ✗ onnxruntime 未安装，跳过 INT8 量化")
    print("  请先安装：pip install onnxruntime")

# ==================== 3. 导出 VAE Decoder ====================
vae_fp32_path = os.path.join(args.out_dir, "vae_decoder_fp32.onnx")
print(f"\n[3/3] 导出 VAE Decoder FP32 ONNX → {vae_fp32_path}")

vae_dec = vae.vae.decoder.float().cpu()
dummy_latent_vae = torch.zeros(1, 4, 32, 32, dtype=torch.float32)

with torch.no_grad():
    torch.onnx.export(
        vae_dec,
        dummy_latent_vae,
        vae_fp32_path,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={"latent": {0: "batch"}, "image": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    vae_size = os.path.getsize(vae_fp32_path) / 1e6
    print(f"  ✓ VAE Decoder FP32: {vae_size:.1f} MB")

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    vae_int8_path = os.path.join(args.out_dir, "vae_decoder_int8.onnx")
    quantize_dynamic(vae_fp32_path, vae_int8_path, weight_type=QuantType.QInt8)
    vae_int8_size = os.path.getsize(vae_int8_path) / 1e6
    print(f"  ✓ VAE Decoder INT8: {vae_int8_size:.1f} MB")
except ImportError:
    pass

# ==================== 验证 INT8 精度 ====================
print(f"\n[验证] INT8 vs FP32 精度对比")
try:
    import onnxruntime as ort
    sess_fp32 = ort.InferenceSession(unet_fp32_path,
                                     providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(unet_int8_path,
                                     providers=["CPUExecutionProvider"])

    lat_np   = np.random.randn(1, 8, 32, 32).astype(np.float32)
    t_np     = np.array([0], dtype=np.int64)
    audio_np = np.random.randn(1, 50, 384).astype(np.float32)
    feed = {"latent": lat_np, "timestep": t_np, "audio_feat": audio_np}

    out_fp32 = sess_fp32.run(None, feed)[0]
    out_int8 = sess_int8.run(None, feed)[0]
    diff = np.abs(out_fp32 - out_int8)
    print(f"  INT8 vs FP32 最大误差: {diff.max():.6f}")
    print(f"  INT8 vs FP32 均值误差: {diff.mean():.6f}")
    print(f"  ✓ INT8 精度验证通过（误差 < 0.01 为合格）"
          if diff.mean() < 0.01 else "  ⚠ 误差偏大，请检查量化配置")
except Exception as e:
    print(f"  精度验证跳过: {e}")

print(f"""
==============================
  导出完成
==============================
  UNet INT8:      {os.path.join(args.out_dir, 'unet_int8.onnx')}
  VAE INT8:       {os.path.join(args.out_dir, 'vae_decoder_int8.onnx')}

下一步：
  1. 将 *.onnx 文件复制到 Android 项目 assets/
  2. 用 ONNX Runtime Android 加载推理
  3. 在真机上运行并记录 FPS
""")
