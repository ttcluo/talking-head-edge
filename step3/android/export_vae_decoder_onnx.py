"""
导出 SD VAE Decoder 为 ONNX，供 Android 端将 UNet 输出的 4ch latent 解码为 RGB 图像。

输入: [1, 4, 32, 32] float32（UNet 输出 / scaling_factor 后）
输出: [1, 3, 256, 256] float32（-1~1，端上再 (x+1)/2*255）

使用方式（在 $MUSE_ROOT 下）：
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/export_vae_decoder_onnx.py \\
      --vae_dir models/sd-vae \\
      --out_dir models/student_onnx
"""
import argparse
import os
import sys

import torch

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.getcwd())
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--vae_dir", default="models/sd-vae")
parser.add_argument("--out_dir", default="models/student_onnx")
parser.add_argument("--opset", type=int, default=17)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.chdir(MUSE_ROOT)

# 关闭 SDPA，否则 ONNX 导出会报 aten::scaled_dot_product_attention 不支持
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

print("=" * 50)
print("  VAE Decoder ONNX 导出")
print("=" * 50)

from diffusers import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor

vae = AutoencoderKL.from_pretrained(args.vae_dir)
vae.set_attn_processor(AttnProcessor())  # 避免 SDPA，用 ONNX 可导出的 attention
vae.eval()

# 只导出 decoder：输入 [1,4,32,32]，输出 [1,3,256,256]
# 兼容 diffusers：decoder 有的版本返回 .sample，有的直接返回 Tensor
class DecoderWrapper(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
    def forward(self, z):
        out = self.dec(z)
        return out.sample if hasattr(out, "sample") else out

dec = DecoderWrapper(vae.decoder)
dummy = torch.randn(1, 4, 32, 32)
out_path = os.path.join(args.out_dir, "vae_decoder.onnx")

print(f"  导出 → {out_path}")
with torch.no_grad():
    torch.onnx.export(
        dec,
        dummy,
        out_path,
        opset_version=args.opset,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={"latent": {0: "batch"}, "image": {0: "batch"}},
        do_constant_folding=True,
    )

size_mb = os.path.getsize(out_path) / 1e6
print(f"  ✓ vae_decoder.onnx: {size_mb:.1f} MB")
print(f"  端上: latent 先除以 scaling_factor=0.18215 再送入此模型")
print("=" * 50)
