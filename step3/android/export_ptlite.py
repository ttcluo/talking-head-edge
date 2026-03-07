"""
为 Android 端侧部署导出 PyTorch Mobile (.ptl) 格式模型。

PyTorch Mobile 路径优势：
  - 与 PyTorch 2.0.1 完全兼容，无需 ONNX 中间格式
  - Linear INT8 动态量化已验证可用（attention QKV projection 全部量化）
  - Android 端用 org.pytorch:pytorch_android 直接加载
  - 支持 Vulkan GPU 后端（Snapdragon Adreno GPU 加速）

产出：
  models/ptlite/unet_int8.ptl          (~400-500 MB，Linear INT8）
  models/ptlite/vae_decoder_fp32.ptl   (~200 MB）

使用方式：
  cd $MUSE_ROOT
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/export_ptlite.py \
      --out_dir models/ptlite/

Android 集成：
  build.gradle: implementation 'org.pytorch:pytorch_android_lite:1.13.1'
  Java:
    Module unet = LiteModuleLoader.load(assetFilePath("unet_int8.ptl"));
    Tensor out = unet.forward(IValue.from(latent), IValue.from(t), IValue.from(audio))
                     .toTensor();
"""

import argparse
import os
import sys
import time
import torch
import torch.quantization as tq
from torch.utils.mobile_optimizer import optimize_for_mobile

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",         type=str, default="models/ptlite")
parser.add_argument("--unet_config",     type=str, default="models/musetalkV15/musetalk.json")
parser.add_argument("--unet_model_path", type=str, default="models/musetalkV15/unet.pth")
parser.add_argument("--skip_unet",       action="store_true", help="跳过 UNet 导出（仅导出 VAE）")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

print("=" * 60)
print("  PyTorch Mobile (.ptl) 端侧导出")
print("=" * 60)

# ==================== 加载模型 ====================
from musetalk.utils.utils import load_all_model
from diffusers.models.attention_processor import AttnProcessor

print("\n[加载模型]")
vae, unet, pe = load_all_model(
    unet_model_path=args.unet_model_path,
    unet_config=args.unet_config,
    device="cpu",
)
unet.model = unet.model.float().cpu().eval()
vae.vae    = vae.vae.float().cpu().eval()

# 切换为 eager attention，使 JIT tracer 可以追踪 attention 计算图
unet.model.set_attn_processor(AttnProcessor())
vae.vae.set_attn_processor(AttnProcessor())
print("  ✓ 模型加载完成（CPU FP32，eager attention）")

# ==================== UNet wrapper ====================
class _UNetWrapper(torch.nn.Module):
    """forward 返回 tensor，固定 encoder_hidden_states 参数名。"""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, latent: torch.Tensor,
                timestep: torch.Tensor,
                audio_feat: torch.Tensor) -> torch.Tensor:
        return self.m(latent, timestep,
                      encoder_hidden_states=audio_feat,
                      return_dict=False)[0]

dummy_latent = torch.zeros(1, 8, 32, 32, dtype=torch.float32)
dummy_t      = torch.tensor([0], dtype=torch.long)
dummy_audio  = torch.zeros(1, 50, 384, dtype=torch.float32)

# ==================== 1. UNet INT8 动态量化 + .ptl ====================
if not args.skip_unet:
    print("\n[1/2] UNet：Linear INT8 量化 → JIT trace → optimize_for_mobile")

    # quantize_dynamic 量化所有 Linear（QKV projection、FFN），Conv 保持 FP32
    t0 = time.time()
    unet_q = tq.quantize_dynamic(
        unet.model,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8,
    )
    print(f"  ✓ 量化完成 ({time.time()-t0:.1f}s)")

    unet_wrapper = _UNetWrapper(unet_q)

    t0 = time.time()
    with torch.no_grad():
        traced_unet = torch.jit.trace(
            unet_wrapper,
            (dummy_latent, dummy_t, dummy_audio),
            strict=False,
        )
    print(f"  ✓ JIT trace 完成 ({time.time()-t0:.1f}s)")

    optimized_unet = optimize_for_mobile(traced_unet)
    unet_ptl_path  = os.path.join(args.out_dir, "unet_int8.ptl")
    optimized_unet._save_for_lite_interpreter(unet_ptl_path)
    unet_size = os.path.getsize(unet_ptl_path) / 1e6
    print(f"  ✓ UNet INT8: {unet_ptl_path}  ({unet_size:.1f} MB)")

    # 快速验证：加载 .ptl 并做一次前向
    print("  [验证] 加载 .ptl 并推理...")
    from torch.jit import load as jit_load
    loaded = jit_load(unet_ptl_path)
    with torch.no_grad():
        out = loaded(dummy_latent, dummy_t, dummy_audio)
    assert out.shape == (1, 4, 32, 32), f"输出形状异常: {out.shape}"
    print(f"  ✓ 推理验证通过，输出 {out.shape}")
else:
    print("\n[1/2] UNet 导出已跳过（--skip_unet）")
    unet_size = 0

# ==================== 2. VAE Decoder FP32 → .ptl ====================
print("\n[2/2] VAE Decoder FP32 → JIT trace → optimize_for_mobile")

vae_dec = vae.vae.decoder.eval()
dummy_latent_vae = torch.zeros(1, 4, 32, 32, dtype=torch.float32)

with torch.no_grad():
    traced_vae = torch.jit.trace(vae_dec, dummy_latent_vae, strict=False)

optimized_vae = optimize_for_mobile(traced_vae)
vae_ptl_path  = os.path.join(args.out_dir, "vae_decoder_fp32.ptl")
optimized_vae._save_for_lite_interpreter(vae_ptl_path)
vae_size = os.path.getsize(vae_ptl_path) / 1e6
print(f"  ✓ VAE Decoder FP32: {vae_ptl_path}  ({vae_size:.1f} MB)")

# 验证 VAE
with torch.no_grad():
    loaded_vae = jit_load(vae_ptl_path)
    vae_out = loaded_vae(dummy_latent_vae)
print(f"  ✓ VAE 推理验证通过，输出 {vae_out.shape}")

# ==================== 汇总 ====================
print(f"""
==============================
  导出完成
==============================
  UNet INT8 (Linear量化): {unet_size:.1f} MB
  VAE Decoder FP32:       {vae_size:.1f} MB
  总计:                   {unet_size + vae_size:.1f} MB

Android 集成步骤：
  1. build.gradle 添加依赖：
       implementation 'org.pytorch:pytorch_android_lite:1.13.1'
       implementation 'org.pytorch:pytorch_android_torchvision_lite:1.13.1'

  2. 将 .ptl 文件放入 app/src/main/assets/

  3. Java 加载推理：
       Module unet = LiteModuleLoader.load(assetFilePath("unet_int8.ptl"));
       Tensor out = unet.forward(IValue.from(latent), IValue.from(t), IValue.from(audio))
                        .toTensor();

  4. 启用 Vulkan GPU 后端（Snapdragon Adreno）：
       PyTorchAndroid.setNumThreads(1);
       // 在 Module.load 前设置 VulkanAPI
""")
