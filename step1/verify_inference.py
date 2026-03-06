"""
MuseTalk 推理验证脚本
用途：验证环境安装正确，各模块可正常加载和前向推理
使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    python /path/to/step1/verify_inference.py
"""

import sys
import time
import torch
import numpy as np
import os

# 确保可以找到 musetalk 包（默认 ~/MuseTalk）
MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)


def check(name, fn):
    """执行检查项，打印结果"""
    try:
        result = fn()
        print(f"  ✓ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: 失败 → {e}")
        return False


# ==================== 1. 基础环境检查 ====================
print("\n" + "=" * 50)
print("1. 基础环境检查")
print("=" * 50)

check("Python 版本", lambda: sys.version.split()[0])
check("PyTorch 版本", lambda: torch.__version__)
check("CUDA 可用", lambda: torch.cuda.is_available())
check("GPU 数量", lambda: torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / 1024**3
        check(f"GPU {i}", lambda i=i, props=props, total_mem=total_mem:
              f"{props.name}, 显存 {total_mem:.1f}GB")

# ==================== 2. 依赖包检查 ====================
print("\n" + "=" * 50)
print("2. 依赖包检查")
print("=" * 50)

packages = [
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("opencv-python", "cv2"),
    ("mmcv", "mmcv"),
    ("mmdet", "mmdet"),
    ("mmpose", "mmpose"),
    ("librosa", "librosa"),
    ("imageio", "imageio"),
    ("omegaconf", "omegaconf"),
]

all_ok = True
for display_name, import_name in packages:
    ok = check(
        display_name,
        lambda n=import_name: __import__(n).__version__
    )
    all_ok = all_ok and ok

if not all_ok:
    print("\n⚠️  部分包未安装，请检查 requirements.txt")
    sys.exit(1)

# ==================== 3. 模型文件检查 ====================
print("\n" + "=" * 50)
print("3. 模型权重文件检查")
print("=" * 50)

import os
model_files = {
    "MuseTalk V1.5 UNet": "models/musetalkV15/unet.pth",
    "MuseTalk V1.5 Config": "models/musetalkV15/musetalk.json",
    "SD VAE Config": "models/sd-vae/config.json",
    "SD VAE Weights": "models/sd-vae/diffusion_pytorch_model.bin",
    "Whisper Config": "models/whisper/config.json",
    "Whisper Weights": "models/whisper/pytorch_model.bin",
    "DWPose Weights": "models/dwpose/dw-ll_ucoco_384.pth",
    "Face Parse Weights": "models/face-parse-bisent/79999_iter.pth",
    "SyncNet Weights": "models/syncnet/latentsync_syncnet.pt",
}

all_files_ok = True
for name, path in model_files.items():
    exists = os.path.exists(path)
    size = os.path.getsize(path) / 1024**2 if exists else 0
    status = f"存在 ({size:.1f}MB)" if exists else "❌ 文件不存在"
    check(name, lambda s=status: s)
    if not exists:
        all_files_ok = False

if not all_files_ok:
    print("\n⚠️  部分模型文件缺失，请运行 bash ./download_weights.sh")
    sys.exit(1)

# ==================== 4. 模型加载测试 ====================
print("\n" + "=" * 50)
print("4. 模型加载测试（仅加载，不推理）")
print("=" * 50)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  使用设备: {device}")

# 加载 VAE
print("\n  加载 VAE...")
t0 = time.time()
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("models/sd-vae").to(device)
vae.eval()
print(f"  ✓ VAE 加载完成，耗时 {time.time()-t0:.2f}s")
vae_params = sum(p.numel() for p in vae.parameters()) / 1e6
print(f"    参数量: {vae_params:.1f}M")

# 加载 UNet
print("\n  加载 UNet...")
t0 = time.time()
import json
from musetalk.models.unet import UNet2DConditionModel

with open("models/musetalkV15/musetalk.json") as f:
    unet_config = json.load(f)
unet = UNet2DConditionModel(**unet_config).to(device)
state_dict = torch.load("models/musetalkV15/unet.pth", map_location=device)
unet.load_state_dict(state_dict)
unet.eval()
print(f"  ✓ UNet 加载完成，耗时 {time.time()-t0:.2f}s")
unet_params = sum(p.numel() for p in unet.parameters()) / 1e6
print(f"    参数量: {unet_params:.1f}M")

# 加载 Whisper（可选，失败不影响 UNet 计时）
print("\n  加载 Whisper-tiny...")
t0 = time.time()
try:
    from musetalk.whisper.audio2feature import Audio2Feature
    # MuseTalk 签名：whisper_model_type, model_path（需 .pt 或兼容格式）
    audio_processor = Audio2Feature(
        whisper_model_type="tiny",
        model_path="models/whisper/pytorch_model.bin"
    )
    print(f"  ✓ Whisper 加载完成，耗时 {time.time()-t0:.2f}s")
except Exception as e:
    print(f"  ⚠ Whisper 跳过（{e}），继续 UNet 计时")

# ==================== 5. 单次前向推理测试 ====================
print("\n" + "=" * 50)
print("5. 单次前向推理测试（随机输入）")
print("=" * 50)

with torch.no_grad():
    # 模拟输入
    batch_size = 1
    dummy_latent = torch.randn(batch_size, 32, 64, 64).to(device)  # UNet输入
    dummy_audio = torch.randn(batch_size, 1, 384).to(device)       # Whisper特征
    timestep = torch.tensor([0.0], device=device)

    print(f"  输入形状: latent={dummy_latent.shape}, audio={dummy_audio.shape}")

    # 测量推理时间（预热 3 次，正式 10 次）
    print("  预热...")
    for _ in range(3):
        _ = unet(dummy_latent, timestep=timestep,
                 encoder_hidden_states=dummy_audio).sample

    print("  正式计时（10次取平均）...")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out = unet(dummy_latent, timestep=timestep,
                   encoder_hidden_states=dummy_audio).sample
    torch.cuda.synchronize()
    avg_ms = (time.time() - t0) / 10 * 1000

    print(f"  ✓ UNet 单次推理: {avg_ms:.1f}ms → 理论最高 {1000/avg_ms:.1f} FPS")
    print(f"  输出形状: {out.shape}")

# ==================== 6. 显存占用报告 ====================
print("\n" + "=" * 50)
print("6. 显存占用报告")
print("=" * 50)

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  已分配显存: {allocated:.2f}GB")
    print(f"  已预留显存: {reserved:.2f}GB")
    print(f"  总显存:     {total:.2f}GB")
    print(f"  剩余可用:   {total - reserved:.2f}GB")

# ==================== 总结 ====================
print("\n" + "=" * 50)
print("✅ 验证完成！环境正常，可以进行下一步。")
print("\n下一步：运行完整推理")
print("  sh inference.sh v1.5 normal")
print("=" * 50 + "\n")
