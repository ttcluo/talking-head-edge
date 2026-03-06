"""
MuseTalk 性能 Profiling 脚本（第 2 周核心任务）
用途：精确测量各模块耗时和显存占用，确定优化瓶颈
使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    python ~/tad/step1/profile_musetalk.py --video data/video/sun.mp4 --audio data/audio/sun.wav

输出：
    - 各模块耗时分布表
    - 显存占用峰值
    - 理论最高 FPS
    - profiling 结果保存到 profile_results/
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default="data/video/sun.mp4")
parser.add_argument("--audio", type=str, default="data/audio/sun.wav")
parser.add_argument("--num_frames", type=int, default=50, help="测试帧数")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--use_fp16", action="store_true", help="使用 FP16 推理")
parser.add_argument("--output_dir", type=str, default="profile_results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if args.use_fp16 else torch.float32
print(f"\n设备: {device} | 精度: {'FP16' if args.use_fp16 else 'FP32'} | 批大小: {args.batch_size}")

# ==================== 计时工具 ====================
class Timer:
    def __init__(self, name):
        self.name = name
        self.times = []

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.time() - self.t0)

    @property
    def avg_ms(self):
        return np.mean(self.times) * 1000 if self.times else 0

    @property
    def total_ms(self):
        return np.sum(self.times) * 1000


timers = {
    "audio_encode":    Timer("Whisper 音频编码"),
    "face_detect":     Timer("人脸检测 (DWPose)"),
    "face_parse":      Timer("人脸分割 (mask 生成)"),
    "vae_encode":      Timer("VAE Encode"),
    "unet_forward":    Timer("UNet 前向推理"),
    "vae_decode":      Timer("VAE Decode"),
    "blending":        Timer("图像融合 (blending)"),
    "total_per_frame": Timer("单帧端到端"),
}

# ==================== 加载模型 ====================
print("\n[加载模型]")

from diffusers import AutoencoderKL
from musetalk.models.unet import UNet2DConditionModel
from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import get_file_type, get_video_fps

with open("models/musetalkV15/musetalk.json") as f:
    unet_config = json.load(f)

t_load = time.time()
vae = AutoencoderKL.from_pretrained("models/sd-vae").to(device, dtype)
unet = UNet2DConditionModel(**unet_config).to(device, dtype)
state_dict = torch.load("models/musetalkV15/unet.pth", map_location=device)
unet.load_state_dict(state_dict)
vae.eval()
unet.eval()

audio_processor = Audio2Feature(
    model_path="models/whisper/pytorch_model.bin",
    device=device, fps=25
)
face_parser = FaceParsing()
print(f"  模型加载完成，耗时 {time.time()-t_load:.1f}s")

# 打印模型参数量
def count_params(model, name):
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {name}: {n:.1f}M 参数")

count_params(vae, "VAE")
count_params(unet, "UNet")

# ==================== 预处理视频 ====================
print(f"\n[预处理视频] {args.video}")
cap = cv2.VideoCapture(args.video)
frames = []
while len(frames) < args.num_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"  读取 {len(frames)} 帧")

# ==================== 音频预处理（只做一次）====================
print(f"\n[预处理音频] {args.audio}")
with timers["audio_encode"]:
    audio_features = audio_processor.audio2feat(args.audio)
    audio_chunks = audio_processor.feature2chunks(
        feature_array=audio_features, fps=25
    )
print(f"  音频特征形状: {audio_features.shape}")
print(f"  音频块数量: {len(audio_chunks)}")

# ==================== 逐帧 Profiling ====================
print(f"\n[开始逐帧 Profiling，共 {len(frames)} 帧]")

# 显存采样
peak_memory = {}

def record_peak_memory(stage):
    if torch.cuda.is_available():
        peak_memory[stage] = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()

# 预热
print("  预热 3 帧...")
with torch.no_grad():
    for i in range(min(3, len(frames))):
        frame = frames[i]
        face_crop = cv2.resize(frame[:256, :256], (256, 256))
        face_tensor = torch.from_numpy(face_crop).permute(2,0,1).float() / 127.5 - 1
        face_tensor = face_tensor.unsqueeze(0).to(device, dtype)
        audio_chunk = torch.from_numpy(audio_chunks[i % len(audio_chunks)]).unsqueeze(0).to(device, dtype)
        latent = vae.encode(face_tensor).latent_dist.sample()
        mask = torch.zeros_like(latent)
        unet_input = torch.cat([latent, mask], dim=1)
        out_latent = unet(unet_input, encoder_hidden_states=audio_chunk).sample
        _ = vae.decode(out_latent).sample

record_peak_memory("warmup")
print("  预热完成，开始正式计时...\n")

# 正式测量
with torch.no_grad():
    for i, frame in enumerate(frames):
        audio_chunk = torch.from_numpy(
            audio_chunks[i % len(audio_chunks)]
        ).unsqueeze(0).to(device, dtype)

        with timers["total_per_frame"]:
            # 1. 人脸检测（简化：直接裁剪，实际应用中用 DWPose）
            with timers["face_detect"]:
                face_crop = frame[:256, :256]
                face_resized = cv2.resize(face_crop, (256, 256))

            # 2. 人脸分割
            with timers["face_parse"]:
                # 简化版本，实际用 face_parser
                mask_np = np.zeros((256, 256), dtype=np.uint8)
                mask_np[180:256, 60:196] = 255  # 模拟嘴部 mask

            # 3. VAE Encode
            with timers["vae_encode"]:
                face_tensor = torch.from_numpy(face_resized).permute(2,0,1).float() / 127.5 - 1
                face_tensor = face_tensor.unsqueeze(0).to(device, dtype)
                latent = vae.encode(face_tensor).latent_dist.sample()
                mask_tensor = torch.from_numpy(mask_np).float() / 255.
                mask_latent = mask_tensor.unsqueeze(0).unsqueeze(0).to(device, dtype)
                mask_latent = torch.nn.functional.interpolate(
                    mask_latent, size=(64, 64)
                ).expand(-1, 16, -1, -1)
                unet_input = torch.cat([latent, mask_latent], dim=1)

            record_peak_memory(f"after_vae_encode_{i}")

            # 4. UNet 前向
            with timers["unet_forward"]:
                out_latent = unet(
                    unet_input,
                    encoder_hidden_states=audio_chunk
                ).sample

            record_peak_memory(f"after_unet_{i}")

            # 5. VAE Decode
            with timers["vae_decode"]:
                generated_face = vae.decode(out_latent).sample

            # 6. 图像融合
            with timers["blending"]:
                generated_np = (generated_face[0].permute(1,2,0).cpu().float().numpy() + 1) * 127.5
                generated_np = generated_np.clip(0, 255).astype(np.uint8)
                output_frame = frame.copy()
                output_frame[:256, :256] = generated_np

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(frames)} 帧...")

# ==================== 输出结果 ====================
print("\n" + "=" * 60)
print("  性能 Profiling 结果")
print("=" * 60)

total_per_frame_ms = timers["total_per_frame"].avg_ms
fps = 1000 / total_per_frame_ms if total_per_frame_ms > 0 else 0

rows = []
for key, timer in timers.items():
    if key == "total_per_frame":
        continue
    pct = (timer.avg_ms / total_per_frame_ms * 100) if total_per_frame_ms > 0 else 0
    rows.append((timer.name, timer.avg_ms, pct))

print(f"\n  {'模块':<25} {'平均耗时(ms)':>12} {'占比':>8}")
print(f"  {'-'*25} {'-'*12} {'-'*8}")
for name, ms, pct in sorted(rows, key=lambda x: -x[1]):
    bar = "█" * int(pct / 5)
    print(f"  {name:<25} {ms:>10.1f}ms {pct:>6.1f}% {bar}")

print(f"\n  {'单帧端到端':<25} {total_per_frame_ms:>10.1f}ms")
print(f"  {'理论最高 FPS':<25} {fps:>10.1f}")

# 显存统计
if torch.cuda.is_available():
    print(f"\n  显存占用（峰值）:")
    peak_unet = max(
        (v for k, v in peak_memory.items() if "unet" in k),
        default=0
    )
    peak_vae = max(
        (v for k, v in peak_memory.items() if "vae" in k),
        default=0
    )
    print(f"    UNet 推理峰值: {peak_unet:.2f}GB")
    print(f"    VAE 操作峰值: {peak_vae:.2f}GB")
    total_alloc = torch.cuda.memory_allocated() / 1024**3
    print(f"    当前已分配:   {total_alloc:.2f}GB")

# ==================== 保存结果 ====================
result = {
    "config": {
        "video": args.video,
        "num_frames": len(frames),
        "batch_size": args.batch_size,
        "fp16": args.use_fp16,
        "device": device,
    },
    "timing_ms": {k: v.avg_ms for k, v in timers.items()},
    "fps": fps,
    "bottleneck": sorted(rows, key=lambda x: -x[1])[0][0],
}

output_file = os.path.join(
    args.output_dir,
    f"profile_{'fp16' if args.use_fp16 else 'fp32'}.json"
)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n  结果已保存到: {output_file}")

# ==================== 优化建议 ====================
print("\n" + "=" * 60)
print("  优化建议")
print("=" * 60)
bottleneck = sorted(rows, key=lambda x: -x[1])[0]
print(f"\n  主要瓶颈: {bottleneck[0]} ({bottleneck[1]:.1f}ms, {bottleneck[2]:.1f}%)")

if bottleneck[0] == "UNet 前向推理":
    print("""
  UNet 是主要瓶颈（符合预期），建议研究方向：
  → 候选创新点 A：区域感知混合精度量化（嘴唇 INT8，背景 INT4）
  → 候选创新点 B：针对 Talking Head 的知识蒸馏
  → 候选创新点 C：时序感知特征缓存（背景特征复用）
    """)
elif bottleneck[0] == "VAE Decode":
    print("""
  VAE Decode 是瓶颈，建议：
  → 轻量化 VAE Decoder（GAN-based 直接解码）
  → Token 压缩（减少 latent 空间大小）
    """)
elif bottleneck[0] == "人脸检测 (DWPose)":
    print("""
  人脸检测是瓶颈，建议：
  → 预处理时缓存所有 bbox，实时推理不做检测
  → 用轻量关键点检测器替换 DWPose（如 MediaPipe）
    """)

print("\n下一步：运行 FP16 对比")
print(f"  python profile_musetalk.py --use_fp16 --video {args.video} --audio {args.audio}")
print()
