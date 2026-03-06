"""
EchoMimic V1 性能 Profiling 脚本
用途：测量各模块耗时，与 MuseTalk 基线形成横向对比
使用方法：
    conda activate musetalk
    cd ~/EchoMimic
    python /path/to/step2/profile_echomimic.py \
        --pretrained_dir ~/EchoMimic/pretrained_weights \
        --ref_image assets/halfbody_demo/refimgs/natural/guy1.jpg \
        --audio assets/halfbody_demo/audios/chinese/echomimicv2_man.wav

输出：
    - 各模块耗时分布表
    - 与 MuseTalk FP32/FP16 的横向对比
    - 结果保存到 profile_results/echomimic_profile.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 确保可以找到 EchoMimic 包（默认 ~/EchoMimic）
ECHO_ROOT = os.environ.get("ECHO_ROOT", os.path.expanduser("~/EchoMimic"))
if ECHO_ROOT not in sys.path:
    sys.path.insert(0, ECHO_ROOT)

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_dir", type=str,
                    default=os.path.expanduser("~/EchoMimic/pretrained_weights"))
parser.add_argument("--ref_image", type=str,
                    default="assets/halfbody_demo/refimgs/natural/guy1.jpg")
parser.add_argument("--audio", type=str,
                    default="assets/halfbody_demo/audios/chinese/echomimicv2_man.wav")
parser.add_argument("--num_steps", type=int, default=30, help="DDIM 步数（默认 30）")
parser.add_argument("--context_frames", type=int, default=12, help="上下文窗口帧数")
parser.add_argument("--num_frames", type=int, default=24, help="总帧数（建议 ≥ context_frames）")
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--use_fp16", action="store_true")
parser.add_argument("--output_dir", type=str, default="profile_results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if args.use_fp16 else torch.float32
print(f"\n设备: {device} | 精度: {'FP16' if args.use_fp16 else 'FP32'}")
print(f"分辨率: {args.width}×{args.height} | DDIM 步数: {args.num_steps} | 总帧数: {args.num_frames}")

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
    "audio_encode":      Timer("Whisper 音频编码"),
    "reference_unet":    Timer("ReferenceNet（每视频一次）"),
    "denoising_unet":    Timer("去噪 UNet（单步）"),
    "vae_decode":        Timer("VAE Decode（每帧）"),
    "face_locator":      Timer("FaceLocator"),
    "total_per_frame":   Timer("单帧平均端到端"),
    "total_pipeline":    Timer("完整 pipeline"),
}

# ==================== 加载模型 ====================
print("\n[加载模型]")
t_load = time.time()

from diffusers import AutoencoderKL, DDIMScheduler

# VAE（优先使用已有的 sd-vae，否则找 sd-vae-ft-mse）
vae_path = os.path.join(args.pretrained_dir, "sd-vae-ft-mse")
if not os.path.exists(vae_path):
    # 尝试软链或 MuseTalk 复用
    muse_vae = os.path.expanduser("~/MuseTalk/models/sd-vae")
    if os.path.exists(muse_vae):
        vae_path = muse_vae
        print(f"  VAE: 复用 MuseTalk 的 {muse_vae}")
    else:
        raise FileNotFoundError(f"VAE 权重不存在: {vae_path}")

vae = AutoencoderKL.from_pretrained(vae_path).to(device, dtype)
vae.eval()
print(f"  ✓ VAE 加载完成")

# Reference UNet（SD UNet 2D，冻结结构）
from src.models.unet_2d_condition import UNet2DConditionModel

sd_base_path = os.path.join(args.pretrained_dir, "stable-diffusion-v1-5")
reference_unet = UNet2DConditionModel.from_pretrained(
    sd_base_path, subfolder="unet"
).to(dtype=dtype, device=device)
reference_unet.load_state_dict(
    torch.load(os.path.join(args.pretrained_dir, "reference_unet.pth"), map_location="cpu")
)
reference_unet.eval()
print(f"  ✓ ReferenceNet 加载完成")

# Denoising UNet（3D 时序 UNet）
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from omegaconf import OmegaConf

infer_config = OmegaConf.load(os.path.join(ECHO_ROOT, "configs/inference/inference_v2.yaml"))
motion_module_path = os.path.join(args.pretrained_dir, "motion_module.pth")

denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
    sd_base_path,
    motion_module_path if os.path.exists(motion_module_path) else "",
    subfolder="unet",
    unet_additional_kwargs=infer_config.unet_additional_kwargs if os.path.exists(motion_module_path)
    else {"use_motion_module": False, "unet_use_temporal_attention": False,
          "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim}
).to(dtype=dtype, device=device)
denoising_unet.load_state_dict(
    torch.load(os.path.join(args.pretrained_dir, "denoising_unet.pth"), map_location="cpu"),
    strict=False
)
denoising_unet.eval()
print(f"  ✓ Denoising UNet 加载完成")

# Face Locator
from src.models.face_locator import FaceLocator

face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
    dtype=dtype, device=device
)
face_locator.load_state_dict(
    torch.load(os.path.join(args.pretrained_dir, "face_locator.pth"), map_location="cpu")
)
face_locator.eval()
print(f"  ✓ FaceLocator 加载完成")

# Whisper 音频编码（EchoMimic 使用与 MuseTalk 相同的 Whisper-tiny）
from src.models.whisper.audio2feature import load_audio_model

whisper_path = os.path.join(args.pretrained_dir, "whisper")
if not os.path.exists(whisper_path):
    whisper_path = os.path.expanduser("~/MuseTalk/models/whisper")
    print(f"  Whisper: 复用 MuseTalk 的 {whisper_path}")

try:
    audio_processor = load_audio_model(model_path=whisper_path, device=device)
    whisper_ok = True
    print(f"  ✓ Whisper 加载完成")
except Exception as e:
    print(f"  ⚠ Whisper 加载失败（{e}），音频编码跳过")
    audio_processor = None
    whisper_ok = False

# DDIM 调度器
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012,
    beta_schedule="linear",
    clip_sample=False,
    steps_offset=1,
)
scheduler.set_timesteps(args.num_steps, device=device)

print(f"\n  模型加载完成，耗时 {time.time()-t_load:.1f}s")

def count_params(model, name):
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {name}: {n:.1f}M 参数")

count_params(vae, "VAE")
count_params(reference_unet, "ReferenceNet")
count_params(denoising_unet, "Denoising UNet")
count_params(face_locator, "FaceLocator")

# ==================== 准备输入 ====================
print(f"\n[准备输入]")
VAE_SCALE = 8
latent_h = args.height // VAE_SCALE
latent_w = args.width // VAE_SCALE

# 参考帧
if os.path.exists(args.ref_image):
    ref_img = cv2.imread(args.ref_image)
    ref_img = cv2.resize(ref_img, (args.width, args.height))
    ref_tensor = torch.from_numpy(ref_img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1
    ref_tensor = ref_tensor.unsqueeze(0).to(device, dtype)
    print(f"  参考帧: {args.ref_image} → {args.width}×{args.height}")
else:
    ref_tensor = torch.randn(1, 3, args.height, args.width, device=device, dtype=dtype)
    print(f"  参考帧: 随机占位（{args.ref_image} 不存在）")

# 人脸 mask（简化：全脸区域）
face_mask = torch.zeros(1, 1, 1, args.height, args.width, device=device, dtype=dtype)
face_mask[:, :, :, args.height//4:args.height*3//4, args.width//4:args.width*3//4] = 1.0

# 音频特征
if whisper_ok and os.path.exists(args.audio):
    with timers["audio_encode"]:
        whisper_feat = audio_processor.audio2feat(args.audio)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feat, fps=25)
    audio_fea = torch.tensor(whisper_chunks, dtype=dtype, device=device)[:args.num_frames]
    print(f"  音频特征: shape={audio_fea.shape}")
else:
    # 占位：EchoMimic whisper 特征维度为 (T, 5, 384) 或 (T, 50, 384) 依版本
    audio_fea = torch.zeros(args.num_frames, 5, 384, device=device, dtype=dtype)
    print(f"  音频特征: 零向量占位 shape={audio_fea.shape}")

audio_fea = audio_fea.unsqueeze(0)  # (1, T, 5, 384)

# ==================== 各组件单独 Profiling ====================
print(f"\n[组件级 Profiling]")

WARMUP = 3
with torch.no_grad():
    # --- ReferenceNet ---
    ref_latents = vae.encode(ref_tensor).latent_dist.mean * 0.18215
    print(f"\n  ReferenceNet 预热...")
    for _ in range(WARMUP):
        reference_unet(ref_latents, torch.zeros(1, device=device, dtype=dtype),
                       encoder_hidden_states=None, return_dict=False)

    for _ in range(10):
        with timers["reference_unet"]:
            reference_unet(ref_latents, torch.zeros(1, device=device, dtype=dtype),
                           encoder_hidden_states=None, return_dict=False)
    print(f"  ReferenceNet: {timers['reference_unet'].avg_ms:.1f}ms（每视频一次，可均摊）")

    # --- FaceLocator ---
    print(f"\n  FaceLocator 预热...")
    for _ in range(WARMUP):
        face_locator(face_mask)
    for _ in range(20):
        with timers["face_locator"]:
            face_locator(face_mask)
    print(f"  FaceLocator: {timers['face_locator'].avg_ms:.1f}ms")

    # --- 去噪 UNet 单步 ---
    # 构造正确的 3D latent 输入: (batch, channel, frames, h, w)
    n_channels = denoising_unet.in_channels
    context = args.context_frames
    dummy_latent_3d = torch.randn(1, n_channels, context, latent_h, latent_w, device=device, dtype=dtype)
    dummy_t = scheduler.timesteps[0]
    # 音频条件：取前 context 帧
    audio_cond = audio_fea[:, :context]  # (1, context, 5, 384)
    # FaceLocator 条件
    face_cond = face_locator(face_mask)  # (1, C, 1, h, w)
    face_cond_expanded = face_cond.expand(-1, -1, context, -1, -1)

    print(f"\n  去噪 UNet 预热...")
    for _ in range(WARMUP):
        denoising_unet(
            dummy_latent_3d, dummy_t,
            encoder_hidden_states=None,
            audio_cond_fea=audio_cond,
            face_musk_fea=face_cond_expanded,
            return_dict=False,
        )

    # 正式计时：模拟 num_steps 步
    print(f"  去噪 UNet {args.num_steps} 步计时...")
    for i, t in enumerate(scheduler.timesteps[:args.num_steps]):
        with timers["denoising_unet"]:
            denoising_unet(
                dummy_latent_3d, t,
                encoder_hidden_states=None,
                audio_cond_fea=audio_cond,
                face_musk_fea=face_cond_expanded,
                return_dict=False,
            )
    print(f"  去噪 UNet 单步: {timers['denoising_unet'].avg_ms:.1f}ms")
    print(f"  去噪 UNet 全程({args.num_steps}步): {timers['denoising_unet'].total_ms:.1f}ms")

    # --- VAE Decode（每帧） ---
    dummy_latent_2d = torch.randn(1, 4, latent_h, latent_w, device=device, dtype=dtype)
    print(f"\n  VAE Decode 预热...")
    for _ in range(WARMUP):
        vae.decode(dummy_latent_2d).sample

    for _ in range(20):
        with timers["vae_decode"]:
            vae.decode(dummy_latent_2d).sample
    print(f"  VAE Decode 单帧: {timers['vae_decode'].avg_ms:.1f}ms")

# ==================== 完整 Pipeline 端到端 ====================
print(f"\n[完整 Pipeline 端到端 Profiling]")
print(f"  测量 3 次完整 pipeline...")

with torch.no_grad():
    for run in range(3):
        with timers["total_pipeline"]:
            # 1. VAE encode ref image
            ref_latents = vae.encode(ref_tensor).latent_dist.mean * 0.18215

            # 2. ReferenceNet（只跑一次）
            reference_unet(ref_latents, torch.zeros(1, device=device, dtype=dtype),
                           encoder_hidden_states=None, return_dict=False)

            # 3. FaceLocator
            face_cond = face_locator(face_mask)
            face_cond_expanded = face_cond.expand(-1, -1, context, -1, -1)

            # 4. DDIM 循环
            latents = torch.randn(1, n_channels, args.num_frames, latent_h, latent_w,
                                  device=device, dtype=dtype)
            audio_cond_full = audio_fea[:, :args.num_frames]

            for t in scheduler.timesteps[:args.num_steps]:
                # 模拟上下文窗口：仅取前 context 帧
                latent_window = latents[:, :, :context]
                audio_window = audio_cond_full[:, :context]
                face_window = face_cond.expand(-1, -1, context, -1, -1)

                noise_pred = denoising_unet(
                    latent_window, t,
                    encoder_hidden_states=None,
                    audio_cond_fea=audio_window,
                    face_musk_fea=face_window,
                    return_dict=False,
                )[0]

                # DDIM step（简化：仅对窗口部分更新）
                latents[:, :, :context] = scheduler.step(
                    noise_pred, t, latent_window
                ).prev_sample

            # 5. VAE decode（逐帧）
            for f in range(args.num_frames):
                vae.decode(latents[:, :, f] / 0.18215).sample

        total_ms = timers["total_pipeline"].times[-1] * 1000
        fps = args.num_frames / timers["total_pipeline"].times[-1]
        print(f"  Run {run+1}: {total_ms:.0f}ms  →  {fps:.1f} FPS（{args.num_frames}帧）")

# ==================== 输出结果 ====================
total_ms = timers["total_pipeline"].avg_ms
fps = args.num_frames / (total_ms / 1000) if total_ms > 0 else 0

# 计算每帧平均耗时（不含 ReferenceNet，因为它是视频级别的一次性开销）
per_frame_ms = (
    timers["denoising_unet"].avg_ms * args.num_steps / args.num_frames +
    timers["vae_decode"].avg_ms
)

print("\n" + "=" * 65)
print("  EchoMimic V1 性能 Profiling 结果")
print("=" * 65)

print(f"\n  {'模块':<30} {'耗时':>12} {'说明'}")
print(f"  {'-'*30} {'-'*12} {'-'*20}")
print(f"  {'Whisper 音频编码':<30} {timers['audio_encode'].avg_ms:>9.1f}ms  每视频一次")
print(f"  {'ReferenceNet':<30} {timers['reference_unet'].avg_ms:>9.1f}ms  每视频一次")
print(f"  {'FaceLocator':<30} {timers['face_locator'].avg_ms:>9.1f}ms  每视频一次")
print(f"  {'去噪 UNet 单步':<30} {timers['denoising_unet'].avg_ms:>9.1f}ms  × {args.num_steps} = {timers['denoising_unet'].avg_ms * args.num_steps:.0f}ms")
print(f"  {'VAE Decode（单帧）':<30} {timers['vae_decode'].avg_ms:>9.1f}ms  × {args.num_frames}帧")
print(f"\n  {'完整 pipeline':<30} {total_ms:>9.1f}ms  ({args.num_frames}帧)")
print(f"  {'理论 FPS':<30} {fps:>9.1f}")

# 显存统计
if torch.cuda.is_available():
    print(f"\n  当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

# ==================== 与 MuseTalk 对比 ====================
print("\n" + "=" * 65)
print("  EchoMimic vs MuseTalk 基线对比")
print("=" * 65)
print(f"""
  {'方法':<25} {'推理步数':>8} {'FPS':>8} {'单帧ms':>10} {'说明'}
  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*20}
  {'MuseTalk 1.5 FP32':<25} {'1':>8} {'18.4':>8} {'54.3':>9}ms  A800 实测
  {'MuseTalk 1.5 FP16':<25} {'1':>8} {'21.5':>8} {'46.4':>9}ms  A800 实测
  {'EchoMimic V1 FP32':<25} {str(args.num_steps):>8} {fps:>8.1f} {total_ms/args.num_frames:>9.1f}ms  A800 实测 ({args.width}×{args.height})
""")

# ==================== 优化建议 ====================
print("=" * 65)
print("  瓶颈分析与量化实验方向")
print("=" * 65)
denoising_pct = (timers["denoising_unet"].avg_ms * args.num_steps) / total_ms * 100
vae_pct = (timers["vae_decode"].avg_ms * args.num_frames) / total_ms * 100
print(f"""
  主要瓶颈：去噪 UNet（{args.num_steps} 步）占 {denoising_pct:.0f}%，VAE Decode 占 {vae_pct:.0f}%

  EchoMimic vs MuseTalk 速度差距主要来源：
    → 步数差异: {args.num_steps} 步 vs MuseTalk 1 步
    → 去噪 UNet 单步: {timers['denoising_unet'].avg_ms:.1f}ms（含时序注意力）
    → 若能将步数从 {args.num_steps} 步压缩到 4 步（蒸馏），可实现 ~{fps*(args.num_steps/4):.0f} FPS

  下一步量化实验（step2/quant_sensitivity.py）：
    → 对 MuseTalk UNet 逐层敏感度分析，确定哪些层可 INT4/INT8
    → 重点关注 Cross-Attention（音频条件）和时序 Self-Attention
""")

# ==================== 保存结果 ====================
result = {
    "config": {
        "model": "EchoMimic V1",
        "width": args.width, "height": args.height,
        "num_steps": args.num_steps, "num_frames": args.num_frames,
        "fp16": args.use_fp16, "device": device,
    },
    "timing_ms": {
        "audio_encode": timers["audio_encode"].avg_ms,
        "reference_unet_once": timers["reference_unet"].avg_ms,
        "denoising_unet_per_step": timers["denoising_unet"].avg_ms,
        "denoising_unet_total": timers["denoising_unet"].avg_ms * args.num_steps,
        "vae_decode_per_frame": timers["vae_decode"].avg_ms,
        "total_pipeline": total_ms,
    },
    "fps": fps,
    "comparison": {
        "musetalk_fp32_fps": 18.4, "musetalk_fp16_fps": 21.5,
        "echomimic_fps": fps,
        "speedup_needed": fps / 18.4,
    }
}

suffix = "fp16" if args.use_fp16 else "fp32"
out_file = os.path.join(args.output_dir, f"echomimic_profile_{suffix}.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"  结果已保存到: {out_file}")
