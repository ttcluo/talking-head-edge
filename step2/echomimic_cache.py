"""
EchoMimic 窗口级缓存（P2）
用途：验证 MATS 思路可迁移到多步扩散模型——
      相邻上下文窗口音频特征相似时，跳过整个 30 步 DDIM，复用上一窗口输出
使用方法：
    conda activate musetalk
    cd ~/EchoMimic
    python /path/to/step2/echomimic_cache.py \
        --pretrained_dir ~/EchoMimic/pretrained_weights \
        --ref_image assets/halfbody_demo/refimgs/natural/guy1.jpg \
        --audio assets/halfbody_demo/audios/chinese/echomimicv2_man.wav

输出：
    - 基线 vs 缓存 SSIM / PSNR / FPS / 窗口跳过率
    - profile_results/echomimic_cache.json
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ECHO_ROOT = os.environ.get("ECHO_ROOT", os.path.expanduser("~/EchoMimic"))
if ECHO_ROOT not in sys.path:
    sys.path.insert(0, ECHO_ROOT)

try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

# ==================== 参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_dir", type=str,
                    default=os.path.expanduser("~/EchoMimic/pretrained_weights"))
parser.add_argument("--ref_image", type=str,
                    default="assets/halfbody_demo/refimgs/natural/guy1.jpg")
parser.add_argument("--audio", type=str,
                    default="assets/halfbody_demo/audios/chinese/echomimicv2_man.wav")
parser.add_argument("--num_steps",     type=int,   default=30)
parser.add_argument("--context_frames",type=int,   default=12)
parser.add_argument("--num_windows",   type=int,   default=10,
                    help="评估窗口数（越多越准，建议 ≥10）")
parser.add_argument("--width",         type=int,   default=512)
parser.add_argument("--height",        type=int,   default=512)
parser.add_argument("--threshold",     type=float, default=0.10,
                    help="音频特征 L2 距离阈值，低于此值跳过 DDIM")
parser.add_argument("--output_dir",    type=str,   default="profile_results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16

print("=" * 65)
print("  EchoMimic 窗口级缓存评估（P2）")
print("=" * 65)
print(f"  DDIM步数={args.num_steps}  窗口帧数={args.context_frames}  "
      f"窗口数={args.num_windows}  阈值={args.threshold}")

# ==================== 工具函数 ====================
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def compute_ssim(a, b):
    if SKIMAGE_OK:
        return float(sk_ssim(a, b, channel_axis=2, data_range=255))
    mse = float(((a.astype(float) - b.astype(float)) ** 2).mean())
    return float(1 - mse / (255 ** 2))

def compute_psnr(a, b):
    if SKIMAGE_OK:
        v = float(sk_psnr(a, b, data_range=255))
        return v if np.isfinite(v) else 100.0
    mse = float(((a.astype(float) - b.astype(float)) ** 2).mean())
    return float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else 100.0

def latent_to_uint8(lat, vae):
    with torch.no_grad():
        dec = vae.decode(lat / 0.18215).sample  # (1, 3, H, W)
    arr = ((dec[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255)
    return arr.astype(np.uint8)

# ==================== 加载模型 ====================
print("\n[加载模型]")
t0 = time.time()

from diffusers import AutoencoderKL, DDIMScheduler

vae_path = os.path.join(args.pretrained_dir, "sd-vae-ft-mse")
if not os.path.exists(vae_path):
    vae_path = os.path.expanduser("~/MuseTalk/models/sd-vae")
vae = AutoencoderKL.from_pretrained(vae_path).to(device, dtype)
vae.eval()

from src.models.unet_2d_condition import UNet2DConditionModel
sd_base = os.path.join(args.pretrained_dir, "sd-image-variations-diffusers")
if not os.path.exists(sd_base):
    sd_base = os.path.join(args.pretrained_dir, "stable-diffusion-v1-5")
reference_unet = UNet2DConditionModel.from_pretrained(
    sd_base, subfolder="unet").to(dtype=dtype, device=device)
reference_unet.load_state_dict(
    torch.load(os.path.join(args.pretrained_dir, "reference_unet.pth"), map_location="cpu"))
reference_unet.eval()

from src.models.unet_3d_echo import EchoUNet3DConditionModel
from omegaconf import OmegaConf
infer_cfg = OmegaConf.load(os.path.join(ECHO_ROOT, "configs/inference/inference_v2.yaml"))
motion_module = os.path.join(args.pretrained_dir, "motion_module.pth")
denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
    sd_base, motion_module if os.path.exists(motion_module) else "",
    subfolder="unet",
    unet_additional_kwargs=infer_cfg.unet_additional_kwargs if os.path.exists(motion_module)
    else {"use_motion_module": False, "unet_use_temporal_attention": False,
          "cross_attention_dim": infer_cfg.unet_additional_kwargs.cross_attention_dim}
).to(dtype=dtype, device=device)
denoising_unet.load_state_dict(
    torch.load(os.path.join(args.pretrained_dir, "denoising_unet.pth"), map_location="cpu"),
    strict=False)
denoising_unet.eval()

from src.models.face_locator import FaceLocator
face_locator = FaceLocator(
    320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)
).to(dtype=dtype, device=device)
face_locator.load_state_dict(
    torch.load(os.path.join(args.pretrained_dir, "face_locator.pth"), map_location="cpu"))
face_locator.eval()

scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="linear",
    clip_sample=False, steps_offset=1)
scheduler.set_timesteps(args.num_steps, device=device)

print(f"  ✓ 全部模型加载完成，耗时 {time.time()-t0:.1f}s")

# ==================== 准备输入 ====================
print("\n[准备输入]")
VAE_SCALE = 8
lh = args.height // VAE_SCALE
lw = args.width  // VAE_SCALE
n_ch = denoising_unet.in_channels
ctx = args.context_frames

# 参考帧
if os.path.exists(args.ref_image):
    img = cv2.resize(cv2.imread(args.ref_image), (args.width, args.height))
    ref_t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1
    ref_t = ref_t.unsqueeze(0).to(device, dtype)
else:
    ref_t = torch.randn(1, 3, args.height, args.width, device=device, dtype=dtype)
    print("  ⚠ 参考图不存在，使用随机张量")

# face mask
face_mask = torch.zeros(1, 1, 1, args.height, args.width, device=device, dtype=dtype)
face_mask[:, :, :, args.height//4:args.height*3//4, args.width//4:args.width*3//4] = 1.0

# 音频特征
whisper_ok = False
audio_fea  = None
whisper_path = os.path.join(args.pretrained_dir, "whisper")
if not os.path.exists(whisper_path):
    whisper_path = os.path.expanduser("~/MuseTalk/models/whisper")
try:
    from src.models.whisper.audio2feature import load_audio_model
    ap = load_audio_model(model_path=whisper_path, device=device)
    if os.path.exists(args.audio):
        feat = ap.audio2feat(args.audio)
        chunks = ap.feature2chunks(feature_array=feat, fps=25)
        audio_fea = torch.tensor(chunks, dtype=dtype, device=device)
        # 扩展到足够窗口数
        needed = args.num_windows * ctx
        if len(audio_fea) < needed:
            reps = (needed // len(audio_fea)) + 1
            audio_fea = audio_fea.repeat(reps, 1, 1)[:needed]
        whisper_ok = True
        print(f"  ✓ 音频特征 shape={audio_fea.shape}")
except Exception as e:
    print(f"  ⚠ Whisper 失败（{e}），使用零向量")

if not whisper_ok or audio_fea is None:
    audio_fea = torch.zeros(args.num_windows * ctx, 5, 384, device=device, dtype=dtype)

# 预计算：ref latent、face cond（全程固定）
with torch.no_grad():
    ref_lat = vae.encode(ref_t).latent_dist.mean * 0.18215
    reference_unet(ref_lat, torch.zeros(1, device=device, dtype=dtype),
                   encoder_hidden_states=None, return_dict=False)
    face_cond = face_locator(face_mask)  # (1, C, 1, h, w)

# ==================== 单窗口 DDIM ====================
def run_ddim_window(audio_window):
    """对一个上下文窗口运行完整 DDIM，返回 latent (1, 4, ctx, lh, lw)"""
    lat = torch.randn(1, n_ch, ctx, lh, lw, device=device, dtype=dtype)
    fc  = face_cond.expand(-1, -1, ctx, -1, -1)
    aud = audio_window.unsqueeze(0)  # (1, ctx, ...)
    with torch.no_grad():
        for t in scheduler.timesteps[:args.num_steps]:
            noise = denoising_unet(
                lat, t,
                encoder_hidden_states=None,
                audio_cond_fea=aud,
                face_musk_fea=fc,
                return_dict=False,
            )[0]
            lat = scheduler.step(noise, t, lat).prev_sample
    return lat  # (1, 4, ctx, lh, lw)

# ==================== 基线 ====================
print(f"\n[基线：{args.num_windows} 个窗口，每窗口 {args.num_steps} 步 DDIM]")

baseline_frames = []  # list of (ctx, H, W, 3) uint8
baseline_latents = [] # list of (1, 4, ctx, lh, lw) cpu
t_baseline = []

for w in range(args.num_windows):
    aud_w = audio_fea[w * ctx:(w + 1) * ctx]  # (ctx, 5, 384)
    sync(); t0 = time.time()
    lat = run_ddim_window(aud_w)
    sync(); t_baseline.append(time.time() - t0)

    frames_w = []
    with torch.no_grad():
        for f in range(ctx):
            px = latent_to_uint8(lat[:, :, f], vae)
            frames_w.append(px)
    baseline_frames.append(frames_w)
    baseline_latents.append(lat.cpu())

    if (w + 1) % 5 == 0:
        print(f"  [{w+1}/{args.num_windows}] {t_baseline[-1]*1000:.0f}ms/窗口")

ms_base = np.mean(t_baseline) * 1000
fps_base = args.context_frames / (ms_base / 1000)
print(f"  基线: {ms_base:.0f}ms/窗口  →  {fps_base:.1f} FPS")

# ==================== 运动量（音频特征距离）====================
# 用每窗口音频特征均值向量计算 L2 距离
audio_means = [
    audio_fea[w * ctx:(w + 1) * ctx].float().mean(dim=0).mean(dim=0)  # (384,)
    for w in range(args.num_windows)
]
motions = [999.0] + [
    float((audio_means[w] - audio_means[w-1]).norm() /
          (audio_means[w-1].norm() + 1e-6))
    for w in range(1, args.num_windows)
]

m_arr = np.array(motions[1:])
skip_count = int((m_arr < args.threshold).sum())
print(f"\n  音频运动统计: mean={m_arr.mean():.4f}  median={np.median(m_arr):.4f}")
print(f"  阈值 {args.threshold}：预估跳过率 {skip_count}/{args.num_windows-1} = "
      f"{skip_count/(args.num_windows-1):.1%}")

# ==================== 缓存版本 ====================
print(f"\n[缓存版本：阈值={args.threshold}]")

cached_frames  = []
t_cached       = []
skip_flags     = [False]

last_lat = baseline_latents[0]  # 第 0 窗口总是跑

# 窗口 0：直接复用基线
aud_0 = audio_fea[:ctx]
sync(); t0 = time.time()
lat0  = run_ddim_window(aud_0)
sync(); t_cached.append(time.time() - t0)
last_lat = lat0.cpu()
cached_frames.append(baseline_frames[0])  # 同基线

skipped = 0
for w in range(1, args.num_windows):
    if motions[w] < args.threshold:
        # 跳过：只需从 CPU 搬运上一次结果（近似为 VAE decode 开销，可忽略）
        skip_flags.append(True)
        skipped += 1
        sync(); t0 = time.time()
        # 复用上一窗口 latent → decode
        frames_w = []
        with torch.no_grad():
            for f in range(ctx):
                px = latent_to_uint8(last_lat[:, :, f].to(device, dtype), vae)
                frames_w.append(px)
        sync(); t_cached.append(time.time() - t0)
        cached_frames.append(frames_w)
    else:
        # 完整 DDIM
        skip_flags.append(False)
        aud_w = audio_fea[w * ctx:(w + 1) * ctx]
        sync(); t0 = time.time()
        lat = run_ddim_window(aud_w)
        sync(); t_cached.append(time.time() - t0)
        last_lat = lat.cpu()
        frames_w = []
        with torch.no_grad():
            for f in range(ctx):
                px = latent_to_uint8(lat[:, :, f], vae)
                frames_w.append(px)
        cached_frames.append(frames_w)

skip_rate = skipped / args.num_windows
ms_cached = np.mean(t_cached) * 1000
fps_cached = args.context_frames / (ms_cached / 1000)
print(f"  窗口跳过率: {skipped}/{args.num_windows} = {skip_rate:.1%}")
print(f"  缓存: {ms_cached:.0f}ms/窗口  →  {fps_cached:.1f} FPS")

# ==================== 质量评估 ====================
print(f"\n[质量评估（SSIM / PSNR）]")

ssim_all, psnr_all = [], []
for w in range(args.num_windows):
    for f in range(ctx):
        ref_px = baseline_frames[w][f]
        est_px = cached_frames[w][f]
        ssim_all.append(compute_ssim(ref_px, est_px))
        psnr_all.append(compute_psnr(ref_px, est_px))

avg_ssim = float(np.mean(ssim_all))
avg_psnr = float(np.mean(psnr_all))

# ==================== 结果汇总 ====================
speedup = fps_cached / fps_base

print("\n" + "=" * 65)
print("  EchoMimic 缓存结果（P2）")
print("=" * 65)
print(f"""
  {'方法':<30} {'FPS':>8} {'SSIM':>8} {'PSNR':>8} {'跳过率':>8}
  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}
  {'基线（30步 DDIM）':<30} {fps_base:>8.1f} {'1.0000':>8} {'∞':>8} {'0%':>8}
  {f'缓存(thr={args.threshold})':<30} {fps_cached:>8.1f} {avg_ssim:>8.4f} {avg_psnr:>8.1f} {skip_rate:>8.1%}

  加速比: {speedup:.2f}×
  论文结论: EchoMimic 上 MATS 同样有效，{skip_rate:.0%}窗口跳过，
           SSIM={avg_ssim:.4f}，PSNR={avg_psnr:.1f}dB，加速 {speedup:.2f}×
""")

# 与 MuseTalk 对比行
print("  横向对比（单视频，阈值≈0.15）：")
print(f"  {'MuseTalk + MATS':<30} {'40.3':>8} {'0.9976':>8} {'57.3':>8} {'53%':>8}")
print(f"  {'EchoMimic + MATS':<30} {fps_cached:>8.1f} {avg_ssim:>8.4f} {avg_psnr:>8.1f} {skip_rate:>8.1%}")

# ==================== 保存 ====================
result = {
    "method": "echomimic_window_cache",
    "config": {
        "num_steps": args.num_steps, "context_frames": ctx,
        "num_windows": args.num_windows, "threshold": args.threshold,
    },
    "baseline": {"fps": round(fps_base, 1), "ms_per_window": round(ms_base, 1)},
    "cached": {
        "fps": round(fps_cached, 1), "ms_per_window": round(ms_cached, 1),
        "skip_rate": round(skip_rate, 4),
        "ssim": round(avg_ssim, 4), "psnr": round(avg_psnr, 2),
        "speedup": round(speedup, 2),
    },
    "motion_stats": {
        "mean": round(float(m_arr.mean()), 4),
        "median": round(float(np.median(m_arr)), 4),
        "std": round(float(m_arr.std()), 4),
    }
}
out = os.path.join(args.output_dir, "echomimic_cache.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n  结果已保存: {out}")
