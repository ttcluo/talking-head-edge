"""
MATS 视频对比 Demo（基于 MuseTalk 真实推理流程）

流程：
  1. 加载 MuseTalk 预处理数据（face detection + VAE encode + mask）
  2. 提取 Whisper 音频特征
  3. 基线推理（全量 UNet，每帧计算）
  4. MATS 推理（像素帧缓存，跳帧复用）
  5. 生成左右对比视频

使用方法：
    conda activate musetalk && cd $MUSE_ROOT

    # 先确认 MuseTalk 已经对该视频做过预处理（会在 results/v15/avatars/ 下生成数据）
    # 如果没有，先运行一次：
    # python scripts/realtime_inference.py --version v15 --preparation True \
    #     --avatar_id yongen --video_path data/video/yongen.mp4

    python $REPO/step2/demo_video.py \
        --avatar_id yongen \
        --audio data/audio/yongen.wav \
        --threshold 0.15 \
        --num_frames 200 \
        --out profile_results/mats_demo.mp4
"""

import argparse
import copy
import glob
import json
import os
import pickle
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--avatar_id",  type=str,   default="yongen",
                    help="MuseTalk 预处理过的角色 ID（对应 results/v15/avatars/<id>/）")
parser.add_argument("--audio",      type=str,   default="data/audio/yongen.wav")
parser.add_argument("--threshold",  type=float, default=0.15,
                    help="MATS 视觉跳帧阈值（越低跳帧越少）")
parser.add_argument("--audio_threshold", type=float, default=0.10,
                    help="MATS 音频门控阈值：audio motion 超过此值则强制计算（0=禁用）")
parser.add_argument("--num_frames", type=int,   default=200,
                    help="生成帧数（0=全部）")
parser.add_argument("--fps",        type=int,   default=25)
parser.add_argument("--batch_size", type=int,   default=4)
parser.add_argument("--out",        type=str,   default="profile_results/mats_demo.mp4")
parser.add_argument("--version",    type=str,   default="v15")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("  MATS 视频对比 Demo（真实 MuseTalk 推理）")
print("=" * 65)

# ==================== 加载预处理数据 ====================
avatar_path = f"results/{args.version}/avatars/{args.avatar_id}"
if not os.path.exists(avatar_path):
    print(f"\n  ✗ 预处理数据不存在: {avatar_path}")
    print("  请先运行 MuseTalk 预处理：")
    print(f"    python scripts/realtime_inference.py --version v15 --preparation True \\")
    print(f"        --avatar_id {args.avatar_id} --video_path data/video/{args.avatar_id}.mp4")
    sys.exit(1)

print(f"\n[加载预处理数据] {avatar_path}")
input_latent_list_cycle = torch.load(f"{avatar_path}/latents.pt")
with open(f"{avatar_path}/coords.pkl", "rb") as f:
    coord_list_cycle = pickle.load(f)
with open(f"{avatar_path}/mask_coords.pkl", "rb") as f:
    mask_coords_list_cycle = pickle.load(f)

img_list = sorted(glob.glob(os.path.join(avatar_path, "full_imgs", "*.[jpJP][pnPN]*[gG]")),
                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
mask_list_files = sorted(glob.glob(os.path.join(avatar_path, "mask", "*.[jpJP][pnPN]*[gG]")),
                         key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
from musetalk.utils.preprocessing import read_imgs
frame_list_cycle = read_imgs(img_list)
mask_list_cycle  = read_imgs(mask_list_files)
cycle_len = len(frame_list_cycle)
print(f"  ✓ 预处理帧数: {cycle_len}")

# ==================== 加载模型 ====================
print("\n[加载模型]")
from transformers import WhisperModel
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending

vae, unet, pe = load_all_model(
    unet_model_path="models/musetalkV15/unet.pth",
    unet_config="models/musetalkV15/musetalk.json",
)
weight_dtype = unet.model.dtype
timesteps = torch.tensor([0], device=device)

audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
whisper = WhisperModel.from_pretrained("models/whisper")
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)
print("  ✓ 所有模型加载完成")

# ==================== 提取音频特征 ====================
print(f"\n[提取音频特征] {args.audio}")
whisper_input_features, librosa_length = audio_processor.get_audio_feature(
    args.audio, weight_dtype=weight_dtype)
whisper_chunks = audio_processor.get_whisper_chunk(
    whisper_input_features, device, weight_dtype, whisper, librosa_length,
    fps=args.fps,
    audio_padding_length_left=2,
    audio_padding_length_right=2,
)
total_frames = len(whisper_chunks)
if args.num_frames > 0:
    total_frames = min(total_frames, args.num_frames)
whisper_chunks = whisper_chunks[:total_frames]
print(f"  ✓ 音频块: {total_frames} 帧")

# ==================== 工具函数 ====================
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def compose_frame(idx, res_face):
    """将生成人脸 blend 回原帧"""
    bbox = coord_list_cycle[idx % cycle_len]
    ori  = copy.deepcopy(frame_list_cycle[idx % cycle_len])
    x1, y1, x2, y2 = bbox
    try:
        res = cv2.resize(res_face.astype(np.uint8), (x2 - x1, y2 - y1))
    except Exception:
        return ori
    mask  = mask_list_cycle[idx % cycle_len]
    mcbox = mask_coords_list_cycle[idx % cycle_len]
    return get_image_blending(ori, res, bbox, mask, mcbox)

def add_overlay(frame, lines, fps_val, skipped, color):
    out = frame.copy()
    h, w = out.shape[:2]
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (w, 62), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, out, 0.45, 0, out)
    for j, line in enumerate(lines):
        cv2.putText(out, line, (8, 22 + j * 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)
    fps_str = f"{fps_val:.1f} FPS"
    tw = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0][0]
    cv2.putText(out, fps_str, (w - tw - 8, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 128), 2, cv2.LINE_AA)
    dot_c = (0, 80, 255) if skipped else (0, 220, 0)
    cv2.circle(out, (w - 16, h - 16), 8, dot_c, -1)
    cv2.putText(out, "SKIP" if skipped else "CALC", (w - 65, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_c, 1, cv2.LINE_AA)
    return out

def rolling_fps(timings, i, win=20):
    s = max(0, i - win + 1)
    return min(1.0 / max(np.mean(timings[s:i+1]), 1e-9), 9999)

# ==================== 基线推理 ====================
print(f"\n[基线推理：全量 UNet，{total_frames} 帧]")
baseline_out = []
baseline_t   = []

with torch.no_grad():
    for i in range(0, total_frames, args.batch_size):
        bs = min(args.batch_size, total_frames - i)
        chunk_slice = whisper_chunks[i:i+bs]
        if isinstance(chunk_slice[0], torch.Tensor):
            w_batch = torch.stack([c.cpu() for c in chunk_slice]).to(device)
        else:
            w_batch = torch.from_numpy(np.stack(chunk_slice)).to(device)
        lat_batch = torch.cat(
            [input_latent_list_cycle[j % cycle_len] for j in range(i, i+bs)], 0
        ).to(device=device, dtype=weight_dtype)

        audio_feat = pe(w_batch)

        sync(); t0 = time.time()
        pred = unet.model(lat_batch, timesteps,
                          encoder_hidden_states=audio_feat).sample
        pred = pred.to(dtype=vae.vae.dtype)
        recon = vae.decode_latents(pred)
        sync(); elapsed = time.time() - t0

        per_frame = elapsed / bs
        for k, face in enumerate(recon):
            idx = i + k
            composed = compose_frame(idx, face)
            baseline_out.append(composed)
            baseline_t.append(per_frame)

        if (i + bs) % 50 < args.batch_size or (i + bs) >= total_frames:
            print(f"  [{i+bs}/{total_frames}] {1/per_frame:.1f} FPS")

fps_base = total_frames / sum(baseline_t)
print(f"  基线完成：{fps_base:.1f} FPS")

# ==================== MATS 推理（P3+ 像素帧缓存）====================
print(f"\n[MATS 推理：像素帧缓存，阈值={args.threshold}]")
mats_out     = []
mats_t       = []
skip_flags   = []
prev_lat     = None
cached_pixel = None
skip_count = 0
unet_count = 0
prev_audio_feat = None

with torch.no_grad():
    for i in range(total_frames):
        lat = input_latent_list_cycle[i % cycle_len].to(device=device, dtype=weight_dtype)

        # 视觉运动检测
        if prev_lat is not None:
            motion = float((lat.float() - prev_lat.float()).norm() /
                           (prev_lat.float().norm() + 1e-6))
        else:
            motion = 999.0

        wc = whisper_chunks[i]
        if isinstance(wc, torch.Tensor):
            w = wc.cpu().unsqueeze(0).to(device)
        else:
            w = torch.from_numpy(np.array([wc])).to(device)
        audio_feat = pe(w)

        # 音频运动检测：音频变化大时强制计算
        if args.audio_threshold > 0 and prev_audio_feat is not None:
            audio_motion = float((audio_feat.float() - prev_audio_feat.float()).norm() /
                                 (prev_audio_feat.float().norm() + 1e-6))
        else:
            audio_motion = 0.0

        sync(); t0 = time.time()

        audio_gate = (args.audio_threshold > 0 and audio_motion >= args.audio_threshold)
        if motion >= args.threshold or cached_pixel is None or audio_gate:
            pred = unet.model(lat, timesteps,
                              encoder_hidden_states=audio_feat).sample
            pred = pred.to(dtype=vae.vae.dtype)
            faces = vae.decode_latents(pred)
            face  = faces[0]
            cached_pixel = face.copy()
            skipped = False
            unet_count += 1
        else:
            face  = cached_pixel
            skipped = True
            skip_count += 1

        prev_audio_feat = audio_feat.detach()

        sync(); elapsed = time.time() - t0

        composed = compose_frame(i, face)
        mats_out.append(composed)
        mats_t.append(elapsed)
        skip_flags.append(skipped)
        prev_lat = lat.clone()

        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            print(f"  [{i+1}/{total_frames}] {1/np.mean(mats_t[-20:]):.1f} FPS  "
                  f"跳过率={skip_count/(i+1):.1%}")

fps_mats = total_frames / sum(mats_t)
skip_rate = skip_count / total_frames
print(f"  MATS 完成：{fps_mats:.1f} FPS  跳过率={skip_rate:.1%}")

# ==================== SSIM / PSNR ====================
def _ssim_psnr(img1, img2, win_size=11, sigma=1.5):
    """标准 Gaussian 窗口 SSIM + PSNR（与 skimage 实现一致）"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 1e-10 else 100.0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    kernel = cv2.getGaussianKernel(win_size, sigma)
    kernel2d = (kernel @ kernel.T).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.filter2D(g1, -1, kernel2d)
    mu2 = cv2.filter2D(g2, -1, kernel2d)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sig1 = cv2.filter2D(g1 * g1, -1, kernel2d) - mu1_sq
    sig2 = cv2.filter2D(g2 * g2, -1, kernel2d) - mu2_sq
    sig12 = cv2.filter2D(g1 * g2, -1, kernel2d) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1 + sig2 + C2))
    return float(ssim_map.mean()), float(psnr)

print(f"\n[质量评估：SSIM / PSNR（真实音频）]")
ssim_vals, psnr_vals = [], []
for b_f, m_f in zip(baseline_out, mats_out):
    s, p = _ssim_psnr(b_f, m_f)
    ssim_vals.append(s)
    psnr_vals.append(p)
mean_ssim = float(np.mean(ssim_vals))
mean_psnr = float(np.mean(psnr_vals))
print(f"  SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.2f} dB")

# ==================== 合成对比视频 ====================
print(f"\n[合成对比视频]")
speedup = fps_mats / fps_base

target_h = min(baseline_out[0].shape[0], 540)

def resize_h(f, h):
    oh, ow = f.shape[:2]
    nw = int(ow * h / oh)
    return cv2.resize(f, (nw, h))

comparison_frames = []
for i in range(total_frames):
    b_fps = rolling_fps(baseline_t, i)
    m_fps = rolling_fps(mats_t, i)
    b = resize_h(baseline_out[i], target_h)
    m = resize_h(mats_out[i], target_h)
    b = add_overlay(b, ["Baseline (MuseTalk FP16)", "Full UNet every frame"],
                    b_fps, False, (200, 200, 255))
    m = add_overlay(m, [f"MATS (thr={args.threshold:.2f})",
                        f"Skip: {skip_flags[i]}"],
                    m_fps, skip_flags[i], (200, 255, 200))
    div  = np.zeros((target_h, 4, 3), dtype=np.uint8)
    div[:] = (80, 80, 80)
    row  = np.concatenate([b, div, m], axis=1)
    bw   = row.shape[1]
    stat = np.zeros((30, bw, 3), dtype=np.uint8)
    info = (f"Frame {i+1}/{total_frames}   "
            f"Baseline {fps_base:.1f} FPS   "
            f"MATS {fps_mats:.1f} FPS   "
            f"Speedup {speedup:.2f}x   "
            f"Skip {skip_rate:.1%}")
    cv2.putText(stat, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 220, 180), 1, cv2.LINE_AA)
    row = np.concatenate([row, stat], axis=0)
    comparison_frames.append(row)

def write_video_with_audio(frames, audio_path, out_path, fps):
    tmp = out_path.replace(".mp4", "_tmp.mp4")
    h_o, w_o = frames[0].shape[:2]
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_o, h_o))
    for fr in frames:
        wr.write(fr)
    wr.release()
    cmd = (f"ffmpeg -loglevel error -nostdin -y "
           f"-i {tmp} -i {audio_path} "
           f"-c:v libx264 -crf 20 -preset fast "
           f"-c:a aac -b:a 128k -shortest {out_path}")
    ret = subprocess.call(cmd, shell=True)
    if ret == 0:
        os.remove(tmp)
    else:
        os.rename(tmp, out_path)

write_video_with_audio(comparison_frames, args.audio, args.out, args.fps)
print(f"\n  ✓ 对比视频: {args.out}")

base_dir = os.path.dirname(args.out)
baseline_path = os.path.join(base_dir, "baseline.mp4")
mats_path     = os.path.join(base_dir, "mats.mp4")
h1, w1 = baseline_out[0].shape[:2]
write_video_with_audio(baseline_out, args.audio, baseline_path, args.fps)
print(f"  ✓ 单独基线: {baseline_path}")
write_video_with_audio(mats_out, args.audio, mats_path, args.fps)
print(f"  ✓ 单独MATS: {mats_path}")
print(f"""
======================================================================
  Demo 汇总
======================================================================
  基线：{fps_base:.1f} FPS
  MATS：{fps_mats:.1f} FPS  (加速 {speedup:.2f}×)
  跳过率：{skip_rate:.1%}  (UNet 实际调用 {unet_count}/{total_frames} 帧)
  SSIM：{mean_ssim:.4f}  PSNR：{mean_psnr:.2f} dB  (真实音频，baseline vs MATS)
  输出：{args.out}
""")
