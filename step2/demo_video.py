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
                    help="MATS 跳帧阈值（越低跳帧越少）")
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
from musetalk.utils.utils import read_imgs
frame_list_cycle = read_imgs(img_list)
mask_list_cycle  = read_imgs(mask_list_files)
cycle_len = len(frame_list_cycle)
print(f"  ✓ 预处理帧数: {cycle_len}")

# ==================== 加载模型 ====================
print("\n[加载模型]")
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending

audio_processor, vae, unet, pe, timesteps = load_all_model()
weight_dtype = unet.model.dtype
print("  ✓ 所有模型加载完成")

# ==================== 提取音频特征 ====================
print(f"\n[提取音频特征] {args.audio}")
whisper_input_features, librosa_length = audio_processor.get_audio_feature(
    args.audio, weight_dtype=weight_dtype)
whisper_chunks = audio_processor.get_whisper_chunk(
    whisper_input_features, device, weight_dtype, unet.model, librosa_length,
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
        w_batch = torch.from_numpy(
            np.stack(whisper_chunks[i:i+bs])).to(device)
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

with torch.no_grad():
    for i in range(total_frames):
        lat = input_latent_list_cycle[i % cycle_len].to(device=device, dtype=weight_dtype)

        # 运动检测
        if prev_lat is not None:
            motion = float((lat.float() - prev_lat.float()).norm() /
                           (prev_lat.float().norm() + 1e-6))
        else:
            motion = 999.0

        w = torch.from_numpy(whisper_chunks[i:i+1]).to(device)
        audio_feat = pe(w)

        sync(); t0 = time.time()

        if motion >= args.threshold or cached_pixel is None:
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

tmp = args.out.replace(".mp4", "_tmp.mp4")
h_o, w_o = comparison_frames[0].shape[:2]
wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w_o, h_o))
for fr in comparison_frames:
    wr.write(fr)
wr.release()

cmd = (f"ffmpeg -loglevel error -nostdin -y "
       f"-i {tmp} -i {args.audio} "
       f"-c:v libx264 -crf 20 -preset fast "
       f"-c:a aac -b:a 128k -shortest {args.out}")
ret = subprocess.call(cmd, shell=True)
if ret == 0:
    os.remove(tmp)
else:
    os.rename(tmp, args.out)

print(f"\n  ✓ 对比视频: {args.out}")
print(f"""
======================================================================
  Demo 汇总
======================================================================
  基线：{fps_base:.1f} FPS
  MATS：{fps_mats:.1f} FPS  (加速 {speedup:.2f}×)
  跳过率：{skip_rate:.1%}  (UNet 实际调用 {unet_count}/{total_frames} 帧)
  输出：{args.out}
""")
