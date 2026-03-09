"""
MATS 视频对比 Demo（4 路对比）

4 种模式：基线(Teacher全帧) | MATS(Teacher+缓存) | Student(全帧) | MATS+Student
输出：4 条单独视频 + 2×2 对比视频

流程：
  1. 加载 MuseTalk 预处理数据
  2. 提取 Whisper 音频特征（从 wav 实时提取）
  3. 基线：Teacher 全帧
  4. MATS：Teacher + 像素帧缓存
  5. Student：Student 全帧
  6. MATS+Student：Student + 像素帧缓存
  7. 生成 2×2 对比视频

使用方法：
    推荐（REPO 自动推导，避免路径错误）：
        cd 项目根 && bash step2/run_demo_student.sh
    或手动（需 REPO=项目根，服务器上勿用 tad）：
        cd $MUSE_ROOT
        PYTHONPATH=$MUSE_ROOT python $REPO/step2/demo_video_student.py \
        --student_ckpt exp_out/distill/distill_lipsync/student_unet-2000.pth \
        --student_config $REPO/step3/distill/configs/student_musetalk.json \
        --avatar_id yongen \
        --audio data/audio/yongen.wav \
        --threshold 0.15 \
        --num_frames 200 \
        --out profile_results/mats_demo_student.mp4
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
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--student_ckpt", type=str, required=True, help="Student checkpoint 路径")
parser.add_argument("--student_config", type=str, required=True, help="Student config json 路径")
parser.add_argument("--avatar_id", type=str, default="yongen")
parser.add_argument("--audio", type=str, default="data/audio/yongen.wav")
parser.add_argument("--threshold", type=float, default=0.15)
parser.add_argument("--max_skip", type=int, default=2)
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--out", type=str, default="profile_results/mats_demo_student.mp4")
parser.add_argument("--version", type=str, default="v15")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("  MATS 视频对比 Demo（4 路：基线 | MATS | Student | MATS+Student）")
print("=" * 65)

# ==================== 加载预处理数据（与 demo_video 完全一致）====================
avatar_path = f"results/{args.version}/avatars/{args.avatar_id}"
if not os.path.exists(avatar_path):
    print(f"\n  ✗ 预处理数据不存在: {avatar_path}")
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
from musetalk.utils.blending import get_image_blending

frame_list_cycle = read_imgs(img_list)
mask_list_cycle = read_imgs(mask_list_files)
cycle_len = len(frame_list_cycle)
print(f"  ✓ 预处理帧数: {cycle_len}")

# ==================== 加载模型（VAE + Teacher + Student + PE）====================
print("\n[加载模型]")
from transformers import WhisperModel
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

vae, teacher_wrapper, pe_module = load_all_model(
    unet_model_path="models/musetalkV15/unet.pth",
    unet_config="models/musetalkV15/musetalk.json",
)
teacher_unet = teacher_wrapper.model.to(device).float().eval()
vae.vae = vae.vae.to(device).float()
pe = pe_module.to(device)

with open(args.student_config) as f:
    student_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
student_unet = UNet2DConditionModel(**student_cfg)
student_unet.set_attn_processor(AttnProcessor())
ckpt = torch.load(args.student_ckpt, map_location="cpu")
student_unet.load_state_dict(ckpt, strict=False)
student_unet = student_unet.to(device).float().eval()

audio_processor = AudioProcessor(feature_extractor_path="models/whisper")
whisper = WhisperModel.from_pretrained("models/whisper")
whisper = whisper.to(device=device, dtype=torch.float32).eval()
whisper.requires_grad_(False)
print("  ✓ VAE + Teacher + Student + PE + Whisper 加载完成")

# ==================== 提取音频特征（与 demo_video 完全一致：从 wav 实时提取）====================
print(f"\n[提取音频特征] {args.audio}")
whisper_input_features, librosa_length = audio_processor.get_audio_feature(
    args.audio, weight_dtype=torch.float32)
whisper_chunks = audio_processor.get_whisper_chunk(
    whisper_input_features, device, torch.float32, whisper, librosa_length,
    fps=args.fps,
    audio_padding_length_left=2,
    audio_padding_length_right=2,
)
total_frames = len(whisper_chunks)
if args.num_frames > 0:
    total_frames = min(total_frames, args.num_frames)
whisper_chunks = whisper_chunks[:total_frames]
print(f"  ✓ 音频块: {total_frames} 帧")

# ==================== 工具函数（与 demo_video 完全一致）====================
timesteps = torch.tensor([0], device=device)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def compose_frame(idx, res_face):
    bbox = coord_list_cycle[idx % cycle_len]
    ori = copy.deepcopy(frame_list_cycle[idx % cycle_len])
    x1, y1, x2, y2 = bbox
    try:
        res = cv2.resize(res_face.astype(np.uint8), (x2 - x1, y2 - y1))
    except Exception:
        return ori
    mask = mask_list_cycle[idx % cycle_len]
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

# ==================== 基线推理（Teacher 全帧，原始 MuseTalk）====================
print(f"\n[基线推理：Teacher 全帧，{total_frames} 帧]")
baseline_out = []
baseline_t = []

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
        ).to(device=device, dtype=torch.float32)

        audio_feat = pe(w_batch)

        sync()
        t0 = time.time()
        pred = teacher_unet(lat_batch, timesteps, encoder_hidden_states=audio_feat, return_dict=False)[0]
        pred = pred.to(dtype=vae.vae.dtype)
        recon = vae.decode_latents(pred)
        sync()
        elapsed = time.time() - t0

        per_frame = elapsed / bs
        for k, face in enumerate(recon):
            idx = i + k
            if hasattr(face, "permute"):
                face = (face.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            composed = compose_frame(idx, face)
            baseline_out.append(composed)
            baseline_t.append(per_frame)

        if (i + bs) % 50 < args.batch_size or (i + bs) >= total_frames:
            print(f"  [{i+bs}/{total_frames}] {1/per_frame:.1f} FPS")

fps_base = total_frames / sum(baseline_t)
print(f"  基线完成：{fps_base:.1f} FPS")

# ==================== MATS 推理（Teacher + 像素帧缓存）====================
# 复用 demo_video 的 MATS 逻辑，用 Teacher
print(f"\n[MATS 推理：Teacher + 像素帧缓存，阈值={args.threshold}，max_skip={args.max_skip}]")
mats_teacher_out = []
mats_teacher_t = []
skip_flags_teacher = []
prev_lat = None
cached_pixel = None
skip_count_t = 0
consec_skip = 0

with torch.no_grad():
    for i in range(total_frames):
        lat = input_latent_list_cycle[i % cycle_len].to(device=device, dtype=torch.float32)
        if prev_lat is not None:
            motion = float((lat.float() - prev_lat.float()).norm() /
                          (prev_lat.float().norm() + 1e-6))
        else:
            motion = 999.0
        max_skip_hit = (args.max_skip > 0 and consec_skip >= args.max_skip)
        need_compute = (motion >= args.threshold or cached_pixel is None or max_skip_hit)
        sync()
        t0 = time.time()
        if need_compute:
            wc = whisper_chunks[i]
            w = wc.cpu().unsqueeze(0).to(device) if isinstance(wc, torch.Tensor) \
                else torch.from_numpy(np.array([wc])).to(device)
            audio_feat = pe(w)
            pred = teacher_unet(lat.unsqueeze(0) if lat.dim() == 3 else lat, timesteps,
                               encoder_hidden_states=audio_feat, return_dict=False)[0]
            pred = pred.to(dtype=vae.vae.dtype)
            faces = vae.decode_latents(pred)
            face = faces[0]
            if hasattr(face, "permute"):
                face = (face.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cached_pixel = face.copy()
            skipped = False
            consec_skip = 0
        else:
            face = cached_pixel
            skipped = True
            skip_count_t += 1
            consec_skip += 1
        sync()
        elapsed = time.time() - t0
        composed = compose_frame(i, face)
        mats_teacher_out.append(composed)
        mats_teacher_t.append(elapsed)
        skip_flags_teacher.append(skipped)
        prev_lat = lat.clone()
        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            print(f"  [{i+1}/{total_frames}] {1/np.mean(mats_teacher_t[-20:]):.1f} FPS  "
                  f"跳过率={skip_count_t/(i+1):.1%}")

fps_mats_teacher = total_frames / sum(mats_teacher_t)
print(f"  MATS(Teacher) 完成：{fps_mats_teacher:.1f} FPS")

# ==================== Student 全帧推理 ====================
print(f"\n[Student 全帧推理，{total_frames} 帧]")
student_out = []
student_t = []

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
        ).to(device=device, dtype=torch.float32)
        audio_feat = pe(w_batch)
        sync()
        t0 = time.time()
        pred = student_unet(lat_batch, timesteps, encoder_hidden_states=audio_feat, return_dict=False)[0]
        pred = pred.to(dtype=vae.vae.dtype)
        recon = vae.decode_latents(pred)
        sync()
        elapsed = time.time() - t0
        per_frame = elapsed / bs
        for k, face in enumerate(recon):
            idx = i + k
            if hasattr(face, "permute"):
                face = (face.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            composed = compose_frame(idx, face)
            student_out.append(composed)
            student_t.append(per_frame)
        if (i + bs) % 50 < args.batch_size or (i + bs) >= total_frames:
            print(f"  [{i+bs}/{total_frames}] {1/per_frame:.1f} FPS")

fps_student = total_frames / sum(student_t)
print(f"  Student 全帧完成：{fps_student:.1f} FPS")

# ==================== MATS + Student 推理 ====================
print(f"\n[MATS + Student 推理，阈值={args.threshold}，max_skip={args.max_skip}]")
mats_student_out = []
mats_student_t = []
skip_flags_student = []
prev_lat = None
cached_pixel = None
skip_count_s = 0
unet_count = 0
consec_skip = 0

with torch.no_grad():
    for i in range(total_frames):
        lat = input_latent_list_cycle[i % cycle_len].to(device=device, dtype=torch.float32)
        if prev_lat is not None:
            motion = float((lat.float() - prev_lat.float()).norm() /
                          (prev_lat.float().norm() + 1e-6))
        else:
            motion = 999.0
        max_skip_hit = (args.max_skip > 0 and consec_skip >= args.max_skip)
        need_compute = (motion >= args.threshold or cached_pixel is None or max_skip_hit)
        sync()
        t0 = time.time()
        if need_compute:
            wc = whisper_chunks[i]
            w = wc.cpu().unsqueeze(0).to(device) if isinstance(wc, torch.Tensor) \
                else torch.from_numpy(np.array([wc])).to(device)
            audio_feat = pe(w)
            pred = student_unet(lat.unsqueeze(0) if lat.dim() == 3 else lat, timesteps,
                               encoder_hidden_states=audio_feat, return_dict=False)[0]
            pred = pred.to(dtype=vae.vae.dtype)
            faces = vae.decode_latents(pred)
            face = faces[0]
            if hasattr(face, "permute"):
                face = (face.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cached_pixel = face.copy()
            skipped = False
            unet_count += 1
            consec_skip = 0
        else:
            face = cached_pixel
            skipped = True
            skip_count_s += 1
            consec_skip += 1
        sync()
        elapsed = time.time() - t0
        composed = compose_frame(i, face)
        mats_student_out.append(composed)
        mats_student_t.append(elapsed)
        skip_flags_student.append(skipped)
        prev_lat = lat.clone()
        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            print(f"  [{i+1}/{total_frames}] {1/np.mean(mats_student_t[-20:]):.1f} FPS  "
                  f"跳过率={skip_count_s/(i+1):.1%}")

fps_mats_student = total_frames / sum(mats_student_t)
skip_rate = skip_count_s / total_frames
print(f"  MATS+Student 完成：{fps_mats_student:.1f} FPS  跳过率={skip_rate:.1%}")

# ==================== SSIM / PSNR ====================
def _ssim_psnr(img1, img2, win_size=11, sigma=1.5):
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

print(f"\n[质量评估：SSIM / PSNR（MATS+Student vs 基线）]")
ssim_vals, psnr_vals = [], []
for b_f, m_f in zip(baseline_out, mats_student_out):
    s, p = _ssim_psnr(b_f, m_f)
    ssim_vals.append(s)
    psnr_vals.append(p)
mean_ssim = float(np.mean(ssim_vals))
mean_psnr = float(np.mean(psnr_vals))
print(f"  SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.2f} dB")

# ==================== 合成 2×2 对比视频 ====================
print(f"\n[合成 2×2 对比视频]")
target_h = min(baseline_out[0].shape[0], 360)

def resize_h(f, h):
    oh, ow = f.shape[:2]
    nw = int(ow * h / oh)
    return cv2.resize(f, (nw, h))

def add_label(f, label, color):
    out = f.copy()
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out

comparison_frames = []
for i in range(total_frames):
    b = resize_h(baseline_out[i], target_h)
    mt = resize_h(mats_teacher_out[i], target_h)
    s = resize_h(student_out[i], target_h)
    ms = resize_h(mats_student_out[i], target_h)
    b = add_label(add_overlay(b, ["Baseline", "Teacher 全帧"], rolling_fps(baseline_t, i), False, (200, 200, 255)), "1.Baseline", (200, 200, 255))
    mt = add_label(add_overlay(mt, ["MATS", f"Skip:{skip_flags_teacher[i]}"], rolling_fps(mats_teacher_t, i), skip_flags_teacher[i], (200, 255, 200)), "2.MATS(Teacher)", (200, 255, 200))
    s = add_label(add_overlay(s, ["Student", "全帧"], rolling_fps(student_t, i), False, (255, 200, 200)), "3.Student", (255, 200, 200))
    ms = add_label(add_overlay(ms, ["MATS+Student", f"Skip:{skip_flags_student[i]}"], rolling_fps(mats_student_t, i), skip_flags_student[i], (255, 255, 200)), "4.MATS+Student", (255, 255, 200))
    # 统一宽度
    w_target = b.shape[1]
    mt = cv2.resize(mt, (w_target, target_h))
    s = cv2.resize(s, (w_target, target_h))
    ms = cv2.resize(ms, (w_target, target_h))
    div_v = np.zeros((target_h, 4, 3), dtype=np.uint8)
    div_v[:] = (60, 60, 60)
    div_h = np.zeros((4, w_target * 2 + 4, 3), dtype=np.uint8)
    div_h[:] = (60, 60, 60)
    row1 = np.concatenate([b, div_v, mt], axis=1)
    row2 = np.concatenate([s, div_v, ms], axis=1)
    grid = np.concatenate([row1, div_h, row2], axis=0)
    stat = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
    stat[:] = (40, 40, 40)
    info = (f"Frame {i+1}/{total_frames}  "
            f"Baseline {fps_base:.1f}  MATS(T) {fps_mats_teacher:.1f}  "
            f"Student {fps_student:.1f}  MATS+S {fps_mats_student:.1f} FPS")
    cv2.putText(stat, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA)
    grid = np.concatenate([grid, stat], axis=0)
    comparison_frames.append(grid)

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

base_dir = os.path.dirname(args.out)
write_video_with_audio(comparison_frames, args.audio, args.out, args.fps)
print(f"\n  ✓ 2×2 对比视频: {args.out}")

baseline_path = os.path.join(base_dir, "1_baseline_musetalk.mp4")
mats_teacher_path = os.path.join(base_dir, "2_mats_teacher.mp4")
student_path = os.path.join(base_dir, "3_student_full.mp4")
mats_student_path = os.path.join(base_dir, "4_mats_student.mp4")
write_video_with_audio(baseline_out, args.audio, baseline_path, args.fps)
write_video_with_audio(mats_teacher_out, args.audio, mats_teacher_path, args.fps)
write_video_with_audio(student_out, args.audio, student_path, args.fps)
write_video_with_audio(mats_student_out, args.audio, mats_student_path, args.fps)
print(f"  ✓ 1. 基线: {baseline_path}")
print(f"  ✓ 2. MATS(Teacher): {mats_teacher_path}")
print(f"  ✓ 3. Student: {student_path}")
print(f"  ✓ 4. MATS+Student: {mats_student_path}")
print(f"""
======================================================================
  Demo 汇总（4 路对比）
======================================================================
  1. 基线（Teacher 全帧）：{baseline_path}  {fps_base:.1f} FPS
  2. MATS（Teacher+缓存）：{mats_teacher_path}  {fps_mats_teacher:.1f} FPS
  3. Student（全帧）：{student_path}  {fps_student:.1f} FPS
  4. MATS+Student：{mats_student_path}  {fps_mats_student:.1f} FPS  跳过率 {skip_rate:.1%}
  2×2 对比：{args.out}
  质量（MATS+Student vs 基线）：SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.2f} dB
""")
