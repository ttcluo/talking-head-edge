"""
完整优化管线对比：原始 MuseTalk（Teacher 全帧） vs 本方法（MATS + 蒸馏 Student）。

与 step2/demo_video.py 一致：同一套预处理数据（含 mask）、get_image_blending 合成、
输出左右对比视频 + 两条单独视频 + FPS/加速比 + SSIM/PSNR。
使用方式：
  cd $MUSE_ROOT
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/distill/run_full_pipeline.py \\
      --student_ckpt exp_out/distill/distill_v1/student_unet_final.pth \\
      --student_config $REPO/step3/distill/configs/student_musetalk.json \\
      --avatar_id yongen \\
      --num_frames 200 \\
      --threshold 0.15 --max_skip 2 \\
      --out_dir profile_results/full_pipeline \\
      --audio data/audio/yongen.wav
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
from tqdm import tqdm

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.getcwd())
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from musetalk.models.unet import PositionalEncoding
from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending

parser = argparse.ArgumentParser()
parser.add_argument("--student_ckpt",   type=str, required=True)
parser.add_argument("--student_config", type=str, required=True)
parser.add_argument("--avatar_id",      type=str, default="avator_1")
parser.add_argument("--num_frames",     type=int, default=200)
parser.add_argument("--threshold",      type=float, default=0.15, help="MATS 运动阈值")
parser.add_argument("--max_skip",       type=int, default=2, help="MATS 最大连续跳帧")
parser.add_argument("--version",        type=str, default="v15")
parser.add_argument("--audio_feat_dir", type=str, default="dataset/distill/audio_feats")
parser.add_argument("--out_dir",        type=str, default="profile_results/full_pipeline")
parser.add_argument("--fps",            type=int, default=25)
parser.add_argument("--audio",         type=str, default="", help="可选，合成时混入的音频 wav")
args = parser.parse_args()

os.chdir(MUSE_ROOT)
os.makedirs(args.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("  对比：原始 MuseTalk vs MATS + Student（与 demo_video 同流程）")
print("=" * 60)

# ==================== 加载预处理数据（与 demo_video.py 一致：含 mask）====================
avatar_path = f"results/{args.version}/avatars/{args.avatar_id}"
if not os.path.exists(avatar_path):
    print(f"  ✗ 预处理数据不存在: {avatar_path}")
    sys.exit(1)

input_latent_list_cycle = torch.load(os.path.join(avatar_path, "latents.pt"), map_location=device)
with open(os.path.join(avatar_path, "coords.pkl"), "rb") as f:
    coord_list_cycle = pickle.load(f)
with open(os.path.join(avatar_path, "mask_coords.pkl"), "rb") as f:
    mask_coords_list_cycle = pickle.load(f)

img_list = sorted(
    glob.glob(os.path.join(avatar_path, "full_imgs", "*.[jpJP][pnPN]*[gG]")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)
mask_list_files = sorted(
    glob.glob(os.path.join(avatar_path, "mask", "*.[jpJP][pnPN]*[gG]")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)
frame_list_cycle = read_imgs(img_list)
mask_list_cycle = read_imgs(mask_list_files)
cycle_len = len(frame_list_cycle)
print(f"  ✓ 预处理帧数: {cycle_len}  latents: {len(input_latent_list_cycle)}")

audio_feat_path = os.path.join(args.audio_feat_dir, f"{args.avatar_id}.pt")
if not os.path.exists(audio_feat_path):
    raise FileNotFoundError(
        f"找不到预计算音频特征: {audio_feat_path}\n请先运行 precompute_audio_feats.py"
    )
audio_chunks = torch.load(audio_feat_path, map_location=device)
num_frames = min(args.num_frames, len(audio_chunks))
print(f"  ✓ 音频特征: 使用前 {num_frames} 帧")

# ==================== 加载模型（VAE + Teacher + Student + PE）====================
print("\n[加载模型]")
vae, teacher_wrapper, pe_module = load_all_model(device=device)
teacher_unet = teacher_wrapper.model.to(device).float()
teacher_unet.eval()
vae.vae = vae.vae.to(device).float()
pe = pe_module.to(device)

with open(args.student_config) as f:
    student_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
student_unet = UNet2DConditionModel(**student_cfg)
student_unet.set_attn_processor(AttnProcessor())
ckpt = torch.load(args.student_ckpt, map_location="cpu")
student_unet.load_state_dict(ckpt, strict=False)
student_unet = student_unet.to(device).float()
student_unet.eval()

teacher_params = sum(p.numel() for p in teacher_unet.parameters())
student_params = sum(p.numel() for p in student_unet.parameters())
print(f"  Teacher: {teacher_params/1e6:.1f}M 参数  Student: {student_params/1e6:.1f}M 参数  压缩比: {teacher_params/student_params:.1f}×")

# ==================== 帧合成（与 demo_video 一致：get_image_blending）====================
def decode_latent_to_face(latent_pred):
    if hasattr(vae, "decode_latents"):
        recon = vae.decode_latents(latent_pred.to(dtype=vae.vae.dtype))
        face = recon[0] if hasattr(recon, "__getitem__") else recon
        if hasattr(face, "cpu"):
            face = (face.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        return face
    with torch.no_grad():
        img = vae.vae.decode(latent_pred / vae.vae.config.scaling_factor).sample
    img = (img.clamp(-1, 1) + 1) / 2 * 255
    img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def compose_frame(idx, res_face):
    """与 demo_video 一致：bbox + mask + get_image_blending"""
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
        cv2.putText(out, line, (8, 22 + j * 22), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)
    fps_str = f"{fps_val:.1f} FPS"
    tw = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0][0]
    cv2.putText(out, fps_str, (w - tw - 8, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 128), 2, cv2.LINE_AA)
    dot_c = (0, 80, 255) if skipped else (0, 220, 0)
    cv2.circle(out, (w - 16, h - 16), 8, dot_c, -1)
    cv2.putText(out, "SKIP" if skipped else "CALC", (w - 65, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_c, 1, cv2.LINE_AA)
    return out

def rolling_fps(timings, i, win=20):
    s = max(0, i - win + 1)
    return min(1.0 / max(np.mean(timings[s : i + 1]), 1e-9), 9999)

def ssim_psnr(a, b):
    """a, b: uint8 BGR [H,W,3]。用于本方法 vs 基线 画质对比。"""
    kernel = cv2.getGaussianKernel(11, 1.5)
    kernel = (kernel @ kernel.T).astype(np.float32)
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_vals = []
    for ch in range(3):
        ac, bc = a[:, :, ch], b[:, :, ch]
        mu_a = cv2.filter2D(ac, -1, kernel)
        mu_b = cv2.filter2D(bc, -1, kernel)
        sig_a = cv2.filter2D(ac * ac, -1, kernel) - mu_a ** 2
        sig_b = cv2.filter2D(bc * bc, -1, kernel) - mu_b ** 2
        sig_ab = cv2.filter2D(ac * bc, -1, kernel) - mu_a * mu_b
        ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / (
            (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a + sig_b + C2)
        )
        ssim_vals.append(ssim_map.mean())
    mse = np.mean((a - b) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    return float(np.mean(ssim_vals)), float(psnr)

# ==================== 基线：原始 MuseTalk（Teacher 全帧，无 MATS）====================
print(f"\n[基线] 原始 MuseTalk（Teacher 全帧）共 {num_frames} 帧")
t_zero_teacher = torch.tensor([0], dtype=torch.long, device=device)
baseline_frames = []
baseline_timings = []

for i in tqdm(range(num_frames), desc="Baseline"):
    lat = input_latent_list_cycle[i % cycle_len].to(device).float()
    if lat.dim() == 3:
        lat = lat.unsqueeze(0)
    af = audio_chunks[i].unsqueeze(0).to(device).float()
    af = pe(af)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        pred = teacher_unet(lat, t_zero_teacher, encoder_hidden_states=af, return_dict=False)[0]
    face_bgr = decode_latent_to_face(pred)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    full_frame = compose_frame(i, face_bgr)
    baseline_frames.append(full_frame)
    baseline_timings.append(elapsed)

fps_baseline = num_frames / sum(baseline_timings)
print(f"  基线 FPS: {fps_baseline:.1f}")

# ==================== 本方法：MATS + Student 推理 ====================
print(f"\n[MATS + Student 推理] 阈值={args.threshold} max_skip={args.max_skip} 共 {num_frames} 帧")
frames_out = []
timings = []
skip_flags = []
skip_count = 0
prev_lat = None
cached_face_bgr = None
consec_skip = 0
t_zero = torch.tensor([0], dtype=torch.long, device=device)

for i in tqdm(range(num_frames), desc="MATS+Student"):
    lat = input_latent_list_cycle[i % cycle_len].to(device).float()
    if lat.dim() == 3:
        lat = lat.unsqueeze(0)

    if prev_lat is not None:
        motion = float((lat.float() - prev_lat.float()).norm() /
                       (prev_lat.float().norm() + 1e-6))
    else:
        motion = 999.0

    max_skip_hit = (args.max_skip > 0 and consec_skip >= args.max_skip)
    need_compute = (motion >= args.threshold or cached_face_bgr is None or max_skip_hit)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    if need_compute:
        af = audio_chunks[i].unsqueeze(0).to(device).float()
        af = pe(af)
        with torch.no_grad():
            pred = student_unet(lat, t_zero, encoder_hidden_states=af, return_dict=False)[0]
        face_bgr = decode_latent_to_face(pred)
        cached_face_bgr = face_bgr.copy()
        consec_skip = 0
        skipped = False
    else:
        face_bgr = cached_face_bgr
        skip_count += 1
        consec_skip += 1
        skipped = True

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    full_frame = compose_frame(i, face_bgr)
    frames_out.append(full_frame)
    timings.append(elapsed)
    skip_flags.append(skipped)
    prev_lat = lat.clone()

fps_ours = num_frames / sum(timings)
skip_rate = skip_count / num_frames
speedup = fps_ours / fps_baseline
print(f"  本方法 FPS: {fps_ours:.1f}  跳过率: {skip_rate:.1%}  相对基线加速: {speedup:.2f}×")

# ==================== 质量对比：本方法 vs 基线（原始 MuseTalk）====================
print("\n[质量对比] 本方法（MATS+Student）vs 基线（原始 MuseTalk）")
ssim_vals, psnr_vals = [], []
for b_f, o_f in zip(baseline_frames, frames_out):
    s, p = ssim_psnr(b_f, o_f)
    ssim_vals.append(s)
    psnr_vals.append(p)
mean_ssim = float(np.mean(ssim_vals))
mean_psnr = float(np.mean(psnr_vals))
print(f"  SSIM: {mean_ssim:.4f}  PSNR: {mean_psnr:.2f} dB")

# ==================== 合成左右对比视频 + 两条单独视频（与 demo_video 一致）====================
def write_video_with_audio(frames, audio_path, out_path, fps):
    tmp = out_path.replace(".mp4", "_tmp.mp4")
    h_o, w_o = frames[0].shape[:2]
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_o, h_o))
    for fr in frames:
        wr.write(fr)
    wr.release()
    cmd = (f"ffmpeg -loglevel error -nostdin -y -i {tmp} -i {audio_path} "
           f"-c:v libx264 -crf 20 -preset fast -c:a aac -b:a 128k -shortest {out_path}")
    ret = subprocess.call(cmd, shell=True)
    if ret == 0:
        os.remove(tmp)
    else:
        os.rename(tmp, out_path)

target_h = min(baseline_frames[0].shape[0], 540)

def resize_h(f, h):
    oh, ow = f.shape[:2]
    nw = int(ow * h / oh)
    return cv2.resize(f, (nw, h))

comparison_frames = []
for i in range(num_frames):
    b_fps = rolling_fps(baseline_timings, i)
    m_fps = rolling_fps(timings, i)
    b = resize_h(baseline_frames[i], target_h)
    m = resize_h(frames_out[i], target_h)
    b = add_overlay(b, ["Baseline (MuseTalk)", "Full UNet every frame"], b_fps, False, (200, 200, 255))
    m = add_overlay(m, [f"MATS+Student (thr={args.threshold:.2f})", f"Skip: {skip_flags[i]}"], m_fps, skip_flags[i], (200, 255, 200))
    div = np.zeros((target_h, 4, 3), dtype=np.uint8)
    div[:] = (80, 80, 80)
    row = np.concatenate([b, div, m], axis=1)
    bw = row.shape[1]
    stat = np.zeros((30, bw, 3), dtype=np.uint8)
    info = (f"Frame {i+1}/{num_frames}   "
            f"Baseline {fps_baseline:.1f} FPS   "
            f"MATS+Student {fps_ours:.1f} FPS   "
            f"Speedup {speedup:.2f}x   "
            f"Skip {skip_rate:.1%}")
    cv2.putText(stat, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA)
    row = np.concatenate([row, stat], axis=0)
    comparison_frames.append(row)

comparison_mp4 = os.path.join(args.out_dir, "comparison_baseline_vs_MATS_Student.mp4")
baseline_mp4 = os.path.join(args.out_dir, "baseline_musetalk.mp4")
ours_mp4 = os.path.join(args.out_dir, "full_pipeline_MATS_Student.mp4")
audio_path = args.audio if (args.audio and os.path.isfile(args.audio)) else None
if audio_path:
    write_video_with_audio(comparison_frames, audio_path, comparison_mp4, args.fps)
    print(f"  ✓ 左右对比视频: {comparison_mp4}")
    write_video_with_audio(baseline_frames, audio_path, baseline_mp4, args.fps)
    write_video_with_audio(frames_out, audio_path, ours_mp4, args.fps)
else:
    for f, p in [(comparison_frames, comparison_mp4), (baseline_frames, baseline_mp4), (frames_out, ours_mp4)]:
        wr = cv2.VideoWriter(p.replace(".mp4", "_tmp.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (f[0].shape[1], f[0].shape[0]))
        for fr in f:
            wr.write(fr)
        wr.release()
        os.rename(p.replace(".mp4", "_tmp.mp4"), p)
    print(f"  ✓ 左右对比视频: {comparison_mp4}（无音轨）")
print(f"  ✓ 单独基线: {baseline_mp4}")
print(f"  ✓ 本方法: {ours_mp4}")

result = {
    "baseline_fps": round(fps_baseline, 1),
    "ours_fps": round(fps_ours, 1),
    "speedup": round(speedup, 2),
    "ssim_vs_baseline": round(mean_ssim, 4),
    "psnr_vs_baseline_db": round(mean_psnr, 2),
    "teacher_params_M": round(teacher_params / 1e6, 1),
    "student_params_M": round(student_params / 1e6, 1),
    "mats_threshold": args.threshold,
    "mats_max_skip": args.max_skip,
    "skip_rate": round(skip_rate, 4),
    "num_frames": num_frames,
}
with open(os.path.join(args.out_dir, "full_pipeline_results.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"""
======================================================================
  对比汇总：原始 MuseTalk vs 本方法（MATS + Student）
======================================================================
  基线（原始 MuseTalk）: {teacher_params/1e6:.1f}M 参数  {fps_baseline:.1f} FPS
  本方法（MATS+Student）: {student_params/1e6:.1f}M 参数  {fps_ours:.1f} FPS
  加速比:                {speedup:.2f}×
  质量（本方法 vs 基线）: SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.2f} dB
  输出: {args.out_dir}/
    - comparison_baseline_vs_MATS_Student.mp4  （左右对比）
    - baseline_musetalk.mp4                    （单独基线）
    - full_pipeline_MATS_Student.mp4            （本方法）
======================================================================
""")
