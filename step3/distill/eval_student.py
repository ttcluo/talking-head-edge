"""
学生模型评估脚本：与 Teacher 和 Baseline 对比 SSIM/PSNR/LSE-C/FPS。

使用方式：
  cd $MUSE_ROOT
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/distill/eval_student.py \
      --student_ckpt exp_out/distill/distill_v1/student_unet_final.pth \
      --student_config $REPO/step3/distill/configs/student_musetalk.json \
      --avatar_id avator_1 \
      --audio data/audio/yongen.wav \
      --num_frames 200
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
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from tqdm import tqdm

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import read_imgs
from musetalk.whisper.audio2feature import Audio2Feature

parser = argparse.ArgumentParser()
parser.add_argument("--student_ckpt",   type=str, required=True)
parser.add_argument("--student_config", type=str, required=True)
parser.add_argument("--avatar_id",      type=str, default="avator_1")
parser.add_argument("--audio",          type=str, default="data/audio/yongen.wav")
parser.add_argument("--num_frames",     type=int, default=200)
parser.add_argument("--version",        type=str, default="v15")
parser.add_argument("--unet_config",    type=str, default="models/musetalkV15/musetalk.json")
parser.add_argument("--unet_model_path",type=str, default="models/musetalkV15/unet.pth")
parser.add_argument("--out_dir",        type=str, default="profile_results/student_eval")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("  Student UNet 评估")
print("=" * 60)

# ==================== 加载预处理数据 ====================
avatar_dir = f"results/{args.version}/avatars/{args.avatar_id}"
coord_path = os.path.join(avatar_dir, "coords.pkl")
input_dir  = os.path.join(avatar_dir, "full_imgs")
latent_dir = os.path.join(avatar_dir, "latents")

import pickle
coords_list = pickle.load(open(coord_path, "rb"))
input_img_list = read_imgs(input_dir)
latent_list    = torch.load(os.path.join(latent_dir, "unet_input_latent_list.pt"),
                             map_location=device)
print(f"  ✓ 预处理帧数: {len(input_img_list)}")

# ==================== 加载模型 ====================
print("\n[加载模型]")
vae, teacher_unet_wrapper, pe = load_all_model(
    unet_model_path=args.unet_model_path,
    unet_config=args.unet_config,
    device=device,
)
teacher_unet = teacher_unet_wrapper.model.to(device).half()
teacher_unet.eval()
vae.vae = vae.vae.to(device).half()

# 加载学生模型
with open(args.student_config) as f:
    student_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
student_unet = UNet2DConditionModel(**student_cfg)
student_unet.set_attn_processor(AttnProcessor())
ckpt = torch.load(args.student_ckpt, map_location="cpu")
student_unet.load_state_dict(ckpt, strict=False)
student_unet = student_unet.to(device).half()
student_unet.eval()

teacher_params = sum(p.numel() for p in teacher_unet.parameters())
student_params = sum(p.numel() for p in student_unet.parameters())
print(f"  Teacher 参数量: {teacher_params/1e6:.1f}M  (~{teacher_params*4/1e6:.0f}MB FP32)")
print(f"  Student 参数量: {student_params/1e6:.1f}M  (~{student_params*4/1e6:.0f}MB FP32)")
print(f"  压缩比: {teacher_params/student_params:.1f}×")

# ==================== 提取音频特征 ====================
print(f"\n[提取音频特征] {args.audio}")
audio2feat = Audio2Feature(model_path="models/whisper/tiny.pt")
whisper_feature = audio2feat.get_hubert_from_whisper(args.audio)
whisper_chunks   = audio2feat.feature2chunks(feature_array=whisper_feature, fps=25)
num_frames = min(args.num_frames, len(whisper_chunks))
print(f"  ✓ 音频块: {num_frames} 帧")

# ==================== SSIM/PSNR 工具 ====================

def _gaussian_kernel(size=11, sigma=1.5):
    x = np.arange(size) - size // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g).astype(np.float32)

_KERNEL = _gaussian_kernel()

def ssim_psnr(a, b):
    """a, b: uint8 BGR [H,W,3]"""
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    # SSIM
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_vals = []
    for ch in range(3):
        ac, bc = a[:, :, ch], b[:, :, ch]
        mu_a  = cv2.filter2D(ac, -1, _KERNEL)
        mu_b  = cv2.filter2D(bc, -1, _KERNEL)
        sig_a = cv2.filter2D(ac * ac, -1, _KERNEL) - mu_a ** 2
        sig_b = cv2.filter2D(bc * bc, -1, _KERNEL) - mu_b ** 2
        sig_ab= cv2.filter2D(ac * bc, -1, _KERNEL) - mu_a * mu_b
        ssim_map = ((2*mu_a*mu_b+C1)*(2*sig_ab+C2)) / ((mu_a**2+mu_b**2+C1)*(sig_a+sig_b+C2))
        ssim_vals.append(ssim_map.mean())
    ssim = np.mean(ssim_vals)
    mse  = np.mean((a - b) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    return float(ssim), float(psnr)

def compose_frame(latent_pred, idx):
    """解码 latent → 合成完整帧"""
    with torch.no_grad():
        img = vae.vae.decode(latent_pred / vae.vae.config.scaling_factor).sample
    img = (img.clamp(-1, 1) + 1) / 2 * 255
    img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 粘回原图
    frame_idx = idx % len(input_img_list)
    full = input_img_list[frame_idx].copy()
    y1, y2, x1, x2 = coords_list[frame_idx]
    patch = cv2.resize(img, (x2 - x1, y2 - y1))
    full[y1:y2, x1:x2] = patch
    return full

# ==================== 推理 + 评估 ====================

def run_inference(unet, label):
    frames = []
    ssim_list, psnr_list = [], []
    t_start = time.time()

    for i in tqdm(range(num_frames), desc=label):
        af = whisper_chunks[i]
        if isinstance(af, torch.Tensor):
            af = af.cpu().numpy()
        af = torch.from_numpy(af).unsqueeze(0).to(device).half()
        af = pe(af)

        lat = latent_list[i % len(latent_list)].to(device).half()
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)

        t = torch.tensor([0], dtype=torch.long, device=device)

        with torch.no_grad():
            pred = unet(lat, t, encoder_hidden_states=af, return_dict=False)[0]

        frame = compose_frame(pred, i)
        frames.append(frame)

    elapsed = time.time() - t_start
    fps = num_frames / elapsed

    # 和 Teacher 帧对比 SSIM/PSNR（如果已有 teacher_frames）
    return frames, fps

print(f"\n[Teacher 推理]")
teacher_frames, teacher_fps = run_inference(teacher_unet, "Teacher")
print(f"  Teacher FPS: {teacher_fps:.1f}")

print(f"\n[Student 推理]")
student_frames, student_fps = run_inference(student_unet, "Student")
print(f"  Student FPS: {student_fps:.1f}")

# ==================== 对比质量 ====================
print(f"\n[质量评估：Student vs Teacher]")
ssim_vals, psnr_vals = [], []
for tf, sf in zip(teacher_frames, student_frames):
    s, p = ssim_psnr(tf, sf)
    ssim_vals.append(s)
    psnr_vals.append(p)
mean_ssim = np.mean(ssim_vals)
mean_psnr = np.mean(psnr_vals)

print(f"  SSIM={mean_ssim:.4f}  PSNR={mean_psnr:.2f} dB  (Student vs Teacher)")

# ==================== 保存视频 ====================
h, w = teacher_frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer_t = cv2.VideoWriter(os.path.join(args.out_dir, "teacher.mp4"), fourcc, 25, (w, h))
writer_s = cv2.VideoWriter(os.path.join(args.out_dir, "student.mp4"), fourcc, 25, (w, h))
for tf, sf in zip(teacher_frames, student_frames):
    writer_t.write(tf)
    writer_s.write(sf)
writer_t.release()
writer_s.release()

# ==================== 汇总 ====================
print(f"""
======================================================================
  评估汇总
======================================================================
  Teacher: {teacher_params/1e6:.1f}M 参数  {teacher_fps:.1f} FPS
  Student: {student_params/1e6:.1f}M 参数  {student_fps:.1f} FPS
  压缩比:   {teacher_params/student_params:.1f}×  速度提升: {student_fps/teacher_fps:.2f}×
  SSIM (Student vs Teacher): {mean_ssim:.4f}
  PSNR (Student vs Teacher): {mean_psnr:.2f} dB
  输出: {args.out_dir}/teacher.mp4  /student.mp4
======================================================================
""")

# 保存 JSON 结果
import json as _json
result = {
    "teacher_params_M": round(teacher_params/1e6, 1),
    "student_params_M": round(student_params/1e6, 1),
    "compression_ratio": round(teacher_params/student_params, 2),
    "teacher_fps": round(teacher_fps, 1),
    "student_fps": round(student_fps, 1),
    "speedup": round(student_fps/teacher_fps, 2),
    "ssim": round(mean_ssim, 4),
    "psnr_db": round(mean_psnr, 2),
}
with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
    _json.dump(result, f, indent=2)
print(f"  ✓ 结果 JSON: {args.out_dir}/eval_results.json")
