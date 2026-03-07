"""
MATS 视频对比 Demo
生成左右拼接对比视频：左=基线（全量 UNet），右=MATS（像素帧缓存）
文字叠加：方法名、实时 FPS、跳帧指示

使用方法：
    conda activate musetalk && cd $MUSE_ROOT
    python $REPO/step2/demo_video.py \
        --video data/video/yongen.mp4 \
        --audio data/audio/yongen.wav \
        --threshold 0.15 \
        --num_frames 200 \
        --out profile_results/mats_demo.mp4
"""

import argparse
import json
import os
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
parser.add_argument("--video",      type=str,   default="data/video/yongen.mp4")
parser.add_argument("--audio",      type=str,   default="data/audio/yongen.wav")
parser.add_argument("--threshold",  type=float, default=0.15)
parser.add_argument("--num_frames", type=int,   default=200)
parser.add_argument("--out",        type=str,   default="profile_results/mats_demo.mp4")
parser.add_argument("--fps_out",    type=int,   default=25,  help="输出视频帧率")
parser.add_argument("--face_size",  type=int,   default=256, help="人脸区域大小")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16

print("=" * 65)
print("  MATS 视频对比 Demo")
print("=" * 65)
print(f"  视频={args.video}  阈值={args.threshold}  帧数={args.num_frames}")

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ==================== 加载模型 ====================
print("\n[加载模型]")
import json as _json
from diffusers import AutoencoderKL
from musetalk.models.unet import UNet2DConditionModel

with open("models/musetalkV15/musetalk.json") as f:
    unet_config = _json.load(f)

vae  = AutoencoderKL.from_pretrained("models/sd-vae").to(device, dtype)
unet = UNet2DConditionModel(**unet_config).to(device, dtype)
unet.load_state_dict(torch.load("models/musetalkV15/unet.pth", map_location=device))
vae.eval(); unet.eval()
print("  ✓ VAE + UNet 加载完成")

# ==================== 加载音频特征 ====================
def load_audio_chunks(audio_path, num_frames):
    """优先用 OpenAI .pt，备用 HuggingFace，最后退路零向量"""
    for wp in ["models/whisper/tiny.pt", "models/whisper/whisper_tiny.pt"]:
        if not os.path.exists(wp):
            continue
        try:
            from musetalk.whisper.audio2feature import Audio2Feature
            ap = Audio2Feature(whisper_model_type="tiny", model_path=wp)
            af = ap.audio2feat(audio_path)
            chunks = ap.feature2chunks(feature_array=af, fps=25)
            print(f"  ✓ Whisper (OpenAI .pt): {len(chunks)} 块")
            return chunks
        except Exception:
            pass

    if os.path.exists("models/whisper/config.json"):
        try:
            from transformers import WhisperFeatureExtractor, WhisperModel
            import librosa as _librosa
            print("  → 使用 HuggingFace Whisper 格式...")
            hf_feat = WhisperFeatureExtractor.from_pretrained("models/whisper")
            hf_model = WhisperModel.from_pretrained("models/whisper").to(device)
            hf_model.eval()
            wav, _ = _librosa.load(audio_path, sr=16000)
            chunk_samp = 30 * 16000
            feat_list = []
            for off in range(0, max(len(wav), chunk_samp), chunk_samp):
                seg = wav[off: off + chunk_samp]
                if len(seg) == 0:
                    break
                inp = hf_feat(seg, sampling_rate=16000, return_tensors="pt",
                              padding="max_length", max_length=chunk_samp).input_features.to(device)
                with torch.no_grad():
                    enc = hf_model.encoder(inp).last_hidden_state[0]
                feat_list.append(enc.float().cpu().numpy())
            all_feats = np.concatenate(feat_list, axis=0)
            win, step = 8, 2
            chunks = []
            for fi in range(num_frames):
                center = fi * step
                s = max(0, center - win)
                e = s + 16
                if e > all_feats.shape[0]:
                    e = all_feats.shape[0]; s = max(0, e - 16)
                chunk = all_feats[s:e]
                if chunk.shape[0] < 16:
                    chunk = np.pad(chunk, ((0, 16 - chunk.shape[0]), (0, 0)), mode="edge")
                chunks.append(chunk.mean(axis=0, keepdims=True))
            print(f"  ✓ Whisper (HuggingFace): {len(chunks)} 块")
            return chunks
        except Exception as e:
            print(f"  ✗ HuggingFace Whisper 失败: {e}")

    print("  ⚠ 使用零向量音频（FPS 测量有效，但无唇形驱动）")
    return [np.zeros((1, 384), dtype=np.float32)] * num_frames

print("\n[加载音频特征]")
audio_chunks = load_audio_chunks(args.audio, args.num_frames)

# ==================== 读取视频帧 ====================
print(f"\n[读取视频] {args.video}")
cap = cv2.VideoCapture(args.video)
src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames  = []
while len(frames) < args.num_frames:
    ret, frm = cap.read()
    if not ret:
        break
    frames.append(frm)
cap.release()
N = len(frames)
print(f"  读取 {N} 帧，原始分辨率 {src_w}×{src_h}，{src_fps:.1f}fps")

S = args.face_size  # 256

def get_face_tensor(frame):
    """从帧中提取并预处理人脸区域（简单中心区域 crop）"""
    h, w = frame.shape[:2]
    # 取帧中央的正方形区域（大多数正面说话人视频的人脸在中央偏上）
    cx, cy = w // 2, h // 3
    half = min(cx, cy, S)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + 2 * half)
    y2 = min(h, y1 + 2 * half)
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (S, S))
    rgb  = face[:, :, ::-1].copy()
    t    = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1
    return t.unsqueeze(0).to(device, dtype), (x1, y1, x2, y2)

def decode_px(lat):
    out = vae.decode(lat).sample
    px  = ((out[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_RGB2BGR)

def blend_back(frame, face_bgr, bbox):
    """将生成的人脸 paste 回原帧"""
    x1, y1, x2, y2 = bbox
    rh, rw = y2 - y1, x2 - x1
    face_r = cv2.resize(face_bgr, (rw, rh))
    out = frame.copy()
    out[y1:y2, x1:x2] = face_r
    return out

ts = torch.tensor([0.0], device=device, dtype=dtype)

# ==================== 预热 ====================
print("\n[预热]")
with torch.no_grad():
    _d = torch.randn(1, 8, S // 8, S // 8, device=device, dtype=dtype)
    _a = torch.zeros(1, 1, 384, device=device, dtype=dtype)
    for _ in range(5):
        unet(_d, ts, encoder_hidden_states=_a)
        vae.decode(torch.randn(1, 4, S // 8, S // 8, device=device, dtype=dtype))
print("  ✓ 预热完成")

# ==================== 基线推理 ====================
print(f"\n[基线推理：全量 UNet，{N} 帧]")
baseline_frames, baseline_timings = [], []
t_total_base = 0.0

with torch.no_grad():
    for i, frm in enumerate(frames):
        face_t, bbox = get_face_tensor(frm)
        ac = torch.from_numpy(audio_chunks[i % len(audio_chunks)]).unsqueeze(0).to(device, dtype)
        mask = torch.zeros(1, 4, S // 8, S // 8, device=device, dtype=dtype)
        lat  = vae.encode(face_t).latent_dist.mean

        sync(); t0 = time.time()
        out = unet(torch.cat([lat, mask], 1), ts, encoder_hidden_states=ac).sample
        px  = decode_px(out)
        sync(); elapsed = time.time() - t0

        composed = blend_back(frm, px, bbox)
        baseline_frames.append(composed)
        baseline_timings.append(elapsed)
        t_total_base += elapsed

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] {1/np.mean(baseline_timings[-20:]):.1f} FPS")

fps_base = N / t_total_base
print(f"  基线完成：{fps_base:.1f} FPS")

# ==================== MATS 推理（P3+ 像素帧缓存）====================
print(f"\n[MATS 推理：像素帧缓存，阈值={args.threshold}]")
mats_frames, mats_timings, skip_flags = [], [], []
t_total_mats = 0.0
prev_lat, cached_pixel = None, None
skip_count, unet_count = 0, 0

with torch.no_grad():
    for i, frm in enumerate(frames):
        face_t, bbox = get_face_tensor(frm)
        lat  = vae.encode(face_t).latent_dist.mean

        if prev_lat is not None:
            motion = float((lat.float() - prev_lat.float()).norm() /
                           (prev_lat.float().norm() + 1e-6))
        else:
            motion = 999.0

        ac   = torch.from_numpy(audio_chunks[i % len(audio_chunks)]).unsqueeze(0).to(device, dtype)
        mask = torch.zeros(1, 4, S // 8, S // 8, device=device, dtype=dtype)

        sync(); t0 = time.time()

        if motion >= args.threshold or cached_pixel is None:
            out = unet(torch.cat([lat, mask], 1), ts, encoder_hidden_states=ac).sample
            px  = decode_px(out)
            cached_pixel = px
            unet_count += 1
            skipped = False
        else:
            px = cached_pixel
            skip_count += 1
            skipped = True

        sync(); elapsed = time.time() - t0

        composed = blend_back(frm, px, bbox)
        mats_frames.append(composed)
        mats_timings.append(elapsed)
        skip_flags.append(skipped)
        t_total_mats += elapsed
        prev_lat = lat.clone()

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] {1/np.mean(mats_timings[-20:]):.1f} FPS  "
                  f"跳过率={skip_count/(i+1):.1%}")

fps_mats = N / t_total_mats
skip_rate = skip_count / N
print(f"  MATS 完成：{fps_mats:.1f} FPS  跳过率={skip_rate:.1%}")

# ==================== 合成对比视频 ====================
print(f"\n[合成对比视频]")

def add_overlay(frame, text_lines, fps_val, skipped=False, color=(255, 255, 255)):
    """在帧上叠加文字信息"""
    out = frame.copy()
    h, w = out.shape[:2]
    # 半透明背景条
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    for j, line in enumerate(text_lines):
        cv2.putText(out, line, (8, 22 + j * 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA)

    # FPS 右上角
    fps_str = f"{fps_val:.1f} FPS"
    tw, _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0], 0
    cv2.putText(out, fps_str, (w - tw[0] - 8, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 128), 2, cv2.LINE_AA)

    # 跳帧指示（右下角红点/绿点）
    indicator_color = (0, 80, 255) if skipped else (0, 220, 0)
    cv2.circle(out, (w - 16, h - 16), 8, indicator_color, -1)
    label = "SKIP" if skipped else "CALC"
    cv2.putText(out, label, (w - 65, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, indicator_color, 1, cv2.LINE_AA)
    return out

# 调整帧高度一致
def resize_h(f, h):
    oh, ow = f.shape[:2]
    nw = int(ow * h / oh)
    return cv2.resize(f, (nw, h))

target_h = min(src_h, 480)  # 限制输出高度
speedup = fps_mats / fps_base

# 计算滑动窗口 FPS（最近 20 帧）
def rolling_fps(timings, i, win=20):
    s = max(0, i - win + 1)
    return min(1.0 / max(np.mean(timings[s:i+1]), 1e-6), 9999)

comparison_frames = []
for i in range(N):
    b_fps = rolling_fps(baseline_timings, i)
    m_fps = rolling_fps(mats_timings, i)

    b_fr = resize_h(baseline_frames[i], target_h)
    m_fr = resize_h(mats_frames[i], target_h)

    b_fr = add_overlay(b_fr,
                       ["Baseline (MuseTalk FP16)", "Full UNet every frame"],
                       b_fps, skipped=False, color=(200, 200, 255))
    m_fr = add_overlay(m_fr,
                       [f"MATS (thr={args.threshold:.2f})", f"Skip: {skip_flags[i]}"],
                       m_fps, skipped=skip_flags[i], color=(200, 255, 200))

    # 中间分割线
    divider = np.zeros((target_h, 4, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)

    row = np.concatenate([b_fr, divider, m_fr], axis=1)

    # 底部统计条
    bh, bw = row.shape[:2]
    stat = np.zeros((32, bw, 3), dtype=np.uint8)
    info = (f"Frame {i+1}/{N}   "
            f"Baseline {fps_base:.1f} FPS   "
            f"MATS {fps_mats:.1f} FPS   "
            f"Speedup {speedup:.2f}x   "
            f"Skip {skip_rate:.1%}")
    cv2.putText(stat, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 220, 180), 1, cv2.LINE_AA)
    row = np.concatenate([row, stat], axis=0)
    comparison_frames.append(row)

# 写入视频
tmp_mp4 = args.out.replace(".mp4", "_tmp.mp4")
h_out, w_out = comparison_frames[0].shape[:2]
writer = cv2.VideoWriter(tmp_mp4,
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps_out, (w_out, h_out))
for fr in comparison_frames:
    writer.write(fr)
writer.release()

# 添加音频
cmd = (f"ffmpeg -loglevel error -nostdin -y "
       f"-i {tmp_mp4} -i {args.audio} "
       f"-c:v libx264 -crf 20 -preset fast "
       f"-c:a aac -b:a 128k -shortest {args.out}")
ret = subprocess.call(cmd, shell=True)
if ret == 0:
    os.remove(tmp_mp4)
    print(f"  ✓ 对比视频保存到: {args.out}")
else:
    os.rename(tmp_mp4, args.out)
    print(f"  ⚠ ffmpeg 编码失败，保存为未压缩版: {args.out}")

# ==================== 汇总 ====================
print(f"""
======================================================================
  Demo 结果汇总
======================================================================
  基线：{fps_base:.1f} FPS
  MATS：{fps_mats:.1f} FPS  (加速 {speedup:.2f}×)
  跳过率：{skip_rate:.1%}  (UNet 实际调用 {unet_count}/{N} 帧)
  输出：{args.out}
""")

result = {
    "baseline_fps": round(fps_base, 1),
    "mats_fps":     round(fps_mats, 1),
    "speedup":      round(speedup, 2),
    "skip_rate":    round(skip_rate, 4),
    "threshold":    args.threshold,
    "num_frames":   N,
    "output":       args.out,
}
json_out = args.out.replace(".mp4", ".json")
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"  数字已保存: {json_out}")
