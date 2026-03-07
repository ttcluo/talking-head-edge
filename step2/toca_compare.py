"""
ToCa / PAB 帧间 Attention 缓存 vs MATS vs Baseline 对比

ToCa 原始论文（ICLR 2025）在同一 denoising step 内缓存"不重要"的 token；
本实现将其思路适配到帧间场景：

  AttnCache（帧间 Attention 广播）：
    - refresh 帧：正常运行 UNet 所有 attention，保存 cross-attention 输出
    - cache 帧：用上一 refresh 帧的 cross-attention 输出替换当前帧的运算
                self-attention 仍正常运行（保留几何/空间信息）

  这与 PAB（Pyramid Attention Broadcast）的 frame-level 变体思路一致：
  跨帧广播 cross-attention（audio 条件），self-attention 每帧运算。

比较方案：
  1. Baseline     - 全量 UNet（batch_size=4）
  2. MATS         - 输出像素缓存，运动自适应（max_skip=2）
  3. AttnCache-2  - cross-attn 每 2 帧 refresh，self-attn 每帧运行
  4. AttnCache-4  - cross-attn 每 4 帧 refresh，self-attn 每帧运行
  5. AttnCache-Adp- cross-attn 运动自适应 refresh（同 MATS 阈值）

使用方式：
  cd $MUSE_ROOT
  PYTHONPATH=$MUSE_ROOT python $REPO/step2/toca_compare.py \\
      --avatar_id avator_1 --audio data/audio/yongen.wav \\
      --threshold 0.15 --max_skip 2 --num_frames 200 \\
      --out profile_results/toca_compare.mp4
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
import torch.nn as nn

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--avatar_id",    type=str,   default="avator_1")
parser.add_argument("--audio",        type=str,   default="data/audio/yongen.wav")
parser.add_argument("--threshold",    type=float, default=0.15)
parser.add_argument("--max_skip",     type=int,   default=2)
parser.add_argument("--ac_intervals", type=int,   nargs="+", default=[2, 4],
                    help="AttnCache 固定 refresh 间隔列表")
parser.add_argument("--num_frames",   type=int,   default=200)
parser.add_argument("--fps",          type=int,   default=25)
parser.add_argument("--batch_size",   type=int,   default=4)
parser.add_argument("--out",          type=str,   default="profile_results/toca_compare.mp4")
parser.add_argument("--version",      type=str,   default="v15")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("  ToCa/PAB 帧间 Attention 缓存 vs MATS vs Baseline")
print("=" * 70)

# ==================== 加载预处理数据 ====================
avatar_path = f"results/{args.version}/avatars/{args.avatar_id}"
if not os.path.exists(avatar_path):
    print(f"  ✗ 预处理数据不存在: {avatar_path}")
    sys.exit(1)

print(f"\n[加载预处理数据] {avatar_path}")
input_latent_list_cycle = torch.load(f"{avatar_path}/latents.pt")
with open(f"{avatar_path}/coords.pkl", "rb") as f:
    coord_list_cycle = pickle.load(f)
with open(f"{avatar_path}/mask_coords.pkl", "rb") as f:
    mask_coords_list_cycle = pickle.load(f)

img_list = sorted(
    glob.glob(os.path.join(avatar_path, "full_imgs", "*.[jpJP][pnPN]*[gG]")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)
mask_list_files = sorted(
    glob.glob(os.path.join(avatar_path, "mask", "*.[jpJP][pnPN]*[gG]")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)
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


# ==================== AttnCache 引擎 ====================
class AttnCacheHook:
    """
    帧间 Cross-Attention 广播（ToCa / PAB 帧间变体）。

    策略：
      - cross-attention（audio 条件）：帧间缓存，跨 N 帧共享
      - self-attention（空间/几何信息）：每帧正常运行

    实现方式：用 register_forward_hook 拦截 BasicTransformerBlock.attn2
    的输出，在 cache 帧直接返回上一 refresh 帧的输出。

    注意：hook 只能修改输出，无法跳过计算。因此：
      - cache 帧仍然运行 cross-attention 的前向，然后被 hook 替换输出
      - 这与 ToCa 的"跳过 KV 投影"有所不同，但等价于最简单的广播策略
      - 计算节省来自 VAE decode 和 pe() 的优化（通过 MATS 对比体现差异）
    """

    def __init__(self, unet_model):
        self.m = unet_model
        self._hooks = []
        self._cache = {}        # block_id -> cached cross-attn output
        self._is_refresh = True
        self._block_idx = 0
        self._install_hooks()

    def _install_hooks(self):
        """在所有 BasicTransformerBlock 的 attn2（cross-attention）上注册 hook。"""
        def make_hook(block_id):
            def hook(module, input, output):
                if self._is_refresh:
                    self._cache[block_id] = output.detach().clone()
                    return output
                else:
                    # cache 帧：用缓存输出替换当前输出
                    if block_id in self._cache:
                        return self._cache[block_id]
                    return output
            return hook

        for name, module in self.m.named_modules():
            # BasicTransformerBlock.attn2 是 cross-attention
            if name.endswith(".attn2") and hasattr(module, "to_q"):
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

        print(f"  ✓ AttnCache：注册 {len(self._hooks)} 个 cross-attn hook")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        self._cache.clear()
        self._is_refresh = True
        self._block_idx = 0

    @torch.no_grad()
    def forward(self, latent, timestep, audio_feat, is_refresh):
        self._is_refresh = is_refresh
        out = self.m(latent, timestep, encoder_hidden_states=audio_feat).sample
        return out


# ==================== 工具函数 ====================
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def compose_frame(idx, res_face):
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


def _ssim_psnr(img1, img2, win_size=11, sigma=1.5):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 1e-10 else 100.0
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    kernel = cv2.getGaussianKernel(win_size, sigma)
    k2d = (kernel @ kernel.T).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.filter2D(g1, -1, k2d); mu2 = cv2.filter2D(g2, -1, k2d)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
    s1  = cv2.filter2D(g1*g1, -1, k2d) - mu1_sq
    s2  = cv2.filter2D(g2*g2, -1, k2d) - mu2_sq
    s12 = cv2.filter2D(g1*g2, -1, k2d) - mu12
    ssim_map = ((2*mu12+C1)*(2*s12+C2)) / ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))
    return float(ssim_map.mean()), float(psnr)


def batch_quality(ref_list, tgt_list):
    ss, ps = [], []
    for r, t in zip(ref_list, tgt_list):
        s, p = _ssim_psnr(r, t)
        ss.append(s); ps.append(p)
    return float(np.mean(ss)), float(np.mean(ps))


def write_video(frames, audio_path, out_path, fps):
    tmp = out_path.replace(".mp4", "_tmp.mp4")
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames: wr.write(fr)
    wr.release()
    cmd = (f"ffmpeg -loglevel error -nostdin -y "
           f"-i {tmp} -i {audio_path} "
           f"-c:v libx264 -crf 20 -preset fast "
           f"-c:a aac -b:a 128k -shortest {out_path}")
    ret = subprocess.call(cmd, shell=True)
    if ret == 0: os.remove(tmp)
    else: os.rename(tmp, out_path)


def get_audio_feat(i):
    wc = whisper_chunks[i]
    w = wc.cpu().unsqueeze(0).to(device) if isinstance(wc, torch.Tensor) \
        else torch.from_numpy(np.array([wc])).to(device)
    return pe(w)


def get_lat(i):
    return input_latent_list_cycle[i % cycle_len].to(
        device=device, dtype=weight_dtype)


def decode_face(pred):
    pred = pred.to(dtype=vae.vae.dtype)
    return vae.decode_latents(pred)[0]


# ==================== 1. 基线推理 ====================
print(f"\n[1/N] 基线推理（batch_size={args.batch_size}，{total_frames} 帧）")
baseline_out, baseline_t = [], []

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

        per = elapsed / bs
        for k, face in enumerate(recon):
            baseline_out.append(compose_frame(i + k, face))
            baseline_t.append(per)

        if (i + bs) % 50 < args.batch_size or (i + bs) >= total_frames:
            print(f"  [{i+bs}/{total_frames}] {1/per:.1f} FPS")

fps_base = total_frames / sum(baseline_t)
print(f"  基线完成：{fps_base:.1f} FPS")

# ==================== 2. MATS 推理 ====================
print(f"\n[2/N] MATS 推理（thr={args.threshold}，max_skip={args.max_skip}）")
mats_out, mats_t = [], []
prev_lat_mats = None
cached_pixel = None
skip_count = unet_count = consec_skip = 0

with torch.no_grad():
    for i in range(total_frames):
        lat = get_lat(i)
        motion = float((lat.float() - prev_lat_mats.float()).norm() /
                       (prev_lat_mats.float().norm() + 1e-6)) \
                 if prev_lat_mats is not None else 999.0
        max_skip_hit = (args.max_skip > 0 and consec_skip >= args.max_skip)
        need_compute = (motion >= args.threshold or cached_pixel is None or max_skip_hit)

        sync(); t0 = time.time()
        if need_compute:
            af = get_audio_feat(i)
            pred = unet.model(lat, timesteps, encoder_hidden_states=af).sample
            face = decode_face(pred)
            cached_pixel = face.copy()
            unet_count += 1; consec_skip = 0; skipped = False
        else:
            face = cached_pixel
            skip_count += 1; consec_skip += 1; skipped = True
        sync(); elapsed = time.time() - t0

        mats_out.append(compose_frame(i, face))
        mats_t.append(elapsed)
        prev_lat_mats = lat.clone()

        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            print(f"  [{i+1}/{total_frames}] {1/np.mean(mats_t[-20:]):.1f} FPS  "
                  f"跳过率={skip_count/(i+1):.1%}")

fps_mats  = total_frames / sum(mats_t)
skip_mats = skip_count / total_frames
ssim_mats, psnr_mats = batch_quality(baseline_out, mats_out)
print(f"  MATS 完成：{fps_mats:.1f} FPS  SSIM={ssim_mats:.4f}  PSNR={psnr_mats:.2f} dB")

# ==================== 3. AttnCache（固定间隔）====================
ac_results = {}
ac_engine = AttnCacheHook(unet.model)

for interval in args.ac_intervals:
    label = f"AC-{interval}"
    print(f"\n[3/N] AttnCache-{interval}（cross-attn 每 {interval} 帧 refresh）")
    ac_engine.reset()
    ac_out, ac_t = [], []
    refresh_count = 0
    frame_idx = 0

    with torch.no_grad():
        for i in range(total_frames):
            lat = get_lat(i)
            af  = get_audio_feat(i)
            is_refresh = (frame_idx % interval == 0)
            frame_idx += 1

            sync(); t0 = time.time()
            pred = ac_engine.forward(lat, timesteps, af, is_refresh)
            face = decode_face(pred)
            sync(); elapsed = time.time() - t0

            ac_out.append(compose_frame(i, face))
            ac_t.append(elapsed)
            if is_refresh:
                refresh_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == total_frames:
                print(f"  [{i+1}/{total_frames}] {1/np.mean(ac_t[-20:]):.1f} FPS  "
                      f"refresh率={refresh_count/(i+1):.1%}")

    fps_ac  = total_frames / sum(ac_t)
    ssim_ac, psnr_ac = batch_quality(baseline_out, ac_out)
    ac_results[label] = {
        "fps": fps_ac, "refresh_rate": refresh_count / total_frames,
        "ssim": ssim_ac, "psnr": psnr_ac, "frames": ac_out,
    }
    print(f"  {label} 完成：{fps_ac:.1f} FPS  SSIM={ssim_ac:.4f}  PSNR={psnr_ac:.2f} dB")

    vid = args.out.replace(".mp4", f"_{label}.mp4")
    write_video(ac_out, args.audio, vid, args.fps)
    print(f"  ✓ 视频: {vid}")

# ==================== 4. AttnCache-Adp（运动自适应）====================
label_adp = "AC-Adp"
print(f"\n[4/N] AttnCache-Adp（运动自适应 thr={args.threshold}）")
ac_engine.reset()
acadp_out, acadp_t = [], []
prev_lat_adp = None
refresh_count_adp = 0
frame_idx_adp = 0

with torch.no_grad():
    for i in range(total_frames):
        lat = get_lat(i)
        motion = float((lat.float() - prev_lat_adp.float()).norm() /
                       (prev_lat_adp.float().norm() + 1e-6)) \
                 if prev_lat_adp is not None else 999.0
        af = get_audio_feat(i)
        is_refresh = (motion >= args.threshold or frame_idx_adp == 0)
        frame_idx_adp += 1

        sync(); t0 = time.time()
        pred = ac_engine.forward(lat, timesteps, af, is_refresh)
        face = decode_face(pred)
        sync(); elapsed = time.time() - t0

        acadp_out.append(compose_frame(i, face))
        acadp_t.append(elapsed)
        if is_refresh:
            refresh_count_adp += 1
        prev_lat_adp = lat.clone()

        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            print(f"  [{i+1}/{total_frames}] {1/np.mean(acadp_t[-20:]):.1f} FPS  "
                  f"refresh率={refresh_count_adp/(i+1):.1%}")

fps_acadp  = total_frames / sum(acadp_t)
ssim_acadp, psnr_acadp = batch_quality(baseline_out, acadp_out)
ac_results[label_adp] = {
    "fps": fps_acadp, "refresh_rate": refresh_count_adp / total_frames,
    "ssim": ssim_acadp, "psnr": psnr_acadp,
}
print(f"  {label_adp} 完成：{fps_acadp:.1f} FPS  SSIM={ssim_acadp:.4f}  PSNR={psnr_acadp:.2f} dB")
vid_adp = args.out.replace(".mp4", f"_{label_adp}.mp4")
write_video(acadp_out, args.audio, vid_adp, args.fps)
print(f"  ✓ 视频: {vid_adp}")

ac_engine.remove_hooks()

# ==================== 汇总 ====================
all_methods = {
    "Baseline": {"fps": fps_base, "speedup": 1.0, "ssim": 1.0, "psnr": float("inf"), "skip": 0.0},
    "MATS":     {"fps": fps_mats, "speedup": fps_mats/fps_base, "ssim": ssim_mats, "psnr": psnr_mats, "skip": skip_mats},
}
for k, v in ac_results.items():
    all_methods[k] = {
        "fps": v["fps"], "speedup": v["fps"]/fps_base,
        "ssim": v["ssim"], "psnr": v["psnr"],
        "skip": 1.0 - v["refresh_rate"],
    }

print(f"\n{'='*70}\n  对比结果汇总\n{'='*70}")
print(f"{'方法':<18} {'FPS':>7} {'加速':>7} {'cache率':>8} {'SSIM':>8} {'PSNR(dB)':>10}")
print("-" * 70)
for name, m in all_methods.items():
    psnr_s = f"{m['psnr']:.2f}" if m["psnr"] != float("inf") else "∞"
    print(f"  {name:<16} {m['fps']:>7.1f} {m['speedup']:>7.2f}x "
          f"{m['skip']:>8.1%} {m['ssim']:>8.4f} {psnr_s:>10}")
print("=" * 70)

# 保存 JSON
result_path = args.out.replace(".mp4", "_results.json")
save_data = {}
for name, m in all_methods.items():
    save_data[name] = {k: (str(v) if isinstance(v, float) and v == float("inf") else v)
                       for k, v in m.items()}
with open(result_path, "w") as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)
print(f"\n  ✓ 结果 JSON: {result_path}")

# 保存独立视频
write_video(mats_out,     args.audio, args.out.replace(".mp4", "_MATS.mp4"),     args.fps)
write_video(baseline_out, args.audio, args.out.replace(".mp4", "_Baseline.mp4"), args.fps)
print(f"  ✓ 所有视频已保存")
