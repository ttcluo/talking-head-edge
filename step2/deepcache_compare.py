"""
DeepCache vs MATS vs Baseline 对比实验（MuseTalk 帧级缓存）

DeepCache 原始论文将编码器特征跨去噪 timestep 复用；
本实现将其适配到 MuseTalk 的帧间场景：
  - refresh 帧：运行完整 UNet，缓存 (down_block_res_samples, mid_sample)
  - cache 帧：跳过 Encoder（down_blocks + mid_block），
              仅运行 Decoder（up_blocks）+ 当前帧音频 cross-attention

三种方案对比：
  1. Baseline      ：每帧完整 UNet（batch_size=4 基线）
  2. MATS          ：输出像素级缓存，运动自适应（max_skip=2）
  3. DeepCache-N   ：特征级缓存，固定每 N 帧 refresh（N=2, 4）
  4. DeepCache-Adp ：特征级缓存，运动自适应 refresh（与 MATS 同阈值）

使用方法（在 MuseTalk 根目录下运行）：
    conda activate musetalk && cd $MUSE_ROOT
    PYTHONPATH=$MUSE_ROOT python $REPO/step2/deepcache_compare.py \\
        --avatar_id avator_1 --audio data/audio/yongen.wav \\
        --threshold 0.15 --num_frames 200 \\
        --out profile_results/deepcache_compare.mp4
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
parser.add_argument("--avatar_id",  type=str,   default="avator_1")
parser.add_argument("--audio",      type=str,   default="data/audio/yongen.wav")
parser.add_argument("--threshold",  type=float, default=0.15,
                    help="MATS / DeepCache-Adp 运动阈值")
parser.add_argument("--max_skip",   type=int,   default=2,
                    help="MATS 最大连续跳帧数")
parser.add_argument("--dc_intervals", type=int, nargs="+", default=[2, 4],
                    help="DeepCache 固定 refresh 间隔列表，例如 --dc_intervals 2 4")
parser.add_argument("--num_frames", type=int,   default=200)
parser.add_argument("--fps",        type=int,   default=25)
parser.add_argument("--batch_size", type=int,   default=4)
parser.add_argument("--out",        type=str,   default="profile_results/deepcache_compare.mp4")
parser.add_argument("--version",    type=str,   default="v15")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("  DeepCache vs MATS vs Baseline 对比（MuseTalk 帧级缓存）")
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


# ==================== DeepCache 引擎 ====================
class DeepCacheEngine:
    """
    帧级 DeepCache：缓存 UNet Encoder 特征，Decoder 每帧运行（含当前音频）。

    UNet 分段：
      Encoder: conv_in → down_blocks → mid_block   (~60% 计算量)
      Decoder: up_blocks → conv_norm_out → conv_out (~40% 计算量)

    refresh 帧：运行完整 UNet，更新缓存
    cache 帧：  跳过 Encoder，用缓存 skip-connections + mid，
               仅运行 Decoder（cross-attention 用当前音频）
    """

    def __init__(self, unet_model):
        self.m = unet_model
        self.cache = None   # {'down_samples': tuple[Tensor], 'mid': Tensor}
        self.frame_idx = 0

    def reset(self):
        self.cache = None
        self.frame_idx = 0

    @torch.no_grad()
    def _time_emb(self, timestep, dtype):
        t = self.m.time_proj(timestep).to(dtype=dtype)
        return self.m.time_embedding(t)

    @torch.no_grad()
    def _run_encoder(self, x, emb, audio_feat):
        """运行 Encoder：down_blocks + mid_block，返回 (down_samples, mid)。"""
        m = self.m
        down_samples = (x,)
        for blk in m.down_blocks:
            if getattr(blk, "has_cross_attention", False):
                x, res = blk(
                    hidden_states=x,
                    temb=emb,
                    encoder_hidden_states=audio_feat,
                )
            else:
                x, res = blk(hidden_states=x, temb=emb)
            down_samples += res

        x = m.mid_block(x, emb, encoder_hidden_states=audio_feat)
        return down_samples, x

    @torch.no_grad()
    def _run_decoder(self, mid, down_samples, emb, audio_feat):
        """运行 Decoder：up_blocks，使用给定 skip-connections 和 mid。"""
        m = self.m
        x = mid
        for blk in m.up_blocks:
            n = len(blk.resnets)
            res = down_samples[-n:]
            down_samples = down_samples[:-n]
            if getattr(blk, "has_cross_attention", False):
                x = blk(
                    hidden_states=x,
                    temb=emb,
                    res_hidden_states_tuple=res,
                    encoder_hidden_states=audio_feat,
                )
            else:
                x = blk(
                    hidden_states=x,
                    temb=emb,
                    res_hidden_states_tuple=res,
                )

        if getattr(m, "conv_norm_out", None) is not None:
            x = m.conv_norm_out(x)
            x = m.conv_act(x)
        return m.conv_out(x)

    @torch.no_grad()
    def forward_fixed(self, latent, timestep, audio_feat, interval):
        """固定间隔 refresh（DeepCache-N）。"""
        is_refresh = (self.cache is None or self.frame_idx % interval == 0)
        self.frame_idx += 1

        emb = self._time_emb(timestep, latent.dtype)

        if is_refresh:
            x = self.m.conv_in(latent)
            down_samples, mid = self._run_encoder(x, emb, audio_feat)
            self.cache = {
                "down_samples": tuple(t.detach().clone() for t in down_samples),
                "mid": mid.detach().clone(),
            }
            return self._run_decoder(mid, down_samples, emb, audio_feat), True
        else:
            return self._run_decoder(
                self.cache["mid"], self.cache["down_samples"], emb, audio_feat
            ), False

    @torch.no_grad()
    def forward_adaptive(self, latent, timestep, audio_feat, motion, threshold):
        """运动自适应 refresh（DeepCache-Adp）。"""
        is_refresh = (self.cache is None or motion >= threshold)
        self.frame_idx += 1

        emb = self._time_emb(timestep, latent.dtype)

        if is_refresh:
            x = self.m.conv_in(latent)
            down_samples, mid = self._run_encoder(x, emb, audio_feat)
            self.cache = {
                "down_samples": tuple(t.detach().clone() for t in down_samples),
                "mid": mid.detach().clone(),
            }
            return self._run_decoder(mid, down_samples, emb, audio_feat), True
        else:
            return self._run_decoder(
                self.cache["mid"], self.cache["down_samples"], emb, audio_feat
            ), False


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
    mu1 = cv2.filter2D(g1, -1, k2d)
    mu2 = cv2.filter2D(g2, -1, k2d)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1  = cv2.filter2D(g1 * g1, -1, k2d) - mu1_sq
    s2  = cv2.filter2D(g2 * g2, -1, k2d) - mu2_sq
    s12 = cv2.filter2D(g1 * g2, -1, k2d) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return float(ssim_map.mean()), float(psnr)


def batch_quality(ref_list, tgt_list):
    ss, ps = [], []
    for r, t in zip(ref_list, tgt_list):
        s, p = _ssim_psnr(r, t)
        ss.append(s)
        ps.append(p)
    return float(np.mean(ss)), float(np.mean(ps))


def write_video(frames, audio_path, out_path, fps):
    tmp = out_path.replace(".mp4", "_tmp.mp4")
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
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


def get_audio_feat(i):
    wc = whisper_chunks[i]
    w = wc.cpu().unsqueeze(0).to(device) if isinstance(wc, torch.Tensor) \
        else torch.from_numpy(np.array([wc])).to(device)
    return pe(w)


def get_lat(i):
    return input_latent_list_cycle[i % cycle_len].to(
        device=device, dtype=weight_dtype)


def decode_face(pred):
    """仅 VAE decode，不含 compose（与基线/MATS 计时口径一致）。"""
    pred = pred.to(dtype=vae.vae.dtype)
    return vae.decode_latents(pred)[0]


# ==================== 1. 基线推理（批量）====================
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
mats_out, mats_t, mats_skip_flags = [], [], []
prev_lat_mats = None
cached_pixel  = None
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
            audio_feat = get_audio_feat(i)
            pred = unet.model(lat, timesteps,
                              encoder_hidden_states=audio_feat).sample
            pred = pred.to(dtype=vae.vae.dtype)
            face = vae.decode_latents(pred)[0]
            cached_pixel = face.copy()
            skipped = False; unet_count += 1; consec_skip = 0
        else:
            face = cached_pixel
            skipped = True; skip_count += 1; consec_skip += 1
        sync(); elapsed = time.time() - t0

        mats_out.append(compose_frame(i, face))
        mats_t.append(elapsed)
        mats_skip_flags.append(skipped)
        prev_lat_mats = lat.clone()

        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            print(f"  [{i+1}/{total_frames}] {1/np.mean(mats_t[-20:]):.1f} FPS  "
                  f"跳过率={skip_count/(i+1):.1%}")

fps_mats  = total_frames / sum(mats_t)
skip_mats = skip_count / total_frames
print(f"  MATS 完成：{fps_mats:.1f} FPS  跳过率={skip_mats:.1%}")

# ==================== 3. DeepCache-N（固定间隔）====================
dc_results = {}
dc_engine = DeepCacheEngine(unet.model)

for interval in args.dc_intervals:
    label = f"DC-{interval}"
    print(f"\n[3/N] DeepCache-{interval}（固定每 {interval} 帧 refresh）")
    dc_engine.reset()
    dc_out, dc_t, dc_refresh_flags = [], [], []
    refresh_count = 0
    prev_lat_dc = None

    with torch.no_grad():
        for i in range(total_frames):
            lat = get_lat(i)
            audio_feat = get_audio_feat(i)

            sync(); t0 = time.time()
            pred, is_refresh = dc_engine.forward_fixed(
                lat, timesteps, audio_feat, interval)
            face = decode_face(pred)
            sync(); elapsed = time.time() - t0

            composed = compose_frame(i, face)   # compose 不计入 timing（与基线/MATS 口径一致）
            dc_out.append(composed)
            dc_t.append(elapsed)
            dc_refresh_flags.append(is_refresh)
            if is_refresh:
                refresh_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == total_frames:
                print(f"  [{i+1}/{total_frames}] {1/np.mean(dc_t[-20:]):.1f} FPS  "
                      f"refresh率={refresh_count/(i+1):.1%}")

    fps_dc = total_frames / sum(dc_t)
    refresh_rate = refresh_count / total_frames
    ssim_dc, psnr_dc = batch_quality(baseline_out, dc_out)
    dc_results[label] = {
        "fps": fps_dc,
        "refresh_rate": refresh_rate,
        "ssim": ssim_dc,
        "psnr": psnr_dc,
        "frames": dc_out,
        "timings": dc_t,
        "flags": dc_refresh_flags,
    }
    print(f"  {label} 完成：{fps_dc:.1f} FPS  SSIM={ssim_dc:.4f}  PSNR={psnr_dc:.2f} dB")

    # 保存单独视频
    dc_vid = args.out.replace(".mp4", f"_{label}.mp4")
    write_video(dc_out, args.audio, dc_vid, args.fps)
    print(f"  ✓ 视频: {dc_vid}")

# ==================== 4. DeepCache-Adp（运动自适应）====================
label_adp = "DC-Adp"
print(f"\n[4/N] DeepCache-Adp（运动自适应 thr={args.threshold}）")
dc_engine.reset()
dcadp_out, dcadp_t, dcadp_refresh_flags = [], [], []
prev_lat_adp = None
refresh_count_adp = 0

with torch.no_grad():
    for i in range(total_frames):
        lat = get_lat(i)
        motion = float((lat.float() - prev_lat_adp.float()).norm() /
                       (prev_lat_adp.float().norm() + 1e-6)) \
                 if prev_lat_adp is not None else 999.0
        audio_feat = get_audio_feat(i)

        sync(); t0 = time.time()
        pred, is_refresh = dc_engine.forward_adaptive(
            lat, timesteps, audio_feat, motion, args.threshold)
        face = decode_face(pred)
        sync(); elapsed = time.time() - t0

        composed = compose_frame(i, face)   # compose 不计入 timing
        dcadp_out.append(composed)
        dcadp_t.append(elapsed)
        dcadp_refresh_flags.append(is_refresh)
        if is_refresh:
            refresh_count_adp += 1
        prev_lat_adp = lat.clone()

        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            print(f"  [{i+1}/{total_frames}] {1/np.mean(dcadp_t[-20:]):.1f} FPS  "
                  f"refresh率={refresh_count_adp/(i+1):.1%}")

fps_dcadp  = total_frames / sum(dcadp_t)
refresh_adp = refresh_count_adp / total_frames
ssim_dcadp, psnr_dcadp = batch_quality(baseline_out, dcadp_out)
dc_results[label_adp] = {
    "fps": fps_dcadp,
    "refresh_rate": refresh_adp,
    "ssim": ssim_dcadp,
    "psnr": psnr_dcadp,
    "frames": dcadp_out,
    "timings": dcadp_t,
    "flags": dcadp_refresh_flags,
}
print(f"  {label_adp} 完成：{fps_dcadp:.1f} FPS  SSIM={ssim_dcadp:.4f}  PSNR={psnr_dcadp:.2f} dB")
dcadp_vid = args.out.replace(".mp4", f"_{label_adp}.mp4")
write_video(dcadp_out, args.audio, dcadp_vid, args.fps)
print(f"  ✓ 视频: {dcadp_vid}")

# ==================== MATS 质量指标 ====================
ssim_mats, psnr_mats = batch_quality(baseline_out, mats_out)

# ==================== 汇总结果 ====================
all_methods = {
    "Baseline": {
        "fps": fps_base, "speedup": 1.0,
        "ssim": 1.0, "psnr": float("inf"),
        "skip_rate": 0.0, "note": "batch_size=4",
    },
    "MATS": {
        "fps": fps_mats, "speedup": fps_mats / fps_base,
        "ssim": ssim_mats, "psnr": psnr_mats,
        "skip_rate": skip_mats,
        "note": f"thr={args.threshold}, max_skip={args.max_skip}",
    },
}
for k, v in dc_results.items():
    all_methods[k] = {
        "fps": v["fps"], "speedup": v["fps"] / fps_base,
        "ssim": v["ssim"], "psnr": v["psnr"],
        "skip_rate": 1.0 - v["refresh_rate"],
        "note": "feature-level cache",
    }

print(f"""
{'='*70}
  对比结果汇总
{'='*70}
{'方法':<18} {'FPS':>7} {'加速':>7} {'跳过率':>8} {'SSIM':>8} {'PSNR(dB)':>10}
{'-'*70}""")
for name, m in all_methods.items():
    skip_str = f"{m['skip_rate']:.1%}"
    psnr_str = f"{m['psnr']:.2f}" if m['psnr'] != float('inf') else "∞"
    print(f"  {name:<16} {m['fps']:>7.1f} {m['speedup']:>7.2f}x "
          f"{skip_str:>8} {m['ssim']:>8.4f} {psnr_str:>10}")
print(f"{'='*70}")

# ==================== 保存 JSON ====================
result_path = args.out.replace(".mp4", "_results.json")
save_data = {}
for name, m in all_methods.items():
    save_data[name] = {k: v for k, v in m.items() if k != "frames"}
    # psnr inf → string
    if save_data[name]["psnr"] == float("inf"):
        save_data[name]["psnr"] = "inf"
with open(result_path, "w") as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)
print(f"\n  ✓ 结果 JSON: {result_path}")

# ==================== 保存各方法单独视频 ====================
mats_vid = args.out.replace(".mp4", "_MATS.mp4")
write_video(mats_out, args.audio, mats_vid, args.fps)
print(f"  ✓ MATS 视频: {mats_vid}")

baseline_vid = args.out.replace(".mp4", "_Baseline.mp4")
write_video(baseline_out, args.audio, baseline_vid, args.fps)
print(f"  ✓ Baseline 视频: {baseline_vid}")
