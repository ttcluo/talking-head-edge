"""
LSE-C 唇形同步置信度评估
使用 LatentSync StableSyncNet（潜在空间版）

架构自动推断：
  从 checkpoint weight shapes 推断实际模型架构（不依赖 yaml）
  视觉输入：N帧 × 4通道 × 32×32（N从权重自动检测）
  音频输入：mel 频谱 (1, 80, 52)，hop=200, win=800, sr=16000

运行方式：
    conda activate musetalk && cd ~/MuseTalk

    python ~/tad/step2/lse_eval.py --mode check_env

    python ~/tad/step2/lse_eval.py \\
        --mode generate_and_eval \\
        --video data/video/yongen.mp4 \\
        --audio data/audio/yongen.wav \\
        --threshold 0.15 \\
        --num_frames 150
"""

import argparse
import json
import os
import sys
import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MUSETALK_ROOT   = os.environ.get("MUSE_ROOT",       os.path.expanduser("~/MuseTalk"))
LATENTSYNC_ROOT = os.environ.get("LATENTSYNC_ROOT", os.path.expanduser("~/LatentSync"))

for p in [MUSETALK_ROOT, LATENTSYNC_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ==================== 参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["generate_and_eval", "eval_only", "check_env"],
                    default="check_env")
parser.add_argument("--video",           type=str,   default="data/video/yongen.mp4")
parser.add_argument("--audio",           type=str,   default="data/audio/yongen.wav")
parser.add_argument("--threshold",       type=float, default=0.15)
parser.add_argument("--baseline_video",  type=str,   default="")
parser.add_argument("--cached_video",    type=str,   default="")
parser.add_argument("--syncnet_ckpt",    type=str,   default="models/syncnet/latentsync_syncnet.pt")
parser.add_argument("--output_dir",      type=str,   default="profile_results")
parser.add_argument("--num_frames",      type=int,   default=150)
parser.add_argument("--vshift",          type=int,   default=5)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("  LSE-C 唇形同步置信度评估（LatentSync StableSyncNet）")
print("=" * 65)

# ==================== 环境检查 ====================
def check_environment():
    print("\n[环境检查]")
    ok = True
    try:
        import librosa
        print("  ✓ librosa")
    except ImportError:
        print("  ✗ librosa → pip install librosa")
        ok = False

    if os.path.exists(args.syncnet_ckpt):
        mb = os.path.getsize(args.syncnet_ckpt) / 1024 / 1024
        print(f"  ✓ SyncNet 权重: {args.syncnet_ckpt} ({mb:.1f}MB)")
    else:
        print(f"  ✗ SyncNet 权重不存在: {args.syncnet_ckpt}")
        ok = False

    sn_py = os.path.join(LATENTSYNC_ROOT, "latentsync/models/stable_syncnet.py")
    if os.path.exists(sn_py):
        print(f"  ✓ LatentSync 仓库: {LATENTSYNC_ROOT}")
    else:
        print(f"  ✗ LatentSync 仓库: {LATENTSYNC_ROOT}")
        ok = False

    for label, path in [("MuseTalk VAE", "models/sd-vae"),
                         ("MuseTalk UNet", "models/musetalkV15/unet.pth")]:
        e = os.path.exists(path)
        print(f"  {'✓' if e else '✗'} {label}")
        ok = ok and e

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        print("  ✓ ffmpeg")
    except Exception:
        print("  ✗ ffmpeg 不可用")
        ok = False
    return ok

# ==================== 从权重自动推断架构配置 ====================
def infer_config_from_state(state: dict) -> tuple:
    """
    从 checkpoint 的 weight shapes 推断 StableSyncNet 架构配置，
    完全不依赖 yaml 文件，兼容任意训练版本。
    返回 (model_config_dict, num_frames_per_window)
    """
    def infer_encoder(prefix):
        in_channels = state[f"{prefix}.conv_in.weight"].shape[1]
        block_out_channels, downsample_factors, attn_blocks_cfg = [], [], []

        i = 0
        while True:
            rk = f"{prefix}.down_blocks.{i}.conv1.weight"
            ak = f"{prefix}.down_blocks.{i}.conv_in.weight"  # AttentionBlock 标志
            if rk in state:
                out_ch = state[rk].shape[0]
                has_down = f"{prefix}.down_blocks.{i}.downsample_conv.weight" in state
                block_out_channels.append(out_ch)
                downsample_factors.append(2 if has_down else 1)
                attn_blocks_cfg.append(0)
            elif ak in state:
                attn_blocks_cfg[-1] = 1   # 这个 block 是 Attn，附属于上一个 ResNet
            else:
                break
            i += 1

        return {
            "in_channels":        in_channels,
            "block_out_channels": block_out_channels,
            "downsample_factors": downsample_factors,
            "attn_blocks":        attn_blocks_cfg,
            "dropout":            0.0,
        }

    audio_cfg  = infer_encoder("audio_encoder")
    visual_cfg = infer_encoder("visual_encoder")

    # visual in_channels = num_frames × 4 (VAE channels)
    num_frames = visual_cfg["in_channels"] // 4

    model_cfg = {"audio_encoder": audio_cfg, "visual_encoder": visual_cfg}
    return model_cfg, num_frames

# ==================== 加载 SyncNet ====================
def load_stable_syncnet():
    from latentsync.models.stable_syncnet import StableSyncNet

    state = torch.load(args.syncnet_ckpt, map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]

    model_cfg, num_frames = infer_config_from_state(state)

    a_cfg = model_cfg["audio_encoder"]
    v_cfg = model_cfg["visual_encoder"]
    print(f"  推断架构: 视频窗口={num_frames}帧, "
          f"audio={len(a_cfg['block_out_channels'])}blocks({a_cfg['block_out_channels'][-1]}ch), "
          f"visual={len(v_cfg['block_out_channels'])}blocks({v_cfg['block_out_channels'][-1]}ch)")

    syncnet = StableSyncNet(model_cfg).to(device)
    syncnet.load_state_dict(state)
    syncnet.eval()
    total = sum(p.numel() for p in syncnet.parameters())
    print(f"  ✓ StableSyncNet 加载完成（{total:,} 参数）")
    return syncnet, num_frames

# ==================== 音频 mel 频谱 ====================
def load_audio_mel(audio_path: str) -> np.ndarray:
    """
    按 LatentSync 参数计算 mel 频谱，返回 (80, T)
    hop_size=200 → 25fps 视频每帧对应 3.2 个时间步
    """
    import librosa
    from scipy import signal as scipy_signal

    sr, n_fft, hop_size, win_size = 16000, 800, 200, 800
    n_mels, fmin, fmax = 80, 55, 7600
    pre_coef = 0.97
    ref_level_db, min_level_db, max_abs_value = 20, -100, 4.0

    wav, _ = librosa.load(audio_path, sr=sr)
    wav = scipy_signal.lfilter([1, -pre_coef], [1], wav)

    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = np.dot(mel_basis, np.abs(D))

    min_level = np.exp(min_level_db / 20 * np.log(10))
    mel = 20 * np.log10(np.maximum(min_level, mel)) - ref_level_db
    mel = np.clip(
        (2 * max_abs_value) * ((mel - min_level_db) / (-min_level_db)) - max_abs_value,
        -max_abs_value, max_abs_value
    )
    return mel  # (80, T)

def get_mel_window(mel: np.ndarray, frame_idx: int, mel_length: int = 52) -> np.ndarray:
    """以 frame_idx 为中心提取 mel 窗口，25fps → hop=200 → 3.2步/帧"""
    center = int(round(frame_idx * 3.2))
    half   = mel_length // 2
    start  = max(0, center - half)
    end    = start + mel_length
    T      = mel.shape[1]
    if end > T:
        end   = T
        start = max(0, end - mel_length)
    chunk = mel[:, start:end]
    if chunk.shape[1] < mel_length:
        chunk = np.pad(chunk, ((0, 0), (0, mel_length - chunk.shape[1])), mode="edge")
    return chunk  # (80, 52)

# ==================== 视频帧 → VAE 潜变量 ====================
def encode_frames(frames: list, vae, dtype) -> list:
    """list of BGR uint8 → list of float32 CPU tensor (4, 32, 32)"""
    latents = []
    with torch.no_grad():
        for f in frames:
            face = cv2.resize(f, (256, 256))
            rgb  = face[:, :, ::-1].copy()
            t    = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1
            t    = t.unsqueeze(0).to(device, dtype)
            lat  = vae.encode(t).latent_dist.mean  # (1, 4, 32, 32)
            latents.append(lat[0].float().cpu())
    return latents

# ==================== LSE-C 计算 ====================
def compute_lse(syncnet, latents: list, mel: np.ndarray,
                num_frames_per_window: int, vshift: int = 5) -> tuple:
    """
    滑动窗口计算 LSE-C / LSE-D
    """
    N = len(latents)
    if N < num_frames_per_window + vshift:
        return None, None, None

    vis_feats, aud_feats = [], []
    with torch.no_grad():
        for i in range(0, N - num_frames_per_window):
            # 视觉：N帧潜变量拼接 → (1, N*4, 32, 32)
            win = torch.stack(latents[i:i + num_frames_per_window], 0)  # (N, 4, 32, 32)
            vis_in = win.view(num_frames_per_window * 4, 32, 32).unsqueeze(0).to(device)

            # 音频：以窗口中心帧为基准
            center = i + num_frames_per_window // 2
            mel_ch = get_mel_window(mel, center)  # (80, 52)
            aud_in = torch.from_numpy(mel_ch).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 80, 52)

            ve, ae = syncnet(vis_in.float(), aud_in.float())
            vis_feats.append(ve.cpu())
            aud_feats.append(ae.cpu())

    vis_feats = torch.cat(vis_feats, 0)  # (M, D)
    aud_feats = torch.cat(aud_feats, 0)  # (M, D)
    M = vis_feats.shape[0]

    # 余弦距离偏移搜索（参考标准 SyncNet）
    aud_pad = F.pad(aud_feats, (0, 0, vshift, vshift))
    win_size = vshift * 2 + 1
    dists = []
    for i in range(M):
        sim  = F.cosine_similarity(
            vis_feats[i].unsqueeze(0).expand(win_size, -1),
            aud_pad[i:i + win_size]
        )
        dists.append(1.0 - sim)  # 余弦距离

    mean_dists = torch.stack(dists, 1).mean(1)  # (win_size,)
    min_dist, min_idx = mean_dists.min(0)
    lse_d   = min_dist.item()
    lse_c   = (mean_dists.median() - min_dist).item()
    offset  = vshift - min_idx.item()
    return lse_c, lse_d, offset

# ==================== 工具函数 ====================
def read_frames(path: str, max_n: int = 9999) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_n:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames

def write_video(frames: list, audio: str, out: str, fps: int = 25):
    h, w = frames[0].shape[:2]
    tmp  = out.replace(".mp4", "_tmp.mp4")
    wrt  = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        wrt.write(f)
    wrt.release()
    subprocess.call(
        f"ffmpeg -loglevel error -nostdin -y -i {tmp} -i {audio} "
        f"-c:v copy -c:a aac -shortest {out}", shell=True
    )
    os.remove(tmp)

# ==================== 主流程 ====================
if args.mode == "check_env":
    ok = check_environment()
    print("\n  ✅ 环境就绪" if ok else "\n  ⚠ 请先解决上述问题")
    sys.exit(0)

# ——————————————————————————————————————————————
if args.mode == "generate_and_eval":
    print("\n[模式：生成对比视频 + LSE-C 评估]")
    if not check_environment():
        sys.exit(1)

    import json as _json
    from diffusers import AutoencoderKL
    from musetalk.models.unet import UNet2DConditionModel

    print("\n[加载 MuseTalk 模型]")
    with open("models/musetalkV15/musetalk.json") as f:
        unet_config = _json.load(f)
    dtype = torch.float16
    vae  = AutoencoderKL.from_pretrained("models/sd-vae").to(device, dtype)
    unet = UNet2DConditionModel(**unet_config).to(device, dtype)
    unet.load_state_dict(torch.load("models/musetalkV15/unet.pth", map_location=device))
    vae.eval(); unet.eval()

    # Whisper 音频特征（使用旧版 Audio2Feature 接口）
    audio_chunks = None
    try:
        from musetalk.whisper.audio2feature import Audio2Feature
        ap = Audio2Feature(model_path="models/whisper/pytorch_model.bin",
                           device=device, fps=25)
        audio_feats = ap.get_sliced_feature(
            feature_array=ap.audio2feat(args.audio), fps=25,
            audio_length_in_s=None, weight_dtype=dtype
        )
        audio_chunks = audio_feats
        print(f"  ✓ Whisper 音频特征: {len(audio_chunks)} 块")
    except Exception:
        pass

    if audio_chunks is None:
        try:
            from musetalk.whisper.audio2feature import Audio2Feature
            ap = Audio2Feature(whisper_model_type="tiny",
                               model_path="models/whisper/pytorch_model.bin")
            af = ap.audio2feat(args.audio)
            audio_chunks = ap.feature2chunks(feature_array=af, fps=25)
            print(f"  ✓ Whisper 特征(v2): {len(audio_chunks)} 块")
        except Exception as e:
            print(f"  ⚠ Whisper 失败({e}), 用零向量——跳过率会失真，不建议评估 LSE")
            audio_chunks = [np.zeros((1, 384), dtype=np.float32)] * args.num_frames

    frames = read_frames(args.video, args.num_frames)
    N      = len(frames)
    print(f"  读取 {N} 帧")

    ts = torch.tensor([0.0], device=device, dtype=dtype)

    # 基线视频
    print(f"\n[生成基线视频（全量 UNet，{N} 帧）]")
    baseline_pixels, input_latents = [], []
    with torch.no_grad():
        for i, frm in enumerate(frames):
            face = cv2.resize(frm, (256, 256))
            rgb  = face[:, :, ::-1].copy()
            t    = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1
            t    = t.unsqueeze(0).to(device, dtype)
            lat  = vae.encode(t).latent_dist.mean
            ac   = torch.from_numpy(audio_chunks[i % len(audio_chunks)]).unsqueeze(0).to(device, dtype)
            mask = torch.zeros_like(lat)
            out  = unet(torch.cat([lat, mask], 1), ts, encoder_hidden_states=ac).sample
            dec  = vae.decode(out).sample
            px   = ((dec[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            baseline_pixels.append(cv2.cvtColor(px, cv2.COLOR_RGB2BGR))
            input_latents.append(lat[0].detach().cpu().float())

    baseline_path = os.path.join(args.output_dir, "baseline.mp4")
    write_video(baseline_pixels, args.audio, baseline_path)
    print(f"  ✓ {baseline_path}")

    # 运动量计算
    motions = [999.0] + [
        float((input_latents[i] - input_latents[i-1]).norm() /
              (input_latents[i-1].norm() + 1e-6))
        for i in range(1, N)
    ]
    skip_count = sum(1 for m in motions[1:] if m < args.threshold)
    print(f"  运动统计: mean={np.mean(motions[1:]):.4f}, "
          f"跳过率预估={skip_count/(N-1):.1%}")

    if skip_count / max(N - 1, 1) > 0.95:
        print("  ⚠ 跳过率 >95%，可能是 Whisper 零音频条件导致特征退化")
        print("    LSE-C 数值仍有参考意义（两视频均无音频条件），但绝对值不可信")

    # 缓存视频
    print(f"\n[生成缓存视频（阈值={args.threshold}）]")
    cached_pixels, skip = [], 0
    last_px = None
    with torch.no_grad():
        for i, frm in enumerate(frames):
            if i == 0 or motions[i] >= args.threshold:
                face = cv2.resize(frm, (256, 256))
                rgb  = face[:, :, ::-1].copy()
                t    = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1
                t    = t.unsqueeze(0).to(device, dtype)
                lat  = vae.encode(t).latent_dist.mean
                ac   = torch.from_numpy(audio_chunks[i % len(audio_chunks)]).unsqueeze(0).to(device, dtype)
                mask = torch.zeros_like(lat)
                out  = unet(torch.cat([lat, mask], 1), ts, encoder_hidden_states=ac).sample
                dec  = vae.decode(out).sample
                px   = ((dec[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                last_px = px
            else:
                skip += 1
            cached_pixels.append(cv2.cvtColor(last_px, cv2.COLOR_RGB2BGR))

    print(f"  跳过率: {skip}/{N} = {skip/N:.1%}")
    cached_path = os.path.join(args.output_dir, "cached.mp4")
    write_video(cached_pixels, args.audio, cached_path)
    print(f"  ✓ {cached_path}")

    args.baseline_video = baseline_path
    args.cached_video   = cached_path

# ——————————————————————————————————————————————
if args.mode in ["eval_only", "generate_and_eval"]:
    if args.mode == "eval_only":
        print("\n[模式：仅 LSE-C 评估]")
        if not check_environment():
            sys.exit(1)

    for p in [args.baseline_video, args.cached_video]:
        if not p or not os.path.exists(p):
            print(f"  ✗ 视频不存在: {p}")
            sys.exit(1)

    # 加载 SyncNet（自动推断架构）
    print("\n[加载 StableSyncNet]")
    syncnet, num_frames_win = load_stable_syncnet()
    print(f"  视频窗口大小: {num_frames_win} 帧")

    # 加载 VAE（eval_only 模式）
    if args.mode == "eval_only":
        import json as _json
        from diffusers import AutoencoderKL
        dtype = torch.float16
        vae = AutoencoderKL.from_pretrained("models/sd-vae").to(device, dtype)
        vae.eval()

    # 音频 mel
    audio_src = args.audio if (args.audio and os.path.exists(args.audio)) else None
    if audio_src is None:
        audio_src = os.path.join(args.output_dir, "_tmp_audio.wav")
        subprocess.call(
            f"ffmpeg -loglevel error -nostdin -y -i {args.baseline_video} "
            f"-ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_src}", shell=True
        )
    mel = load_audio_mel(audio_src)
    print(f"\n  mel 频谱: {mel.shape} (~{mel.shape[1]/3.2/25:.1f}s)")

    def eval_one(path, label):
        print(f"\n  [{label}] 编码视频帧...")
        frames = read_frames(path)
        lats   = encode_frames(frames, vae, dtype)
        lse_c, lse_d, offset = compute_lse(syncnet, lats, mel, num_frames_win, args.vshift)
        if lse_c is None:
            print(f"  ✗ 帧数不足（需要 ≥ {num_frames_win + args.vshift} 帧，实际 {len(frames)}）")
            return None
        print(f"  {label}: LSE-C={lse_c:.4f}  LSE-D={lse_d:.4f}  offset={offset:+d}")
        return {"lse_c": lse_c, "lse_d": lse_d, "offset": offset, "frames": len(frames)}

    rb = eval_one(args.baseline_video, "基线")
    rc = eval_one(args.cached_video,   f"缓存(thr={args.threshold})")
    if rb is None or rc is None:
        sys.exit(1)

    dc = rc["lse_c"] - rb["lse_c"]
    dd = rc["lse_d"] - rb["lse_d"]

    print("\n" + "=" * 65)
    print("  LSE-C 对比结果")
    print("=" * 65)
    print(f"""
  {'方法':<22} {'LSE-C↑':>8} {'LSE-D↓':>8} {'offset':>8}
  {'-'*22} {'-'*8} {'-'*8} {'-'*8}
  {'基线（全量）':<22} {rb['lse_c']:>8.4f} {rb['lse_d']:>8.4f} {rb['offset']:>+8d}
  {f'缓存(thr={args.threshold})':<22} {rc['lse_c']:>8.4f} {rc['lse_d']:>8.4f} {rc['offset']:>+8d}
  {'Δ':<22} {dc:>+8.4f} {dd:>+8.4f}

  唇形同步保持: {'✓ 可接受 (|ΔLSE-C| < 0.5)' if abs(dc) < 0.5 else '⚠ 退化 (|ΔLSE-C| ≥ 0.5)'}
""")

    print("=" * 65)
    print("  论文 Table 完整数字（填写用）")
    print("=" * 65)
    print(f"""
  方法               FPS    SSIM    PSNR    LSE-C↑  LSE-D↓
  MuseTalk 基线      22.2   1.000   ∞       {rb['lse_c']:.3f}   {rb['lse_d']:.3f}
  + 帧跳过(0.15)    40.3   0.9976  57.3dB  {rc['lse_c']:.3f}   {rc['lse_d']:.3f}
""")

    result = {
        "threshold": args.threshold,
        "vshift": args.vshift,
        "num_frames_window": num_frames_win,
        "baseline": rb,
        "cached":   rc,
        "delta": {"lse_c": dc, "lse_d": dd},
        "sync_preserved": bool(abs(dc) < 0.5),
    }
    out_f = os.path.join(args.output_dir, "lse_eval.json")
    with open(out_f, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {out_f}")
