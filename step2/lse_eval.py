"""
LSE-C 唇形同步置信度评估
使用 LatentSync StableSyncNet（潜在空间版）计算 LSE-C / LSE-D

架构说明：
  - latentsync_syncnet.pt 是 StableSyncNet，工作在 VAE 潜在空间
  - 视觉输入：16帧 × 4通道 32×32 VAE 潜变量 → (64, 32, 32)
  - 音频输入：mel 频谱 (1, 80, 52)，hop=200, win=800, sr=16000
  - MuseTalk 与 LatentSync 使用同一个 SD VAE，完全兼容

运行方式：
    conda activate musetalk && cd ~/MuseTalk

    # 检查环境
    python ~/tad/step2/lse_eval.py --mode check_env

    # 生成视频并评估 LSE-C（推荐）
    python ~/tad/step2/lse_eval.py \\
        --mode generate_and_eval \\
        --video data/video/yongen.mp4 \\
        --audio data/audio/yongen.wav \\
        --threshold 0.15

    # 仅评估已有视频
    python ~/tad/step2/lse_eval.py \\
        --mode eval_only \\
        --baseline_video profile_results/baseline.mp4 \\
        --cached_video profile_results/cached.mp4
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
LATENTSYNC_ROOT = os.environ.get("LATENTSYNC_ROOT", os.path.expanduser("~/LatentSync"))

for p in [MUSETALK_ROOT, LATENTSYNC_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ==================== 参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["generate_and_eval", "eval_only", "check_env"],
                    default="check_env")
parser.add_argument("--video", type=str, default="data/video/yongen.mp4")
parser.add_argument("--audio", type=str, default="data/audio/yongen.wav")
parser.add_argument("--threshold", type=float, default=0.15,
                    help="帧跳过运动阈值")
parser.add_argument("--baseline_video", type=str, default="")
parser.add_argument("--cached_video",   type=str, default="")
parser.add_argument("--syncnet_ckpt", type=str,
                    default="models/syncnet/latentsync_syncnet.pt")
parser.add_argument("--output_dir", type=str, default="profile_results")
parser.add_argument("--num_frames", type=int, default=150,
                    help="生成/评估帧数（≥50 帧，LSE-C 才稳定）")
parser.add_argument("--vshift", type=int, default=5,
                    help="时序偏移搜索范围（帧数），LatentSync 默认 5")
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

    # librosa（mel 频谱计算）
    try:
        import librosa
        print(f"  ✓ librosa")
    except ImportError:
        print(f"  ✗ librosa 未安装 → pip install librosa")
        ok = False

    # SyncNet 权重
    if os.path.exists(args.syncnet_ckpt):
        size_mb = os.path.getsize(args.syncnet_ckpt) / 1024 / 1024
        print(f"  ✓ SyncNet 权重: {args.syncnet_ckpt} ({size_mb:.1f}MB)")
    else:
        print(f"  ✗ SyncNet 权重不存在: {args.syncnet_ckpt}")
        ok = False

    # LatentSync 仓库（需要 stable_syncnet.py 和 configs/audio.yaml）
    sn_code = os.path.join(LATENTSYNC_ROOT, "latentsync/models/stable_syncnet.py")
    audio_cfg = os.path.join(LATENTSYNC_ROOT, "configs/audio.yaml")
    if os.path.exists(sn_code) and os.path.exists(audio_cfg):
        print(f"  ✓ LatentSync 仓库: {LATENTSYNC_ROOT}")
    else:
        print(f"  ✗ LatentSync 仓库缺失文件: {LATENTSYNC_ROOT}")
        print(f"    需要: latentsync/models/stable_syncnet.py + configs/audio.yaml")
        ok = False

    # MuseTalk VAE + UNet
    vae_ok = os.path.exists("models/sd-vae")
    unet_ok = os.path.exists("models/musetalkV15/unet.pth")
    print(f"  {'✓' if vae_ok else '✗'} MuseTalk VAE")
    print(f"  {'✓' if unet_ok else '✗'} MuseTalk UNet")
    ok = ok and vae_ok and unet_ok

    # ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        print(f"  ✓ ffmpeg")
    except Exception:
        print(f"  ✗ ffmpeg 不可用")
        ok = False

    return ok

# ==================== StableSyncNet 加载 ====================
def load_stable_syncnet():
    """加载 LatentSync 的潜在空间 SyncNet（StableSyncNet）"""
    from omegaconf import OmegaConf
    from latentsync.models.stable_syncnet import StableSyncNet

    cfg_path = os.path.join(LATENTSYNC_ROOT, "configs/syncnet/syncnet_16_latent.yaml")
    cfg = OmegaConf.load(cfg_path)
    model_cfg = {
        "audio_encoder": dict(cfg.model.audio_encoder),
        "visual_encoder": dict(cfg.model.visual_encoder),
    }

    syncnet = StableSyncNet(model_cfg).to(device)
    state = torch.load(args.syncnet_ckpt, map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    syncnet.load_state_dict(state)
    syncnet.eval()
    print(f"  ✓ StableSyncNet 加载完成（潜在空间，{sum(p.numel() for p in syncnet.parameters()):,} 参数）")
    return syncnet

# ==================== 音频预处理：mel 频谱 ====================
def load_audio_mel(audio_path: str):
    """
    按 LatentSync 的参数计算 mel 频谱
    返回 shape: (80, T)，对应 25fps 视频每帧约 3.2 个时间步
    """
    import librosa
    from scipy import signal as scipy_signal

    sr = 16000
    n_fft = 800
    hop_size = 200
    win_size = 800
    n_mels = 80
    fmin = 55
    fmax = 7600
    preemphasis_coef = 0.97
    ref_level_db = 20
    min_level_db = -100
    max_abs_value = 4.0

    wav, _ = librosa.load(audio_path, sr=sr)

    # 预加重
    wav = scipy_signal.lfilter([1, -preemphasis_coef], [1], wav)

    # STFT → mel
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel = np.dot(mel_basis, np.abs(D))

    # dB 归一化（与 LatentSync audio.py 保持一致）
    min_level = np.exp(min_level_db / 20 * np.log(10))
    mel = 20 * np.log10(np.maximum(min_level, mel)) - ref_level_db
    mel = np.clip(
        (2 * max_abs_value) * ((mel - min_level_db) / (-min_level_db)) - max_abs_value,
        -max_abs_value, max_abs_value
    )
    return mel  # (80, T)

def mel_for_frame(mel_full: np.ndarray, frame_idx: int, mel_length: int = 52) -> np.ndarray:
    """
    提取第 frame_idx 帧对应的 mel 窗口
    25fps → hop_size=200 → 每帧约 3.2 个时间步，取整为 3
    mel 时间步 i 对应 frame_idx ~= i / 3.2
    中心对齐：取 frame_idx * 3.2 为中心
    """
    center = int(round(frame_idx * 3.2))  # 约等于 16000/25/200 × frame_idx
    half = mel_length // 2
    start = max(0, center - half)
    end = start + mel_length

    T = mel_full.shape[1]
    if end > T:
        end = T
        start = max(0, end - mel_length)

    chunk = mel_full[:, start:end]
    if chunk.shape[1] < mel_length:
        chunk = np.pad(chunk, ((0, 0), (0, mel_length - chunk.shape[1])), mode="edge")
    return chunk  # (80, 52)

# ==================== 视频预处理：VAE 潜变量 ====================
def encode_frames_to_latents(frames: list, vae, dtype):
    """
    frames: list of BGR uint8 np.ndarray (H, W, 3)
    返回: list of float32 CPU tensor (4, 32, 32)
    """
    latents = []
    with torch.no_grad():
        for frame in frames:
            face = cv2.resize(frame, (256, 256))
            face_t = torch.from_numpy(face[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1
            face_t = face_t.unsqueeze(0).to(device, dtype)
            lat = vae.encode(face_t).latent_dist.mean  # (1, 4, 32, 32)
            latents.append(lat[0].float().cpu())       # (4, 32, 32)
    return latents

# ==================== LSE-C 计算核心 ====================
def compute_lse_latent(syncnet, latents: list, mel_full: np.ndarray,
                       num_frames_per_window: int = 16,
                       vshift: int = 5) -> tuple:
    """
    基于潜在空间 SyncNet 计算 LSE-C / LSE-D
    latents: list of (4, 32, 32) float32 tensors
    mel_full: (80, T) mel 频谱
    返回: (lse_c, lse_d, best_offset)
    """
    N = len(latents)
    if N < num_frames_per_window + vshift:
        return None, None, None

    vis_feats = []
    aud_feats = []

    with torch.no_grad():
        # 滑动窗口，步长 1 帧
        for i in range(0, N - num_frames_per_window, 1):
            # 视觉：16帧潜变量拼接 → (64, 32, 32)
            window_lats = torch.stack(latents[i:i + num_frames_per_window], dim=0)  # (16, 4, 32, 32)
            vis_in = window_lats.view(num_frames_per_window * 4, 32, 32).unsqueeze(0).to(device)  # (1, 64, 32, 32)

            # 音频：以窗口中间帧为基准
            center_frame = i + num_frames_per_window // 2
            mel_chunk = mel_for_frame(mel_full, center_frame)  # (80, 52)
            aud_in = torch.from_numpy(mel_chunk).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 80, 52)

            vis_emb, aud_emb = syncnet(vis_in.float(), aud_in.float())
            vis_feats.append(vis_emb.cpu())
            aud_feats.append(aud_emb.cpu())

    if not vis_feats:
        return None, None, None

    vis_feats = torch.cat(vis_feats, 0)   # (M, D)
    aud_feats = torch.cat(aud_feats, 0)   # (M, D)

    # 与标准 SyncNet 一致：用余弦距离 (1 - cosine_similarity)
    M = vis_feats.shape[0]
    win_size = vshift * 2 + 1
    aud_feats_pad = F.pad(aud_feats, (0, 0, vshift, vshift))  # (M + 2*vshift, D)

    dists = []
    for i in range(M):
        sim = F.cosine_similarity(
            vis_feats[i].unsqueeze(0).expand(win_size, -1),
            aud_feats_pad[i:i + win_size]
        )  # (win_size,)
        dist = 1.0 - sim  # 余弦距离
        dists.append(dist)

    dists = torch.stack(dists, dim=1)       # (win_size, M)
    mean_dists = dists.mean(dim=1)          # (win_size,)

    min_dist, min_idx = mean_dists.min(0)
    best_offset = vshift - min_idx.item()
    lse_d = min_dist.item()
    lse_c = (mean_dists.median() - min_dist).item()

    return lse_c, lse_d, best_offset

# ==================== 视频合成工具 ====================
def frames_to_video(frames: list, audio_path: str, output_path: str, fps: int = 25):
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    tmp = output_path.replace(".mp4", "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    cmd = (f"ffmpeg -loglevel error -nostdin -y "
           f"-i {tmp} -i {audio_path} "
           f"-c:v copy -c:a aac -shortest {output_path}")
    subprocess.call(cmd, shell=True)
    os.remove(tmp)
    return os.path.exists(output_path)

def read_video_frames(path: str, max_frames: int = 9999) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# ==================== 主流程 ====================
if args.mode == "check_env":
    env_ok = check_environment()
    if env_ok:
        print("\n  ✅ 环境就绪，可运行 generate_and_eval 或 eval_only")
    else:
        print("\n  ⚠ 解决上述问题后重试")
        print("  pip install librosa  # 若缺少")
    sys.exit(0)

# ——————————————————————————————————————————
elif args.mode == "generate_and_eval":
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
    vae = AutoencoderKL.from_pretrained("models/sd-vae").to(device, dtype)
    unet = UNet2DConditionModel(**unet_config).to(device, dtype)
    unet.load_state_dict(torch.load("models/musetalkV15/unet.pth", map_location=device))
    vae.eval(); unet.eval()
    print("  ✓ VAE + UNet 加载完成")

    # 音频特征（Whisper）
    try:
        from musetalk.whisper.audio2feature import Audio2Feature
        audio_processor = Audio2Feature(
            whisper_model_type="tiny",
            model_path="models/whisper/pytorch_model.bin"
        )
        audio_features = audio_processor.audio2feat(args.audio)
        audio_chunks = audio_processor.feature2chunks(feature_array=audio_features, fps=25)
        print(f"  ✓ Whisper 音频特征：{len(audio_chunks)} 块")
    except Exception as e:
        print(f"  ⚠ Whisper 失败 ({e})，使用零向量（会影响口型）")
        audio_chunks = [np.zeros((1, 384), dtype=np.float32)] * args.num_frames

    # 读取视频帧
    frames = read_video_frames(args.video, args.num_frames)
    N = len(frames)
    print(f"  读取 {N} 帧")

    timestep = torch.tensor([0.0], device=device, dtype=dtype)

    # 生成基线
    print(f"\n[生成基线视频（全量 UNet，{N} 帧）]")
    baseline_pixels = []
    input_latents = []
    with torch.no_grad():
        for i, frame in enumerate(frames):
            face = cv2.resize(frame, (256, 256))
            face_bgr = face.copy()
            face_t = torch.from_numpy(face[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1
            face_t = face_t.unsqueeze(0).to(device, dtype)
            lat = vae.encode(face_t).latent_dist.mean
            audio_c = torch.from_numpy(audio_chunks[i % len(audio_chunks)]).unsqueeze(0).to(device, dtype)
            mask = torch.zeros_like(lat)
            out = unet(torch.cat([lat, mask], 1), timestep, encoder_hidden_states=audio_c).sample
            decoded = vae.decode(out).sample
            pixel = (decoded[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5
            pixel = pixel.clip(0, 255).astype(np.uint8)
            baseline_pixels.append(cv2.cvtColor(pixel, cv2.COLOR_RGB2BGR))
            input_latents.append(lat.detach().cpu().float()[0])  # (4, 32, 32)

    baseline_path = os.path.join(args.output_dir, "baseline.mp4")
    frames_to_video(baseline_pixels, args.audio, baseline_path)
    print(f"  ✓ 基线视频: {baseline_path}")

    # 计算运动
    motions = [999.0]
    for i in range(1, N):
        m = (input_latents[i] - input_latents[i-1]).norm() / \
            (input_latents[i-1].norm() + 1e-6)
        motions.append(float(m))

    # 生成缓存版本
    print(f"\n[生成缓存视频（阈值={args.threshold}）]")
    cached_pixels = []
    skip_count = 0
    last_decoded = None
    with torch.no_grad():
        for i, frame in enumerate(frames):
            if i == 0 or motions[i] >= args.threshold:
                face = cv2.resize(frame, (256, 256))
                face_t = torch.from_numpy(face[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1
                face_t = face_t.unsqueeze(0).to(device, dtype)
                lat = vae.encode(face_t).latent_dist.mean
                audio_c = torch.from_numpy(audio_chunks[i % len(audio_chunks)]).unsqueeze(0).to(device, dtype)
                mask = torch.zeros_like(lat)
                out = unet(torch.cat([lat, mask], 1), timestep, encoder_hidden_states=audio_c).sample
                decoded = vae.decode(out).sample
                pixel = (decoded[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5
                last_decoded = pixel.clip(0, 255).astype(np.uint8)
            else:
                skip_count += 1
            cached_pixels.append(cv2.cvtColor(last_decoded, cv2.COLOR_RGB2BGR))

    print(f"  跳过率: {skip_count}/{N} = {skip_count/N:.1%}")
    cached_path = os.path.join(args.output_dir, "cached.mp4")
    frames_to_video(cached_pixels, args.audio, cached_path)
    print(f"  ✓ 缓存视频: {cached_path}")

    args.baseline_video = baseline_path
    args.cached_video   = cached_path
    # 继续下方评估逻辑

# ——————————————————————————————————————————
if args.mode in ["eval_only", "generate_and_eval"]:
    if args.mode == "eval_only":
        print(f"\n[模式：仅 LSE-C 评估]")
        if not check_environment():
            sys.exit(1)

    for p in [args.baseline_video, args.cached_video]:
        if not p or not os.path.exists(p):
            print(f"  ✗ 视频不存在: {p}")
            sys.exit(1)

    # 加载 StableSyncNet
    print(f"\n[加载 StableSyncNet]")
    syncnet = load_stable_syncnet()

    # 加载 MuseTalk VAE（用于将视频帧编码到潜在空间）
    print(f"\n[加载 VAE 编码器（视频 → 潜变量）]")
    if args.mode == "eval_only":
        import json as _json
        from diffusers import AutoencoderKL
        dtype = torch.float16
        vae = AutoencoderKL.from_pretrained("models/sd-vae").to(device, dtype)
        vae.eval()
        print("  ✓ VAE 加载完成")

    # 提取音频 mel 频谱（只计算一次，两个视频共用相同 audio）
    audio_path = args.audio if args.audio and os.path.exists(args.audio) else None
    if audio_path is None:
        # 从 baseline 视频提取音频
        audio_path = os.path.join(args.output_dir, "_eval_audio.wav")
        subprocess.call(
            f"ffmpeg -loglevel error -nostdin -y -i {args.baseline_video} "
            f"-async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_path}",
            shell=True
        )
        print(f"  从视频提取音频: {audio_path}")
    mel_full = load_audio_mel(audio_path)
    print(f"  ✓ mel 频谱: shape={mel_full.shape} (~{mel_full.shape[1]/3.2/25:.1f}s)")

    def eval_video(video_path: str, label: str):
        print(f"\n  计算 [{label}] ...")
        frames = read_video_frames(video_path)
        print(f"  读取 {len(frames)} 帧")
        latents = encode_frames_to_latents(frames, vae, dtype)
        lse_c, lse_d, offset = compute_lse_latent(syncnet, latents, mel_full, vshift=args.vshift)
        if lse_c is None:
            print(f"  ✗ 帧数不足，评估失败")
            return None
        print(f"  {label}: LSE-C={lse_c:.4f}  LSE-D={lse_d:.4f}  offset={offset:+d}帧")
        return {"lse_c": lse_c, "lse_d": lse_d, "offset": offset}

    r_base  = eval_video(args.baseline_video, "基线")
    r_cache = eval_video(args.cached_video,   "缓存")

    if r_base is None or r_cache is None:
        print("  评估失败，退出")
        sys.exit(1)

    # ==================== 结果汇报 ====================
    lse_c_diff = r_cache["lse_c"] - r_base["lse_c"]
    lse_d_diff = r_cache["lse_d"] - r_base["lse_d"]
    sync_ok = abs(lse_c_diff) < 0.5

    print("\n" + "=" * 65)
    print("  唇形同步对比结果（LatentSync StableSyncNet）")
    print("=" * 65)
    print(f"""
  {'方法':<22} {'LSE-C↑':>8} {'LSE-D↓':>8} {'offset':>8}
  {'-'*22} {'-'*8} {'-'*8} {'-'*8}
  {'基线（全量计算）':<22} {r_base['lse_c']:>8.4f} {r_base['lse_d']:>8.4f} {r_base['offset']:>+8d}
  {f'缓存（thr={args.threshold}）':<22} {r_cache['lse_c']:>8.4f} {r_cache['lse_d']:>8.4f} {r_cache['offset']:>+8d}
  {'Δ':<22} {lse_c_diff:>+8.4f} {lse_d_diff:>+8.4f}

  唇形同步保持：{'✓ 可接受（ΔLSE-C < 0.5）' if sync_ok else '⚠ 退化（ΔLSE-C ≥ 0.5）'}
    """)

    print("=" * 65)
    print("  论文 Table 数字（供填写）")
    print("=" * 65)
    print(f"""
  方法                FPS     SSIM    PSNR     LSE-C↑  LSE-D↓
  MuseTalk 基线       22.2    1.000   ∞        {r_base['lse_c']:.3f}   {r_base['lse_d']:.3f}
  + 帧跳过(0.15)     40.3    0.9976  57.3dB   {r_cache['lse_c']:.3f}   {r_cache['lse_d']:.3f}
    """)

    result = {
        "threshold": args.threshold,
        "vshift": args.vshift,
        "baseline": r_base,
        "cached":   r_cache,
        "delta": {"lse_c": lse_c_diff, "lse_d": lse_d_diff},
        "sync_preserved": sync_ok,
    }
    out_file = os.path.join(args.output_dir, "lse_eval.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {out_file}")
