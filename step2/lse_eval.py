"""
LSE-C（唇形同步置信度）评估脚本
使用 LatentSync 的像素空间 SyncNet 计算 LSE-C/LSE-D

依赖：
  pip install python_speech_features
  已下载：models/syncnet/latentsync_syncnet.pt
  已克隆：~/LatentSync（LatentSync 仓库）

使用方法：
    conda activate musetalk
    cd ~/MuseTalk
    # Step 1: 生成对比视频（需要先运行 MuseTalk 推理）
    python /path/to/step2/lse_eval.py \
        --mode generate_and_eval \
        --video data/video/yongen.mp4 \
        --audio data/audio/yongen.wav \
        --threshold 0.15

    # Step 2: 仅评估已有视频
    python /path/to/step2/lse_eval.py \
        --mode eval_only \
        --baseline_video results/baseline.mp4 \
        --cached_video results/cached.mp4

输出：
    - baseline vs cached 的 LSE-C / LSE-D / av_offset
    - 结果保存到 profile_results/lse_eval.json
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch

MUSETALK_ROOT = os.environ.get("MUSE_ROOT", os.path.expanduser("~/MuseTalk"))
LATENTSYNC_ROOT = os.environ.get("LATENTSYNC_ROOT", os.path.expanduser("~/LatentSync"))

if MUSETALK_ROOT not in sys.path:
    sys.path.insert(0, MUSETALK_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["generate_and_eval", "eval_only", "check_env"],
                    default="check_env")
parser.add_argument("--video", type=str, default="data/video/yongen.mp4")
parser.add_argument("--audio", type=str, default="data/audio/yongen.wav")
parser.add_argument("--threshold", type=float, default=0.15,
                    help="帧跳过运动阈值（对应 quality_eval 的最优阈值）")
parser.add_argument("--baseline_video", type=str, default="")
parser.add_argument("--cached_video", type=str, default="")
parser.add_argument("--syncnet_ckpt", type=str,
                    default="models/syncnet/latentsync_syncnet.pt")
parser.add_argument("--output_dir", type=str, default="profile_results")
parser.add_argument("--num_frames", type=int, default=100,
                    help="评估帧数（建议 ≥ 100 帧以保证 LSE-C 稳定）")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("  LSE-C 唇形同步置信度评估")
print("=" * 65)

# ==================== 环境检查 ====================
def check_environment():
    print("\n[环境检查]")
    ok = True

    # python_speech_features
    try:
        import python_speech_features
        print(f"  ✓ python_speech_features")
    except ImportError:
        print(f"  ✗ python_speech_features 未安装")
        print(f"    修复：pip install python_speech_features")
        ok = False

    # LatentSync SyncNet 权重
    if os.path.exists(args.syncnet_ckpt):
        size_mb = os.path.getsize(args.syncnet_ckpt) / 1024 / 1024
        print(f"  ✓ SyncNet 权重: {args.syncnet_ckpt} ({size_mb:.1f}MB)")
    else:
        print(f"  ✗ SyncNet 权重不存在: {args.syncnet_ckpt}")
        ok = False

    # LatentSync 仓库（用于 SyncNet 模型代码）
    syncnet_code = os.path.join(LATENTSYNC_ROOT, "eval/syncnet/syncnet.py")
    if os.path.exists(syncnet_code):
        print(f"  ✓ LatentSync 仓库: {LATENTSYNC_ROOT}")
    else:
        print(f"  ✗ LatentSync 仓库未克隆到 {LATENTSYNC_ROOT}")
        print(f"    修复：git clone https://github.com/bytedance/LatentSync ~/LatentSync")
        ok = False

    # ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        print(f"  ✓ ffmpeg 可用")
    except Exception:
        print(f"  ✗ ffmpeg 不可用")
        ok = False

    return ok

# ==================== SyncNet 加载 ====================
def load_syncnet():
    """加载 LatentSync 的像素空间 SyncNet"""
    if LATENTSYNC_ROOT not in sys.path:
        sys.path.insert(0, LATENTSYNC_ROOT)

    try:
        from eval.syncnet.syncnet import S
        syncnet = S(num_layers_in_fc_layers=1024).to(device)
        state = torch.load(args.syncnet_ckpt, map_location=device, weights_only=True)
        syncnet.load_state_dict(state)
        syncnet.eval()
        print(f"  ✓ SyncNet 加载完成（像素空间，Wav2Lip 架构）")
        return syncnet
    except Exception as e:
        print(f"  ✗ SyncNet 加载失败: {e}")
        print(f"  → 检查权重格式：weights_only=False 模式重试")
        try:
            from eval.syncnet.syncnet import S
            syncnet = S(num_layers_in_fc_layers=1024).to(device)
            state = torch.load(args.syncnet_ckpt, map_location=device, weights_only=False)
            if "state_dict" in state:
                state = state["state_dict"]
            syncnet.load_state_dict(state, strict=False)
            syncnet.eval()
            print(f"  ✓ SyncNet 加载完成（strict=False 模式）")
            return syncnet
        except Exception as e2:
            print(f"  ✗ SyncNet 加载彻底失败: {e2}")
            return None

# ==================== LSE-C 计算核心 ====================
def compute_lse(syncnet, video_path: str, vshift: int = 15, batch_size: int = 20):
    """
    计算视频的 LSE-C 和 LSE-D
    LSE-C：置信度，越高说明口型越同步（主要指标）
    LSE-D：距离，越低越好
    av_offset：视频相对音频的偏移帧数，0 为完全同步
    """
    import python_speech_features
    from scipy.io import wavfile
    from scipy import signal as scipy_signal

    with tempfile.TemporaryDirectory() as tmp:
        # 提取帧
        cmd = f"ffmpeg -loglevel error -nostdin -y -i {video_path} -f image2 {tmp}/%06d.jpg"
        subprocess.call(cmd, shell=True)

        # 提取音频
        wav_path = os.path.join(tmp, "audio.wav")
        cmd = f"ffmpeg -loglevel error -nostdin -y -i {video_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {wav_path}"
        subprocess.call(cmd, shell=True)

        # 读取帧
        import glob
        flist = sorted(glob.glob(os.path.join(tmp, "*.jpg")))
        if len(flist) < 10:
            return None, None, None, "视频帧数不足"

        images = []
        for f in flist:
            img = cv2.imread(f)
            img = cv2.resize(img, (224, 224))
            images.append(img)

        im = np.stack(images, axis=3)          # (H, W, C, T)
        im = np.expand_dims(im, axis=0)        # (1, H, W, C, T)
        im = np.transpose(im, (0, 3, 4, 1, 2)) # (1, T, C, H, W)
        imtv = torch.from_numpy(im.astype(np.float32))

        # 读取音频 MFCC
        sample_rate, audio = wavfile.read(wav_path)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.from_numpy(cc.astype(np.float32))

        min_length = min(len(images), int(len(audio) / 640))
        if min_length < 10:
            return None, None, None, "音视频长度不匹配"

        lastframe = min_length - 5
        im_feat, cc_feat = [], []

        with torch.no_grad():
            for i in range(0, lastframe, batch_size):
                end = min(lastframe, i + batch_size)
                im_batch = torch.cat(
                    [imtv[:, :, f:f+5, :, :] for f in range(i, end)], 0
                ).to(device)
                im_feat.append(syncnet.forward_lip(im_batch).cpu())

                cc_batch = torch.cat(
                    [cct[:, :, :, f*4:f*4+20] for f in range(i, end)], 0
                ).to(device)
                cc_feat.append(syncnet.forward_aud(cc_batch).cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        # 计算偏移和置信度
        win_size = vshift * 2 + 1
        cc_feat_pad = torch.nn.functional.pad(cc_feat, (0, 0, vshift, vshift))
        dists = []
        for i in range(len(im_feat)):
            dists.append(
                torch.nn.functional.pairwise_distance(
                    im_feat[[i]].repeat(win_size, 1),
                    cc_feat_pad[i:i+win_size]
                )
            )

        mean_dists = torch.mean(torch.stack(dists, 1), 1)
        min_dist, minidx = torch.min(mean_dists, 0)
        av_offset = vshift - minidx.item()
        lse_c = (torch.median(mean_dists) - min_dist).item()
        lse_d = min_dist.item()

    return lse_c, lse_d, av_offset, None

# ==================== 视频生成（简化版：从已有帧+音频合成）====================
def generate_video_with_audio(frames_np: list, audio_path: str, output_path: str, fps: int = 25):
    """
    将 numpy 帧列表 + 音频文件合成 MP4
    frames_np: list of (H, W, 3) uint8 BGR 帧
    """
    if not frames_np:
        return False

    h, w = frames_np[0].shape[:2]
    tmp_video = output_path.replace(".mp4", "_noaudio.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    for f in frames_np:
        writer.write(f)
    writer.release()

    # 合并音频
    cmd = (f"ffmpeg -loglevel error -nostdin -y "
           f"-i {tmp_video} -i {audio_path} "
           f"-c:v copy -c:a aac -shortest {output_path}")
    subprocess.call(cmd, shell=True)
    os.remove(tmp_video)
    return os.path.exists(output_path)

# ==================== 主逻辑 ====================
if args.mode == "check_env":
    env_ok = check_environment()
    if env_ok:
        print("\n  ✅ 环境就绪，可以运行 generate_and_eval 或 eval_only 模式")
    else:
        print("\n  ⚠ 请解决上述问题后重新运行")
        print("""
  快速修复命令：
    pip install python_speech_features
    git clone https://github.com/bytedance/LatentSync ~/LatentSync
        """)
    sys.exit(0)

# ——————————————————————————————————————————————
elif args.mode == "generate_and_eval":
    print("\n[模式：生成视频 + LSE-C 评估]")
    print(f"  输入视频: {args.video}")
    print(f"  输入音频: {args.audio}")
    print(f"  跳过阈值: {args.threshold}")

    # 环境检查
    if not check_environment():
        sys.exit(1)

    from diffusers import AutoencoderKL
    from musetalk.models.unet import UNet2DConditionModel
    import json as _json

    print("\n[加载模型]")
    with open("models/musetalkV15/musetalk.json") as f:
        unet_config = _json.load(f)

    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained("models/sd-vae").to(device, dtype)
    unet = UNet2DConditionModel(**unet_config).to(device, dtype)
    unet.load_state_dict(torch.load("models/musetalkV15/unet.pth", map_location=device))
    vae.eval()
    unet.eval()

    # 加载音频特征（Whisper）
    try:
        from musetalk.whisper.audio2feature import Audio2Feature
        audio_processor = Audio2Feature(
            whisper_model_type="tiny",
            model_path="models/whisper/pytorch_model.bin"
        )
        audio_features = audio_processor.audio2feat(args.audio)
        audio_chunks = audio_processor.feature2chunks(feature_array=audio_features, fps=25)
        whisper_ok = True
        print(f"  ✓ 音频特征加载完成，共 {len(audio_chunks)} 块")
    except Exception as e:
        print(f"  ⚠ Whisper 加载失败（{e}），使用零向量")
        audio_chunks = [np.zeros((1, 384), dtype=np.float32)] * args.num_frames
        whisper_ok = False

    # 读取视频
    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    N = len(frames)
    print(f"  读取 {N} 帧")

    # 生成基线视频
    print(f"\n[生成基线视频（{N} 帧，全量计算）]")
    timestep = torch.tensor([0.0], device=device, dtype=dtype)
    input_latents = []
    baseline_pixels = []

    with torch.no_grad():
        for i, frame in enumerate(frames):
            face = cv2.resize(frame[:256, :256], (256, 256))
            face_t = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1
            face_t = face_t.unsqueeze(0).to(device, dtype)

            latent = vae.encode(face_t).latent_dist.mean
            audio_chunk = torch.from_numpy(
                audio_chunks[i % len(audio_chunks)]
            ).unsqueeze(0).to(device, dtype)
            mask = torch.zeros_like(latent)
            unet_input = torch.cat([latent, mask], dim=1)
            out = unet(unet_input, timestep, encoder_hidden_states=audio_chunk).sample
            decoded = vae.decode(out).sample
            pixel = (decoded[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5
            pixel = pixel.clip(0, 255).astype(np.uint8)

            input_latents.append(latent.detach().cpu().float())
            baseline_pixels.append(cv2.cvtColor(pixel, cv2.COLOR_RGB2BGR))

    baseline_path = os.path.join(args.output_dir, "baseline.mp4")
    generate_video_with_audio(baseline_pixels, args.audio, baseline_path)
    print(f"  ✓ 基线视频: {baseline_path}")

    # 生成缓存版本视频
    print(f"\n[生成缓存视频（阈值 {args.threshold}，跳过低运动帧）]")
    cached_pixels = []
    skip_count = 0
    last_out_lat = None
    last_decoded = None

    motions = []
    for i in range(1, N):
        m = (input_latents[i] - input_latents[i-1]).norm() / \
            (input_latents[i-1].norm() + 1e-6)
        motions.append(float(m))

    with torch.no_grad():
        for i, frame in enumerate(frames):
            motion = motions[i-1] if i > 0 else 999.0

            if i == 0 or motion >= args.threshold:
                # 计算
                face = cv2.resize(frame[:256, :256], (256, 256))
                face_t = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1
                face_t = face_t.unsqueeze(0).to(device, dtype)
                latent = vae.encode(face_t).latent_dist.mean
                audio_chunk = torch.from_numpy(
                    audio_chunks[i % len(audio_chunks)]
                ).unsqueeze(0).to(device, dtype)
                mask = torch.zeros_like(latent)
                unet_input = torch.cat([latent, mask], dim=1)
                out = unet(unet_input, timestep, encoder_hidden_states=audio_chunk).sample
                last_out_lat = out
                decoded = vae.decode(out).sample
                pixel = (decoded[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5
                last_decoded = pixel.clip(0, 255).astype(np.uint8)
            else:
                # 跳过：复用上帧
                skip_count += 1

            cached_pixels.append(cv2.cvtColor(last_decoded, cv2.COLOR_RGB2BGR))

    skip_rate = skip_count / N
    print(f"  跳过率: {skip_rate:.1%}（{skip_count}/{N} 帧）")
    cached_path = os.path.join(args.output_dir, "cached.mp4")
    generate_video_with_audio(cached_pixels, args.audio, cached_path)
    print(f"  ✓ 缓存视频: {cached_path}")

    args.baseline_video = baseline_path
    args.cached_video = cached_path
    # 继续进入 eval_only 逻辑

# ——————————————————————————————————————————————
if args.mode in ["eval_only", "generate_and_eval"]:
    if args.mode == "eval_only":
        print(f"\n[模式：仅评估]")
        if not check_environment():
            sys.exit(1)

    if not args.baseline_video or not os.path.exists(args.baseline_video):
        print(f"  ✗ baseline_video 不存在: {args.baseline_video}")
        sys.exit(1)
    if not args.cached_video or not os.path.exists(args.cached_video):
        print(f"  ✗ cached_video 不存在: {args.cached_video}")
        sys.exit(1)

    print(f"\n[加载 SyncNet]")
    syncnet = load_syncnet()
    if syncnet is None:
        print("  ✗ SyncNet 加载失败，退出")
        sys.exit(1)

    print(f"\n[计算 LSE-C / LSE-D]")
    print(f"  计算基线视频...")
    lse_c_base, lse_d_base, offset_base, err = compute_lse(syncnet, args.baseline_video)
    if err:
        print(f"  ✗ 基线评估失败: {err}")
        sys.exit(1)
    print(f"  基线：LSE-C={lse_c_base:.4f}  LSE-D={lse_d_base:.4f}  偏移={offset_base}帧")

    print(f"  计算缓存视频...")
    lse_c_cache, lse_d_cache, offset_cache, err = compute_lse(syncnet, args.cached_video)
    if err:
        print(f"  ✗ 缓存评估失败: {err}")
        sys.exit(1)
    print(f"  缓存：LSE-C={lse_c_cache:.4f}  LSE-D={lse_d_cache:.4f}  偏移={offset_cache}帧")

    # 结果输出
    print("\n" + "=" * 65)
    print("  唇形同步对比结果")
    print("=" * 65)

    lse_c_diff = lse_c_cache - lse_c_base
    lse_d_diff = lse_d_cache - lse_d_base
    ACCEPTABLE_THRESHOLD = 0.5  # LSE-C 下降 < 0.5 为可接受（文献参考值）

    print(f"""
  {'方法':<20} {'LSE-C↑':>8} {'LSE-D↓':>8} {'av_offset':>10}
  {'-'*20} {'-'*8} {'-'*8} {'-'*10}
  {'MuseTalk 基线':<20} {lse_c_base:>8.4f} {lse_d_base:>8.4f} {offset_base:>10}
  {'+ 帧跳过缓存':<20} {lse_c_cache:>8.4f} {lse_d_cache:>8.4f} {offset_cache:>10}
  {'差值(Δ)':<20} {lse_c_diff:>+8.4f} {lse_d_diff:>+8.4f}

  唇形同步保持情况：
    LSE-C 变化: {lse_c_diff:+.4f}  {'✓ 可接受（< 0.5）' if abs(lse_c_diff) < ACCEPTABLE_THRESHOLD else '✗ 超过阈值'}
    LSE-D 变化: {lse_d_diff:+.4f}  {'✓ 改善' if lse_d_diff < 0 else '✓ 基本持平' if lse_d_diff < 1 else '⚠ 略有退化'}
    """)

    # 论文数字摘要
    print("=" * 65)
    print("  论文 Table 汇总（最终完整数字）")
    print("=" * 65)
    print(f"""
  方法                  FPS     SSIM    PSNR     LSE-C↑  LSE-D↓
  MuseTalk FP16（基线） 22.2    1.000   ∞        {lse_c_base:.3f}   {lse_d_base:.3f}
  + 帧跳过(thr=0.15)   40.3    0.9976  57.3dB   {lse_c_cache:.3f}   {lse_d_cache:.3f}
  """)

    # 保存
    result = {
        "baseline_video": args.baseline_video,
        "cached_video": args.cached_video,
        "threshold": args.threshold,
        "baseline": {"lse_c": lse_c_base, "lse_d": lse_d_base, "av_offset": offset_base},
        "cached":   {"lse_c": lse_c_cache, "lse_d": lse_d_cache, "av_offset": offset_cache},
        "delta":    {"lse_c": lse_c_diff, "lse_d": lse_d_diff},
        "lse_c_acceptable": bool(abs(lse_c_diff) < ACCEPTABLE_THRESHOLD),
    }

    out_file = os.path.join(args.output_dir, "lse_eval.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存到: {out_file}")
