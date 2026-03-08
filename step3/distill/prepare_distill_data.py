#!/usr/bin/env python3
"""
从视频目录生成蒸馏训练所需：realtime_distill.yaml、data/audio/<id>.wav、train_avatars.txt。

每个视频对应一个 avatar（avator_1, avator_2, ...）。需在 MuseTalk 项目根下执行 realtime_inference
使用生成的 realtime_distill.yaml 才能得到 results/v15/avatars/<id>/latents.pt。

用法（在 tad 仓库根或任意处）：
  python step3/distill/prepare_distill_data.py \\
      --video_dir /path/to/MuseTalk/dataset/HDTF/source \\
      --muse_root  /path/to/MuseTalk \\
      --max_avatars 20
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="准备蒸馏数据：视频列表 → 配置 + 抽音频 + train_avatars.txt")
    parser.add_argument("--video_dir", required=True, help="放 MP4 的目录（如 HDTF source）")
    parser.add_argument("--muse_root", required=True, help="MuseTalk 项目根目录")
    parser.add_argument("--max_avatars", type=int, default=20, help="最多使用多少个视频（默认 20）")
    parser.add_argument("--skip_audio", action="store_true", help="不抽音频，仅生成配置与 train_avatars.txt")
    args = parser.parse_args()

    video_dir = os.path.abspath(args.video_dir)
    muse_root = os.path.abspath(args.muse_root)
    if not os.path.isdir(video_dir):
        print(f"错误: 视频目录不存在: {video_dir}")
        sys.exit(1)
    os.makedirs(muse_root, exist_ok=True)

    # 扫描 MP4，按文件名排序
    videos = sorted(
        [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")],
        key=lambda x: x,
    )
    videos = videos[: args.max_avatars]
    if not videos:
        print(f"错误: 在 {video_dir} 下未找到 .mp4 文件")
        sys.exit(1)
    print(f"使用 {len(videos)} 个视频: {video_dir}")

    # 目录
    audio_dir = os.path.join(muse_root, "data", "audio")
    distill_dir = os.path.join(muse_root, "dataset", "distill")
    config_dir = os.path.join(muse_root, "configs", "inference")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(distill_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # 1）抽音频
    if not args.skip_audio:
        for i, name in enumerate(videos, start=1):
            vid_path = os.path.join(video_dir, name)
            wav_path = os.path.join(audio_dir, f"{i}.wav")
            if os.path.exists(wav_path):
                print(f"  已存在 {wav_path}，跳过")
                continue
            cmd = [
                "ffmpeg", "-y", "-v", "warning", "-i", vid_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                wav_path,
            ]
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"  警告: 抽音频失败 {vid_path} -> {wav_path}")
            else:
                print(f"  {name} -> data/audio/{i}.wav")

    # 2）生成 realtime_distill.yaml（realtime_inference 在 muse_root 下运行，用相对路径即可）
    lines = []
    for i in range(1, len(videos) + 1):
        vid_name = videos[i - 1]
        vid_abs = os.path.join(video_dir, vid_name)
        wav_rel = f"data/audio/{i}.wav"
        lines.append(f"avator_{i}:")
        lines.append("  preparation: True")
        lines.append("  bbox_shift: 0")
        # 若视频在 muse_root 外，写绝对路径；否则写相对 muse_root 的路径
        try:
            vid_rel = os.path.relpath(vid_abs, muse_root)
        except ValueError:
            vid_rel = vid_abs
        lines.append(f"  video_path: \"{vid_rel}\"")
        lines.append("  audio_clips:")
        lines.append(f"    audio_0: \"{wav_rel}\"")
        lines.append("")
    yaml_path = os.path.join(config_dir, "realtime_distill.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已写入 {yaml_path}")

    # 3）train_avatars.txt
    list_path = os.path.join(distill_dir, "train_avatars.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for i in range(1, len(videos) + 1):
            f.write(f"avator_{i}\n")
    print(f"已写入 {list_path}")

    print("\n下一步（在 MuseTalk 项目根执行，需设置 PYTHONPATH）：")
    print(f"  cd {muse_root}")
    print("  export PYTHONPATH=$PWD")
    print("  python scripts/realtime_inference.py \\")
    print("      --inference_config configs/inference/realtime_distill.yaml \\")
    print("      --unet_model_path models/musetalkV15/unet.pth \\")
    print("      --unet_config models/musetalkV15/musetalk.json \\")
    print("      --version v15")
    print("\n再运行 precompute_audio_feats.py（见 step3/distill/README_dataset.md）。")


if __name__ == "__main__":
    main()
