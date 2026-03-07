"""
离线预计算各 avatar 的 Whisper 音频特征并保存到磁盘。

运行一次后，AvatarDistillDataset 直接加载预计算结果，无需运行时 Whisper。

使用方式（在 $MUSE_ROOT 下）：
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/distill/precompute_audio_feats.py \
      --avatar_list dataset/distill/train_avatars.txt \
      --out_dir     dataset/distill/audio_feats/ \
      --audio_dir   data/audio/
"""

import argparse
import os
import sys

import torch

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.getcwd())
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from musetalk.whisper.audio2feature import Audio2Feature


def find_whisper_model(muse_root: str) -> str:
    """在常见位置查找 Whisper 模型文件。"""
    candidates = [
        os.path.join(muse_root, "models/whisper/tiny.pt"),
        os.path.join(muse_root, "models/whisper/tiny"),
        os.path.join(muse_root, "models/whisper"),
        "tiny",   # 让 whisper 从缓存加载
    ]
    for c in candidates:
        if os.path.isfile(c) or (c == "tiny"):
            return c
    # 搜索 models/whisper/ 下的任意 .pt 文件
    whisper_dir = os.path.join(muse_root, "models/whisper")
    if os.path.isdir(whisper_dir):
        for f in os.listdir(whisper_dir):
            if f.endswith(".pt"):
                return os.path.join(whisper_dir, f)
    return "tiny"


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    model_path = find_whisper_model(MUSE_ROOT)
    print(f"[Whisper 路径] {model_path}")

    os.chdir(MUSE_ROOT)  # 保证相对路径可用
    audio2feat = Audio2Feature(model_path=model_path)

    with open(args.avatar_list) as f:
        avatar_ids = [l.strip() for l in f if l.strip()]

    for avatar_id in avatar_ids:
        out_path = os.path.join(args.out_dir, f"{avatar_id}.pt")
        if os.path.exists(out_path) and not args.force:
            print(f"  ✓ {avatar_id}: 已存在，跳过")
            continue

        # 找音频文件
        vname = avatar_id.replace("avator_", "")
        audio_path = os.path.join(args.audio_dir, f"{vname}.wav")
        if not os.path.exists(audio_path):
            audio_path = os.path.join(args.audio_dir, "yongen.wav")
        if not os.path.exists(audio_path):
            print(f"  ⚠ {avatar_id}: 找不到音频，跳过")
            continue

        try:
            whisper_feat = audio2feat.get_hubert_from_whisper(audio_path)
            chunks = audio2feat.feature2chunks(
                feature_array=whisper_feat, fps=args.fps
            )
            chunks_t = [
                torch.tensor(c, dtype=torch.float32) if not isinstance(c, torch.Tensor)
                else c.float()
                for c in chunks
            ]
            torch.save(chunks_t, out_path)
            print(f"  ✓ {avatar_id}: {len(chunks_t)} 帧 → {out_path}")
        except Exception as e:
            print(f"  ✗ {avatar_id}: 失败 ({e})")

    print(f"\n[完成] 音频特征已保存至 {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar_list", default="dataset/distill/train_avatars.txt")
    parser.add_argument("--out_dir",     default="dataset/distill/audio_feats/")
    parser.add_argument("--audio_dir",   default="data/audio/")
    parser.add_argument("--fps",         type=int, default=25)
    parser.add_argument("--force",       action="store_true")
    args = parser.parse_args()
    main(args)
