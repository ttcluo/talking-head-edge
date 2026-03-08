"""
离线预计算各 avatar 的 Whisper 音频特征并保存到磁盘。

使用 MuseTalk V15 的 AudioProcessor（HuggingFace WhisperModel），
与训练/推理管线完全一致，输出 [T, 50, 384] 的逐帧音频特征。

使用方式（在 $MUSE_ROOT 下）：
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/distill/precompute_audio_feats.py \\
      --avatar_list dataset/distill/train_avatars.txt \\
      --out_dir     dataset/distill/audio_feats/ \\
      --audio_dir   data/audio/ \\
      --whisper_dir models/whisper/
"""

import argparse
import os
import sys

import torch

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.getcwd())
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from transformers import WhisperModel
from musetalk.utils.audio_processor import AudioProcessor


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    whisper_dir = args.whisper_dir
    if not os.path.isabs(whisper_dir):
        whisper_dir = os.path.join(MUSE_ROOT, whisper_dir)

    print(f"[加载 AudioProcessor] {whisper_dir}")
    os.chdir(MUSE_ROOT)

    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    whisper = WhisperModel.from_pretrained(whisper_dir).to("cuda")
    whisper.eval()

    if args.avatar_id:
        avatar_ids = [args.avatar_id]
        audio_override = {args.avatar_id: args.audio} if args.audio else {}
    else:
        with open(args.avatar_list) as f:
            avatar_ids = [l.strip() for l in f if l.strip()]
        audio_override = {}

    for avatar_id in avatar_ids:
        out_path = os.path.join(args.out_dir, f"{avatar_id}.pt")
        if os.path.exists(out_path) and not args.force:
            print(f"  ✓ {avatar_id}: 已存在，跳过")
            continue

        if avatar_id in audio_override:
            audio_path = audio_override[avatar_id]
        else:
            vname = avatar_id.replace("avator_", "")
            audio_path = os.path.join(args.audio_dir, f"{vname}.wav")
            if not os.path.exists(audio_path):
                audio_path = os.path.join(args.audio_dir, "yongen.wav")
        if not os.path.exists(audio_path):
            print(f"  ⚠ {avatar_id}: 找不到音频 {audio_path}，跳过")
            continue

        try:
            with torch.no_grad():
                input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
                # audio_prompts: [T, 50, 384]
                audio_prompts = audio_processor.get_whisper_chunk(
                    whisper_input_features=input_features,
                    device="cuda",
                    weight_dtype=torch.float32,
                    whisper=whisper,
                    librosa_length=librosa_length,
                    fps=args.fps,
                    audio_padding_length_left=2,
                    audio_padding_length_right=2,
                )
            # 逐帧拆分存储
            chunks = [audio_prompts[i] for i in range(audio_prompts.shape[0])]
            torch.save(chunks, out_path)
            print(f"  ✓ {avatar_id}: {len(chunks)} 帧 → {out_path}")
        except Exception as e:
            import traceback
            print(f"  ✗ {avatar_id}: 失败")
            traceback.print_exc()

    print(f"\n[完成] 音频特征已保存至 {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar_list", default="dataset/distill/train_avatars.txt",
                        help="avatar 列表；若指定 --avatar_id 则忽略")
    parser.add_argument("--avatar_id",   type=str, default="",
                        help="单 avatar 模式，如 yongen；需配合 --audio 或 data/audio/<id>.wav")
    parser.add_argument("--audio",       type=str, default="",
                        help="单 avatar 时指定音频 wav 路径，如 data/audio/yongen.wav")
    parser.add_argument("--out_dir",     default="dataset/distill/audio_feats/")
    parser.add_argument("--audio_dir",   default="data/audio/")
    parser.add_argument("--whisper_dir", default="models/whisper/")
    parser.add_argument("--fps",         type=int, default=25)
    parser.add_argument("--force",       action="store_true")
    args = parser.parse_args()
    main(args)
