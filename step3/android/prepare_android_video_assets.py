"""
为 Android 视频预览生成多帧序列资源。

输出（放入 app/src/main/assets/）：
  latents_seq.bin   N × [1,8,32,32] float32
  audio_seq.bin     N × [1,50,384]  float32（已 PE）
  video_meta.json   num_frames, shapes, vae_scaling_factor

使用方式（在 $MUSE_ROOT 下）：
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/prepare_android_video_assets.py \\
      --avatar_id      avator_1 \\
      --audio_feat_dir dataset/distill/audio_feats \\
      --out_dir        $REPO/android/TalkingHeadDemo/app/src/main/assets/ \\
      --num_frames     80
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

MUSE_ROOT = os.environ.get("MUSE_ROOT", os.getcwd())
if MUSE_ROOT not in sys.path:
    sys.path.insert(0, MUSE_ROOT)

from musetalk.models.unet import PositionalEncoding

parser = argparse.ArgumentParser()
parser.add_argument("--avatar_id",      default="avator_1")
parser.add_argument("--avatar_base",    default="results/v15/avatars")
parser.add_argument("--audio_feat_dir", default="dataset/distill/audio_feats")
parser.add_argument("--out_dir",        required=True)
parser.add_argument("--num_frames",     type=int, default=80)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.chdir(MUSE_ROOT)

print("=" * 50)
print("  Android 视频序列资产生成")
print("=" * 50)

latent_path = os.path.join(args.avatar_base, args.avatar_id, "latents.pt")
audio_path  = os.path.join(args.audio_feat_dir, f"{args.avatar_id}.pt")
latents     = torch.load(latent_path, map_location="cpu")
audio_chunks = torch.load(audio_path, map_location="cpu")

n = min(args.num_frames, len(latents), len(audio_chunks))
pe = PositionalEncoding(d_model=384)

lat_list = []
aud_list = []
for i in range(n):
    lat = latents[i].float()
    if lat.dim() == 3:
        lat = lat.unsqueeze(0)
    af = audio_chunks[i].float().unsqueeze(0)
    af = pe(af)
    lat_list.append(lat.numpy().astype(np.float32))
    aud_list.append(af.numpy().astype(np.float32))

lat_arr = np.concatenate(lat_list, axis=0)   # [n, 1, 8, 32, 32] -> flatten to (n, 1*8*32*32)
aud_arr = np.concatenate(aud_list, axis=0)   # [n, 1, 50, 384]
lat_arr.tofile(os.path.join(args.out_dir, "latents_seq.bin"))
aud_arr.tofile(os.path.join(args.out_dir, "audio_seq.bin"))

meta = {
    "num_frames": n,
    "latent_shape": [1, 8, 32, 32],
    "audio_shape":  [1, 50, 384],
    "vae_scaling_factor": 0.18215,
}
with open(os.path.join(args.out_dir, "video_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

lat_size = os.path.getsize(os.path.join(args.out_dir, "latents_seq.bin")) / 1024
aud_size = os.path.getsize(os.path.join(args.out_dir, "audio_seq.bin")) / 1024
print(f"  ✓ {n} 帧  latent_seq: {lat_size:.1f} KB  audio_seq: {aud_size:.1f} KB")
print(f"  ✓ video_meta.json  vae_scaling_factor={meta['vae_scaling_factor']}")
print("=" * 50)
