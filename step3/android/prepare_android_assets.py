"""
为 Android 基准测试生成测试输入文件。

输出（放入 Android app/src/main/assets/）：
  latent_test.bin     [1, 8, 32, 32] float32
  audio_test.bin      [1, 50, 384]   float32
  meta.json           维度信息

使用方式（在 $MUSE_ROOT 下）：
  PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/prepare_android_assets.py \\
      --avatar_id      avator_1 \\
      --audio_feat_dir dataset/distill/audio_feats \\
      --out_dir        $REPO/android/TalkingHeadDemo/app/src/main/assets/
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
parser.add_argument("--frame_idx",      type=int, default=0)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
os.chdir(MUSE_ROOT)

print("=" * 50)
print("  Android 测试资产生成")
print("=" * 50)

# ==================== 加载一帧 latent ====================
latent_path = os.path.join(args.avatar_base, args.avatar_id, "latents.pt")
latents = torch.load(latent_path, map_location="cpu")
lat = latents[args.frame_idx % len(latents)].float()
if lat.dim() == 3:
    lat = lat.unsqueeze(0)  # [1, 8, 32, 32]
print(f"  latent shape: {lat.shape}")

# ==================== 加载音频特征 + 应用 PE ====================
audio_path = os.path.join(args.audio_feat_dir, f"{args.avatar_id}.pt")
audio_chunks = torch.load(audio_path, map_location="cpu")
af = audio_chunks[args.frame_idx % len(audio_chunks)].float().unsqueeze(0)  # [1, 50, 384]

pe = PositionalEncoding(d_model=384)
af = pe(af)
print(f"  audio_feat shape (after PE): {af.shape}")

# ==================== 保存为二进制 ====================
lat_np = lat.numpy().astype(np.float32)
af_np  = af.numpy().astype(np.float32)

lat_path  = os.path.join(args.out_dir, "latent_test.bin")
audio_path_out = os.path.join(args.out_dir, "audio_test.bin")
meta_path = os.path.join(args.out_dir, "meta.json")

lat_np.tofile(lat_path)
af_np.tofile(audio_path_out)

meta = {
    "latent_shape":     list(lat_np.shape),
    "audio_feat_shape": list(af_np.shape),
    "latent_dtype":     "float32",
    "audio_dtype":      "float32",
    "timestep":         0,
    "avatar_id":        args.avatar_id,
    "frame_idx":        args.frame_idx,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

lat_size   = os.path.getsize(lat_path) / 1024
audio_size = os.path.getsize(audio_path_out) / 1024

print(f"""
  ✓ latent_test.bin:  {lat_size:.1f} KB  → {lat_path}
  ✓ audio_test.bin:   {audio_size:.1f} KB  → {audio_path_out}
  ✓ meta.json         → {meta_path}
""")
print("下一步：将这 3 个文件复制到 Android assets/ 目录")
