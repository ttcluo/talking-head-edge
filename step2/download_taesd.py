"""
TAESD 模型离线下载

在可访问 HuggingFace 的机器上执行，将 madebyollin/taesd 下载到本地目录，
再 scp 到无法访问 HF 的服务器。

用法：
  python step2/download_taesd.py --out_dir ./taesd_cache

  然后：scp -r taesd_cache root@GPU1:/data/luochuan/talking-head-edge/talking-head-edge/models/

  服务器上运行 compare_vae_taesd.py 时加：
  --taesd_dir models/taesd_cache
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="./taesd_cache",
                    help="下载目标目录")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
print(f"下载 TAESD 到 {os.path.abspath(args.out_dir)}")

try:
    from huggingface_hub import snapshot_download
    path = snapshot_download("madebyollin/taesd", local_dir=args.out_dir, local_dir_use_symlinks=False)
    print(f"✓ 下载完成: {path}")
    print(f"\n  上传到服务器: scp -r {os.path.abspath(args.out_dir)} user@server:/path/to/models/")
    print(f"  运行对比时加: --taesd_dir /path/to/models/taesd_cache")
except Exception as e:
    print(f"✗ 下载失败: {e}")
    print("  请确保: pip install huggingface_hub 且网络可访问 huggingface.co")
