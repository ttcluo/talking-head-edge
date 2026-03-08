#!/usr/bin/env python3
"""
解压 HDTF 的 videos.zip，并整理为同一目录下的 MP4，便于后续 prepare_distill_data.py 使用。

用法（在服务器上，指向 HDTF source 目录）：
  python step3/distill/unzip_hdtf_videos.py --dir /data/luochuan/.../MuseTalk/dataset/HDTF/source

或直接指定 zip 路径：
  python step3/distill/unzip_hdtf_videos.py --zip /path/to/dataset/HDTF/source/videos.zip
"""
import argparse
import os
import shutil
import zipfile
import sys


def main():
    parser = argparse.ArgumentParser(description="解压 HDTF videos.zip 并整理为单目录 MP4")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", help="含 videos.zip 的目录（如 .../HDTF/source）")
    g.add_argument("--zip", help="videos.zip 的完整路径")
    parser.add_argument("--remove_zip", action="store_true", help="解压后删除 zip 以省空间")
    args = parser.parse_args()

    if args.zip:
        zip_path = os.path.abspath(args.zip)
        target_dir = os.path.dirname(zip_path)
    else:
        target_dir = os.path.abspath(args.dir)
        zip_path = os.path.join(target_dir, "videos.zip")

    if not os.path.isfile(zip_path):
        print(f"错误: 未找到 {zip_path}")
        sys.exit(1)

    print(f"解压: {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

    # 若解压后仅有一个子目录且其中为视频，则把视频提到 target_dir，便于 --video_dir 直接指 source
    subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    mp4_in_root = [f for f in os.listdir(target_dir) if f.lower().endswith(".mp4")]
    if len(subdirs) == 1 and not mp4_in_root:
        subdir = os.path.join(target_dir, subdirs[0])
        files = os.listdir(subdir)
        mp4s = [f for f in files if f.lower().endswith(".mp4")]
        if mp4s:
            print(f"将 {subdirs[0]}/ 下 {len(mp4s)} 个 MP4 移到 {target_dir}")
            for f in mp4s:
                src = os.path.join(subdir, f)
                dst = os.path.join(target_dir, f)
                if os.path.exists(dst):
                    continue
                shutil.move(src, dst)
            try:
                os.rmdir(subdir)
            except OSError:
                for f in files:
                    if f not in mp4s:
                        shutil.move(os.path.join(subdir, f), os.path.join(target_dir, f))
                os.rmdir(subdir)

    if args.remove_zip:
        os.remove(zip_path)
        print("已删除 zip 文件")

    mp4_count = len([f for f in os.listdir(target_dir) if f.lower().endswith(".mp4")])
    print(f"完成。当前目录 MP4 数量: {mp4_count}")
    print(f"后续可将 --video_dir 设为: {target_dir}")


if __name__ == "__main__":
    main()
