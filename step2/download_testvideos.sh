#!/bin/bash
# 在本地 Mac 上运行：下载 3 个说话人测试视频
# 使用方法：bash step2/download_testvideos.sh

set -e
OUT="$HOME/Downloads/tad_testvideos"
mkdir -p "$OUT"

# 确保 yt-dlp 可用，统一用 python3 -m yt_dlp 调用（避免 PATH 问题）
if ! python3 -m yt_dlp --version &>/dev/null 2>&1; then
    echo "安装 yt-dlp..."
    pip3 install -q yt-dlp
fi
YT="python3 -m yt_dlp"

echo "下载目录: $OUT"
echo ""

# 3 个 TED 演讲（正面人脸、说话清晰、多年稳定可用）
# 下载完整视频，multi_eval.py 只取前 100 帧，不需要 ffmpeg

VIDEOS=(
    # (YouTube ID) (说话人描述)
    "qp0HIF3SfI4 Simon_Sinek"
    "iG9CE55wbtY Richard_StJohn"
    "psNPSuFoEvE Amy_Cuddy"
)

for entry in "${VIDEOS[@]}"; do
    vid_id=$(echo "$entry" | awk '{print $1}')
    name=$(echo "$entry" | awk '{print $2}')
    out_file="$OUT/${name}.mp4"

    if [ -f "$out_file" ]; then
        echo "  已存在: $out_file"
        continue
    fi

    echo "  下载 $name ($vid_id)..."
    $YT \
        -f "best[height<=480][ext=mp4]/best[height<=480]/best" \
        -o "$out_file" \
        "https://www.youtube.com/watch?v=$vid_id" \
    && echo "  ✓ $out_file" \
    || echo "  ✗ $name 下载失败，请手动替换"
done

echo ""
echo "================================================================"
echo "  完成！将视频传到服务器："
echo "================================================================"
echo ""
echo "  scp $OUT/*.mp4  <用户名>@<服务器IP>:~/MuseTalk/data/video/"
echo ""
echo "  然后在服务器上运行:"
echo "  python .../step2/multi_eval.py --video_dir data/video/ --num_frames 100"
