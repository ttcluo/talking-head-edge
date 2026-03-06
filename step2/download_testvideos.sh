#!/bin/bash
# 在本地 Mac 上运行：下载 3 个说话人测试视频
# 使用方法：bash step2/download_testvideos.sh

set -e
OUT="$HOME/Downloads/tad_testvideos"
mkdir -p "$OUT"

# 确保 yt-dlp 可用
if ! command -v yt-dlp &>/dev/null; then
    echo "安装 yt-dlp..."
    pip3 install -q yt-dlp
fi

echo "下载目录: $OUT"
echo ""

# 3 个经典 TED 谈话片段（正面人脸、清晰说话、多年稳定可用）
# 只取前 30 秒，分辨率限制 480p，输出 mp4

VIDEOS=(
    # (YouTube ID) (说话人描述)
    "qp0HIF3SfI4 Simon_Sinek"
    "iG9CE55wbtY Richard_StJohn"
    "3clUtJMM3nU Brene_Brown"
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
    yt-dlp \
        --download-sections "*0:00-0:30" \
        --force-keyframes-at-cuts \
        -f "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/mp4" \
        --merge-output-format mp4 \
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
