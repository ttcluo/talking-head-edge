#!/bin/bash
# P1 多视频评估：搜索服务器现有视频 + 不足时从 HDTF 下载
# 使用方法：
#   cd ~/MuseTalk
#   bash /data/luochuan/talking-head-edge/talking-head-edge/step2/prepare_testdata.sh

set -e
TARGET_DIR="data/video/eval"
MIN_VIDEOS=5
mkdir -p "$TARGET_DIR"

echo "================================================================"
echo "  P1 测试数据准备"
echo "================================================================"

# ── 第一步：搜索服务器现有 mp4 ──────────────────────────────────────
echo ""
echo "[1/3] 搜索服务器现有视频..."

SEARCH_PATHS=(
    "data/video"
    "data"
    "$HOME/MuseTalk/data"
    "$HOME/data"
    "/data"
    "/data/luochuan"
    "$HOME/EchoMimic/assets"
)

found=0
for dir in "${SEARCH_PATHS[@]}"; do
    if [ -d "$dir" ]; then
        while IFS= read -r f; do
            name=$(basename "$f")
            dest="$TARGET_DIR/$name"
            if [ ! -f "$dest" ]; then
                # 只复制时长 > 3s 的视频（避免过短的片段）
                dur=$(ffprobe -v error -show_entries format=duration \
                      -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null || echo 0)
                dur_int=${dur%.*}
                if [ "${dur_int:-0}" -ge 3 ]; then
                    cp "$f" "$dest"
                    echo "  ✓ 找到: $f  (${dur_int}s)"
                    found=$((found + 1))
                fi
            fi
        done < <(find "$dir" -maxdepth 3 -name "*.mp4" 2>/dev/null | head -20)
    fi
done
echo "  当前合计: $found 个视频"

# ── 第二步：若不足，从 HDTF HuggingFace 镜像下载 ────────────────────
if [ "$found" -lt "$MIN_VIDEOS" ]; then
    echo ""
    echo "[2/3] 视频不足 $MIN_VIDEOS 个，尝试从 HuggingFace HDTF 下载..."
    echo "  （需要 huggingface_hub 或 wget，约 100MB）"

    # 方案 A：huggingface_hub
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import os, shutil

# HDTF 公开镜像中几段代表性说话人短视频
FILES = [
    ("WRA_KevinMcCarthy0_000.mp4", "WRA_KevinMcCarthy0_000.mp4"),
    ("WRA_MarcoRubio_000.mp4",     "WRA_MarcoRubio_000.mp4"),
    ("WRA_MitchMcConnell_000.mp4", "WRA_MitchMcConnell_000.mp4"),
    ("RD_Radio34_000.mp4",         "RD_Radio34_000.mp4"),
    ("RD_Radio36_000.mp4",         "RD_Radio36_000.mp4"),
]
TARGET = "data/video/eval"
os.makedirs(TARGET, exist_ok=True)
for hf_name, local_name in FILES:
    dest = os.path.join(TARGET, local_name)
    if os.path.exists(dest):
        print(f"  已存在: {local_name}")
        continue
    try:
        path = hf_hub_download(
            repo_id="OpenTalker/video_retalking",
            filename=f"examples/face/{hf_name}",
            repo_type="space"
        )
        shutil.copy(path, dest)
        print(f"  ✓ {local_name}")
    except Exception as e:
        print(f"  ✗ {local_name}: {e}")
PYEOF
    else
        # 方案 B：直接 wget 从公开可访问的 URL
        echo "  huggingface_hub 不可用，尝试 wget..."
        URLS=(
            "https://github.com/OpenTalker/SadTalker/raw/main/examples/source_image/full3.mp4"
            "https://github.com/OpenTalker/SadTalker/raw/main/examples/source_image/full4.mp4"
            "https://github.com/OpenTalker/SadTalker/raw/main/examples/source_image/full_body_1.mp4"
        )
        for url in "${URLS[@]}"; do
            name=$(basename "$url")
            dest="$TARGET_DIR/$name"
            if [ ! -f "$dest" ]; then
                wget -q --timeout=30 -O "$dest" "$url" && \
                    echo "  ✓ $name" || \
                    echo "  ✗ $name (下载失败，跳过)"
            fi
        done
    fi
fi

# ── 第三步：汇总 ─────────────────────────────────────────────────────
echo ""
echo "[3/3] 最终视频列表:"
count=0
for f in "$TARGET_DIR"/*.mp4; do
    [ -f "$f" ] || continue
    dur=$(ffprobe -v error -show_entries format=duration \
          -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null | cut -d. -f1)
    echo "  $f  (${dur:-?}s)"
    count=$((count + 1))
done

echo ""
if [ "$count" -ge "$MIN_VIDEOS" ]; then
    echo "✓ 准备完成，共 $count 个视频，可以运行:"
    echo ""
    echo "  python /data/luochuan/talking-head-edge/talking-head-edge/step2/multi_eval.py \\"
    echo "      --video_dir $TARGET_DIR \\"
    echo "      --threshold 0.15 \\"
    echo "      --num_frames 100"
else
    echo "⚠ 当前仅 $count 个视频（目标 $MIN_VIDEOS）"
    echo ""
    echo "手动补充方案（选一）："
    echo "  A) yt-dlp 下载（推荐）:"
    echo "     pip install yt-dlp"
    echo "     yt-dlp -f 'bestvideo[height<=480]+bestaudio' \\"
    echo "         --postprocessor-args '-t 30' \\"
    echo "         -o '$TARGET_DIR/%(id)s.%(ext)s' \\"
    echo "         'https://www.youtube.com/watch?v=<VIDEO_ID>'"
    echo ""
    echo "  B) HDTF 官方下载:"
    echo "     git clone https://github.com/MRzzm/HDTF"
    echo "     # 按 HDTF README 用 gdown 下载（需 Google Drive 访问）"
fi
