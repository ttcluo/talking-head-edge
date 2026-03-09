#!/bin/bash
# single_frame_baseline.sh
# 补测单帧基线（batch_size=1），用于公平比较 MATS 与基线的加速比
#
# 背景：原 batch_eval.sh 用 batch_size=4，有批处理吞吐优势；
#       MATS 是逐帧顺序推理（实时模式），公平对照应是 batch_size=1 基线。
#
# 使用方式：
#   cd $MUSE_ROOT
#   PYTHONPATH=$MUSE_ROOT bash $REPO/step2/single_frame_baseline.sh \
#       2>&1 | tee profile_results/single_frame_baseline.log

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MUSE_ROOT="${MUSE_ROOT:-$(pwd)}"
AUDIO="data/audio/yongen.wav"
THRESHOLD=0.15
MAX_SKIP=2
NUM_FRAMES=200
RESULT_DIR="profile_results/single_frame"
mkdir -p "$RESULT_DIR"

echo "================================================================="
echo "  单帧基线测试（batch_size=1 vs MATS）"
echo "================================================================="

VIDEOS=(
    "a"
    "b"
    "c"
    "sun"
    "yongen"
)

# 输出表头
SUMMARY="$RESULT_DIR/summary.tsv"
echo -e "video\tskip_rate\tbase_fps_b4\tbase_fps_b1\tmats_fps\tspeedup_b4\tspeedup_b1\tssim\tpsnr" \
    > "$SUMMARY"

# 已有 batch_size=4 数据（从 batch_eval 结果中读取，手动填入）
declare -A BASE_FPS_B4
BASE_FPS_B4["a"]="30.8"
BASE_FPS_B4["b"]="30.7"
BASE_FPS_B4["c"]="30.6"
BASE_FPS_B4["sun"]="30.6"
BASE_FPS_B4["yongen"]="30.6"

for VNAME in "${VIDEOS[@]}"; do
    AVATAR_ID="avator_${VNAME}"
    LOG="$RESULT_DIR/${VNAME}_b1.log"
    DEMO_OUT="$RESULT_DIR/${VNAME}_demo_b1.mp4"

    echo ""
    echo "=========================================="
    echo "  [${VNAME}.mp4]  batch_size=1"
    echo "=========================================="

    PYTHONPATH="$MUSE_ROOT" python "$REPO/step2/demo_video.py" \
        --avatar_id  "$AVATAR_ID" \
        --audio      "$AUDIO" \
        --threshold  "$THRESHOLD" \
        --max_skip   "$MAX_SKIP" \
        --num_frames "$NUM_FRAMES" \
        --batch_size 1 \
        --out        "$DEMO_OUT" 2>&1 | tee "$LOG"

    # 提取指标
    SKIP_RATE=$(grep -oP '跳过率：\K[\d.]+(?=%)' "$LOG" | tail -1)
    BASE_B1=$(grep -oP '基线：\K[\d.]+' "$LOG" | tail -1)
    MATS_FPS=$(grep -oP 'MATS：\K[\d.]+' "$LOG" | tail -1)
    SSIM=$(grep -oP 'SSIM=\K[\d.]+' "$LOG" | tail -1)
    PSNR=$(grep -oP 'PSNR=\K[\d.]+' "$LOG" | tail -1)

    BASE_B4="${BASE_FPS_B4[$VNAME]}"
    SPEEDUP_B4=$(python3 -c "print(f'{float(\"$MATS_FPS\")/float(\"$BASE_B4\"):.2f}')" 2>/dev/null || echo "?")
    SPEEDUP_B1=$(python3 -c "print(f'{float(\"$MATS_FPS\")/float(\"$BASE_B1\"):.2f}')" 2>/dev/null || echo "?")

    echo ""
    echo "  → batch=4 基线: ${BASE_B4} FPS  batch=1 基线: ${BASE_B1} FPS"
    echo "  → MATS:         ${MATS_FPS} FPS  (vs b4: ${SPEEDUP_B4}×, vs b1: ${SPEEDUP_B1}×)"
    echo "  → SSIM=${SSIM}  PSNR=${PSNR}dB  跳过率=${SKIP_RATE}%"

    echo -e "${VNAME}\t${SKIP_RATE}%\t${BASE_B4}\t${BASE_B1}\t${MATS_FPS}\t${SPEEDUP_B4}x\t${SPEEDUP_B1}x\t${SSIM}\t${PSNR}" \
        >> "$SUMMARY"
done

echo ""
echo "================================================================="
echo "  汇总（与 batch_size=4 的公平对比）"
echo "================================================================="
column -t -s $'\t' "$SUMMARY"
echo ""
echo "  日志目录: $RESULT_DIR/"
