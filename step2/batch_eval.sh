#!/bin/bash
# 批量评估：对 data/video/ 所有视频跑 SSIM/PSNR（max_skip=2），并对 sun.mp4 做 LSE-C
#
# 运行方式：
#   cd $MUSE_ROOT
#   PYTHONPATH=$MUSE_ROOT bash $REPO/step2/batch_eval.sh 2>&1 | tee profile_results/batch_eval.log

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MUSE_ROOT="${MUSE_ROOT:-$(pwd)}"
AUDIO="data/audio/yongen.wav"
THRESHOLD=0.15
MAX_SKIP=2
NUM_FRAMES=200
RESULT_DIR="profile_results/batch"
mkdir -p "$RESULT_DIR"

echo "================================================================="
echo "  批量评估：SSIM/PSNR + LSE-C（max_skip=${MAX_SKIP}）"
echo "================================================================="

mapfile -t VIDEOS < <(ls data/video/*.mp4 2>/dev/null | sort)
echo ""
echo "[视频列表]"
for v in "${VIDEOS[@]}"; do echo "  $v"; done

SUMMARY_FILE="$RESULT_DIR/summary.tsv"
echo -e "video\tskip_rate\tbaseline_fps\tmats_fps\tspeedup\tssim\tpsnr" > "$SUMMARY_FILE"

for VIDEO_PATH in "${VIDEOS[@]}"; do
    VNAME=$(basename "$VIDEO_PATH" .mp4)
    AVATAR_ID="avator_${VNAME}"
    AVATAR_DIR="results/v15/avatars/${AVATAR_ID}"
    DEMO_OUT="$RESULT_DIR/${VNAME}_demo.mp4"
    LOG="$RESULT_DIR/${VNAME}.log"

    echo ""
    echo "=================================================="
    echo "  [${VNAME}]"
    echo "=================================================="

    # 1. 预处理（已有则跳过）
    if [ -d "$AVATAR_DIR" ] && [ -f "$AVATAR_DIR/latents.pt" ]; then
        echo "  ✓ 预处理已存在，跳过"
    else
        echo "  预处理中..."
        cat > /tmp/prep_${VNAME}.yaml << YAML
${AVATAR_ID}:
  preparation: true
  bbox_shift: 0
  video_path: ${VIDEO_PATH}
  audio_clips:
    audio_0: ${AUDIO}
YAML
        PYTHONPATH="$MUSE_ROOT" python scripts/realtime_inference.py \
            --version v15 \
            --unet_config     ./models/musetalkV15/musetalk.json \
            --unet_model_path ./models/musetalkV15/unet.pth \
            --inference_config /tmp/prep_${VNAME}.yaml
    fi

    # 2. demo（输出到固定路径，然后立刻重命名保存）
    PYTHONPATH="$MUSE_ROOT" python "$REPO/step2/demo_video.py" \
        --avatar_id  "$AVATAR_ID" \
        --audio      "$AUDIO" \
        --threshold  "$THRESHOLD" \
        --max_skip   "$MAX_SKIP" \
        --num_frames "$NUM_FRAMES" \
        --out        "$DEMO_OUT" 2>&1 | tee "$LOG"

    # demo_video.py 把独立文件写到 --out 所在目录（即 $RESULT_DIR）
    # 立刻重命名，防止下次运行覆盖
    cp "$RESULT_DIR/baseline.mp4" "$RESULT_DIR/${VNAME}_baseline.mp4"
    cp "$RESULT_DIR/mats.mp4"     "$RESULT_DIR/${VNAME}_mats.mp4"

    # 3. 从日志提取数字
    SKIP_RATE=$(grep -oP '跳过率：\K[\d.]+(?=%)' "$LOG" | tail -1)
    BASE_FPS=$(grep -oP '基线：\K[\d.]+' "$LOG" | tail -1)
    MATS_FPS=$(grep -oP 'MATS：\K[\d.]+' "$LOG" | tail -1)
    SPEEDUP=$(grep -oP '加速 \K[\d.]+(?=×)' "$LOG" | tail -1)
    SSIM=$(grep -oP 'SSIM=\K[\d.]+' "$LOG" | tail -1)
    PSNR=$(grep -oP 'PSNR=\K[\d.]+' "$LOG" | tail -1)

    echo "  → 基线=${BASE_FPS}FPS  MATS=${MATS_FPS}FPS(${SPEEDUP}x)  SSIM=${SSIM}  PSNR=${PSNR}dB  跳过率=${SKIP_RATE}%"
    echo -e "${VNAME}\t${SKIP_RATE}%\t${BASE_FPS}\t${MATS_FPS}\t${SPEEDUP}x\t${SSIM}\t${PSNR}" >> "$SUMMARY_FILE"
done

# ---- sun.mp4 LSE-C ----
echo ""
echo "=================================================="
echo "  LSE-C 评估：sun.mp4（max_skip=${MAX_SKIP}）"
echo "=================================================="
PYTHONPATH="$MUSE_ROOT" python "$REPO/step2/lse_eval.py" \
    --mode           eval_only \
    --baseline_video "$RESULT_DIR/sun_baseline.mp4" \
    --cached_video   "$RESULT_DIR/sun_mats.mp4" \
    --audio          "$AUDIO" \
    --threshold      "$THRESHOLD" \
    --output_dir     "$RESULT_DIR" \
    2>&1 | tee "$RESULT_DIR/sun_lse.log"

# ---- 汇总 ----
echo ""
echo "================================================================="
echo "  全部完成！SSIM/PSNR 汇总："
echo "================================================================="
column -t -s $'\t' "$SUMMARY_FILE"
echo ""
echo "  LSE-C 结果见: $RESULT_DIR/sun_lse.log"
echo "  完整日志:     $RESULT_DIR/"
