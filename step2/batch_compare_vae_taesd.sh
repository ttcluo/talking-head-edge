#!/bin/bash
# 批量 TAESD vs SD VAE 质量对比：avator_1 到 avator_10
#
# 用法（在 MuseTalk 目录下）：
#   cd $MUSE_ROOT
#   PYTHONPATH=$PWD bash ../step2/batch_compare_vae_taesd.sh 2>&1 | tee profile_results/vae_taesd_batch.log

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MUSE_ROOT="${MUSE_ROOT:-$(pwd)}"
RESULT_DIR="${RESULT_DIR:-profile_results/vae_taesd_batch}"
TAESD_DIR="${TAESD_DIR:-models/taesd_cache}"
mkdir -p "$RESULT_DIR"

echo "================================================================="
echo "  批量 TAESD vs SD VAE 对比（avator_1 ~ avator_10）"
echo "================================================================="

SUMMARY="$RESULT_DIR/summary.tsv"
echo -e "avatar_id\tsd_ms\ttaesd_ms\tspeedup\tssim\tpsnr\tstatus" > "$SUMMARY"

for i in $(seq 1 10); do
    AID="avator_$i"
    AVATAR_DIR="$MUSE_ROOT/results/v15/avatars/$AID"
    AUDIO="$MUSE_ROOT/data/audio/${AID}.wav"
    OUT_SUBDIR="$RESULT_DIR/$AID"

    if [ ! -d "$AVATAR_DIR" ] || [ ! -f "$AVATAR_DIR/latents.pt" ]; then
        echo ""
        echo "[$AID] 跳过：预处理数据不存在"
        echo -e "$AID\t-\t-\t-\t-\t-\tno_data" >> "$SUMMARY"
        continue
    fi

    if [ ! -f "$AUDIO" ]; then
        AUDIO="$MUSE_ROOT/data/audio/avator_1.wav"
        [ -f "$AUDIO" ] || AUDIO=""
    fi
    if [ -z "$AUDIO" ] || [ ! -f "$AUDIO" ]; then
        echo "[$AID] 跳过：无音频文件"
        echo -e "$AID\t-\t-\t-\t-\t-\tno_audio" >> "$SUMMARY"
        continue
    fi

    echo ""
    echo "=================================================="
    echo "  [$AID]"
    echo "=================================================="

    PYTHONPATH="$MUSE_ROOT" python "$SCRIPT_DIR/compare_vae_taesd.py" \
        --avatar_id "$AID" \
        --audio "$AUDIO" \
        --num_frames 50 \
        --taesd_dir "$TAESD_DIR" \
        --out_dir "$OUT_SUBDIR" 2>&1 | tee "$RESULT_DIR/${AID}.log" || true

    # 从 compare_result.json 提取
    JSON="$OUT_SUBDIR/compare_result.json"
    if [ -f "$JSON" ]; then
        SD_MS=$(python3 -c "import json; d=json.load(open('$JSON')); print(d.get('vae_sd_ms','-'))" 2>/dev/null || echo "-")
        TAESD_MS=$(python3 -c "import json; d=json.load(open('$JSON')); print(d.get('vae_taesd_ms','-'))" 2>/dev/null || echo "-")
        SP=$(python3 -c "import json; d=json.load(open('$JSON')); print(d.get('speedup','-'))" 2>/dev/null || echo "-")
        SSIM=$(python3 -c "import json; d=json.load(open('$JSON')); print(d.get('ssim','-'))" 2>/dev/null || echo "-")
        PSNR=$(python3 -c "import json; d=json.load(open('$JSON')); print(d.get('psnr_db','-'))" 2>/dev/null || echo "-")
        echo -e "$AID\t$SD_MS\t$TAESD_MS\t$SP\t$SSIM\t$PSNR\tok" >> "$SUMMARY"
    else
        echo -e "$AID\t-\t-\t-\t-\t-\tfail" >> "$SUMMARY"
    fi
done

echo ""
echo "================================================================="
echo "  汇总"
echo "================================================================="
cat "$SUMMARY"
echo ""
echo "  详情: $RESULT_DIR/"
