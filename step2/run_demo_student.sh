#!/bin/bash
# MATS 4 路对比 Demo（基线 | MATS | Student | MATS+Student）
# REPO 从脚本位置推导，兼容本地(tad)与服务器(talking-head-edge)路径
#
# 用法（任选其一）：
#   从项目根：bash step2/run_demo_student.sh
#   从 MuseTalk 根：cd MuseTalk && bash ../step2/run_demo_student.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MUSE_ROOT="${MUSE_ROOT:-$REPO/MuseTalk}"

STUDENT_CKPT="${STUDENT_CKPT:-exp_out/distill/distill_lipsync/student_unet-2000.pth}"
STUDENT_CONFIG="${STUDENT_CONFIG:-$REPO/step3/distill/configs/student_musetalk.json}"
AVATAR_ID="${AVATAR_ID:-yongen}"
AUDIO="${AUDIO:-data/audio/yongen.wav}"
THRESHOLD="${THRESHOLD:-0.15}"
NUM_FRAMES="${NUM_FRAMES:-200}"
OUT="${OUT:-profile_results/mats_demo_student.mp4}"

echo "=============================================="
echo "  MATS 4 路对比 Demo"
echo "  REPO=$REPO"
echo "  MUSE_ROOT=$MUSE_ROOT"
echo "  student_ckpt=$STUDENT_CKPT"
echo "=============================================="

cd "$MUSE_ROOT"
export PYTHONPATH="$MUSE_ROOT"

python "$REPO/step2/demo_video_student.py" \
    --student_ckpt   "$STUDENT_CKPT" \
    --student_config "$STUDENT_CONFIG" \
    --avatar_id     "$AVATAR_ID" \
    --audio         "$AUDIO" \
    --threshold     "$THRESHOLD" \
    --num_frames    "$NUM_FRAMES" \
    --out           "$OUT" \
    "$@"
