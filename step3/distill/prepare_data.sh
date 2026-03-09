#!/bin/bash
# MuseTalk 蒸馏数据集准备脚本（轻量版）
#
# 不依赖 mmpose/decord，直接复用 MuseTalk realtime_inference.py 的预处理输出
# 将多个 avatar 的预处理结果打包为蒸馏训练可用的数据集
#
# 使用方式：
#   cd $MUSE_ROOT
#   bash $REPO/step3/distill/prepare_data.sh
#
# 依赖：
#   1. 已运行过 scripts/realtime_inference.py（有 results/v15/avatars/*/）
#   2. 或手动指定视频目录

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MUSE_ROOT="${MUSE_ROOT:-$(pwd)}"
DISTILL_DATA_DIR="./dataset/distill"

echo "============================================================"
echo "  蒸馏数据集准备（轻量版，复用 avatar 预处理结果）"
echo "============================================================"

# ---------- Step 1：预处理所有视频为 avatar ----------
AVATAR_BASE="results/v15/avatars"
VIDEO_DIR="data/video"
AUDIO_DIR="data/audio"

mkdir -p "$DISTILL_DATA_DIR"

echo "[Step 1] 检查已有 avatar 预处理结果..."
EXISTING=$(ls -d "$AVATAR_BASE"/avator_* 2>/dev/null | wc -l)
echo "  已有 avatar 数量: $EXISTING"

# 对每个测试视频做预处理（如果还没有的话）
for VID in "$VIDEO_DIR"/*.mp4; do
    VNAME=$(basename "$VID" .mp4)
    AVATAR_ID="avator_${VNAME}"
    AVATAR_DIR="$AVATAR_BASE/$AVATAR_ID"

    if [ -d "$AVATAR_DIR" ]; then
        echo "  ✓ $AVATAR_ID 已存在，跳过"
        continue
    fi

    # 找对应音频，没有就用 yongen.wav 作为占位
    AUDIO="$AUDIO_DIR/${VNAME}.wav"
    if [ ! -f "$AUDIO" ]; then
        AUDIO="$AUDIO_DIR/yongen.wav"
    fi

    echo "  预处理: $VNAME → $AVATAR_ID"
    cat > /tmp/prep_${VNAME}.yaml << EOF
${AVATAR_ID}:
  preparation: true
  bbox_shift: 0
  video_path: ${VID}
  audio_clips:
    audio_0: ${AUDIO}
EOF

    PYTHONPATH="$MUSE_ROOT" python scripts/realtime_inference.py \
        --version v15 \
        --unet_config  ./models/musetalkV15/musetalk.json \
        --unet_model_path ./models/musetalkV15/unet.pth \
        --inference_config /tmp/prep_${VNAME}.yaml 2>&1 | tail -5
    echo "  ✓ $AVATAR_ID 预处理完成"
done

# ---------- Step 2：生成蒸馏数据集索引 ----------
echo ""
echo "[Step 2] 生成数据集索引..."

INDEX_FILE="$DISTILL_DATA_DIR/avatar_list.txt"
> "$INDEX_FILE"

COUNT=0
for AVATAR_DIR in "$AVATAR_BASE"/avator_*; do
    if [ -d "$AVATAR_DIR/latents" ]; then
        AVATAR_ID=$(basename "$AVATAR_DIR")
        LATENT_COUNT=$(ls "$AVATAR_DIR/latents/"*.pt 2>/dev/null | wc -l)
        if [ "$LATENT_COUNT" -eq 0 ]; then
            # 检查是否有 unet_input_latent_list.pt
            if [ -f "$AVATAR_DIR/latents/unet_input_latent_list.pt" ]; then
                echo "$AVATAR_ID" >> "$INDEX_FILE"
                COUNT=$((COUNT + 1))
            fi
        fi
    fi
done

# 如果上面没找到，直接列出有 latents 目录的
if [ "$COUNT" -eq 0 ]; then
    for AVATAR_DIR in "$AVATAR_BASE"/avator_*; do
        if [ -d "$AVATAR_DIR" ]; then
            AVATAR_ID=$(basename "$AVATAR_DIR")
            echo "$AVATAR_ID" >> "$INDEX_FILE"
            COUNT=$((COUNT + 1))
        fi
    done
fi

echo "  ✓ 数据集 avatar 数量: $COUNT"
echo "  ✓ 索引文件: $INDEX_FILE"

# ---------- Step 3：验证并生成 train/val 分割 ----------
echo ""
echo "[Step 3] 生成 train/val 分割..."

TOTAL=$COUNT
VAL_COUNT=1
TRAIN_COUNT=$((TOTAL - VAL_COUNT))

head -n "$TRAIN_COUNT" "$INDEX_FILE" > "$DISTILL_DATA_DIR/train_avatars.txt"
tail -n "$VAL_COUNT"   "$INDEX_FILE" > "$DISTILL_DATA_DIR/val_avatars.txt"

echo "  ✓ train: $TRAIN_COUNT  val: $VAL_COUNT"

echo ""
echo "============================================================"
echo "  数据准备完成"
echo "============================================================"
echo "  索引: $DISTILL_DATA_DIR/train_avatars.txt"
echo "  Avatar 目录: $AVATAR_BASE/"
echo ""
echo "下一步："
echo "  cd \$MUSE_ROOT"
echo "  PYTHONPATH=\$MUSE_ROOT accelerate launch --num_processes 4 \\"
echo "      \$REPO/step3/distill/train_distill.py \\"
echo "      --config \$REPO/step3/distill/configs/distill.yaml \\"
echo "      --avatar_list \$MUSE_ROOT/dataset/distill/train_avatars.txt"
