#!/bin/bash
# MuseTalk HDTF 数据集预处理脚本
#
# 使用方式：
#   cd $MUSE_ROOT
#   bash $REPO/step3/distill/prepare_data.sh [HDTF_RAW_DIR]
#
# 参数：
#   HDTF_RAW_DIR: HDTF 原始视频目录（默认 ./dataset/HDTF/source/）
#
# 预期输出结构：
#   dataset/HDTF/
#     source/           ← 原始视频（mp4）
#     video_root_25fps/ ← 转码为 25fps
#     video_audio_clip_root/ ← 按 30s 切片
#     meta/             ← 每个视频片段的 pkl 元数据
#     train.txt         ← 训练集列表
#     val.txt           ← 验证集列表

set -e

HDTF_RAW_DIR="${1:-./dataset/HDTF/source/}"
PREPROCESS_CFG="./configs/training/preprocess.yaml"

echo "============================================================"
echo "  MuseTalk HDTF 数据预处理"
echo "============================================================"
echo "原始视频目录: $HDTF_RAW_DIR"

# ---------- Step 0：检查 HDTF 原始数据是否存在 ----------
if [ ! -d "$HDTF_RAW_DIR" ] || [ -z "$(ls -A $HDTF_RAW_DIR 2>/dev/null)" ]; then
    echo ""
    echo "⚠️  HDTF 原始视频不存在或为空: $HDTF_RAW_DIR"
    echo ""
    echo "获取方式（选一）："
    echo ""
    echo "  方式1：从 HDTF 官方源下载（需申请访问）"
    echo "    https://github.com/MRzzm/HDTF"
    echo ""
    echo "  方式2：直接使用已有讲话视频（任意来源）"
    echo "    - 将视频 .mp4 放入 $HDTF_RAW_DIR"
    echo "    - 推荐：人脸清晰、正面、光照良好、单人讲话"
    echo "    - 最少 10 个视频（每个 >30s），建议 50+ 个"
    echo ""
    echo "  方式3：使用已有的测试视频（快速验证流程）"
    echo "    cp data/video/*.mp4 $HDTF_RAW_DIR"
    echo "    （数量少，训练效果有限，仅用于调试）"
    echo ""
    exit 1
fi

VIDEO_COUNT=$(ls "$HDTF_RAW_DIR"/*.mp4 2>/dev/null | wc -l)
echo "✓ 发现 $VIDEO_COUNT 个原始视频"

# ---------- Step 1：修改 preprocess.yaml 以指向 HDTF 目录 ----------
mkdir -p dataset/HDTF/source
# 如果用户指定了不同目录，创建软链接
if [ "$HDTF_RAW_DIR" != "./dataset/HDTF/source/" ]; then
    echo "创建软链接: $HDTF_RAW_DIR → ./dataset/HDTF/source/"
    ln -sfn "$(realpath $HDTF_RAW_DIR)"/* ./dataset/HDTF/source/ 2>/dev/null || true
fi

# ---------- Step 2：运行 MuseTalk 预处理 ----------
echo ""
echo "[Step 2] 运行 MuseTalk preprocess.py ..."
echo "  这将执行：25fps 转码 → 切片 → 人脸检测 → landmark 提取 → 元数据生成"
echo "  预计耗时：~1-2 分钟/视频（取决于 GPU 和视频时长）"
echo ""

PYTHONPATH="$MUSE_ROOT" python scripts/preprocess.py \
    --cfg "$PREPROCESS_CFG"

# ---------- Step 3：验证输出 ----------
echo ""
echo "[Step 3] 验证预处理输出..."
TRAIN_TXT="./dataset/HDTF/train.txt"
META_DIR="./dataset/HDTF/meta"

if [ ! -f "$TRAIN_TXT" ]; then
    echo "✗ train.txt 未生成，预处理可能失败"
    exit 1
fi

TRAIN_COUNT=$(tail -n +2 "$TRAIN_TXT" | wc -l)
META_COUNT=$(ls "$META_DIR"/*.pkl 2>/dev/null | wc -l)

echo "✓ train.txt: $TRAIN_COUNT 条记录"
echo "✓ meta/*.pkl: $META_COUNT 个文件"

echo ""
echo "============================================================"
echo "  数据预处理完成！"
echo "============================================================"
echo ""
echo "下一步：启动蒸馏训练"
echo ""
echo "  cd \$MUSE_ROOT && git pull origin main"
echo "  accelerate launch --num_processes 4 \\"
echo "      \$REPO/step3/distill/train_distill.py \\"
echo "      --config \$REPO/step3/distill/configs/distill.yaml \\"
echo "      --student_config \$REPO/step3/distill/configs/student_musetalk.json"
