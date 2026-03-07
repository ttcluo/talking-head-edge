#!/usr/bin/env bash
# 在 $MUSE_ROOT 下一键生成 Android 所需模型与资源，并写入 TalkingHeadDemo/app/src/main/assets/。
# 依赖：已训练 student_unet_final.pth、avatar latents、distill audio_feats、models/sd-vae。
#
# 用法：
#   export MUSE_ROOT=/path/to/musetalk
#   export REPO=/path/to/tad   # 可选，不设则用本脚本所在仓库根推导
#   bash $REPO/step3/android/build_android_assets.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MUSE_ROOT="${MUSE_ROOT:-$(pwd)}"
ASSETS_DIR="$REPO/android/TalkingHeadDemo/app/src/main/assets"
STUDENT_CKPT="${STUDENT_CKPT:-$MUSE_ROOT/exp_out/distill/distill_v1/student_unet_final.pth}"
STUDENT_CONFIG="${STUDENT_CONFIG:-$REPO/step3/distill/configs/student_musetalk.json}"
AVATAR_ID="${AVATAR_ID:-avator_1}"
AUDIO_FEAT_DIR="${AUDIO_FEAT_DIR:-$MUSE_ROOT/dataset/distill/audio_feats}"
NUM_FRAMES="${NUM_FRAMES:-80}"

echo "=============================================="
echo "  Android 模型与资产生成"
echo "  MUSE_ROOT=$MUSE_ROOT"
echo "  REPO=$REPO"
echo "  ASSETS=$ASSETS_DIR"
echo "=============================================="

mkdir -p "$ASSETS_DIR"
cd "$MUSE_ROOT"
export PYTHONPATH="$MUSE_ROOT"

echo "[1/4] 导出 Student UNet ONNX (FP32 + INT8)..."
python "$REPO/step3/android/export_student_onnx.py" \
    --student_ckpt   "$STUDENT_CKPT" \
    --student_config "$STUDENT_CONFIG" \
    --out_dir        "$MUSE_ROOT/models/student_onnx"

echo "[2/4] 导出 VAE Decoder ONNX..."
python "$REPO/step3/android/export_vae_decoder_onnx.py" \
    --vae_dir "$MUSE_ROOT/models/sd-vae" \
    --out_dir "$MUSE_ROOT/models/student_onnx"

echo "[3/4] 生成单帧测试资源 (latent_test.bin, audio_test.bin, meta.json)..."
python "$REPO/step3/android/prepare_android_assets.py" \
    --avatar_id      "$AVATAR_ID" \
    --audio_feat_dir "$AUDIO_FEAT_DIR" \
    --out_dir        "$ASSETS_DIR"

echo "[4/4] 生成视频预览多帧资源 (latents_seq.bin, audio_seq.bin, video_meta.json)..."
python "$REPO/step3/android/prepare_android_video_assets.py" \
    --avatar_id      "$AVATAR_ID" \
    --audio_feat_dir "$AUDIO_FEAT_DIR" \
    --out_dir        "$ASSETS_DIR" \
    --num_frames     "$NUM_FRAMES"

echo "复制 ONNX 到 assets..."
cp -f "$MUSE_ROOT/models/student_onnx/unet_student_fp32.onnx" "$ASSETS_DIR/"
cp -f "$MUSE_ROOT/models/student_onnx/unet_student_int8.onnx" "$ASSETS_DIR/"
cp -f "$MUSE_ROOT/models/student_onnx/vae_decoder.onnx"       "$ASSETS_DIR/"

echo "=============================================="
echo "  完成。以下文件已就绪："
echo "  $ASSETS_DIR/"
ls -la "$ASSETS_DIR"
echo "=============================================="
