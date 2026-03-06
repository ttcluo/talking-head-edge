#!/bin/bash
# EchoMimic V1 环境搭建与权重下载
# 运行前提：conda activate musetalk（复用已有环境）
# 运行位置：任意目录均可，脚本会在 ~ 下克隆仓库
# 使用方法：bash step2/setup_echomimic.sh

set -e

ECHO_ROOT=~/EchoMimic
PRETRAINED_DIR=$ECHO_ROOT/pretrained_weights

echo "=== EchoMimic V1 环境搭建 ==="

# ==================== 克隆仓库 ====================
if [ ! -d "$ECHO_ROOT" ]; then
    echo "[1/4] 克隆 EchoMimic..."
    git clone https://github.com/antgroup/echomimic "$ECHO_ROOT"
else
    echo "[1/4] 仓库已存在，跳过克隆"
fi

# ==================== 安装依赖 ====================
echo "[2/4] 安装依赖（在 musetalk 环境中）..."
# 检查是否在正确的环境中
if [[ "$CONDA_DEFAULT_ENV" != "musetalk" ]]; then
    echo "  ⚠ 请先执行 conda activate musetalk"
    exit 1
fi

pip install -q diffusers==0.24.0
pip install -q einops==0.4.1 omegaconf==2.3.0
pip install -q facenet_pytorch==2.5.0
pip install -q moviepy==1.0.3
# av 需要 FFmpeg 系统库才能从源码编译；pip --prefer-binary 直接取预编译 wheel
pip install av --prefer-binary -q

echo "  ✓ 依赖安装完成"

# ==================== 检查权重 ====================
echo "[3/4] 检查权重目录..."
mkdir -p "$PRETRAINED_DIR"

# 权重清单
declare -A WEIGHTS=(
    ["denoising_unet.pth"]="EchoMimic 去噪 UNet（3D时序）"
    ["reference_unet.pth"]="ReferenceNet（外观 UNet）"
    ["motion_module.pth"]="时序注意力模块"
    ["audio_projection.pt"]="音频投影层"
    ["face_locator.pth"]="人脸位置控制网络"
)

MISSING=0
for f in "${!WEIGHTS[@]}"; do
    if [ -f "$PRETRAINED_DIR/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ 缺失: $f  (${WEIGHTS[$f]})"
        MISSING=$((MISSING + 1))
    fi
done

# 检查目录型权重
for d in "sd-vae-ft-mse" "whisper"; do
    if [ -d "$PRETRAINED_DIR/$d" ]; then
        echo "  ✓ $d/"
    else
        echo "  ✗ 缺失目录: $d/"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "  ⚠ 缺失 $MISSING 个权重文件，请下载后放到：$PRETRAINED_DIR/"
    echo ""
    echo "  HuggingFace: https://huggingface.co/BadToBest/EchoMimic"
    echo "  modelscope:  https://modelscope.cn/models/BadToBest/EchoMimic"
    echo ""
    echo "  下载命令（若 HuggingFace 可访问）："
    echo "    pip install huggingface_hub"
    echo "    huggingface-cli download BadToBest/EchoMimic --local-dir $PRETRAINED_DIR"
    echo ""
    echo "  注：sd-vae-ft-mse 可直接用 MuseTalk 的 ~/MuseTalk/models/sd-vae"
    echo "      符号链接方式：ln -s ~/MuseTalk/models/sd-vae $PRETRAINED_DIR/sd-vae-ft-mse"
    echo "      Whisper 同理：ln -s ~/MuseTalk/models/whisper $PRETRAINED_DIR/whisper"
fi

# ==================== 快速验证 ====================
echo ""
echo "[4/4] 快速验证..."
cd "$ECHO_ROOT"
python -c "
import sys
sys.path.insert(0, '.')
try:
    from src.models.unet_3d_echo import EchoUNet3DConditionModel
    print('  ✓ EchoUNet3DConditionModel 可导入')
except Exception as e:
    print(f'  ✗ 导入失败: {e}')

try:
    from src.models.unet_2d_condition import UNet2DConditionModel
    print('  ✓ UNet2DConditionModel 可导入')
except Exception as e:
    print(f'  ✗ 导入失败: {e}')

try:
    from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
    print('  ✓ Audio2VideoPipeline 可导入')
except Exception as e:
    print(f'  ✗ 导入失败: {e}')
"

echo ""
echo "=== 搭建完成 ==="
echo "下一步：确认所有权重就绪后，运行 Profiling："
echo "  cd $ECHO_ROOT"
echo "  python /data/luochuan/talking-head-edge/talking-head-edge/step2/profile_echomimic.py \\"
echo "    --pretrained_dir $PRETRAINED_DIR"
