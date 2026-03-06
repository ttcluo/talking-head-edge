#!/bin/bash
# MuseTalk 环境搭建脚本
# 使用方法：在 A800 服务器上执行 bash setup.sh
# 预计耗时：30-60 分钟

set -e  # 任意步骤出错立即停止

echo "=========================================="
echo " MuseTalk 环境搭建"
echo " 预计耗时：30-60 分钟"
echo "=========================================="

# ---------- 0. 前置检查 ----------
echo ""
echo "[检查] GPU 环境..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[检查] CUDA 版本..."
nvcc --version | grep "release"
echo "[检查] 当前目录：$(pwd)"

# ---------- 1. 创建 conda 环境 ----------
echo ""
echo "[步骤 1/7] 创建 conda 环境 musetalk (python 3.10)..."
conda create -n musetalk python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate musetalk
echo "✓ conda 环境创建完成"

# ---------- 2. 安装 PyTorch ----------
echo ""
echo "[步骤 2/7] 安装 PyTorch 2.0.1 (CUDA 11.8)..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
# 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}')"
echo "✓ PyTorch 安装完成"

# ---------- 3. 克隆 MuseTalk ----------
echo ""
echo "[步骤 3/7] 克隆 MuseTalk 仓库..."
cd ~
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
echo "✓ 仓库克隆完成，当前目录：$(pwd)"

# ---------- 4. 安装依赖 ----------
echo ""
echo "[步骤 4/7] 安装 Python 依赖..."
pip install -r requirements.txt
echo "✓ 基础依赖安装完成"

# ---------- 5. 安装 MMLab 套件 ----------
echo ""
echo "[步骤 5/7] 安装 MMLab 套件..."
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
echo "✓ MMLab 安装完成"

# ---------- 6. 安装 FFmpeg ----------
echo ""
echo "[步骤 6/7] 检查 FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg 已安装：$(ffmpeg -version 2>&1 | head -n 1)"
else
    echo "  FFmpeg 未找到，尝试 apt 安装..."
    sudo apt-get install -y ffmpeg
    echo "✓ FFmpeg 安装完成"
fi

# ---------- 7. 下载模型权重 ----------
echo ""
echo "[步骤 7/7] 下载模型权重（需要访问 HuggingFace）..."
echo "  如果网络受限，请手动下载，见 README 中的 Manual Download 部分"
echo "  开始自动下载..."
bash ./download_weights.sh

# ---------- 完成 ----------
echo ""
echo "=========================================="
echo " ✅ 环境搭建完成！"
echo ""
echo " 下一步：运行推理验证"
echo "   sh inference.sh v1.5 normal"
echo "   sh inference.sh v1.5 realtime"
echo "=========================================="
