# 端侧虚拟人推理加速研究

> 基于扩散模型 Talking Head 的端侧推理加速研究项目

## 研究目标

在保持接近 EchoMimic 质量的前提下，实现扩散模型 Talking Head 在端侧设备（手机 GPU/NPU）上的实时推理。

| 指标 | 当前 MuseTalk 基线 | 当前 EchoMimic | 本研究目标 |
|------|-----------------|--------------|----------|
| 推理速度 | 30 FPS（V100）| <5 FPS（A100）| >15 FPS（手机 GPU）|
| 显存占用 | ~8GB | ~16GB | <4GB |
| 模型大小 | ~500MB | ~2GB | <200MB |

## 项目结构

```
├── 研究计划.md          # 完整 12 周研究计划
└── step1/               # 第一步行动脚本
    ├── setup.sh         # A800 服务器环境搭建
    ├── verify_inference.py  # 推理验证
    ├── profile_musetalk.py  # 性能 Profiling（输出至 profile_results/）
    └── papers.md        # 论文阅读清单
```

## 快速开始（A800 服务器）

```bash
# 1. 克隆本仓库
git clone https://github.com/ttcluo/talking-head-edge.git
cd talking-head-edge

# 2. 环境搭建
bash step1/setup.sh

# 3. 克隆 MuseTalk
cd ~ && git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk && bash ./download_weights.sh

# 4. 推理验证
python ~/talking-head-edge/step1/verify_inference.py

# 5. 性能 Profiling
python ~/talking-head-edge/step1/profile_musetalk.py \
    --video data/video/sun.mp4 \
    --audio data/audio/sun.wav
```

## 关键基线

- **MuseTalk 1.5**：https://github.com/TMElyralab/MuseTalk
- **EchoMimic**：https://github.com/antgroup/echomimic
- **ViDiT-Q**（量化工具）：https://github.com/thu-nics/ViDiT-Q

## 进度

- [x] 研究方向确定
- [x] 第一步行动脚本编写
- [x] 环境搭建（A800）
- [x] MuseTalk 1.5 推理跑通
- [x] 性能 Profiling 完成（FP32 18.4 FPS / FP16 21.5 FPS，UNet 瓶颈 53%）
- [ ] EchoMimic 复现
- [ ] 核心方法实现
