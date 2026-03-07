# Talking Head Android Demo — 部署指南

## 目录结构

```
android/TalkingHeadDemo/
├── build.gradle
├── settings.gradle
└── app/
    ├── build.gradle
    └── src/main/
        ├── AndroidManifest.xml
        ├── assets/               ← 测试输入（由服务器生成）
        │   ├── latent_test.bin
        │   ├── audio_test.bin
        │   └── meta.json
        └── java/com/example/talkinghead/
            ├── MainActivity.kt    ← 入口 + UI
            ├── InferenceHelper.kt ← ONNX Runtime 推理封装
            └── AssetLoader.kt     ← 加载二进制测试输入
```

---

## Android 所需模型与资源清单

| 文件 | 用途 | 生成方式 | 放置位置 |
|------|------|----------|----------|
| `unet_student_fp32.onnx` | UNet 推理（CPU/视频预览，约 555MB） | 见下「一键生成」或 `export_student_onnx.py` | `assets/` 或 `/sdcard/Download/` |
| `unet_student_int8.onnx` | UNet 推理（NNAPI 可选，约 142MB） | 同上 | 同上 |
| `vae_decoder.onnx` | 将 latent 解码为 256×256 图像（约 198MB） | `export_vae_decoder_onnx.py` | `assets/` 或 `/sdcard/Download/` |
| `latent_test.bin` + `audio_test.bin` + `meta.json` | 单帧基准测试输入 | `prepare_android_assets.py` | `app/src/main/assets/` |
| `latents_seq.bin` + `audio_seq.bin` + `video_meta.json` | 视频预览多帧序列 | `prepare_android_video_assets.py` | `app/src/main/assets/` |

**一键生成（在 GPU 服务器执行）：**

```bash
# 进入 MuseTalk 项目根目录
cd /path/to/musetalk   # 替换为你的 MuseTalk 实际路径
export MUSE_ROOT=$(pwd)
# 可选：export REPO=/path/to/tad  若不设则从脚本位置推导
bash $REPO/step3/android/build_android_assets.sh
```
需已训练好 `student_unet_final.pth` 及 avatar/audio 数据。

脚本会：导出 UNet FP32/INT8 与 VAE Decoder ONNX → 生成单帧/多帧测试资源 → 写入 `$REPO/android/TalkingHeadDemo/app/src/main/assets/`，并将 ONNX 复制到同一目录。若未运行脚本，可按下面 Step 1～Step 3 分步执行。

---

## Step 1：服务器端 — 生成测试资产

```bash
cd $MUSE_ROOT

# 生成单帧测试输入（基准测试用）
PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/prepare_android_assets.py \
    --avatar_id      avator_1 \
    --audio_feat_dir dataset/distill/audio_feats \
    --out_dir        $REPO/android/TalkingHeadDemo/app/src/main/assets/
```

**视频预览**（端上看到实际生成人脸画面）需额外步骤：

```bash
# 1）生成多帧序列（如 80 帧）
PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/prepare_android_video_assets.py \
    --avatar_id avator_1 \
    --audio_feat_dir dataset/distill/audio_feats \
    --out_dir $REPO/android/TalkingHeadDemo/app/src/main/assets/ \
    --num_frames 80

# 2）导出 VAE Decoder ONNX（约 198MB）
PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/export_vae_decoder_onnx.py \
    --vae_dir models/sd-vae \
    --out_dir models/student_onnx
# 将 models/student_onnx/vae_decoder.onnx 复制到 app/src/main/assets/
```

App 内点击「**视频预览**」→ 进入预览页 → 点击「生成预览」：逐帧跑 UNet + VAE，界面实时显示生成的人脸（256×256）。保存的 MP4 **仅含画面，无音轨**（端上未合成音频；若需带声需在服务端导出对应音频再合成）。

---

## Step 2：服务器端 — 确认 ONNX 输入节点名

```bash
python $REPO/step3/android/inspect_onnx_inputs.py \
    --onnx_path models/student_onnx/unet_student_fp32.onnx
```

根据输出的 `name='...'` 更新 `InferenceHelper.kt` 里的 inputs map key。

---

## Step 3：服务器端 → Android 设备 — 推送模型

```bash
# 确认设备已连接
adb devices

# 推送 INT8 ONNX（141.7MB）
adb push models/student_onnx/unet_student_int8.onnx /sdcard/Download/

# 可选：推送 FP32 ONNX 作为对照（555MB）
adb push models/student_onnx/unet_student_fp32.onnx /sdcard/Download/
```

---

## Step 4：Android Studio — 编译 & 安装

```bash
cd android/TalkingHeadDemo

# 编译 debug APK
./gradlew assembleDebug

# 安装到设备
./gradlew installDebug

# 或者直接在 Android Studio 中 Run
```

---

## Step 5：测试

打开 App，点击：
- **CPU 推理** / **NNAPI 推理**：单次基准
- **完整基准测试**：CPU + NNAPI 各 30 次 + 精度对比（MSE/MaxDiff）
- **视频预览**：进入预览页，点击「生成预览」在端上逐帧跑 UNet+VAE，实时看到蒸馏后人脸生成效果（需先按上文生成多帧资源 + vae_decoder.onnx）

**从 logcat 复制结果**（所有结果会打 Log，便于粘贴）：
```bash
adb logcat -s TalkingHeadDemo
```
单次推理结果见 `-------- [CPU/NNAPI] 单次基准结果 --------`；完整基准测试结束时会输出 `======== 完整基准测试结果（可整段复制）========`，整段复制即可。

**视频预览性能**：生成结束后会打 `[性能] --------- 生成耗时汇总 ---------`，包含总耗时、首帧/稳态 UNet/VAE 毫秒数、等效 FPS。实时 25fps 需单帧总耗时 ≤40ms；若远大于此，瓶颈在 UNet/VAE 端上推理，需依赖 NNAPI/NPU 或进一步蒸馏/量化。

---

## NNAPI 说明

| 项目 | 内容 |
|------|------|
| 最低 API | 28（Android 9）|
| INT8 支持 | 需要 QDQ 格式（本脚本用 QOperator 格式，NNAPI 可能 fallback 到 CPU）|
| 备选方案 | 先用 FP32 + NNAPI 测 FP32 加速效果 |
| 量化格式转换 | 若 NNAPI 不接受 INT8，改用 `quantize_static + QDQ` 格式重导出 |

### 常见报错

**`Error code - ORT_NOT_IMPLEMENTED`（CPU 推理 INT8 模型时）**

- **原因**：INT8 动态量化生成的 **ConvInteger** 算子，ONNX Runtime 的 CPU 执行提供程序未实现。
- **处理**：
  1. **用 NNAPI 推理**：同一 INT8 模型在 NNAPI 下可能由 NPU/DSP 执行，不会报错。
  2. **用 FP32 模型做 CPU 测试**：在服务器导出 `unet_student_fp32.onnx`，推送到手机后 App 会优先用 FP32 做 CPU 推理：
     ```bash
     adb push models/student_onnx/unet_student_fp32.onnx /sdcard/Download/
     ```

**`NNAPI 不可用，回退到 CPU`**

→ 设备不支持或模型含不支持的 op，此时 ORT 可能回退 CPU；若模型为 INT8，回退后仍会报 ORT_NOT_IMPLEMENTED，需改用 FP32 模型。

---

## 预期结果（Snapdragon 8 Gen 系列）

| Provider | 预期延迟 |
|----------|---------|
| CPU (4线程) | 200–400ms/frame |
| NNAPI (NPU) | 30–80ms/frame (目标) |

> 实际延迟以真机测量为准，Student UNet 138M 参数。

---

## 端上为何难以实时：与 TaoAvatar 的对比

实测端上 UNet≈500ms/帧、VAE≈10.7s/帧，80 帧需约 15 分钟，无法达到实时 25fps。下面对比 **TaoAvatar（MNN-TaoAvatar）** 能端上实时的原因，说明本方案在架构上的差异。

| 维度 | 本方案（MuseTalk 蒸馏 + 端上 UNet+VAE） | TaoAvatar（阿里 MNN-TaoAvatar） |
|------|----------------------------------------|---------------------------------|
| **表示形式** | 每帧 **2D 图像生成**：latent → **完整 VAE 解码** → 256×256 RGB 像素 | **参数驱动 + 3D 渲染**：音频 → **A2BS 系数（blend shape）** → 3DGS 点云变形 → **光栅化**出图 |
| **端上算的是什么** | 每帧跑 **大 CNN**：Student UNet（138M）+ **VAE Decoder（~198MB 全量解码）** | 每帧只跑 **Audio2BS**（368MB，RTF≈0.34，即远快于实时）→ 输出低维系数；**渲染**是 3D 高斯光栅化（非大网络），可 60 FPS |
| **瓶颈** | VAE 每帧做「潜空间→像素」的 **整图解码**，计算/带宽都大；CPU 上单 VAE 就 ~10s/帧 | 不做每帧像素级生成；重算在「音频→系数」，渲染是传统图形学，易优化 |
| **蒸馏/压缩** | 只对 UNet 做了蒸馏；**VAE 未替换**，仍是 SD 标准 VAE | StyleUnet 等重网络 **蒸馏进轻量 MLP** + blend shape，端上只跑轻量 MLP |
| **推理引擎** | ONNX Runtime（CPU/NNAPI） | **MNN**（为移动端深度优化，CPU/GPU/NPU） |

**结论**：

- **本质差异**：TaoAvatar 是「**参数驱动 + 3D 渲染**」（音频→系数→已有 3D 模型变形→光栅化），我们是「**每帧 2D 图像合成**」（音频→latent→VAE→图像）。前者端上主要成本在「小网络 + 光栅化」，后者在「大 CNN 解码」。
- **本方案端上慢的主因**：不是 UNet 太慢（~500ms 尚可优化），而是 **每帧都跑完整 VAE Decoder**，在移动 CPU 上约 10s/帧，架构上就难以达到 25fps。
- **若仍要端上实时**：需改架构——例如改为「轻量音频→系数（类 A2BS）+ 轻量 2D/3D 渲染」或「端上只跑 UNet，VAE 解码放云端/边缘」；若坚持端上全链路 2D 生成，则需把 VAE 极度轻量化或换成非扩散的轻量解码器，工程量与效果风险都较大。

当前 Android Demo 保留为 **离线效果验证与基准测试**，不追求端上实时；实时场景建议走云端/边缘解码或参考 TaoAvatar 类参数驱动方案。
