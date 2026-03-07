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

## Step 1：服务器端 — 生成测试资产

```bash
cd $MUSE_ROOT

# 生成测试输入（latent + audio feat）
PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/prepare_android_assets.py \
    --avatar_id      avator_1 \
    --audio_feat_dir dataset/distill/audio_feats \
    --out_dir        $REPO/android/TalkingHeadDemo/app/src/main/assets/
```

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
- **CPU 推理**：仅用 CPU 跑 INT8 ONNX，warmup 3 + bench 20
- **NNAPI 推理**：用 Snapdragon NPU/DSP 加速，同等测试
- **完整基准测试**：CPU + NNAPI 各 30 次 + 精度对比（MSE/MaxDiff）

---

## NNAPI 说明

| 项目 | 内容 |
|------|------|
| 最低 API | 28（Android 9）|
| INT8 支持 | 需要 QDQ 格式（本脚本用 QOperator 格式，NNAPI 可能 fallback 到 CPU）|
| 备选方案 | 先用 FP32 + NNAPI 测 FP32 加速效果 |
| 量化格式转换 | 若 NNAPI 不接受 INT8，改用 `quantize_static + QDQ` 格式重导出 |

### NNAPI 常见报错

```
NNAPI 不可用，回退到 CPU
```
→ 设备不支持或模型含不支持的 op，此时 ORT 自动回退 CPU，结果不变。

---

## 预期结果（Snapdragon 8 Gen 系列）

| Provider | 预期延迟 |
|----------|---------|
| CPU (4线程) | 200–400ms/frame |
| NNAPI (NPU) | 30–80ms/frame (目标) |

> 实际延迟以真机测量为准，Student UNet 138M 参数。
