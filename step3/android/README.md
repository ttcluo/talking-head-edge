# MuseTalk + MATS Android 端侧部署

## 部署架构

```
GPU 服务器（已完成）              Android 设备
─────────────────────           ─────────────────────────────
PyTorch FP16 模型               ONNX Runtime Android (ORT)
    ↓ export_onnx.py                ↓ INT8 量化
UNet.onnx (INT8 可选)           UNet_int8.onnx
VAE.onnx                        VAE_int8.onnx
Whisper.onnx                    Whisper.onnx
    ↓                               ↓
profile_results/                JNI 封装 → Java/Kotlin API
unet_ort_result.json            MuseTalkNative.kt
```

## 当前进度

- [x] ONNX 导出（step2/export_onnx.py）
- [x] ORT-GPU benchmark（yongen.mp4，1.53× UNet 加速）
- [ ] INT8 量化（UNet）
- [ ] ORT Android 集成
- [ ] Android APK 构建
- [ ] 真机测试（目标设备见下）

## 目标设备

| 设备 | SoC | NPU | 预期 FPS |
|------|-----|-----|---------|
| Snapdragon 8 Gen 2 手机 | SM8550 | Hexagon | ~5-8 FPS |
| Snapdragon 8 Gen 3 手机 | SM8650 | Hexagon | ~8-12 FPS |
| Dimensity 9300 手机 | MT6989 | APU | ~4-7 FPS |

目标：**≥ 5 FPS**（端侧首次可用，论文核心卖点）

---

## 部署步骤

### Step 1：在 GPU 服务器上导出 INT8 量化模型

```bash
cd $MUSE_ROOT
PYTHONPATH=$MUSE_ROOT python $REPO/step3/android/export_int8.py \
    --out_dir models/android_onnx/
```

产出：
- `models/android_onnx/unet_int8.onnx`（~180 MB，原始 FP16 ~360 MB）
- `models/android_onnx/vae_decoder_int8.onnx`（~80 MB）
- `models/android_onnx/whisper_encoder_int8.onnx`（~25 MB）

### Step 2：Android Studio 项目设置

```bash
# 克隆 Android 项目模板（已在 step3/android/app/ 下）
# 将模型文件放入 assets/
cp models/android_onnx/*.onnx step3/android/app/src/main/assets/
```

### Step 3：集成 ONNX Runtime Android

在 `build.gradle` 中：
```groovy
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'
}
```

### Step 4：实现 JNI 推理管道

```kotlin
// MuseTalkInference.kt
class MuseTalkInference(context: Context) {
    private val session: OrtSession
    init {
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            addNnapi()           // 启用 NNAPI（NPU 加速）
            setIntraOpNumThreads(4)
        }
        session = env.createSession(loadAsset("unet_int8.onnx"), opts)
    }
    fun infer(latent: FloatArray, audioFeat: FloatArray): FloatArray { ... }
}
```

### Step 5：MATS 集成

```kotlin
// MATSController.kt - 端侧 MATS 逻辑（与 Python 实现等价）
class MATSController(private val threshold: Float = 0.15f, private val maxSkip: Int = 2) {
    private var prevLatent: FloatArray? = null
    private var cachedFrame: Bitmap? = null
    private var consecSkip = 0

    fun shouldSkip(latent: FloatArray): Boolean {
        val prev = prevLatent ?: return false
        val motion = computeMotion(latent, prev)
        val maxSkipHit = consecSkip >= maxSkip
        return motion < threshold && !maxSkipHit
    }
    private fun computeMotion(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
        return 1f - dot / (Math.sqrt((na*nb).toDouble()).toFloat() + 1e-6f)
    }
}
```

---

## 性能预估

基于 GPU 服务器数据推算（A800 → Snapdragon 8 Gen 3）：

| 组件 | A800 (FP16) | SD 8 Gen 3 (INT8, NNAPI) | 估算方式 |
|------|------------|--------------------------|---------|
| UNet 单帧 | ~35ms | ~180-250ms | INT8 ~2×；移动 GPU ~5× 差距 |
| VAE decode | ~8ms | ~40-60ms | 类似比例 |
| Whisper | ~5ms | ~20-30ms | 类似比例 |
| **基线 FPS** | **~22 FPS** | **~3-4 FPS** | — |
| **MATS(55% skip)** | **~51 FPS** | **~6-8 FPS** | skip 帧 ~0ms |

> MATS 在移动端优势更显著：skip 帧节省 240ms+ 的完整推理，端侧加速比预期更高（~2.5-3×）

---

## 关键风险

1. **模型尺寸**：三个模型总计 ~285 MB，需要 INT8 量化
2. **NNAPI 兼容性**：部分算子可能回退到 CPU
3. **内存**：Snapdragon 推理时峰值内存 ~1.5-2 GB，需要测试是否 OOM
4. **精度**：INT8 量化后 SSIM 是否仍满足要求（估计 >0.99）
