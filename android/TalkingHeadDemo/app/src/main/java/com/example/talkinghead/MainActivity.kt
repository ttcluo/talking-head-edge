package com.example.talkinghead

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.talkinghead.databinding.ActivityMainBinding
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val scope = CoroutineScope(Dispatchers.Main)

    /** INT8 模型路径（含 ConvInteger，仅 NNAPI 支持；CPU 会报 ORT_NOT_IMPLEMENTED）*/
    private fun getUnetInt8Path(): String {
        val external = File(Environment.getExternalStorageDirectory(), "Download/unet_student_int8.onnx")
        if (external.canRead()) return external.absolutePath
        val inApp = File(filesDir, "unet_student_int8.onnx")
        if (inApp.exists()) return inApp.absolutePath
        try {
            assets.open("unet_student_int8.onnx").use { ins ->
                FileOutputStream(inApp).use { ins.copyTo(it) }
            }
            Log.i(TAG, "模型已从 assets 复制到 ${inApp.absolutePath}")
            return inApp.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "无法从 assets 加载模型", e)
            return external.absolutePath
        }
    }

    /** FP32 模型路径（CPU 与 NNAPI 均支持，无 ConvInteger）*/
    private fun getUnetFp32Path(): String? {
        val external = File(Environment.getExternalStorageDirectory(), "Download/unet_student_fp32.onnx")
        if (external.canRead()) return external.absolutePath
        val inApp = File(filesDir, "unet_student_fp32.onnx")
        if (inApp.exists()) return inApp.absolutePath
        try {
            assets.open("unet_student_fp32.onnx").use { ins ->
                FileOutputStream(inApp).use { ins.copyTo(it) }
            }
            return inApp.absolutePath
        } catch (_: Exception) {
            return null
        }
    }

    companion object {
        private const val TAG = "TalkingHeadDemo"
        private const val PERM_REQ = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        requestStoragePermission()

        binding.btnCpu.setOnClickListener   { runBench(useNnapi = false) }
        binding.btnNnapi.setOnClickListener { runBench(useNnapi = true)  }
        binding.btnBenchmark.setOnClickListener { runFullBenchmark() }
        binding.btnVideoPreview.setOnClickListener { startActivity(Intent(this, VideoPreviewActivity::class.java)) }
    }

    private fun isConvIntegerError(e: Throwable): Boolean {
        val msg = (e.message ?: "") + (e.cause?.message ?: "")
        return msg.contains("NOT_IMPLEMENTED", ignoreCase = true) || msg.contains("ConvInteger", ignoreCase = true)
    }

    private fun runBench(useNnapi: Boolean) = scope.launch(Dispatchers.IO) {
        val provider = if (useNnapi) "NNAPI" else "CPU"
        updateUi("[$provider] 加载输入...")

        val inputs = tryLoadInputs() ?: return@launch
        Log.i(TAG, "[$provider] 输入已加载 latent=${inputs.latent.size} audioFeat=${inputs.audioFeat.size}")
        var modelPath = if (useNnapi) getUnetInt8Path() else (getUnetFp32Path() ?: getUnetInt8Path())

        suspend fun runWithPath(path: String) {
            val modelType = if (path.contains("fp32")) "FP32" else "INT8"
            Log.i(TAG, "[$provider] 开始推理 模型=$modelType path=${path.takeLast(50)}")
            val helper = InferenceHelper(this@MainActivity, path, useNnapi)
            try {
                helper.load()
                updateUi("[$provider] 推理中（warmup 3次 + bench 20次）...")
                val result = helper.benchmark(inputs.latent, inputs.audioFeat)
                val resultText = result.toString()
                updateUi(resultText)
                Log.i(TAG, "-------- [$provider] 单次基准结果 --------\n$resultText\n----------------------------------------")
            } finally {
                helper.close()
            }
        }

        try {
            runWithPath(modelPath)
        } catch (e: Exception) {
            if (isConvIntegerError(e)) {
                val fp32 = getUnetFp32Path()
                if (fp32 != null && !modelPath.contains("fp32")) {
                    Log.w(TAG, "INT8 ConvInteger 不可用，改用 FP32 模型重试")
                    updateUi("[$provider] INT8 不支持，改用 FP32 模型...")
                    modelPath = fp32
                    try {
                        runWithPath(modelPath)
                        return@launch
                    } catch (e2: Exception) {
                        Log.e(TAG, "FP32 推理失败", e2)
                        updateUi("[$provider] FP32 也失败: ${e2.message}")
                        return@launch
                    }
                }
            }
            Log.e(TAG, "推理失败", e)
            val msg = e.message ?: ""
            val hint = when {
                isConvIntegerError(e) ->
                    "INT8 含 ConvInteger，本机 CPU/NNAPI 均不支持。\n\n请放入 FP32 模型后重试：\nadb push unet_student_fp32.onnx /sdcard/Download/"
                else -> "将 unet_student_int8.onnx 或 unet_student_fp32.onnx 放入 assets/ 或 /sdcard/Download/"
            }
            updateUi("[$provider] 错误: $msg\n\n$hint")
        }
    }

    private fun runFullBenchmark() = scope.launch(Dispatchers.IO) {
        val inputs = tryLoadInputs() ?: return@launch
        Log.i(TAG, "完整基准测试 输入已加载 latent=${inputs.latent.size} audioFeat=${inputs.audioFeat.size}")
        val sb = StringBuilder()

        // -------- FP32 + CPU --------
        sb.appendLine("=".repeat(40))
        sb.appendLine("  Student UNet 端侧基准测试")
        sb.appendLine("  设备: ${Build.MODEL}  API ${Build.VERSION.SDK_INT}")
        sb.appendLine("  模型: INT8 ONNX (141.7MB)")
        sb.appendLine("=".repeat(40))
        updateUi(sb.toString())
        Log.i(TAG, "======== 完整基准测试开始 设备=${Build.MODEL} API=${Build.VERSION.SDK_INT} ========")

        for ((label, useNnapi) in listOf("CPU" to false, "NNAPI" to true)) {
            var modelPath = if (useNnapi) getUnetInt8Path() else (getUnetFp32Path() ?: getUnetInt8Path())
            sb.appendLine("\n[$label] 模型: ${if (modelPath.contains("fp32")) "FP32" else "INT8"}")
            updateUi(sb.toString())
            Log.i(TAG, "[$label] 模型: ${if (modelPath.contains("fp32")) "FP32" else "INT8"}")
            for (attempt in 0..1) {
                val helper = InferenceHelper(this@MainActivity, modelPath, useNnapi)
                try {
                    helper.load()
                    val res = helper.benchmark(inputs.latent, inputs.audioFeat, warmupRuns = 5, benchRuns = 30)
                    sb.append(res.toString())
                    updateUi(sb.toString())
                    Log.i(TAG, "[$label] 结果:\n${res.toString()}")
                    break
                } catch (e: Exception) {
                    if (attempt == 0 && isConvIntegerError(e)) {
                        val fp32 = getUnetFp32Path()
                        if (fp32 != null) {
                            sb.appendLine("INT8 ConvInteger 不可用，改用 FP32 重试...")
                            sb.appendLine("[$label] 实际运行模型: FP32")
                            updateUi(sb.toString())
                            modelPath = fp32
                            continue
                        }
                    }
                    sb.appendLine("错误: ${e.message}")
                    if (isConvIntegerError(e))
                        sb.appendLine("(请 adb push unet_student_fp32.onnx /sdcard/Download/)")
                    updateUi(sb.toString())
                    break
                } finally {
                    helper.close()
                }
            }
        }

        // -------- 精度验证（CPU vs NNAPI 输出差异，需同一模型）--------
        sb.appendLine("\n[精度对比 CPU vs NNAPI]")
        updateUi(sb.toString())
        try {
            val pathForCompare = getUnetFp32Path() ?: getUnetInt8Path()
            val cpuHelper   = InferenceHelper(this@MainActivity, pathForCompare, false).also { it.load() }
            val nnapiHelper = InferenceHelper(this@MainActivity, pathForCompare, true).also { it.load() }

            val cpuOut   = cpuHelper.infer(inputs.latent, inputs.audioFeat)
            val nnapiOut = nnapiHelper.infer(inputs.latent, inputs.audioFeat)

            val mse = cpuOut.indices
                .map { i -> (cpuOut[i] - nnapiOut[i]).let { it * it } }
                .average()
            val maxDiff = cpuOut.indices
                .maxOf { i -> Math.abs(cpuOut[i] - nnapiOut[i]) }

            sb.appendLine("MSE(CPU, NNAPI):     ${"%.6f".format(mse)}")
            sb.appendLine("MaxDiff(CPU, NNAPI): ${"%.4f".format(maxDiff)}")
            Log.i(TAG, "精度对比 MSE=$mse MaxDiff=$maxDiff")

            cpuHelper.close()
            nnapiHelper.close()
        } catch (e: Exception) {
            sb.appendLine("精度对比失败: ${e.message}")
            Log.w(TAG, "精度对比失败", e)
        }

        sb.appendLine("\n" + "=".repeat(40))
        sb.appendLine("  测试完成")
        sb.appendLine("=".repeat(40))
        updateUi(sb.toString())
        Log.i(TAG, "======== 完整基准测试结果（可整段复制）========\n${sb.toString()}\n========================================")
    }

    private suspend fun tryLoadInputs(): AssetLoader.TestInputs? {
        return try {
            AssetLoader.load(this)
        } catch (e: Exception) {
            updateUi("assets 加载失败: ${e.message}\n请运行 prepare_android_assets.py 并复制到 assets/")
            null
        }
    }

    private suspend fun updateUi(text: String) = withContext(Dispatchers.Main) {
        binding.tvResult.text = text
    }

    // ==================== 权限 ====================
    private fun requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) return // Android 13+ 不需要
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                PERM_REQ,
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }
}
