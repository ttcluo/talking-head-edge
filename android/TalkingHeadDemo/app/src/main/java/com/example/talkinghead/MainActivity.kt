package com.example.talkinghead

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.talkinghead.databinding.ActivityMainBinding
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val scope = CoroutineScope(Dispatchers.Main)

    // 模型通过 adb push 到设备后的路径
    // adb push unet_student_int8.onnx  /sdcard/Download/
    // adb push unet_student_fp32.onnx  /sdcard/Download/
    private val MODEL_INT8 = "${Environment.getExternalStorageDirectory()}/Download/unet_student_int8.onnx"
    private val MODEL_FP32 = "${Environment.getExternalStorageDirectory()}/Download/unet_student_fp32.onnx"

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
    }

    private fun runBench(useNnapi: Boolean) = scope.launch(Dispatchers.IO) {
        val provider = if (useNnapi) "NNAPI" else "CPU"
        updateUi("[$provider] 加载输入...")

        val inputs = tryLoadInputs() ?: return@launch
        val modelPath = if (useNnapi) MODEL_INT8 else MODEL_INT8

        val helper = InferenceHelper(this@MainActivity, modelPath, useNnapi)
        try {
            helper.load()
            updateUi("[$provider] 推理中（warmup 3次 + bench 20次）...")
            val result = helper.benchmark(inputs.latent, inputs.audioFeat)
            updateUi(result.toString())
        } catch (e: Exception) {
            Log.e(TAG, "推理失败", e)
            updateUi("[$provider] 错误: ${e.message}\n\n提示：确认 INT8 ONNX 文件已 adb push 到 $MODEL_INT8")
        } finally {
            helper.close()
        }
    }

    private fun runFullBenchmark() = scope.launch(Dispatchers.IO) {
        val inputs = tryLoadInputs() ?: return@launch
        val sb = StringBuilder()

        // -------- FP32 + CPU --------
        sb.appendLine("=".repeat(40))
        sb.appendLine("  Student UNet 端侧基准测试")
        sb.appendLine("  设备: ${Build.MODEL}  API ${Build.VERSION.SDK_INT}")
        sb.appendLine("  模型: INT8 ONNX (141.7MB)")
        sb.appendLine("=".repeat(40))
        updateUi(sb.toString())

        for ((label, useNnapi) in listOf("CPU" to false, "NNAPI" to true)) {
            sb.appendLine("\n[$label]")
            updateUi(sb.toString())
            val helper = InferenceHelper(this@MainActivity, MODEL_INT8, useNnapi)
            try {
                helper.load()
                val res = helper.benchmark(inputs.latent, inputs.audioFeat, warmupRuns = 5, benchRuns = 30)
                sb.append(res.toString())
                updateUi(sb.toString())
            } catch (e: Exception) {
                sb.appendLine("错误: ${e.message}")
                updateUi(sb.toString())
            } finally {
                helper.close()
            }
        }

        // -------- 精度验证（CPU vs NNAPI 输出差异）--------
        sb.appendLine("\n[精度对比 CPU vs NNAPI]")
        updateUi(sb.toString())
        try {
            val cpuHelper   = InferenceHelper(this@MainActivity, MODEL_INT8, false).also { it.load() }
            val nnapiHelper = InferenceHelper(this@MainActivity, MODEL_INT8, true).also { it.load() }

            val cpuOut   = cpuHelper.infer(inputs.latent, inputs.audioFeat)
            val nnapiOut = nnapiHelper.infer(inputs.latent, inputs.audioFeat)

            val mse = cpuOut.indices
                .map { i -> (cpuOut[i] - nnapiOut[i]).let { it * it } }
                .average()
            val maxDiff = cpuOut.indices
                .maxOf { i -> Math.abs(cpuOut[i] - nnapiOut[i]) }

            sb.appendLine("MSE(CPU, NNAPI):     ${"%.6f".format(mse)}")
            sb.appendLine("MaxDiff(CPU, NNAPI): ${"%.4f".format(maxDiff)}")

            cpuHelper.close()
            nnapiHelper.close()
        } catch (e: Exception) {
            sb.appendLine("精度对比失败: ${e.message}")
        }

        sb.appendLine("\n" + "=".repeat(40))
        sb.appendLine("  测试完成")
        sb.appendLine("=".repeat(40))
        updateUi(sb.toString())
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
