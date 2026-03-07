package com.example.talkinghead

import ai.onnxruntime.*
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * 封装 Student UNet ONNX 推理。
 *
 * 支持两种 provider：
 *   - CPU：标准 ONNX Runtime CPU（FP32 或 INT8 MatMul）
 *   - NNAPI：Snapdragon NPU 加速（需要 INT8 QDQ 格式或 FP32）
 */
class InferenceHelper(
    private val context: Context,
    private val modelPath: String,   // 绝对路径，通过 adb push 到 /sdcard/Download/
    private val useNnapi: Boolean = false,
) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession

    companion object {
        private const val TAG = "InferenceHelper"

        // 输入形状
        val LATENT_SHAPE = longArrayOf(1, 8, 32, 32)
        val AUDIO_SHAPE  = longArrayOf(1, 50, 384)
        val TIMESTEP_SHAPE = longArrayOf(1)

        val LATENT_SIZE  = LATENT_SHAPE.fold(1L) { a, b -> a * b }.toInt()
        val AUDIO_SIZE   = AUDIO_SHAPE.fold(1L)  { a, b -> a * b }.toInt()
    }

    fun load() {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            if (useNnapi) {
                try {
                    addNnapi()
                    Log.i(TAG, "NNAPI provider 已启用")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI 不可用，回退到 CPU: ${e.message}")
                }
            }
        }
        session = env.createSession(modelPath, opts)
        Log.i(TAG, "模型加载完成: $modelPath  provider=${if (useNnapi) "NNAPI" else "CPU"}")
    }

    /**
     * 单帧推理。
     * @param latent  FloatArray [1,8,32,32]
     * @param audioFeat FloatArray [1,50,384]
     * @return FloatArray [1,4,32,32] 预测 latent
     */
    fun infer(latent: FloatArray, audioFeat: FloatArray): FloatArray {
        val latentTensor = OnnxTensor.createTensor(env,
            FloatBuffer.wrap(latent), LATENT_SHAPE)
        val timestepTensor = OnnxTensor.createTensor(env,
            LongBuffer.wrap(longArrayOf(0L)), TIMESTEP_SHAPE)
        val audioTensor = OnnxTensor.createTensor(env,
            FloatBuffer.wrap(audioFeat), AUDIO_SHAPE)

        val inputs = mapOf(
            "latent"     to latentTensor,
            "timestep"   to timestepTensor,
            "audio_feat" to audioTensor,
        )

        val output = session.run(inputs)
        // ORT Android 输出通过名称访问，避免 index API 版本差异
        val outTensor = output.get("output").get() as OnnxTensor
        val fb = outTensor.floatBuffer
        val result = FloatArray(fb.remaining()).also { fb.get(it) }

        latentTensor.close()
        timestepTensor.close()
        audioTensor.close()
        output.close()

        return result
    }

    /**
     * 基准测试：warmup + 多次推理，返回平均/最小/最大延迟（ms）。
     */
    fun benchmark(
        latent: FloatArray,
        audioFeat: FloatArray,
        warmupRuns: Int = 3,
        benchRuns: Int = 20,
    ): BenchResult {
        // warmup
        Log.i(TAG, "benchmark: warmup $warmupRuns runs...")
        repeat(warmupRuns) { infer(latent, audioFeat) }

        val latencies = mutableListOf<Long>()
        Log.i(TAG, "benchmark: bench $benchRuns runs...")
        repeat(benchRuns) {
            val t0 = System.nanoTime()
            infer(latent, audioFeat)
            latencies.add((System.nanoTime() - t0) / 1_000_000)
        }

        val result = BenchResult(
            provider    = if (useNnapi) "NNAPI" else "CPU",
            warmupRuns  = warmupRuns,
            benchRuns   = benchRuns,
            avgMs       = latencies.average(),
            minMs       = (latencies.minOrNull() ?: 0L).toDouble(),
            maxMs       = (latencies.maxOrNull() ?: 0L).toDouble(),
            fps         = 1000.0 / latencies.average(),
        )
        Log.i(TAG, "benchmark done: ${"%.1f".format(result.avgMs)} ms/frame, ${"%.1f".format(result.fps)} FPS (${result.provider})")
        return result
    }

    fun close() {
        if (::session.isInitialized) session.close()
    }

    data class BenchResult(
        val provider:   String,
        val warmupRuns: Int,
        val benchRuns:  Int,
        val avgMs:      Double,
        val minMs:      Double,
        val maxMs:      Double,
        val fps:        Double,
    ) {
        override fun toString() = buildString {
            appendLine("Provider: $provider")
            appendLine("Warmup:   $warmupRuns runs")
            appendLine("Bench:    $benchRuns runs")
            appendLine("Avg:      ${"%.1f".format(avgMs)} ms/frame")
            appendLine("Min:      ${"%.1f".format(minMs)} ms")
            appendLine("Max:      ${"%.1f".format(maxMs)} ms")
            appendLine("FPS:      ${"%.1f".format(fps)}")
        }
    }
}
