package com.example.talkinghead

import ai.onnxruntime.*
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer

/**
 * VAE Decoder ONNX：将 UNet 输出的 4ch latent [1,4,32,32] 解码为 RGB [256,256]。
 * 输入需先除以 scaling_factor（0.18215）再送入。
 */
class VaeHelper(
    private val context: Context,
    private val modelPath: String,
    private val scalingFactor: Float = 0.18215f,
    private val useNnapi: Boolean = true,
) {
    private val env = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession

    companion object {
        private const val TAG = "VaeHelper"
        private val INPUT_SHAPE = longArrayOf(1, 4, 32, 32)
        private const val INPUT_SIZE = 1 * 4 * 32 * 32
        private const val OUTPUT_SIZE = 1 * 3 * 256 * 256
    }

    fun load() {
        if (useNnapi) {
            try {
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(2)
                    addNnapi()
                }
                session = env.createSession(modelPath, opts)
                Log.i(TAG, "VAE Decoder 加载完成 (NNAPI): $modelPath")
                return
            } catch (e: Exception) {
                Log.w(TAG, "VAE NNAPI 加载失败，回退 CPU: ${e.message}")
            }
        }
        session = env.createSession(modelPath, OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) })
        Log.i(TAG, "VAE Decoder 加载完成 (CPU): $modelPath")
    }

    /**
     * @param latent4ch UNet 输出 [1,4,32,32]，共 4096 个 float
     * @return 解码后的 RGB Bitmap 256x256
     */
    fun decode(latent4ch: FloatArray): Bitmap {
        require(latent4ch.size == INPUT_SIZE) { "latent4ch.size=${latent4ch.size} != $INPUT_SIZE" }
        val scaled = FloatArray(INPUT_SIZE) { latent4ch[it] / scalingFactor }
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(scaled), INPUT_SHAPE)
        val output = session.run(mapOf("latent" to inputTensor))
        val outTensor = output.get("image").get() as OnnxTensor
        val fb = outTensor.floatBuffer
        val rgb = FloatArray(OUTPUT_SIZE).also { fb.get(it) }
        inputTensor.close()
        output.close()

        // [1,3,256,256] CHW -> Bitmap 256x256 ARGB（用 setPixels 批量写入，避免 6 万次 setPixel）
        val pixels = IntArray(256 * 256)
        val stride = 256 * 256
        for (i in 0 until 256 * 256) {
            val r = ((rgb[0 * stride + i].coerceIn(-1f, 1f) + 1f) * 0.5f * 255).toInt().coerceIn(0, 255)
            val g = ((rgb[1 * stride + i].coerceIn(-1f, 1f) + 1f) * 0.5f * 255).toInt().coerceIn(0, 255)
            val b = ((rgb[2 * stride + i].coerceIn(-1f, 1f) + 1f) * 0.5f * 255).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val bitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, 256, 0, 0, 256, 256)
        return bitmap
    }

    fun close() {
        if (::session.isInitialized) session.close()
    }
}
