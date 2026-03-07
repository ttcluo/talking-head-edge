package com.example.talkinghead

import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.talkinghead.databinding.ActivityVideoPreviewBinding
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream

/**
 * 端上逐帧运行 UNet + VAE 解码，在界面显示生成的人脸视频预览。
 * 需在 assets 中放入：latents_seq.bin, audio_seq.bin, video_meta.json，以及 vae_decoder.onnx、unet 模型。
 */
class VideoPreviewActivity : AppCompatActivity() {

    private lateinit var binding: ActivityVideoPreviewBinding
    private val scope = CoroutineScope(Dispatchers.Main)

    companion object {
        private const val TAG = "VideoPreview"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVideoPreviewBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnGenerate.setOnClickListener {
            if (binding.btnGenerate.isEnabled) runGenerate()
        }
    }

    private fun runGenerate() = scope.launch(Dispatchers.IO) {
        binding.btnGenerate.isEnabled = false
        binding.progressBar.visibility = View.VISIBLE
        binding.progressBar.progress = 0
        setStatus("加载序列...")

        val seq = VideoSequenceLoader.load(this@VideoPreviewActivity)
        if (seq == null) {
            withContext(Dispatchers.Main) {
                binding.btnGenerate.isEnabled = true
                binding.progressBar.visibility = View.GONE
                setStatus("缺少资源：video_meta.json + latents_seq.bin + audio_seq.bin")
                Toast.makeText(this@VideoPreviewActivity, "请先运行 prepare_android_video_assets.py", Toast.LENGTH_LONG).show()
            }
            return@launch
        }

        val unetPath = getUnetFp32Path()
        val vaePath = getVaeDecoderPath()
        if (unetPath == null || vaePath == null) {
            withContext(Dispatchers.Main) {
                binding.btnGenerate.isEnabled = true
                binding.progressBar.visibility = View.GONE
                setStatus("缺少模型：unet_student_fp32.onnx 或 vae_decoder.onnx")
            }
            return@launch
        }

        setStatus("加载 UNet + VAE...")
        val unet = InferenceHelper(this@VideoPreviewActivity, unetPath, useNnapi = false)
        val vae = VaeHelper(this@VideoPreviewActivity, vaePath, seq.scalingFactor)
        try {
            unet.load()
            vae.load()
        } catch (e: Exception) {
            Log.e(TAG, "模型加载失败", e)
            withContext(Dispatchers.Main) {
                binding.btnGenerate.isEnabled = true
                binding.progressBar.visibility = View.GONE
                setStatus("加载失败: ${e.message}")
            }
            return@launch
        }

        val n = seq.numFrames
        binding.progressBar.max = n
        setStatus("生成中 0 / $n...")
        Log.i(TAG, "开始生成 $n 帧")

        for (i in 0 until n) {
            val lat = seq.getLatentFrame(i)
            val aud = seq.getAudioFrame(i)
            val pred4ch = unet.infer(lat, aud)
            val bitmap = vae.decode(pred4ch)
            withContext(Dispatchers.Main) {
                binding.ivPreview.setImageBitmap(bitmap)
                binding.progressBar.progress = i + 1
                binding.tvStatus.text = "生成中 ${i + 1} / $n"
            }
        }

        unet.close()
        vae.close()
        withContext(Dispatchers.Main) {
            binding.btnGenerate.isEnabled = true
            binding.progressBar.visibility = View.GONE
            setStatus("完成，共 $n 帧（可再次点击重新生成）")
            Log.i(TAG, "视频预览生成完成 $n 帧")
        }
    }

    private suspend fun setStatus(text: String) = withContext(Dispatchers.Main) {
        binding.tvStatus.text = text
    }

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

    private fun getVaeDecoderPath(): String? {
        val external = File(Environment.getExternalStorageDirectory(), "Download/vae_decoder.onnx")
        if (external.canRead()) return external.absolutePath
        val inApp = File(filesDir, "vae_decoder.onnx")
        if (inApp.exists()) return inApp.absolutePath
        try {
            assets.open("vae_decoder.onnx").use { ins ->
                FileOutputStream(inApp).use { ins.copyTo(it) }
            }
            return inApp.absolutePath
        } catch (_: Exception) {
            return null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }
}
