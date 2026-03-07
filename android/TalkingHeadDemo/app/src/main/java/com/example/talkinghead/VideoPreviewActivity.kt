package com.example.talkinghead

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
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
import java.nio.ByteBuffer

/**
 * 端上逐帧运行 UNet + VAE 解码，在界面显示生成的人脸视频预览。
 * 生成完成后可「循环播放」或「保存 MP4」。
 */
class VideoPreviewActivity : AppCompatActivity() {

    private lateinit var binding: ActivityVideoPreviewBinding
    private val scope = CoroutineScope(Dispatchers.Main)

    private var generatedFrames: List<Bitmap> = emptyList()
    private var loopJob: Job? = null

    companion object {
        private const val TAG = "VideoPreview"
        private const val LOOP_FPS = 25
        private const val LOOP_DELAY_MS = 1000 / LOOP_FPS
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityVideoPreviewBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnGenerate.setOnClickListener {
            if (binding.btnGenerate.isEnabled) runGenerate()
        }
        binding.btnLoopPlay.setOnClickListener { startLoopPlay() }
        binding.btnSaveVideo.setOnClickListener { saveToMp4() }
    }

    private fun runGenerate() = scope.launch(Dispatchers.IO) {
        withContext(Dispatchers.Main) {
            binding.btnGenerate.isEnabled = false
            binding.progressBar.visibility = View.VISIBLE
            binding.progressBar.progress = 0
            setStatus("加载序列...")
        }

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

        withContext(Dispatchers.Main) { setStatus("加载 UNet + VAE...") }
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
        withContext(Dispatchers.Main) {
            binding.progressBar.max = n
            setStatus("生成中 0 / $n...")
        }
        val totalStartNs = System.nanoTime()
        Log.i(TAG, "[性能] 开始生成 $n 帧")

        val frames = ArrayList<Bitmap>(n)
        val unetMsList = LongArray(n)
        val vaeMsList = LongArray(n)
        for (i in 0 until n) {
            val lat = seq.getLatentFrame(i)
            val aud = seq.getAudioFrame(i)
            val tUnet0 = System.nanoTime()
            val pred4ch = unet.infer(lat, aud)
            unetMsList[i] = (System.nanoTime() - tUnet0) / 1_000_000
            val tVae0 = System.nanoTime()
            val bitmap = vae.decode(pred4ch)
            vaeMsList[i] = (System.nanoTime() - tVae0) / 1_000_000
            frames.add(bitmap)
            withContext(Dispatchers.Main) {
                binding.ivPreview.setImageBitmap(bitmap)
                binding.progressBar.progress = i + 1
                binding.tvStatus.text = "生成中 ${i + 1} / $n"
            }
            // 每 5 帧打一次 + 首帧/次帧(看预热) + 末帧，避免首帧后长时间无日志
            if (i == 0 || i == 1 || i % 5 == 0 || i == n - 1) {
                Log.i(TAG, "[性能] 帧 $i/${n}: UNet=${unetMsList[i]} ms, VAE=${vaeMsList[i]} ms")
            }
        }

        val totalMs = (System.nanoTime() - totalStartNs) / 1_000_000
        val unetAvgFirst = unetMsList[0]
        val vaeAvgFirst = vaeMsList[0]
        val unetAvgSteady = if (n > 1) unetMsList.drop(1).sum() / (n - 1) else unetMsList[0]
        val vaeAvgSteady = if (n > 1) vaeMsList.drop(1).sum() / (n - 1) else vaeMsList[0]
        val frameAvgSteady = unetAvgSteady + vaeAvgSteady
        val fpsSteady = if (frameAvgSteady > 0) 1000.0 / frameAvgSteady else 0.0
        Log.i(TAG, "[性能] --------- 生成耗时汇总 ---------")
        Log.i(TAG, "[性能] 总耗时: ${totalMs} ms (${totalMs / 1000.0} s), 帧数: $n")
        Log.i(TAG, "[性能] 首帧(含预热): UNet=${unetAvgFirst} ms, VAE=${vaeAvgFirst} ms")
        Log.i(TAG, "[性能] 稳态平均(帧2~$n): UNet=${unetAvgSteady} ms, VAE=${vaeAvgSteady} ms, 单帧总=${frameAvgSteady} ms")
        Log.i(TAG, "[性能] 等效 FPS(稳态): ${"%.2f".format(fpsSteady)} (实时 25fps 需单帧≤40ms)")
        Log.i(TAG, "[性能] ------------------------------")

        unet.close()
        vae.close()
        withContext(Dispatchers.Main) {
            generatedFrames = frames
            binding.btnGenerate.isEnabled = true
            binding.progressBar.visibility = View.GONE
            binding.btnLoopPlay.visibility = View.VISIBLE
            binding.btnSaveVideo.visibility = View.VISIBLE
            setStatus("完成，共 $n 帧。可「循环播放」或「保存 MP4」")
            Log.i(TAG, "视频预览生成完成 $n 帧")
        }
    }

    private fun startLoopPlay() {
        if (generatedFrames.isEmpty()) return
        if (loopJob?.isActive == true) {
            loopJob?.cancel()
            loopJob = null
            binding.tvStatus.text = "已停止循环播放"
            return
        }
        var index = 0
        val n = generatedFrames.size
        loopJob = scope.launch(Dispatchers.Main) {
            while (isActive) {
                binding.ivPreview.setImageBitmap(generatedFrames[index % n])
                index++
                delay(LOOP_DELAY_MS.toLong())
            }
        }
        binding.tvStatus.text = "循环播放中（${LOOP_FPS} FPS）… 再点「循环播放」可停"
    }

    override fun onPause() {
        super.onPause()
        loopJob?.cancel()
        loopJob = null
    }

    private fun saveToMp4() {
        if (generatedFrames.isEmpty()) {
            Toast.makeText(this, "请先生成预览", Toast.LENGTH_SHORT).show()
            return
        }
        scope.launch(Dispatchers.IO) {
            val dir = getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
            val outFile = File(dir, "talking_head_preview_${System.currentTimeMillis()}.mp4")
            try {
                withContext(Dispatchers.Main) { setStatus("正在编码 MP4...") }
                encodeBitmapsToMp4(generatedFrames, LOOP_FPS, outFile)
                withContext(Dispatchers.Main) {
                    setStatus("已保存: ${outFile.name}（仅画面，无音轨）")
                    Toast.makeText(this@VideoPreviewActivity, "已保存至 ${outFile.absolutePath}\n（仅画面，无音轨）", Toast.LENGTH_LONG).show()
                }
                Log.i(TAG, "MP4 已保存: ${outFile.absolutePath}")
            } catch (e: Exception) {
                Log.e(TAG, "保存 MP4 失败", e)
                withContext(Dispatchers.Main) {
                    setStatus("保存失败: ${e.message}")
                    Toast.makeText(this@VideoPreviewActivity, "保存失败: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun setStatus(text: String) {
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

    /**
     * 将 Bitmap 列表编码为 H.264 MP4。256×256，25fps。
     */
    private fun encodeBitmapsToMp4(bitmaps: List<Bitmap>, fps: Int, outputFile: File) {
        if (bitmaps.isEmpty()) return
        val width = bitmaps[0].width
        val height = bitmaps[0].height
        val mime = "video/avc"
        val format = MediaFormat.createVideoFormat(mime, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar)
            setInteger(MediaFormat.KEY_BIT_RATE, 1_000_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val codec = MediaCodec.createEncoderByType(mime)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
        val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var muxerTrackIndex = -1
        var presentationTimeUs = 0L
        val frameDurationUs = 1_000_000L / fps

        fun drainEncoder(endOfStream: Boolean) {
            val bufferInfo = MediaCodec.BufferInfo()
            while (true) {
                val status = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                when {
                    status == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        if (muxerTrackIndex >= 0) return
                        muxerTrackIndex = muxer.addTrack(codec.outputFormat)
                        muxer.start()
                    }
                    status == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!endOfStream) return
                    status >= 0 -> {
                        if (muxerTrackIndex < 0) return
                        val data = codec.getOutputBuffer(status) ?: return
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                            bufferInfo.size = 0
                        }
                        if (bufferInfo.size > 0) {
                            bufferInfo.presentationTimeUs = presentationTimeUs
                            muxer.writeSampleData(muxerTrackIndex, data, bufferInfo)
                            presentationTimeUs += frameDurationUs
                        }
                        codec.releaseOutputBuffer(status, false)
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                    }
                }
            }
        }

        val nv12 = ByteArray(width * height * 3 / 2)
        for (i in bitmaps.indices) {
            bitmapToNv12(bitmaps[i], nv12)
            var inputIndex = codec.dequeueInputBuffer(10_000)
            while (inputIndex == MediaCodec.INFO_TRY_AGAIN_LATER) {
                drainEncoder(false)
                inputIndex = codec.dequeueInputBuffer(10_000)
            }
            if (inputIndex >= 0) {
                val buffer = codec.getInputBuffer(inputIndex) ?: continue
                buffer.clear()
                buffer.put(nv12)
                codec.queueInputBuffer(inputIndex, 0, nv12.size, presentationTimeUs, 0)
                presentationTimeUs += frameDurationUs
            }
            drainEncoder(false)
        }

        var inputIndex = codec.dequeueInputBuffer(10_000)
        if (inputIndex >= 0) {
            codec.queueInputBuffer(inputIndex, 0, 0, presentationTimeUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        }
        drainEncoder(true)
        codec.stop()
        codec.release()
        muxer.stop()
        muxer.release()
    }

    private fun bitmapToNv12(bitmap: Bitmap, outNv12: ByteArray) {
        val w = bitmap.width
        val h = bitmap.height
        val ySize = w * h
        val uvSize = w * h / 2
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        var yIdx = 0
        var uvIdx = ySize
        var i = 0
        while (i < h) {
            var j = 0
            while (j < w) {
                val p = pixels[i * w + j]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                outNv12[yIdx++] = ((77 * r + 150 * g + 29 * b) shr 8).coerceIn(0, 255).toByte()
                if (i and 1 == 0 && j and 1 == 0) {
                    val u = (((-43 * r - 84 * g + 127 * b) shr 8) + 128).coerceIn(0, 255)
                    val v = (((127 * r - 106 * g - 21 * b) shr 8) + 128).coerceIn(0, 255)
                    if (uvIdx < outNv12.size - 1) {
                        outNv12[uvIdx++] = u.toByte()
                        outNv12[uvIdx++] = v.toByte()
                    }
                }
                j++
            }
            i++
        }
    }
}
