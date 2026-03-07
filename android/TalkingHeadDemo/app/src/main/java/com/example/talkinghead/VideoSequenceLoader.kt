package com.example.talkinghead

import android.content.Context
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 从 assets 加载多帧序列（prepare_android_video_assets.py 生成）。
 * latents_seq.bin: num_frames × [1,8,32,32]
 * audio_seq.bin:   num_frames × [1,50,384]
 */
object VideoSequenceLoader {

    private const val LATENT_FLOATS_PER_FRAME = 1 * 8 * 32 * 32   // 8192
    private const val AUDIO_FLOATS_PER_FRAME  = 1 * 50 * 384       // 19200

    data class VideoSequence(
        val numFrames: Int,
        val latents: FloatArray,   // numFrames * LATENT_FLOATS_PER_FRAME
        val audioFeats: FloatArray,
        val scalingFactor: Float,
    ) {
        fun getLatentFrame(i: Int): FloatArray =
            latents.copyOfRange(i * LATENT_FLOATS_PER_FRAME, (i + 1) * LATENT_FLOATS_PER_FRAME)
        fun getAudioFrame(i: Int): FloatArray =
            audioFeats.copyOfRange(i * AUDIO_FLOATS_PER_FRAME, (i + 1) * AUDIO_FLOATS_PER_FRAME)
    }

    fun load(context: Context): VideoSequence? {
        return try {
            val meta = JSONObject(context.assets.open("video_meta.json").bufferedReader().use { it.readText() })
            val numFrames = meta.getInt("num_frames")
            val scalingFactor = meta.optDouble("vae_scaling_factor", 0.18215).toFloat()
            val latents = loadBin(context, "latents_seq.bin")
            val audioFeats = loadBin(context, "audio_seq.bin")
            require(latents.size == numFrames * LATENT_FLOATS_PER_FRAME)
            require(audioFeats.size == numFrames * AUDIO_FLOATS_PER_FRAME)
            VideoSequence(numFrames, latents, audioFeats, scalingFactor)
        } catch (e: Exception) {
            android.util.Log.e("VideoSequenceLoader", "load failed", e)
            null
        }
    }

    private fun loadBin(context: Context, name: String): FloatArray {
        val bytes = context.assets.open(name).use { stream ->
            ByteArrayOutputStream().use { buf -> stream.copyTo(buf); buf.toByteArray() }
        }
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }
}
