package com.example.talkinghead

import android.content.Context
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 从 assets 加载由 prepare_android_assets.py 生成的测试输入。
 *
 * 文件：
 *   latent_test.bin   - float32 原始二进制，形状 [1,8,32,32]
 *   audio_test.bin    - float32 原始二进制，形状 [1,50,384]
 *   meta.json         - 维度信息
 */
object AssetLoader {

    data class TestInputs(
        val latent:    FloatArray,
        val audioFeat: FloatArray,
        val meta:      JSONObject,
    )

    fun load(context: Context): TestInputs {
        val meta = loadJson(context, "meta.json")
        val latent    = loadBin(context, "latent_test.bin")
        val audioFeat = loadBin(context, "audio_test.bin")
        return TestInputs(latent, audioFeat, meta)
    }

    private fun loadBin(context: Context, name: String): FloatArray {
        val bytes = context.assets.open(name).use { stream ->
            val buf = ByteArrayOutputStream()
            stream.copyTo(buf)
            buf.toByteArray()
        }
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    private fun loadJson(context: Context, name: String): JSONObject {
        val text = context.assets.open(name).bufferedReader().use { it.readText() }
        return JSONObject(text)
    }
}
