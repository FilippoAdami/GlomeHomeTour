package com.glomehometour.capture

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer

/**
 * Runs the ZipDepth base NPU-export ONNX model (256x256, CPU EP) — the ECCV 2026 ZipDepth
 * checkpoint (github.com/fabiotosi92/ZipDepth), exported via `scripts/export.py --variant base
 * --npu --height 256 --width 256`. Output is affine-invariant inverse depth (near = high value).
 */
class DepthEngine(context: Context) : AutoCloseable {

    companion object {
        const val SIZE = 256
    }

    data class Result(val depth: FloatArray, val latencyMs: Long)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open("zipdepth_base_256x256.onnx").use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            // Multi-thread the CPU EP — needed to keep inference latency under the gate
            // in MainActivity; single-threaded default was the bottleneck.
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
        }
        session = env.createSession(modelBytes, options)
    }

    /** [bitmap] must already be SIZE x SIZE, ARGB_8888. Blocking — call off the main thread. */
    fun run(bitmap: Bitmap): Result {
        val input = bitmapToChwTensor(bitmap)
        val t0 = System.nanoTime()
        OnnxTensor.createTensor(env, input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { tensor ->
            session.run(mapOf("image" to tensor)).use { outputs ->
                val latencyMs = (System.nanoTime() - t0) / 1_000_000
                @Suppress("UNCHECKED_CAST")
                val raw = (outputs[0].value as Array<Array<Array<FloatArray>>>)[0][0]
                val depth = FloatArray(SIZE * SIZE)
                for (y in 0 until SIZE) {
                    System.arraycopy(raw[y], 0, depth, y * SIZE, SIZE)
                }
                return Result(depth, latencyMs)
            }
        }
    }

    /** RGB, [0,1], NCHW — matches ZipDepth's own preprocessing (no mean/std normalization). */
    private fun bitmapToChwTensor(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(SIZE * SIZE)
        bitmap.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val buffer = FloatBuffer.allocate(3 * SIZE * SIZE)
        val plane = SIZE * SIZE
        for (i in pixels.indices) {
            val p = pixels[i]
            buffer.put(i, ((p shr 16) and 0xFF) / 255f)
            buffer.put(plane + i, ((p shr 8) and 0xFF) / 255f)
            buffer.put(2 * plane + i, (p and 0xFF) / 255f)
        }
        return buffer
    }

    override fun close() {
        session.close()
    }
}
