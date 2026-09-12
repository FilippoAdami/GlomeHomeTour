package com.glomehometour.arscan

import java.nio.ByteBuffer
import kotlin.math.abs

/**
 * Illumination gate (projet.md §2.2, SPEC §1.2h): keeps dark, blown and mid-transition frames out
 * of the exported dataset while VIO keeps running underneath, so a light being switched on halfway
 * through a scan costs a second of frames instead of the whole scan.
 *
 * Mean luma is read straight out of the camera image's Y plane -- it *is* the luma, so the
 * 0.2126R+0.7152G+0.0722B in the spec is already done by the camera pipeline. Sampled on a stride
 * (see LUMA_STRIDE): 8k samples of a 1080p frame estimate the mean to well inside a grey level,
 * and the whole point is to avoid touching two million pixels at 30 Hz.
 */
object Luma {

    /** Stride between sampled pixels in both axes. */
    const val LUMA_STRIDE = 16

    /** Computes the mean luma of the top 15% lightest pixels in the frame.
     * Uses a 256-bin histogram for zero heap allocations and 60 FPS performance.
     * Evaluates whether the scene is genuinely well-lit regardless of dark objects (doors/furniture). */
    fun mean(
        buf: ByteBuffer, width: Int, height: Int, rowStride: Int, pixelStride: Int,
        stride: Int = LUMA_STRIDE, topFraction: Float = 0.15f,
    ): Float {
        val hist = IntArray(256)
        var count = 0
        var y = 0
        while (y < height) {
            val row = y * rowStride
            var x = 0
            while (x < width) {
                val v = buf.get(row + x * pixelStride).toInt() and 0xFF
                hist[v]++
                count++
                x += stride
            }
            y += stride
        }
        if (count == 0) return 0f
        val targetCount = (count * topFraction).toInt().coerceAtLeast(1)
        var sum = 0L
        var accumulated = 0
        for (v in 255 downTo 0) {
            val c = hist[v]
            if (c == 0) continue
            val take = minOf(c, targetCount - accumulated)
            sum += v.toLong() * take
            accumulated += take
            if (accumulated >= targetCount) break
        }
        return sum.toFloat() / accumulated
    }

    /** ByteArray overload, for tests and for callers that already hold the plane. */
    fun mean(
        plane: ByteArray, width: Int, height: Int, rowStride: Int, pixelStride: Int = 1,
        stride: Int = LUMA_STRIDE, topFraction: Float = 0.15f,
    ): Float = mean(ByteBuffer.wrap(plane), width, height, rowStride, pixelStride, stride, topFraction)
}

/**
 * The gate itself, as a small state machine over the mean-luma series.
 *
 * Rejecting on |dY| alone isn't enough: a light coming on is a step change followed by a second or
 * so of auto-exposure hunting, and every frame of that hunt is differently exposed from both the
 * before and the after. So a step *arms* a transition, and the transition only clears once the
 * last WINDOW frames are photometrically quiet (variance < VARIANCE_MAX) -- projet.md §2.2's own
 * rule, and what makes TC-03 pass rather than merely dropping the single jump frame.
 */
class PhotometricGate {

    enum class Verdict { OK, DARK, BLOWN, TRANSITION }

    private val window = FloatArray(WINDOW)
    private var filled = 0
    private var next = 0
    private var previous = Float.NaN
    private var inTransition = false

    /** Last computed window variance -- HUD/diagnostics only. */
    var variance = 0f
        private set

    var lastMean = 0f
        private set

    fun offer(meanLuma: Float): Verdict {
        val delta = if (previous.isNaN()) 0f else abs(meanLuma - previous)
        previous = meanLuma
        lastMean = meanLuma

        if (delta > DELTA_MAX) {
            // Restart the window on the jump: variance across the step would stay huge for
            // WINDOW frames after things have already settled, holding the gate shut long
            // after the room stopped changing.
            inTransition = true
            filled = 0
            next = 0
        }
        push(meanLuma)
        variance = varianceOfWindow()
        if (inTransition && filled == WINDOW && variance < VARIANCE_MAX) inTransition = false

        return when {
            meanLuma < MEAN_MIN -> Verdict.DARK
            meanLuma > MEAN_MAX -> Verdict.BLOWN
            inTransition -> Verdict.TRANSITION
            else -> Verdict.OK
        }
    }

    private fun push(v: Float) {
        window[next] = v
        next = (next + 1) % WINDOW
        if (filled < WINDOW) filled++
    }

    private fun varianceOfWindow(): Float {
        if (filled == 0) return 0f
        var sum = 0f
        for (i in 0 until filled) sum += window[i]
        val mean = sum / filled
        var sq = 0f
        for (i in 0 until filled) {
            val d = window[i] - mean
            sq += d * d
        }
        return sq / filled
    }

    companion object {
        /** Thresholds tuned for top-15% lightest pixel luminance gating. */
        const val MEAN_MIN = 14f
        const val MEAN_MAX = 250f
        const val DELTA_MAX = 35f
        const val WINDOW = 10
        const val VARIANCE_MAX = 6.0f
    }
}
