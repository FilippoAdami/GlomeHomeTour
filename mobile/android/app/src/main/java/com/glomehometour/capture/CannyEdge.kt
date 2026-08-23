package com.glomehometour.capture

import android.graphics.Bitmap
import kotlin.math.atan2
import kotlin.math.sqrt

/** Canny edge detector (blur -> Sobel -> non-max suppression -> hysteresis), square input. */
object CannyEdge {

    // Percentile-based, not fixed absolute values: a fixed threshold tuned against a
    // near/high-contrast frame produced garbage (near-random speckle) on lower-contrast far
    // scenes, since their real gradients never reached the fixed bar. Deriving the threshold
    // from each frame's own gradient-magnitude distribution keeps edge density roughly
    // consistent regardless of scene contrast/distance.
    // Percentile must be taken over the PRE-NMS magnitude image, not the post-NMS one: NMS
    // zeroes non-maxima, but sensor noise is spatially uncorrelated so almost every noisy pixel
    // is trivially a local maximum in some direction and survives anyway — the post-NMS array
    // isn't actually zero-inflated the way it looks, so its 90th percentile sat in the noise
    // floor instead of above it (dense "worm" speckle everywhere).
    // ponytail: percentile knobs — raise highPercentile if speckle is still too dense.
    private const val highPercentile = 0.87f
    private const val lowRatio = 0.4f

    private val gaussianKernel = floatArrayOf(
        2f, 4f, 5f, 4f, 2f,
        4f, 9f, 12f, 9f, 4f,
        5f, 12f, 15f, 12f, 5f,
        4f, 9f, 12f, 9f, 4f,
        2f, 4f, 5f, 4f, 2f,
    ).map { it / 159f }.toFloatArray()

    fun detect(bitmap: Bitmap, size: Int): BooleanArray {
        // Canny runs entirely on the CPU (unlike the ONNX model, which has a delegate), so full
        // 256x256 was the actual framerate bottleneck (dropped rate from ~4Hz to <1Hz). Run the
        // whole pipeline at half resolution and nearest-upsample the mask at the end — ~4x less
        // work, edges only need to be roughly pixel-aligned with a heatmap this coarse anyway.
        val fullPixels = IntArray(size * size)
        bitmap.getPixels(fullPixels, 0, size, 0, 0, size, size)
        val size2 = size / 2
        val gray = FloatArray(size2 * size2)
        for (y in 0 until size2) {
            for (x in 0 until size2) {
                val p = fullPixels[(y * 2) * size + (x * 2)]
                gray[y * size2 + x] =
                    0.299f * ((p shr 16) and 0xFF) + 0.587f * ((p shr 8) and 0xFF) + 0.114f * (p and 0xFF)
            }
        }
        val size = size2

        val blurred = convolve(gray, size, gaussianKernel, 5)
        val gx = FloatArray(size * size)
        val gy = FloatArray(size * size)
        val mag = FloatArray(size * size)
        val angle = FloatArray(size * size)
        for (y in 1 until size - 1) {
            for (x in 1 until size - 1) {
                val i = y * size + x
                val sx = blurred[i - size - 1] - blurred[i - size + 1] +
                    2 * blurred[i - 1] - 2 * blurred[i + 1] +
                    blurred[i + size - 1] - blurred[i + size + 1]
                val sy = blurred[i - size - 1] + 2 * blurred[i - size] + blurred[i - size + 1] -
                    blurred[i + size - 1] - 2 * blurred[i + size] - blurred[i + size + 1]
                gx[i] = sx
                gy[i] = sy
                mag[i] = sqrt(sx * sx + sy * sy)
                angle[i] = atan2(sy, sx)
            }
        }

        val suppressed = FloatArray(size * size)
        for (y in 1 until size - 1) {
            for (x in 1 until size - 1) {
                val i = y * size + x
                var deg = Math.toDegrees(angle[i].toDouble()).toFloat()
                if (deg < 0) deg += 180f
                val (n1, n2) = when {
                    deg < 22.5f || deg >= 157.5f -> Pair(mag[i - 1], mag[i + 1])
                    deg < 67.5f -> Pair(mag[i - size + 1], mag[i + size - 1])
                    deg < 112.5f -> Pair(mag[i - size], mag[i + size])
                    else -> Pair(mag[i - size - 1], mag[i + size + 1])
                }
                suppressed[i] = if (mag[i] >= n1 && mag[i] >= n2) mag[i] else 0f
            }
        }

        val sortedMag = mag.copyOf().also { it.sort() }
        val highThreshold = sortedMag[(sortedMag.size * highPercentile).toInt().coerceIn(0, sortedMag.size - 1)]
            .coerceAtLeast(15f) // a near-blank/noisy frame must not flag everything as an edge
        val lowThreshold = highThreshold * lowRatio

        val edges = BooleanArray(size * size)
        val stack = ArrayDeque<Int>()
        for (i in suppressed.indices) {
            if (suppressed[i] >= highThreshold && !edges[i]) {
                edges[i] = true
                stack.addLast(i)
            }
        }
        while (stack.isNotEmpty()) {
            val i = stack.removeLast()
            val x = i % size
            val y = i / size
            for (dy in -1..1) {
                for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = x + dx
                    val ny = y + dy
                    if (nx !in 0 until size || ny !in 0 until size) continue
                    val ni = ny * size + nx
                    if (!edges[ni] && suppressed[ni] >= lowThreshold) {
                        edges[ni] = true
                        stack.addLast(ni)
                    }
                }
            }
        }
        val fullSize = size * 2
        val fullEdges = BooleanArray(fullSize * fullSize)
        for (y in 0 until fullSize) {
            for (x in 0 until fullSize) {
                fullEdges[y * fullSize + x] = edges[(y / 2) * size + (x / 2)]
            }
        }
        return fullEdges
    }

    private fun convolve(src: FloatArray, size: Int, kernel: FloatArray, k: Int): FloatArray {
        val half = k / 2
        val out = FloatArray(size * size)
        for (y in 0 until size) {
            for (x in 0 until size) {
                var sum = 0f
                for (ky in 0 until k) {
                    val sy = (y + ky - half).coerceIn(0, size - 1)
                    for (kx in 0 until k) {
                        val sx = (x + kx - half).coerceIn(0, size - 1)
                        sum += src[sy * size + sx] * kernel[ky * k + kx]
                    }
                }
                out[y * size + x] = sum
            }
        }
        return out
    }
}
