package com.glomehometour.arscan

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import kotlin.math.max

/**
 * Android mobile port of GlomeHomeTour's QualityGate (backend/00_ingestion/quality_gate.py).
 *
 * Evaluates frame sharpness, exposure, and texture on two axes:
 * 1. Absolute detail: mean Laplacian variance of the sharpest 25% of tiles.
 * 2. Normalised detail: high-frequency energy divided by local contrast on the most-textured tiles.
 *
 * Relative sharpness is normalized against the scene median, with a safety reject cap (at most 20%
 * blur rejects) to prevent over-culling in dim or textureless environments.
 */
class QualityGate(
    val blurThreshold: Float? = null,
    val relativeBlurThreshold: Float = 0.35f,
    val minRelativeSharpnessFloor: Float? = null,
    val maxRejectFraction: Float = 0.20f,
    val darkThreshold: Float = 12.0f,
    val blownThreshold: Float = 250.0f,
    val maxSaturatedFraction: Float = 0.30f,
    val maxBlackFraction: Float = 0.50f,
    val minTexture: Float = 60.0f,
    val workResolution: Int = 960,
) {

    data class FrameStats(
        val index: Int,
        val filePath: String,
        val blurScore: Float,          // Absolute detail (Laplacian var of sharpest tiles)
        val normalizedBlur: Float,     // Detail / local contrast
        val texture: Float,            // Intensity variance of most-textured tiles
        val meanLuminance: Float,
        val saturatedFraction: Float,
        val blackFraction: Float,
    )

    data class FrameVerdict(
        val index: Int,
        val filePath: String,
        val blurScore: Float,
        val relativeSharpness: Float,
        val accepted: Boolean,
        val rejectionReason: String? = null,
    )

    data class QualityGateResult(
        val acceptedIndices: List<Int>,
        val discardedIndices: List<Int>,
        val verdicts: List<FrameVerdict>,
        val rejectedBlurCount: Int,
        val rejectedExposureCount: Int,
        val rejectedTextureCount: Int,
    )

    /**
     * Compute stats for an in-memory 8-bit grayscale pixel buffer ([0..255]).
     */
    fun computeStats(gray: ByteArray, width: Int, height: Int, index: Int = 0, filePath: String = ""): FrameStats {
        val longEdge = max(width, height)
        val (workBytes, w, h) = if (longEdge > workResolution) {
            val scale = workResolution.toFloat() / longEdge
            val nw = max(1, (width * scale).toInt())
            val nh = max(1, (height * scale).toInt())
            Triple(downsampleNearest(gray, width, height, nw, nh), nw, nh)
        } else {
            Triple(gray, width, height)
        }

        val tilesPerSide = 4
        val tileW = w / tilesPerSide
        val tileH = h / tilesPerSide
        val numTiles = tilesPerSide * tilesPerSide

        val lapVars = FloatArray(numTiles)
        val intVars = FloatArray(numTiles)

        var saturatedCount = 0
        var blackCount = 0
        var sumLuma = 0.0

        for (ty in 0 until tilesPerSide) {
            for (tx in 0 until tilesPerSide) {
                val tileIdx = ty * tilesPerSide + tx
                val startX = tx * tileW
                val endX = if (tx == tilesPerSide - 1) w else (tx + 1) * tileW
                val startY = ty * tileH
                val endY = if (ty == tilesPerSide - 1) h else (ty + 1) * tileH

                var sumL = 0.0
                var sumL2 = 0.0
                var sumI = 0.0
                var sumI2 = 0.0
                var count = 0

                for (y in startY until endY) {
                    val rowOff = y * w
                    val ym1Off = (if (y > 0) y - 1 else y) * w
                    val yp1Off = (if (y < h - 1) y + 1 else y) * w

                    for (x in startX until endX) {
                        val pix = workBytes[rowOff + x].toInt() and 0xFF
                        if (pix >= 250) saturatedCount++
                        if (pix <= 8) blackCount++
                        sumLuma += pix

                        // Discrete 3x3 Laplacian: L(x, y) = up + down + left + right - 4 * center
                        val xm1 = if (x > 0) x - 1 else x
                        val xp1 = if (x < w - 1) x + 1 else x

                        val up = workBytes[ym1Off + x].toInt() and 0xFF
                        val down = workBytes[yp1Off + x].toInt() and 0xFF
                        val left = workBytes[rowOff + xm1].toInt() and 0xFF
                        val right = workBytes[rowOff + xp1].toInt() and 0xFF
                        val lap = (up + down + left + right - 4 * pix).toFloat()

                        sumL += lap
                        sumL2 += (lap * lap)
                        sumI += pix
                        sumI2 += (pix * pix)
                        count++
                    }
                }

                if (count > 0) {
                    val meanL = sumL / count
                    val varL = (sumL2 / count) - (meanL * meanL)
                    lapVars[tileIdx] = max(0f, varL.toFloat())

                    val meanI = sumI / count
                    val varI = (sumI2 / count) - (meanI * meanI)
                    intVars[tileIdx] = max(0f, varI.toFloat())
                }
            }
        }

        val totalPixels = w * h
        val meanLuminance = if (totalPixels > 0) (sumLuma / totalPixels).toFloat() else 0f
        val saturatedFraction = if (totalPixels > 0) saturatedCount.toFloat() / totalPixels else 0f
        val blackFraction = if (totalPixels > 0) blackCount.toFloat() / totalPixels else 0f

        // Sharpest quarter of tiles (top 4 out of 16)
        val sortedLap = lapVars.sortedDescending()
        val topQuarter = max(1, sortedLap.size / 4)
        var blurScore = 0f
        for (i in 0 until topQuarter) blurScore += sortedLap[i]
        blurScore /= topQuarter

        // Most textured tiles (top 4 out of 16 by intensity variance)
        val texturedIndices = (0 until numTiles).sortedByDescending { intVars[it] }.take(topQuarter)
        var textureScore = 0f
        val normalizedList = FloatArray(topQuarter)
        for (i in 0 until topQuarter) {
            val idx = texturedIndices[i]
            textureScore += intVars[idx]
            normalizedList[i] = lapVars[idx] / (intVars[idx] + 1.0f)
        }
        textureScore /= topQuarter
        normalizedList.sort()
        val normalizedBlur = normalizedList[topQuarter / 2] // median of most-textured

        return FrameStats(
            index = index,
            filePath = filePath,
            blurScore = blurScore,
            normalizedBlur = normalizedBlur,
            texture = textureScore,
            meanLuminance = meanLuminance,
            saturatedFraction = saturatedFraction,
            blackFraction = blackFraction,
        )
    }

    /**
     * Compute stats directly from a file path using downscaled BitmapFactory decoding.
     */
    fun computeStatsFromFile(file: File, index: Int): FrameStats {
        val boundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, boundsOpts)
        val origW = boundsOpts.outWidth
        val origH = boundsOpts.outHeight

        val sampleSize = max(1, max(origW, origH) / workResolution)
        val decodeOpts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = BitmapFactory.decodeFile(file.absolutePath, decodeOpts)
            ?: throw IllegalStateException("Could not decode image at ${file.absolutePath}")

        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        bitmap.recycle()

        val gray = ByteArray(w * h)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            // Rec.601 luma
            gray[i] = ((r * 77 + g * 150 + b * 29) shr 8).toByte()
        }

        return computeStats(gray, w, h, index, file.name)
    }

    /**
     * Evaluate a sequence of FrameStats relative to the scene distribution.
     */
    fun evaluate(statsList: List<FrameStats>): QualityGateResult {
        if (statsList.isEmpty()) {
            return QualityGateResult(emptyList(), emptyList(), emptyList(), 0, 0, 0)
        }

        val absValues = statsList.map { it.blurScore }.sorted()
        val normValues = statsList.map { it.normalizedBlur }.sorted()
        val absMedian = max(1e-6f, absValues[absValues.size / 2])
        val normMedian = max(1e-9f, normValues[normValues.size / 2])

        val relativeScores = FloatArray(statsList.size)
        val blurFlags = BooleanArray(statsList.size)
        val exposureFlags = BooleanArray(statsList.size)
        val textureFlags = BooleanArray(statsList.size)
        val reasons = arrayOfNulls<String>(statsList.size)

        for (i in statsList.indices) {
            val s = statsList[i]
            val rel = max(s.blurScore / absMedian, s.normalizedBlur / normMedian)
            relativeScores[i] = rel

            val isBlur = (rel < relativeBlurThreshold) ||
                (minRelativeSharpnessFloor != null && (s.blurScore / absMedian) < minRelativeSharpnessFloor) ||
                (blurThreshold != null && s.blurScore < blurThreshold)
            val isExposure = (s.saturatedFraction > maxSaturatedFraction) ||
                (s.blackFraction > maxBlackFraction) ||
                (s.meanLuminance < darkThreshold) ||
                (s.meanLuminance > blownThreshold)
            val isTexture = s.texture < minTexture

            blurFlags[i] = isBlur
            exposureFlags[i] = isExposure
            textureFlags[i] = isTexture

            if (isBlur) reasons[i] = "blur"
            else if (isExposure) reasons[i] = "exposure"
            else if (isTexture) reasons[i] = "texture"
        }

        // Apply reject cap on blur flags: re-accept sharpest rejected frames if > maxRejectFraction
        val maxRejectCount = (statsList.size * maxRejectFraction).toInt()
        val rejectedBlurIndices = statsList.indices.filter { blurFlags[it] }
            .sortedByDescending { relativeScores[it] }

        if (rejectedBlurIndices.size > maxRejectCount) {
            val toReAccept = rejectedBlurIndices.take(rejectedBlurIndices.size - maxRejectCount)
            for (idx in toReAccept) {
                blurFlags[idx] = false
                if (reasons[idx] == "blur") {
                    reasons[idx] = if (exposureFlags[idx]) "exposure" else if (textureFlags[idx]) "texture" else null
                }
            }
        }

        val acceptedIndices = mutableListOf<Int>()
        val discardedIndices = mutableListOf<Int>()
        val verdicts = mutableListOf<FrameVerdict>()

        var rejBlur = 0
        var rejExposure = 0
        var rejTexture = 0

        for (i in statsList.indices) {
            val isRejected = blurFlags[i] || exposureFlags[i] || textureFlags[i]
            if (isRejected) {
                discardedIndices.add(i)
                when (reasons[i]) {
                    "blur" -> rejBlur++
                    "exposure" -> rejExposure++
                    "texture" -> rejTexture++
                }
            } else {
                acceptedIndices.add(i)
            }
            verdicts.add(
                FrameVerdict(
                    index = statsList[i].index,
                    filePath = statsList[i].filePath,
                    blurScore = statsList[i].blurScore,
                    relativeSharpness = relativeScores[i],
                    accepted = !isRejected,
                    rejectionReason = reasons[i],
                )
            )
        }

        return QualityGateResult(
            acceptedIndices = acceptedIndices,
            discardedIndices = discardedIndices,
            verdicts = verdicts,
            rejectedBlurCount = rejBlur,
            rejectedExposureCount = rejExposure,
            rejectedTextureCount = rejTexture,
        )
    }

    private fun downsampleNearest(src: ByteArray, sw: Int, sh: Int, dw: Int, dh: Int): ByteArray {
        val dst = ByteArray(dw * dh)
        val xRatio = (sw shl 16) / dw
        val yRatio = (sh shl 16) / dh
        var dstOff = 0
        for (y in 0 until dh) {
            val srcY = (y * yRatio) shr 16
            val srcRowOff = srcY * sw
            for (x in 0 until dw) {
                val srcX = (x * xRatio) shr 16
                dst[dstOff++] = src[srcRowOff + srcX]
            }
        }
        return dst
    }
}
