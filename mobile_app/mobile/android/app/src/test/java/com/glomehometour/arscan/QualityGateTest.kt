package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class QualityGateTest {

    private val gate = QualityGate(
        relativeBlurThreshold = 0.35f,
        minRelativeSharpnessFloor = 0.28f,
        maxRejectFraction = 0.20f,
    )

    private fun createSyntheticFrame(width: Int, height: Int, pattern: String): ByteArray {
        val arr = ByteArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val v = when (pattern) {
                    "sharp" -> if ((x / 4 + y / 4) % 2 == 0) 220 else 30
                    "blurry" -> (128 + 20 * kotlin.math.sin(x * 0.05)).toInt().coerceIn(0, 255)
                    "saturated" -> 255
                    "black" -> 0
                    else -> 128
                }
                arr[y * width + x] = v.toByte()
            }
        }
        return arr
    }

    @Test
    fun `sharp pattern has substantially higher Laplacian variance than blurry pattern`() {
        val w = 64
        val h = 64
        val sharp = createSyntheticFrame(w, h, "sharp")
        val blurry = createSyntheticFrame(w, h, "blurry")

        val sSharp = gate.computeStats(sharp, w, h, 0, "sharp.jpg")
        val sBlurry = gate.computeStats(blurry, w, h, 1, "blurry.jpg")

        assertTrue("sharp blurScore (${sSharp.blurScore}) should exceed blurry (${sBlurry.blurScore})",
            sSharp.blurScore > sBlurry.blurScore * 5f)
    }

    @Test
    fun `scene evaluation rejects blurry frame among sharp frames`() {
        val w = 64
        val h = 64
        val statsList = mutableListOf<QualityGate.FrameStats>()
        // 9 sharp frames and 1 blurry frame
        for (i in 0 until 9) {
            statsList.add(gate.computeStats(createSyntheticFrame(w, h, "sharp"), w, h, i, "frame_$i.jpg"))
        }
        statsList.add(gate.computeStats(createSyntheticFrame(w, h, "blurry"), w, h, 9, "frame_9.jpg"))

        val result = gate.evaluate(statsList)
        assertEquals(9, result.acceptedIndices.size)
        assertEquals(listOf(9), result.discardedIndices)
        assertEquals(1, result.rejectedBlurCount)
        assertFalse(result.verdicts[9].accepted)
        assertEquals("blur", result.verdicts[9].rejectionReason)
    }

    @Test
    fun `saturated and crushed black frames are rejected on exposure`() {
        val w = 64
        val h = 64
        val sat = gate.computeStats(createSyntheticFrame(w, h, "saturated"), w, h, 0, "sat.jpg")
        val blk = gate.computeStats(createSyntheticFrame(w, h, "black"), w, h, 1, "blk.jpg")

        val result = gate.evaluate(listOf(sat, blk))
        assertEquals(0, result.acceptedIndices.size)
        assertEquals(2, result.rejectedExposureCount)
    }

    @Test
    fun `reject cap prevents culling entire scan`() {
        val w = 64
        val h = 64
        val statsList = mutableListOf<QualityGate.FrameStats>()
        // 6 sharp frames
        for (i in 0 until 6) {
            statsList.add(gate.computeStats(createSyntheticFrame(w, h, "sharp"), w, h, i, "sharp_$i.jpg"))
        }
        // 4 blurry frames (40% of scan)
        for (i in 6 until 10) {
            val stats = gate.computeStats(createSyntheticFrame(w, h, "blurry"), w, h, i, "blur_$i.jpg")
            statsList.add(stats.copy(texture = 100f))
        }

        val result = gate.evaluate(statsList)
        // With maxRejectFraction = 0.20, at most 2 frames may be rejected for blur (20% of 10)
        assertTrue("At least 8 frames should be preserved by reject cap", result.acceptedIndices.size >= 8)
        assertTrue("At most 2 frames should be discarded", result.discardedIndices.size <= 2)
    }
}
