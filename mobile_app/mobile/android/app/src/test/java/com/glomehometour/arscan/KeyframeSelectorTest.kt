package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class KeyframeSelectorTest {

    private fun identityMatrix(x: Float = 0f, y: Float = 0f, z: Float = 0f): FloatArray =
        floatArrayOf(
            1f, 0f, 0f, x,
            0f, 1f, 0f, y,
            0f, 0f, 1f, z,
            0f, 0f, 0f, 1f,
        )

    private fun yawMatrix(angleDeg: Float, x: Float = 0f, y: Float = 0f, z: Float = 0f): FloatArray {
        val rad = Math.toRadians(angleDeg.toDouble())
        val c = cos(rad).toFloat()
        val s = sin(rad).toFloat()
        return floatArrayOf(
            c,  0f, s, x,
            0f, 1f, 0f, y,
            -s, 0f, c, z,
            0f, 0f, 0f, 1f,
        )
    }

    @Test
    fun `rotation angle calculation accurately measures yaw`() {
        val m0 = identityMatrix()
        val m90 = yawMatrix(90f)
        val r0 = floatArrayOf(m0[0], m0[1], m0[2], m0[4], m0[5], m0[6], m0[8], m0[9], m0[10])
        val r90 = floatArrayOf(m90[0], m90[1], m90[2], m90[4], m90[5], m90[6], m90[8], m90[9], m90[10])

        val angle = KeyframeSelector.rotationAngleDeg(r0, r90)
        assertEquals(90f, angle, 0.1f)
    }

    @Test
    fun `dilated area of a 4m walk produces expected floor area`() {
        val selector = KeyframeSelector(standoffM = 0.8f, minRoomAreaM2 = 9.0f)
        val path = listOf(
            floatArrayOf(0f, 0f),
            floatArrayOf(2f, 0f),
            floatArrayOf(4f, 0f),
        )
        val area = selector.computeDilatedArea(path, 0.8f)
        // Analytical capsule area: pi * 0.8^2 + 2 * 0.8 * 4.0 = ~8.41 m2
        assertTrue("Dilated area ($area) should be between 7.5 and 9.5 m2", area in 7.5f..9.5f)
    }

    @Test
    fun `redundant stationary frames are decimated`() {
        val selector = KeyframeSelector()
        val frames = mutableListOf<KeyframeSelector.PoseFrame>()
        // 30 stationary frames
        for (i in 0 until 30) {
            frames.add(KeyframeSelector.PoseFrame(i, "frame_$i.jpg", identityMatrix(0f, 0f, 0f), 25.0f))
        }

        val result = selector.selectKeyframes(frames, targetCount = 2)
        // With no motion, gap constraint (max_prune_gap = 6) keeps <= 5 frames out of 30
        assertTrue("Expected 30 frames to be decimated to <= 5", result.selectedCount <= 5)
        assertEquals(0, result.selectedIndices.first())
        assertEquals(29, result.selectedIndices.last())
    }

    @Test
    fun `linear walk selects well-spaced keyframes`() {
        val selector = KeyframeSelector(minTranslationM = 0.20f)
        val frames = mutableListOf<KeyframeSelector.PoseFrame>()
        // 50 frames advancing 5 cm per step (total 2.5 m)
        for (i in 0 until 50) {
            frames.add(KeyframeSelector.PoseFrame(i, "frame_$i.jpg", identityMatrix(x = i * 0.05f), 25.0f))
        }

        val result = selector.selectKeyframes(frames, targetCount = 15)
        assertTrue("Selected count should be at most target", result.selectedCount <= 15)
        assertTrue("Selected count should be at least 10", result.selectedCount >= 10)
    }

    @Test
    fun `neighbor swapping prefers sharper temporal neighbors`() {
        val selector = KeyframeSelector(minTranslationM = 0.30f)
        val frames = mutableListOf<KeyframeSelector.PoseFrame>()
        // Frame 0: at 0.0m, sharp
        frames.add(KeyframeSelector.PoseFrame(0, "f0.jpg", identityMatrix(0f, 0f, 0f), 30.0f))
        // Frame 1: at 0.35m, blurry (sharpness = 12.0)
        frames.add(KeyframeSelector.PoseFrame(1, "f1.jpg", identityMatrix(0.35f, 0f, 0f), 12.0f))
        // Frame 2: at 0.37m (2 cm away from frame 1), much sharper (sharpness = 35.0)
        frames.add(KeyframeSelector.PoseFrame(2, "f2.jpg", identityMatrix(0.37f, 0f, 0f), 35.0f))
        // Frame 3: at 0.80m, sharp
        frames.add(KeyframeSelector.PoseFrame(3, "f3.jpg", identityMatrix(0.80f, 0f, 0f), 30.0f))

        val result = selector.selectKeyframes(frames)
        assertTrue("Should swap candidate frame 1 for sharper neighbor frame 2",
            result.selectedIndices.contains(2) || result.sharpnessSwappedCount > 0)
    }
}
