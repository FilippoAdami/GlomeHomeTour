package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The settings screen writes straight into the capture loop's thresholds, so the two things that
 * silently ruin a scan are checked here: that an untouched install still behaves like the
 * hardcoded constants, and that a bad typed value is clamped instead of stored.
 */
class TunablesTest {

    @Test
    fun `defaults match the hardcoded constants`() {
        assertEquals(VoxelGrid.PARALLAX_MIN_DEG, Tunables.parallaxMinDeg, 1e-6f)
        assertEquals(VoxelGrid.PARALLAX_COS, Tunables.parallaxCos, 1e-6f)
        assertEquals(CaptureActivity.COMPLETION_FRACTION, Tunables.coverageCompleteFraction, 1e-6f)
        assertEquals(PhotometricGate.MEAN_MIN, Tunables.photometricMeanMin, 1e-6f)
        assertEquals(PhotometricGate.MEAN_MAX, Tunables.photometricMeanMax, 1e-6f)
        assertEquals(CaptureActivity.DECIMATION_STRIDE, Tunables.decimationStride)
    }

    /** VoxelGrid compares cosines in its hot loop; a stale cos means the angle setting does nothing. */
    @Test
    fun `parallax cosine tracks the angle`() {
        try {
            Tunables.parallaxMinDeg = 40f
            assertEquals(
                kotlin.math.cos(Math.toRadians(40.0)).toFloat(), Tunables.parallaxCos, 1e-6f,
            )
        } finally {
            Tunables.parallaxMinDeg = VoxelGrid.PARALLAX_MIN_DEG
        }
    }

    @Test
    fun `out of range input is clamped`() {
        val v = Tunables.Values.clamped(
            parallaxDeg = 900f, coveragePercent = 0f, lumaMin = -5f, lumaMax = 2500f, stride = 0L,
        )
        assertEquals(60f, v.parallaxMinDeg, 1e-6f)
        assertEquals(0.1f, v.coverageCompleteFraction, 1e-6f)
        assertEquals(1f, v.photometricMeanMin, 1e-6f)
        assertEquals(255f, v.photometricMeanMax, 1e-6f)
        assertEquals(1L, v.decimationStride)
    }

    /** An inverted gate ([min] above [max]) rejects every frame of the scan. */
    @Test
    fun `luma gate cannot be inverted`() {
        val v = Tunables.Values.clamped(
            parallaxDeg = 25f, coveragePercent = 85f, lumaMin = 200f, lumaMax = 30f, stride = 8L,
        )
        assertTrue(v.photometricMeanMax > v.photometricMeanMin)
    }

    @Test
    fun `in range input is kept`() {
        val v = Tunables.Values.clamped(
            parallaxDeg = 30f, coveragePercent = 75f, lumaMin = 20f, lumaMax = 240f, stride = 6L,
        )
        assertEquals(30f, v.parallaxMinDeg, 1e-6f)
        assertEquals(0.75f, v.coverageCompleteFraction, 1e-6f)
        assertEquals(20f, v.photometricMeanMin, 1e-6f)
        assertEquals(240f, v.photometricMeanMax, 1e-6f)
        assertEquals(6L, v.decimationStride)
    }
}
