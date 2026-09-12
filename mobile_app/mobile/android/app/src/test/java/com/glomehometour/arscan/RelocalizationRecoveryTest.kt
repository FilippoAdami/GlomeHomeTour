package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RelocalizationRecoveryTest {

    @Test
    fun `solve3PointRigid accurately solves known translation and rotation`() {
        val src = arrayOf(
            floatArrayOf(0.0f, 0.0f, 0.0f),
            floatArrayOf(1.0f, 0.0f, 0.0f),
            floatArrayOf(0.0f, 1.0f, 0.0f),
        )

        // Rotate 90 deg around Z (+X -> +Y, +Y -> -X) and translate by (0.5, -0.3, 0.2)
        val dst = arrayOf(
            floatArrayOf(0.5f, -0.3f, 0.2f),
            floatArrayOf(0.5f, 0.7f, 0.2f),
            floatArrayOf(-0.5f, -0.3f, 0.2f),
        )

        val transform = RelocalizationRecovery.solve3PointRigid(src, dst)
        assertNotNull(transform)
        transform!!

        val out = FloatArray(3)
        for (i in 0..2) {
            transform.transformPoint(src[i][0], src[i][1], src[i][2], out)
            assertEquals(dst[i][0], out[0], 1e-4f)
            assertEquals(dst[i][1], out[1], 1e-4f)
            assertEquals(dst[i][2], out[2], 1e-4f)
        }
    }

    @Test
    fun `estimateAlignment detects rigid offset between landmark cloud and newly seen features`() {
        val lx = floatArrayOf(0.0f, 0.4f, 0.8f, -0.3f, 0.2f, 0.6f, -0.1f)
        val ly = floatArrayOf(0.1f, 0.0f, 0.2f, -0.1f, 0.4f, -0.2f, 0.3f)
        val lz = floatArrayOf(1.5f, 1.8f, 1.6f,  1.7f, 1.9f,  1.4f, 2.0f)
        val lCount = lx.size

        // Shift all points by delta = (+0.12m, -0.06m, +0.08m)
        val dx = 0.12f; val dy = -0.06f; val dz = 0.08f
        val currentPoints = FloatArray(lCount * 4)
        for (i in 0 until lCount) {
            currentPoints[i * 4] = lx[i] + dx
            currentPoints[i * 4 + 1] = ly[i] + dy
            currentPoints[i * 4 + 2] = lz[i] + dz
            currentPoints[i * 4 + 3] = 0.9f
        }

        val transform = RelocalizationRecovery.estimateAlignment(
            currentPoints, lCount,
            0f, 0f, 0f,
            lx, ly, lz, lCount,
        )

        assertNotNull("Expected alignment to succeed", transform)
        transform!!
        assertTrue(transform.isSignificant)
        assertTrue(transform.inliers >= 6)
        assertEquals(dx, transform.tx, 0.05f)
        assertEquals(dy, transform.ty, 0.05f)
        assertEquals(dz, transform.tz, 0.05f)
    }
}
