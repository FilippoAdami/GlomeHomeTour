package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Test

class ScanProcessorTest {

    @Test
    fun `rollPose leaves camera center translation unchanged`() {
        val c2w = floatArrayOf(
            1f, 0f, 0f, 2.5f,
            0f, 1f, 0f, -1.2f,
            0f, 0f, 1f, 3.8f,
            0f, 0f, 0f, 1f,
        )
        val rolled = ScanProcessor.rollPose(c2w)
        assertEquals(2.5f, rolled[3], 1e-6f)
        assertEquals(-1.2f, rolled[7], 1e-6f)
        assertEquals(3.8f, rolled[11], 1e-6f)
        assertEquals(1f, rolled[15], 1e-6f)
    }

    @Test
    fun `toPortraitIntrinsics transposes focal lengths and principal point 90 deg clockwise`() {
        val flX = 1000f
        val flY = 1005f
        val cx = 960f
        val cy = 540f
        val w = 1920
        val h = 1080

        val portrait = ScanProcessor.toPortraitIntrinsics(flX, flY, cx, cy, w, h)
        // [flX', flY', cx', cy', w', h', camera_angle_x']
        assertEquals(flY, portrait[0], 1e-6f)
        assertEquals(flX, portrait[1], 1e-6f)
        assertEquals((h - cy), portrait[2], 1e-6f)
        assertEquals(cx, portrait[3], 1e-6f)
        assertEquals(h.toFloat(), portrait[4], 1e-6f)
        assertEquals(w.toFloat(), portrait[5], 1e-6f)
    }
}
