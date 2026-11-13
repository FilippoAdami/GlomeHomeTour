package com.glomehometour.arscan

import android.view.Surface
import org.junit.Assert.assertEquals
import org.junit.Test

class CameraTapFocusTest {
    @Test
    fun `maps portrait preview taps through sensor rotation and center crop`() {
        val left = previewPointToSensorNormalized(
            0f, 0.5f, 1080, 1920, 4000, 3000, 90, Surface.ROTATION_0,
        )
        val right = previewPointToSensorNormalized(
            1f, 0.5f, 1080, 1920, 4000, 3000, 90, Surface.ROTATION_0,
        )

        assertEquals(0.5f, left.first, 1e-4f)
        assertEquals(0.875f, left.second, 1e-4f)
        assertEquals(0.5f, right.first, 1e-4f)
        assertEquals(0.125f, right.second, 1e-4f)
    }

    @Test
    fun `maps invalid viewport dimensions to sensor center`() {
        assertEquals(0.5f to 0.5f, previewPointToSensorNormalized(0f, 1f, 0, 0, 4000, 3000, 90, 0))
    }
}
