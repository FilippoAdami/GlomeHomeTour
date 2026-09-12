package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MainActivityExposureTest {

    @Test
    fun `dark indoor scene increases exposure without exceeding limits`() {
        val initialIso = 100
        val initialShutter = CameraPipeline.FASTEST_SHUTTER_NS
        val darkLuma = 20.0f

        val exposure = MainActivity.meteredExposure(darkLuma, initialIso, initialShutter)

        assertTrue(exposure.shutterNs > initialShutter)
        assertTrue(exposure.shutterNs <= CameraPipeline.SLOWEST_SHUTTER_NS)
    }

    @Test
    fun `bright daylight scene reduces ISO and holds fastest shutter`() {
        val initialIso = 400
        val initialShutter = CameraPipeline.FASTEST_SHUTTER_NS
        val brightLuma = 220.0f

        val exposure = MainActivity.meteredExposure(brightLuma, initialIso, initialShutter)

        assertTrue(exposure.iso < initialIso)
        assertEquals(CameraPipeline.FASTEST_SHUTTER_NS, exposure.shutterNs)
    }

    @Test
    fun `white balance gains normalize color casts`() {
        // Warm yellowish room: Y=128, U=110 (low blue, cb=-18), V=145 (high red, cr=+17)
        val gains = MainActivity.whiteBalanceGains(meanY = 128f, meanU = 110f, meanV = 145f)
        // Red gain should be < 1.0 (to attenuate red), Blue gain should be > 1.0 (to boost blue)
        assertTrue(gains[0] < gains[3])
    }
}
