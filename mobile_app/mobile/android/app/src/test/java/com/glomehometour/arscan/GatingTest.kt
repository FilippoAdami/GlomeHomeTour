package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** The photometric gate (projet.md §2.2) and the luma sampling that feeds it. */
class GatingTest {

    @Test
    fun `mean luma is sampled correctly across row padding`() {
        // Padded rows are what a real camera hands back; sampling must skip the padding, not
        // average it in. Padding filled with 0 so a broken stride shows up as a dark mean.
        val width = 64
        val height = 64
        val rowStride = 96
        val buf = ByteArray(rowStride * height)
        for (row in 0 until height) for (col in 0 until width) buf[row * rowStride + col] = 200.toByte()
        assertEquals(200f, Luma.mean(buf, width, height, rowStride, 1, 8), 0.01f)
    }

    @Test
    fun `mean luma handles interleaved pixel strides`() {
        val width = 32
        val height = 32
        val pixelStride = 2
        val buf = ByteArray(width * height * pixelStride)
        for (i in buf.indices) buf[i] = if (i % 2 == 0) 100.toByte() else 255.toByte()
        assertEquals(100f, Luma.mean(buf, width, height, width * pixelStride, pixelStride, 4), 0.01f)
    }

    @Test
    fun `dark frames are rejected and bright ones accepted`() {
        val gate = PhotometricGate()
        assertEquals(PhotometricGate.Verdict.DARK, gate.offer(10f))
        assertEquals(PhotometricGate.Verdict.OK, gate.offer(120f).let {
            // The 100-unit jump also arms a transition, so settle first, then re-check.
            repeat(PhotometricGate.WINDOW) { _ -> gate.offer(120f) }
            gate.offer(120f)
        })
        assertEquals(PhotometricGate.Verdict.BLOWN, gate.offer(254f))
    }

    /** TC-03: a light switched on mid-scan. The jump *and* the auto-exposure hunt after it must
     * be rejected, and the gate must reopen on its own once the room is stable. */
    @Test
    fun `lighting transition is rejected until the window settles`() {
        val gate = PhotometricGate()
        repeat(PhotometricGate.WINDOW * 2) { gate.offer(60f) }

        assertEquals(PhotometricGate.Verdict.TRANSITION, gate.offer(160f)) // light switched on
        // Auto-exposure hunting: still moving around, variance stays high.
        for (v in listOf(150f, 168f, 155f, 172f, 149f, 165f)) {
            assertEquals(PhotometricGate.Verdict.TRANSITION, gate.offer(v))
        }
        // Settled: a full quiet window is required before the gate reopens.
        var verdict = PhotometricGate.Verdict.TRANSITION
        repeat(PhotometricGate.WINDOW + 2) { verdict = gate.offer(160f) }
        assertEquals(PhotometricGate.Verdict.OK, verdict)
        assertTrue(gate.variance < PhotometricGate.VARIANCE_MAX)
    }

    @Test
    fun `small illumination drift does not trip the gate`() {
        val gate = PhotometricGate()
        var verdict = PhotometricGate.Verdict.OK
        for (v in listOf(100f, 101f, 100f, 99f, 100f, 102f, 101f, 100f, 99f, 100f, 101f, 100f)) {
            verdict = gate.offer(v)
        }
        assertEquals(PhotometricGate.Verdict.OK, verdict)
    }

    // ---- keyframe / kinematic helpers (MainActivity companion) ----

    @Test
    fun `quaternion angle is symmetric and sign-independent`() {
        val identity = floatArrayOf(0f, 0f, 0f, 1f)
        assertEquals(0f, MainActivity.quaternionAngleDeg(identity, identity), 1e-3f)
        // Same rotation, negated representation: must read as 0 degrees apart, not 180.
        assertEquals(0f, MainActivity.quaternionAngleDeg(identity, floatArrayOf(0f, 0f, 0f, -1f)), 1e-3f)

        val yaw90 = floatArrayOf(0f, kotlin.math.sin(Math.PI / 4).toFloat(), 0f, kotlin.math.cos(Math.PI / 4).toFloat())
        assertEquals(90f, MainActivity.quaternionAngleDeg(identity, yaw90), 1e-2f)
        assertEquals(90f, MainActivity.quaternionAngleDeg(yaw90, identity), 1e-2f)
    }

    @Test
    fun `distance is euclidean`() {
        assertEquals(
            5f,
            MainActivity.distance(floatArrayOf(1f, 2f, 3f), floatArrayOf(4f, 6f, 3f)),
            1e-5f,
        )
    }

    // ---- keyframe decimation (README §4/§5): fixed-ratio subsample of the gate-passed stream,
    // not a displacement/angle filter -- see Phase_1.md §10 item 2.

    @Test
    fun `decimation keeps exactly one frame per stride`() {
        val stride = 8L
        val kept = (1L..stride * 5).count { MainActivity.isDecimationKeyframe(it, stride) }
        assertEquals(5, kept)
    }

    @Test
    fun `decimation keeps every Nth frame regardless of motion`() {
        // Unlike the displacement-gated policy this replaced, decimation has no notion of
        // "hasn't moved" -- a stationary stream still yields a keyframe every `stride` frames.
        val stride = 6L
        assertFalse(MainActivity.isDecimationKeyframe(1L, stride))
        assertFalse(MainActivity.isDecimationKeyframe(5L, stride))
        assertTrue(MainActivity.isDecimationKeyframe(6L, stride))
        assertTrue(MainActivity.isDecimationKeyframe(12L, stride))
    }

    @Test
    fun `default stride is within README's 6-10 range`() {
        assertTrue(MainActivity.DECIMATION_STRIDE in 6L..10L)
    }

    @Test
    fun `tilt is zero for a level phone and positive looking down`() {
        // The argument is the camera forward's world-Y, which updateUi feeds as -gravityUpZ:
        // phone upright (up_z = 0) is level, phone flat on its back (up_z = +1) is straight down.
        assertEquals(0f, MainActivity.tiltDownDeg(0f), 1e-4f)
        assertEquals(90f, MainActivity.tiltDownDeg(-1f), 1e-3f)
        assertEquals(-90f, MainActivity.tiltDownDeg(1f), 1e-3f)
        // README §6 step 3's 30-45 deg guidance band, roughly: a forward vector tipped about
        // halfway to straight down should land in the middle of it.
        val fortyFiveDown = kotlin.math.sin(Math.toRadians(-45.0)).toFloat()
        assertEquals(45f, MainActivity.tiltDownDeg(fortyFiveDown), 0.1f)
    }

    @Test
    fun `white balance gains are neutral for a neutral grey wall`() {
        val gains = MainActivity.whiteBalanceGains(128f, 128f, 128f)
        assertEquals(1f, gains[0], 1e-6f) // red
        assertEquals(1f, gains[1], 1e-6f) // green
        assertEquals(1f, gains[3], 1e-6f) // blue
    }

    @Test
    fun `white balance gains correct a blue or red cast`() {
        // U (Cb) above 128: image reads too blue, so the blue gain must come down to compensate.
        // Red is decoded purely from Cr, so a Cb-only cast leaves it untouched.
        val blueCast = MainActivity.whiteBalanceGains(128f, 180f, 128f)
        assertTrue(blueCast[3] < 1f)
        assertEquals(1f, blueCast[0], 1e-6f)

        // V (Cr) above 128: image reads too red/warm, so the red gain must come down. Blue is
        // decoded purely from Cb, so a Cr-only cast leaves it untouched.
        val redCast = MainActivity.whiteBalanceGains(128f, 128f, 180f)
        assertTrue(redCast[0] < 1f)
        assertEquals(1f, redCast[3], 1e-6f)
    }

    @Test
    fun `white balance gains correct a green cast that the old red-blue-only heuristic could not reach`() {
        // A patch reading greener than its own luma (green channel above the Y/Cb/Cr-decoded
        // target) is exactly the cast a red/blue-only gain vector can never cancel, since it
        // pinned green's gain to a fixed 1.0 regardless of the sample. The fixed decode-and-solve
        // approach has to pull green's own gain down (and, since it's still solving for a neutral
        // patch, push red/blue up) to compensate.
        val greenCast = MainActivity.whiteBalanceGains(128f, 100f, 110f)
        assertTrue("expected green gain to drop below 1, was ${greenCast[1]}", greenCast[1] < 1f)
        assertTrue(greenCast[0] > 1f) // red
        assertTrue(greenCast[3] > 1f) // blue
    }

    /** The ISP's AWB gains arrive as an RggbChannelVector (two greens); the trim works in RGB, so
     * the two have to line up channel for channel -- getting this wrong swaps red and blue, which
     * looks like a wildly miscalibrated scene rather than an indexing slip. */
    @Test
    fun `white balance trim keeps the ISP gain order`() {
        val frozen = floatArrayOf(1.9f, 1f, 1.4f) // a typical warm-light AWB estimate
        val gains = MainActivity.whiteBalanceTrim(frozen)
        assertEquals(1.9f, gains[0], 1e-6f) // red
        assertEquals(1f, gains[1], 1e-6f)   // greenEven
        assertEquals(1f, gains[2], 1e-6f)   // greenOdd
        assertEquals(1.4f, gains[3], 1e-6f) // blue
    }

    @Test
    fun `white balance gains stay within a sane clamp for extreme chroma`() {
        val gains = MainActivity.whiteBalanceGains(128f, 255f, 0f)
        for (g in gains) assertTrue("$g out of range", g in 0.5f..4f)
    }

    @Test
    fun `white balance warm-cool bias trims red and blue in opposite directions`() {
        val neutral = MainActivity.whiteBalanceGains(128f, 128f, 128f, warmCoolBias = 0f)
        assertEquals(1f, neutral[0], 1e-6f)
        assertEquals(1f, neutral[3], 1e-6f)

        // Positive bias (slider pushed warm): red up, blue down.
        val warm = MainActivity.whiteBalanceGains(128f, 128f, 128f, warmCoolBias = 1f)
        assertTrue(warm[0] > 1f)
        assertTrue(warm[3] < 1f)

        // Negative bias (slider pushed cool): red down, blue up.
        val cool = MainActivity.whiteBalanceGains(128f, 128f, 128f, warmCoolBias = -1f)
        assertTrue(cool[0] < 1f)
        assertTrue(cool[3] > 1f)
    }

    @Test
    fun `white balance tint bias trims only green, independent of warm-cool`() {
        val neutral = MainActivity.whiteBalanceGains(128f, 128f, 128f, tintBias = 0f)
        assertEquals(1f, neutral[1], 1e-6f)

        // Positive bias (slider pushed magenta): green comes down.
        val magenta = MainActivity.whiteBalanceGains(128f, 128f, 128f, tintBias = 1f)
        assertTrue(magenta[1] < 1f)
        assertEquals(1f, magenta[0], 1e-6f) // red untouched by tint
        assertEquals(1f, magenta[3], 1e-6f) // blue untouched by tint

        // Negative bias (slider pushed green): green goes up.
        val green = MainActivity.whiteBalanceGains(128f, 128f, 128f, tintBias = -1f)
        assertTrue(green[1] > 1f)
    }

    @Test
    fun `validation passes a full-length scan with room in the grid`() {
        assertTrue(MainActivity.validationIssues(gridFull = false, keyframeCount = 200).isEmpty())
    }

    @Test
    fun `validation flags a scan ended seconds in`() {
        val issues = MainActivity.validationIssues(gridFull = false, keyframeCount = 3)
        assertEquals(1, issues.size)
    }

    @Test
    fun `validation flags a full grid even with plenty of frames`() {
        val issues = MainActivity.validationIssues(gridFull = true, keyframeCount = 200)
        assertEquals(1, issues.size)
    }

    @Test
    fun `validation can report both issues at once`() {
        val issues = MainActivity.validationIssues(gridFull = true, keyframeCount = 1)
        assertEquals(2, issues.size)
    }

    // ---- ambient-light shutter+ISO metering (README §4: joint tradeoff curve) ----

    @Test
    fun `metered exposure holds steady once luma is already on target`() {
        val exposure = MainActivity.meteredExposure(175f, 100, CameraPipeline.FASTEST_SHUTTER_NS)
        assertEquals(100, exposure.iso)
        assertEquals(CameraPipeline.FASTEST_SHUTTER_NS, exposure.shutterNs)
    }

    @Test
    fun `metered exposure slows shutter before raising iso for a dark indoor scene`() {
        // A near-black frame like a dim indoor room at the default fast shutter: the meter should
        // spend shutter range first (fast toward slow), only raising ISO once shutter alone isn't
        // enough -- shutter is the "free" knob, ISO is the one that costs grain.
        val exposure = MainActivity.meteredExposure(
            measuredLuma = 30f, currentIso = 100, currentShutterNs = CameraPipeline.FASTEST_SHUTTER_NS,
        )
        assertTrue(exposure.shutterNs > CameraPipeline.FASTEST_SHUTTER_NS)
    }

    @Test
    fun `metered exposure raises iso once shutter has bottomed out and luma is still low`() {
        // Already at the slowest shutter and still dark: only ISO can move further.
        val exposure = MainActivity.meteredExposure(
            measuredLuma = 8f, currentIso = 100, currentShutterNs = CameraPipeline.SLOWEST_SHUTTER_NS,
        )
        assertTrue(exposure.iso > 100)
        assertEquals(CameraPipeline.SLOWEST_SHUTTER_NS, exposure.shutterNs)
    }

    @Test
    fun `metered exposure falls for a bright outdoor scene`() {
        val exposure = MainActivity.meteredExposure(
            measuredLuma = 220f, currentIso = 400, currentShutterNs = CameraPipeline.FASTEST_SHUTTER_NS,
        )
        assertTrue(exposure.iso < 400)
        assertEquals(CameraPipeline.FASTEST_SHUTTER_NS, exposure.shutterNs)
    }

    @Test
    fun `metered exposure never exceeds the app iso ceiling or slowest shutter even for a pitch black frame`() {
        val exposure = MainActivity.meteredExposure(
            measuredLuma = 1f, currentIso = 100, currentShutterNs = CameraPipeline.FASTEST_SHUTTER_NS,
        )
        assertTrue(exposure.iso <= CameraPipeline.MAX_ISO)
        assertTrue(exposure.shutterNs <= CameraPipeline.SLOWEST_SHUTTER_NS)
    }

    @Test
    fun `metered exposure never drops below the app iso floor or fastest shutter for a blown-out frame`() {
        val exposure = MainActivity.meteredExposure(
            measuredLuma = 255f, currentIso = 50, currentShutterNs = CameraPipeline.FASTEST_SHUTTER_NS,
        )
        assertTrue(exposure.iso >= CameraPipeline.MIN_ISO)
        assertTrue(exposure.shutterNs >= CameraPipeline.FASTEST_SHUTTER_NS)
    }

    @Test
    fun `metered exposure treats a zero-luma reading as clipped black, not as no data`() {
        // A fully black frame at the default outdoor-bright starting point (iso 100, 1/500s) is
        // exactly the state a dim indoor room starts in -- if this froze instead of pushing
        // exposure up, it could never leave the default and would stay black forever.
        val exposure = MainActivity.meteredExposure(
            measuredLuma = 0f, currentIso = 100, currentShutterNs = CameraPipeline.FASTEST_SHUTTER_NS,
        )
        assertTrue(exposure.shutterNs > CameraPipeline.FASTEST_SHUTTER_NS || exposure.iso > 100)
    }

    @Test
    fun `tracking failures map to an instruction, never to ARCore's raw enum`() {
        // What the operator is shown has to be something they can do with the phone in their hand;
        // "PAUSED/INSUFFICIENT_FEATURES" is not that.
        assertEquals(
            "Point at furniture or a door frame, not a blank wall",
            MainActivity.trackingAdvice("PAUSED/INSUFFICIENT_FEATURES"),
        )
        assertEquals("Move the phone more slowly", MainActivity.trackingAdvice("PAUSED/EXCESSIVE_MOTION"))
        assertEquals("Turn on the room lights", MainActivity.trackingAdvice("PAUSED/INSUFFICIENT_LIGHT"))
        assertEquals("Tracking", MainActivity.trackingAdvice("TRACKING"))
        // Unknown reason and not-yet-started both still say something actionable.
        assertFalse(MainActivity.trackingAdvice("PAUSED/BAD_STATE").isEmpty())
        assertFalse(MainActivity.trackingAdvice(null).isEmpty())
    }

    @Test
    fun `tilt gauge target band is the 5-20 degree range for environment acquisition`() {
        assertEquals(5f, TiltGaugeView.BAND_MIN_DEG, 0.01f)
        assertEquals(20f, TiltGaugeView.BAND_MAX_DEG, 0.01f)
        // The gauge scale has to contain the band it draws, or the green zone falls off the track.
        assertTrue(TiltGaugeView.MAX_DEG > TiltGaugeView.BAND_MAX_DEG)
    }
}
