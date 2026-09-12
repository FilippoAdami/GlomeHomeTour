package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The monocular depth fallback: crop mapping, metric fit, and the depth image it produces. */
class MonoDepthTest {

    private val w = 1920
    private val h = 1080
    private val size = 256

    @Test
    fun `crop mapping round-trips in every orientation`() {
        val model = FloatArray(2)
        val sensor = FloatArray(2)
        for (rot in listOf(0, 90, 180, 270)) {
            for (mx in listOf(0.5f, 37.5f, 128.5f, 255.5f)) {
                for (my in listOf(0.5f, 100.5f, 255.5f)) {
                    ModelCrop.toSensor(mx, my, w, h, rot, size, sensor)
                    assertTrue("u in frame", sensor[0] >= 0f && sensor[0] <= w)
                    assertTrue("v in frame", sensor[1] >= 0f && sensor[1] <= h)
                    assertTrue(ModelCrop.toModel(sensor[0], sensor[1], w, h, rot, size, model))
                    assertEquals("rot $rot mx", mx, model[0], 1e-2f)
                    assertEquals("rot $rot my", my, model[1], 1e-2f)
                }
            }
        }
    }

    @Test
    fun `pixels outside the square crop are rejected`() {
        val model = FloatArray(2)
        // Portrait capture: the sensor's left and right edges fall outside the centre square.
        assertFalse(ModelCrop.toModel(5f, 540f, w, h, 90, size, model))
        assertFalse(ModelCrop.toModel(1915f, 540f, w, h, 90, size, model))
        assertTrue(ModelCrop.toModel(960f, 540f, w, h, 90, size, model))
        // ...and the centre of the sensor is the centre of the model input.
        assertEquals(size / 2f, model[0], 1f)
        assertEquals(size / 2f, model[1], 1f)
    }

    @Test
    fun `crop mapping is upright, not mirrored`() {
        val sensor = FloatArray(2)
        // The top of the upright model image must come from the sensor's +u edge on a phone whose
        // sensor is a quarter turn from the display; mirroring here would flip the whole map.
        ModelCrop.toSensor(128f, 0f, w, h, 90, size, sensor)
        val top = sensor[0]
        ModelCrop.toSensor(128f, 255f, w, h, 90, size, sensor)
        assertTrue("model row 0 must map above model row 255", top < sensor[0])
    }

    @Test
    fun `affine stepping agrees with the per-pixel crop mapping`() {
        // YuvCrop steps the sensor coordinate instead of calling toSensor 65k times a frame; if
        // the two disagree the model is fed a sheared or offset image and every depth is wrong.
        val map = FloatArray(6)
        val sensor = FloatArray(2)
        for (rot in listOf(0, 90, 180, 270)) {
            ModelCrop.affine(w, h, rot, size, map)
            for (my in listOf(0, 1, 91, 255)) {
                for (mx in listOf(0, 1, 137, 255)) {
                    ModelCrop.toSensor(mx + 0.5f, my + 0.5f, w, h, rot, size, sensor)
                    assertEquals("rot $rot u", sensor[0], map[0] + mx * map[2] + my * map[4], 1e-2f)
                    assertEquals("rot $rot v", sensor[1], map[1] + mx * map[3] + my * map[5], 1e-2f)
                }
            }
        }
    }

    // ---- metric fit ----

    /** Builds a disparity image consistent with disparity = alpha/distance + beta for a scene of
     * ARCore points, then checks the fit recovers alpha and beta. */
    private fun syntheticFit(alpha: Float, beta: Float, count: Int): FloatArray? {
        val fx = 1390f
        val fy = 1390f
        val cx = w / 2f
        val cy = h / 2f
        val disparity = FloatArray(size * size)
        val points = FloatArray(count * 4)
        val model = FloatArray(2)
        var n = 0
        var i = 0
        while (n < count && i < count * 4) {
            // Points spread across the crop at a range of distances, camera at the origin
            // looking down -Z (identity pose), so world == camera space.
            val distance = 1f + (n % 5) * 0.7f
            val u = cx + ((n % 7) - 3) * 90f
            val v = cy + ((n / 7) % 5 - 2) * 90f
            if (ModelCrop.toModel(u, v, w, h, 90, size, model)) {
                val mx = model[0].toInt()
                val my = model[1].toInt()
                disparity[my * size + mx] = alpha / distance + beta
                points[n * 4] = (u - cx) / fx * distance
                points[n * 4 + 1] = -(v - cy) / fy * distance
                points[n * 4 + 2] = -distance
                points[n * 4 + 3] = 0.9f
                n++
            }
            i++
        }
        return DepthScale.fit(
            disparity, size, floatArrayOf(0f, 0f, 0f), floatArrayOf(0f, 0f, 0f, 1f),
            points, n, fx, fy, cx, cy, w, h, 90,
        )
    }

    @Test
    fun `fit recovers the scale that generated the disparity`() {
        val fit = syntheticFit(alpha = 3.5f, beta = 0.2f, count = 20)
        assertNotNull(fit)
        assertEquals(3.5f, fit!![0], 1e-2f)
        assertEquals(0.2f, fit[1], 1e-2f)
    }

    @Test
    fun `fit refuses to guess from too few points`() {
        assertNull(syntheticFit(alpha = 3.5f, beta = 0.2f, count = DepthScale.MIN_POINTS - 1))
    }

    @Test
    fun `fit rejects low-confidence points`() {
        val points = FloatArray(40 * 4)
        for (i in 0 until 40) {
            points[i * 4 + 2] = -2f
            points[i * 4 + 3] = 0.05f // below the confidence floor
        }
        assertNull(
            DepthScale.fit(
                FloatArray(size * size), size, floatArrayOf(0f, 0f, 0f), floatArrayOf(0f, 0f, 0f, 1f),
                points, 40, 1390f, 1390f, w / 2f, h / 2f, w, h, 90,
            )
        )
    }

    // ---- depth image ----

    @Test
    fun `depth frame is metric, sensor-aligned, and unknown outside the crop`() {
        val alpha = 4f
        val beta = 0.5f
        val disparity = FloatArray(size * size) { alpha / 2f + beta } // a wall at exactly 2 m
        val fx = 1390f
        val frame = DepthScale.toDepthFrame(
            disparity, size, alpha, beta, w, h, 90, fx, fx, w / 2f, h / 2f,
            MonoDepthWorker.DEPTH_WIDTH, MonoDepthWorker.DEPTH_HEIGHT,
        )

        assertEquals(MonoDepthWorker.DEPTH_WIDTH, frame.width)
        assertEquals(MonoDepthWorker.DEPTH_HEIGHT, frame.height)
        // Intrinsics must be rescaled to the depth image, or every ray leaves at the wrong angle.
        assertEquals(fx * frame.width / w, frame.fx, 1e-3f)
        assertEquals(frame.width / 2f, frame.cx, 1e-3f)

        val centre = frame.millimetres[frame.height / 2 * frame.width + frame.width / 2].toInt()
        assertEquals(2000, centre)
        // Left edge of a landscape sensor is outside the portrait square crop: unknown, not zero
        // distance, and CoverageWorker treats 0 as "no sample".
        assertEquals(0, frame.millimetres[frame.height / 2 * frame.width].toInt())
    }

    @Test
    fun `distances outside the trusted band are dropped rather than clamped`() {
        val alpha = 4f
        val beta = 0f
        val far = FloatArray(size * size) { alpha / 40f } // 40 m: beyond MAX_RANGE_M
        val frame = DepthScale.toDepthFrame(
            far, size, alpha, beta, w, h, 90, 1390f, 1390f, w / 2f, h / 2f, 64, 36,
        )
        assertTrue(frame.millimetres.all { it.toInt() == 0 })
    }

    // ---- kinematics, the reason TURN SLOWER was stuck on ----

    @Test
    fun `angle between forward vectors is small for small turns`() {
        val a = floatArrayOf(0f, 0f, -1f)
        val b = floatArrayOf(kotlin.math.sin(0.01).toFloat(), 0f, -kotlin.math.cos(0.01).toFloat())
        assertEquals(Math.toDegrees(0.01).toFloat(), MainActivity.angleBetweenDeg(a, b), 1e-3f)
        assertEquals(0f, MainActivity.angleBetweenDeg(a, a), 1e-4f)
        assertEquals(90f, MainActivity.angleBetweenDeg(a, floatArrayOf(1f, 0f, 0f)), 1e-3f)
        assertEquals(180f, MainActivity.angleBetweenDeg(a, floatArrayOf(0f, 0f, 1f)), 1e-3f)
    }

    @Test
    fun `a still camera reports a still camera`() {
        // Quaternion-derived forward, one degree of ARCore jitter per frame at 30 fps: the guard
        // must read ~30 deg/s, not the 500 the pose-axis version produced.
        val q = floatArrayOf(0f, 0f, 0f, 1f)
        val forward = FloatArray(3)
        Unproject.rotate(q, 0f, 0f, -1f, forward)
        assertEquals(-1f, forward[2], 1e-6f)

        val jittered = FloatArray(3)
        val half = kotlin.math.sin(Math.toRadians(0.5) / 2).toFloat()
        Unproject.rotate(floatArrayOf(0f, half, 0f, kotlin.math.cos(Math.toRadians(0.5) / 2).toFloat()), 0f, 0f, -1f, jittered)
        val rate = MainActivity.angleBetweenDeg(forward, jittered) / (1f / 30f)
        assertTrue("rate was $rate", rate < MainActivity.MAX_TURN_RATE_DEG_S)
    }
}
