package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer

/** Export-package formats and the two pure conversions feeding them (SPEC §2.5). */
class DatasetTest {

    @Test
    fun `identity pose is the identity matrix with the translation in the last column`() {
        val m = DatasetFormat.cameraToWorld(1f, 2f, 3f, 0f, 0f, 0f, 1f)
        val expected = floatArrayOf(
            1f, 0f, 0f, 1f,
            0f, 1f, 0f, 2f,
            0f, 0f, 1f, 3f,
            0f, 0f, 0f, 1f,
        )
        for (i in expected.indices) assertEquals("element $i", expected[i], m[i], 1e-6f)
    }

    @Test
    fun `a 90 degree yaw maps camera forward to world -X`() {
        // ARCore camera forward is -Z. After yawing +90 degrees about +Y it must point at -X;
        // a transposed matrix would send it to +X and mirror the whole reconstruction.
        val s = kotlin.math.sin(Math.PI / 4).toFloat()
        val c = kotlin.math.cos(Math.PI / 4).toFloat()
        val m = DatasetFormat.cameraToWorld(0f, 0f, 0f, 0f, s, 0f, c)
        // Third column is where camera +Z lands, so camera forward (-Z) is its negation.
        assertEquals(-1f, -m[2], 1e-5f)
        assertEquals(0f, -m[6], 1e-5f)
        assertEquals(0f, -m[10], 1e-5f)
        assertTrue("camera forward should be world -X", -m[2] < -0.99f)
    }

    @Test
    fun `transforms json carries global and per-frame intrinsics`() {
        val frames = listOf(
            DatasetFormat.Keyframe("frame_00000.jpg", DatasetFormat.cameraToWorld(0f, 0f, 0f, 0f, 0f, 0f, 1f), 111L, 500f, 501f, 320f, 240f),
            DatasetFormat.Keyframe("frame_00001.jpg", DatasetFormat.cameraToWorld(1f, 0f, 0f, 0f, 0f, 0f, 1f), 222L, 505f, 506f, 320f, 240f),
        )
        val json = DatasetFormat.transformsJson(500f, 501f, 320f, 240f, 640, 480, frames)

        assertTrue(json.contains("\"schema_version\": \"1.0.0\""))
        assertTrue(json.contains("\"camera_model\": \"OPENCV\""))
        assertTrue(json.contains("\"w\": 640"))
        assertTrue(json.contains("\"file_path\": \"images/frame_00001.jpg\""))
        assertTrue(json.contains("\"timestamp_ns\": 222"))
        assertTrue("autofocus breathing must survive per frame", json.contains("\"fl_x\": 505.000000"))
        assertEquals("one matrix per frame", 2, Regex("transform_matrix").findAll(json).count())
        // Balanced braces/brackets is the cheapest proxy for "a parser will accept this".
        assertEquals(json.count { it == '{' }, json.count { it == '}' })
        assertEquals(json.count { it == '[' }, json.count { it == ']' })
        assertTrue("no trailing comma before the closing bracket", !json.contains(",\n  ]"))
    }

    @Test
    fun `numbers are formatted with dots regardless of device locale`() {
        val previous = java.util.Locale.getDefault()
        try {
            java.util.Locale.setDefault(java.util.Locale.ITALY) // decimal comma
            val line = DatasetFormat.trajectoryLine(7L, 1.5f, 0f, 0f, 0f, 0f, 0f, 1f, "TRACKING", true)
            assertEquals("7,1.500000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,TRACKING,1", line)
            assertEquals(DatasetFormat.TRAJECTORY_HEADER.split(",").size, line.split(",").size)
        } finally {
            java.util.Locale.setDefault(previous)
        }
    }

    @Test
    fun `summary json reports the drop reasons`() {
        val json = DatasetFormat.summaryJson(
            durationSeconds = 90f, coverageFraction = 0.87f, occupiedVoxels = 10, verifiedVoxels = 9,
            freeVoxels = 100, occludedVoxels = 5, voxelSizeM = 0.1f, gridFull = false, framesSeen = 2700, framesExported = 300,
            droppedDark = 4, droppedBlown = 1, droppedTransition = 12, droppedMotion = 30, droppedQueue = 2,
            trackingLosses = 1, depthSource = "depth16",
        )
        assertTrue(json.contains("\"schema_version\": \"1.0.0\""))
        assertTrue(json.contains("\"illumination_transition\": 12"))
        assertTrue(json.contains("\"depth_source\": \"depth16\""))
        assertEquals(json.count { it == '{' }, json.count { it == '}' })
    }

    @Test
    fun `unprojection puts the centre pixel straight down camera forward`() {
        val out = FloatArray(3)
        val identity = floatArrayOf(0f, 0f, 0f, 1f)
        Unproject.toWorld(320f, 240f, 2f, 500f, 500f, 320f, 240f, floatArrayOf(0f, 0f, 0f), identity, out)
        assertEquals(0f, out[0], 1e-5f)
        assertEquals(0f, out[1], 1e-5f)
        assertEquals(-2f, out[2], 1e-5f) // camera looks down -Z
    }

    @Test
    fun `image v axis points down, world y axis points up`() {
        val out = FloatArray(3)
        val identity = floatArrayOf(0f, 0f, 0f, 1f)
        // A pixel below the principal point is below the camera in the world.
        Unproject.toWorld(320f, 340f, 2f, 500f, 500f, 320f, 240f, floatArrayOf(0f, 0f, 0f), identity, out)
        assertTrue("pixel below centre must map to negative Y", out[1] < 0f)
        // ...and one to the right stays to the right.
        Unproject.toWorld(420f, 240f, 2f, 500f, 500f, 320f, 240f, floatArrayOf(0f, 0f, 0f), identity, out)
        assertTrue(out[0] > 0f)
    }

    @Test
    fun `unprojection applies rotation then translation`() {
        val out = FloatArray(3)
        val s = kotlin.math.sin(Math.PI / 4).toFloat()
        val c = kotlin.math.cos(Math.PI / 4).toFloat()
        Unproject.toWorld(
            320f, 240f, 2f, 500f, 500f, 320f, 240f,
            floatArrayOf(10f, 0f, 0f), floatArrayOf(0f, s, 0f, c), out,
        )
        // Yawed 90 degrees, so 2 m of forward becomes 2 m along -X, offset by the camera position.
        assertEquals(8f, out[0], 1e-4f)
        assertEquals(0f, out[1], 1e-4f)
        assertEquals(0f, out[2], 1e-4f)
    }

    /**
     * shared/schemas §6 "mobile-side contract test": generates transforms.json/coverage_summary.json
     * from the same input values hand-copied into the shared/schemas/fixtures example.json files, and
     * checks the same key values appear in both. This is a manual sync point, not an automated
     * cross-repo check -- if it ever fails because the two have drifted, log it in
     * shared/schemas/project_history.md per CLAUDE.md's per-folder history convention.
     */
    @Test
    fun `transforms json matches shared schemas fixture key values`() {
        val frames = listOf(
            DatasetFormat.Keyframe("frame_00000.jpg", DatasetFormat.cameraToWorld(0f, 0f, 0f, 0f, 0f, 0f, 1f), 111L, 500f, 501f, 320f, 240f),
            DatasetFormat.Keyframe("frame_00001.jpg", DatasetFormat.cameraToWorld(1f, 0f, 0f, 0f, 0f, 0f, 1f), 222L, 505f, 506f, 320f, 240f),
        )
        val json = DatasetFormat.transformsJson(500f, 501f, 320f, 240f, 640, 480, frames)

        assertTrue(json.contains("\"schema_version\": \"1.0.0\""))
        assertTrue(json.contains("\"camera_model\": \"OPENCV\""))
        assertTrue(json.contains("\"fl_x\": 500.000000"))
        assertTrue(json.contains("\"fl_y\": 501.000000"))
        assertTrue(json.contains("\"cx\": 320.000000"))
        assertTrue(json.contains("\"cy\": 240.000000"))
        assertTrue(json.contains("\"w\": 640"))
        assertTrue(json.contains("\"h\": 480"))
        assertTrue(json.contains("\"file_path\": \"images/frame_00000.jpg\""))
        assertTrue(json.contains("\"timestamp_ns\": 111"))
        assertTrue(json.contains("\"file_path\": \"images/frame_00001.jpg\""))
        assertTrue(json.contains("\"timestamp_ns\": 222"))
        assertTrue(json.contains("\"fl_x\": 505.000000"))
        assertTrue(json.contains("\"fl_y\": 506.000000"))
    }

    @Test
    fun `coverage summary json matches shared schemas fixture key values`() {
        val json = DatasetFormat.summaryJson(
            durationSeconds = 90f, coverageFraction = 0.87f, occupiedVoxels = 10, verifiedVoxels = 9,
            freeVoxels = 100, occludedVoxels = 5, voxelSizeM = 0.1f, gridFull = false, framesSeen = 2700, framesExported = 300,
            droppedDark = 4, droppedBlown = 1, droppedTransition = 12, droppedMotion = 30, droppedQueue = 2,
            trackingLosses = 1, depthSource = "depth16",
        )

        assertTrue(json.contains("\"duration_s\": 90.000000"))
        assertTrue(json.contains("\"coverage_fraction\": 0.870000"))
        assertTrue(json.contains("\"voxel_size_m\": 0.100000"))
        assertTrue(json.contains("\"parallax_min_deg\": 25.000000"))
        assertTrue(json.contains("\"occupied\": 10, \"parallax_verified\": 9, \"free\": 100, \"occluded\": 5, \"grid_full\": false"))
        assertTrue(json.contains("\"seen\": 2700, \"exported\": 300"))
        assertTrue(json.contains("\"schema_version\": \"1.0.0\""))
        assertTrue(json.contains("\"illumination_transition\": 12"))
        assertTrue(json.contains("\"tracking_loss_events\": 1"))
        assertTrue(json.contains("\"depth_source\": \"depth16\""))
    }

    @Test
    fun `nv21 conversion copies luma past padding and interleaves V before U`() {
        val w = 4
        val h = 4
        val yRowStride = 6 // padded
        val y = ByteBuffer.allocate(yRowStride * h)
        for (row in 0 until h) for (col in 0 until w) y.put(row * yRowStride + col, (row * w + col).toByte())

        val uvRowStride = 4
        val uvPixelStride = 2
        val u = ByteBuffer.allocate(uvRowStride * h / 2)
        val v = ByteBuffer.allocate(uvRowStride * h / 2)
        for (i in 0 until uvRowStride * h / 2) {
            u.put(i, 100.toByte())
            v.put(i, 200.toByte())
        }

        // Split path: bulk copy on the GL thread, interleave on the writer thread.
        val py = ByteArray(w * h)
        val pu = ByteArray(uvRowStride * h / 2)
        val pv = ByteArray(uvRowStride * h / 2)
        Nv21.copyPlanes(y, u, v, w, h, yRowStride, py, pu, pv)
        val out = ByteArray(Nv21.sizeOf(w, h))
        Nv21.interleave(py, pu, pv, w, h, uvRowStride, uvPixelStride, out)

        for (i in 0 until w * h) assertEquals(i.toByte(), out[i])
        assertEquals(200.toByte(), out[w * h])     // V first
        assertEquals(100.toByte(), out[w * h + 1]) // then U
        assertEquals(out.size, w * h + w * h / 2)
    }

    @Test
    fun `transforms json defaults distortion to zero and carries real k1 k2 when given`() {
        val defaulted = DatasetFormat.transformsJson(500f, 501f, 320f, 240f, 640, 480, emptyList())
        assertTrue(defaulted.contains("\"k1\": 0.000000"))
        assertTrue(defaulted.contains("\"p1\": 0.0"))

        val withDistortion = DatasetFormat.transformsJson(
            500f, 501f, 320f, 240f, 640, 480, emptyList(), k1 = -0.123f, k2 = 0.045f,
        )
        assertTrue(withDistortion.contains("\"k1\": -0.123000"))
        assertTrue(withDistortion.contains("\"k2\": 0.045000"))
        assertTrue(withDistortion.contains("\"p1\": 0.0"))
    }
}
