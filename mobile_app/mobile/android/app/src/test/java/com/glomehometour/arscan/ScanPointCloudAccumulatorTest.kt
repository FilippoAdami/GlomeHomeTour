package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream

class ScanPointCloudAccumulatorTest {

    @Test
    fun testVoxelDeduplication() {
        val acc = ScanPointCloudAccumulator(maxPoints = 100, voxelSizeM = 0.05f)

        // Point 1
        assertTrue(acc.addPoint(1.0f, 2.0f, 3.0f))
        assertEquals(1, acc.pointCount)

        // Point within 5 cm voxel should be deduplicated
        assertFalse(acc.addPoint(1.01f, 2.02f, 3.01f))
        assertEquals(1, acc.pointCount)

        // Point outside 5 cm voxel should be accepted
        assertTrue(acc.addPoint(1.20f, 2.0f, 3.0f))
        assertEquals(2, acc.pointCount)
    }

    @Test
    fun testTrajectorySampling() {
        val acc = ScanPointCloudAccumulator()

        // First point
        assertTrue(acc.addTrajectoryPoint(0f, 0f, 0f))
        assertEquals(1, acc.trajectoryCount)

        // Point too close (< 8 cm) rejected
        assertFalse(acc.addTrajectoryPoint(0.02f, 0.01f, 0f))
        assertEquals(1, acc.trajectoryCount)

        // Point >= 8 cm accepted
        assertTrue(acc.addTrajectoryPoint(0.10f, 0f, 0f))
        assertEquals(2, acc.trajectoryCount)
    }

    @Test
    fun testBinaryPlyRoundTrip() {
        val acc1 = ScanPointCloudAccumulator(maxPoints = 1000)
        acc1.addPoint(1.25f, -0.5f, 3.75f)
        acc1.addPoint(2.50f, 1.0f, -4.25f)
        acc1.addPoint(0.0f, 0.0f, 0.0f)
        assertEquals(3, acc1.pointCount)

        val out = ByteArrayOutputStream()
        acc1.writeBinaryPly(out)

        val inStream = ByteArrayInputStream(out.toByteArray())
        val acc2 = ScanPointCloudAccumulator(maxPoints = 1000)
        val readCount = acc2.readBinaryPly(inStream)

        assertEquals(3, readCount)
        assertEquals(3, acc2.pointCount)
        assertEquals(1.25f, acc2.pointCoords[0], 1e-4f)
        assertEquals(-0.5f, acc2.pointCoords[1], 1e-4f)
        assertEquals(3.75f, acc2.pointCoords[2], 1e-4f)
    }

    @Test
    fun testBoundsComputation() {
        val acc = ScanPointCloudAccumulator()
        acc.addPoint(-1.5f, 0.2f, 3.0f)
        acc.addPoint(2.5f, 2.8f, -1.0f)

        val bounds = FloatArray(6)
        acc.getBounds(bounds)
        assertEquals(-1.5f, bounds[0], 1e-4f)
        assertEquals(2.5f, bounds[1], 1e-4f)
        assertEquals(0.2f, bounds[2], 1e-4f)
        assertEquals(2.8f, bounds[3], 1e-4f)
        assertEquals(-1.0f, bounds[4], 1e-4f)
        assertEquals(3.0f, bounds[5], 1e-4f)

        val center = FloatArray(3)
        acc.getCenter(center)
        assertEquals(0.5f, center[0], 1e-4f)
        assertEquals(1.5f, center[1], 1e-4f)
        assertEquals(1.0f, center[2], 1e-4f)
    }

    @Test
    fun `clips points outside the padded camera path and rebuilds bounds`() {
        val acc = ScanPointCloudAccumulator(maxPoints = 20, voxelSizeM = 0.05f)
        acc.addTrajectoryPoint(0f, 0f, 0f)
        acc.addTrajectoryPoint(0.1f, 0f, 0f)
        acc.addPoint(0.5f, 0f, 0f)
        acc.addPoint(0.1f, 2f, 0f)
        acc.addPoint(20f, 20f, 20f)

        assertEquals(2, acc.clipToTrajectory(maxDistanceM = 1f))
        assertEquals(1, acc.pointCount)
        assertEquals(0.5f, acc.minX, 1e-4f)
        assertEquals(0.5f, acc.maxX, 1e-4f)
        assertFalse(acc.addPoint(0.5f, 0f, 0f))
    }

    @Test
    fun `clips points far from the path even when they are inside its bounding box`() {
        val acc = ScanPointCloudAccumulator(maxPoints = 20, voxelSizeM = 0.05f)
        for (i in 0..40) acc.addTrajectoryPoint(i * 0.1f, 0f, 0f)
        acc.addPoint(2f, 0f, 0.5f)
        acc.addPoint(2f, 0f, 2f)
        acc.addPoint(2f, 0f, 20f)

        assertEquals(2, acc.clipToTrajectory(maxDistanceM = 1f))
        assertEquals(1, acc.pointCount)
        assertEquals(2f, acc.pointCoords[0], 1e-4f)
        assertEquals(0.5f, acc.pointCoords[2], 1e-4f)
    }

    @Test
    fun `uses six meter path clipping by default`() {
        val acc = ScanPointCloudAccumulator(maxPoints = 10, voxelSizeM = 0.01f)
        acc.addTrajectoryPoint(0f, 0f, 0f)
        acc.addPoint(5.9f, 0f, 0f)
        acc.addPoint(6.1f, 0f, 0f)

        assertEquals(1, acc.clipToTrajectory())
        assertEquals(1, acc.pointCount)
        assertEquals(5.9f, acc.pointCoords[0], 1e-4f)
    }

    @Test
    fun `removes isolated points with statistical outlier removal without a trajectory`() {
        val acc = ScanPointCloudAccumulator(maxPoints = 200, voxelSizeM = 0.04f)
        for (x in 0..4) for (y in 0..4) for (z in 0..4) {
            acc.addPoint(x * 0.1f, y * 0.1f, z * 0.1f)
        }
        acc.addPoint(30f, 30f, 30f)

        assertEquals(1, acc.clipToTrajectory())
        assertEquals(125, acc.pointCount)
        assertEquals(0.4f, acc.maxX, 1e-4f)
    }

    @Test
    fun `starts a new trajectory segment after a large jump`() {
        val acc = ScanPointCloudAccumulator()
        assertTrue(acc.addTrajectoryPoint(0f, 0f, 0f))
        assertFalse(acc.addTrajectoryPoint(3f, 0f, 0f))
        assertTrue(acc.addTrajectoryPoint(3.1f, 0f, 0f))
        assertEquals(2, acc.trajectoryCount)
        assertEquals(3.1f, acc.trajectoryCoords[3], 1e-4f)
    }

    @Test
    fun `reanchors trajectory after tracking recovery`() {
        val acc = ScanPointCloudAccumulator()
        assertTrue(acc.addTrajectoryPoint(0f, 0f, 0f))
        acc.reanchorTrajectory(3f, 0f, 0f)
        assertTrue(acc.addTrajectoryPoint(3.1f, 0f, 0f))
        assertEquals(2, acc.trajectoryCount)
        assertEquals(3.1f, acc.trajectoryCoords[3], 1e-4f)
    }

    @Test
    fun testBinaryPlyWithRgbaPropertiesAndStride() {
        // Construct binary PLY with 16 bytes per vertex: float x, y, z + uchar r, g, b, a
        val header = "ply\n" +
                "format binary_little_endian 1.0\n" +
                "element vertex 2\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property uchar red\n" +
                "property uchar green\n" +
                "property uchar blue\n" +
                "property uchar alpha\n" +
                "end_header\n"

        val out = ByteArrayOutputStream()
        out.write(header.toByteArray(Charsets.US_ASCII))
        val bb = java.nio.ByteBuffer.allocate(32).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        // Vertex 0: (1.0, 2.0, 3.0), RGBA (255, 0, 0, 255)
        bb.putFloat(1.0f); bb.putFloat(2.0f); bb.putFloat(3.0f)
        bb.put(255.toByte()); bb.put(0.toByte()); bb.put(0.toByte()); bb.put(255.toByte())
        // Vertex 1: (4.0, 5.0, 6.0), RGBA (0, 255, 0, 255)
        bb.putFloat(4.0f); bb.putFloat(5.0f); bb.putFloat(6.0f)
        bb.put(0.toByte()); bb.put(255.toByte()); bb.put(0.toByte()); bb.put(255.toByte())
        out.write(bb.array())

        val acc = ScanPointCloudAccumulator(maxPoints = 10)
        val readCount = acc.readBinaryPly(ByteArrayInputStream(out.toByteArray()))

        assertEquals(2, readCount)
        assertEquals(1.0f, acc.pointCoords[0], 1e-4f)
        assertEquals(2.0f, acc.pointCoords[1], 1e-4f)
        assertEquals(3.0f, acc.pointCoords[2], 1e-4f)
        assertEquals(4.0f, acc.pointCoords[3], 1e-4f)
        assertEquals(5.0f, acc.pointCoords[4], 1e-4f)
        assertEquals(6.0f, acc.pointCoords[5], 1e-4f)
    }
}
