package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScanRestartTest {

    @Test
    fun `tracker reset clears all verified and candidate landmarks`() {
        val tracker = FeatureParallaxTracker()
        val points = floatArrayOf(1.0f, 0.0f, 2.0f, 0.9f)
        val ids = intArrayOf(101)

        tracker.update(points, ids, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        assertEquals(0f, tracker.coverageFraction, 1e-4f)

        // Reset tracker
        tracker.reset()

        assertEquals(0, tracker.verifiedLandmarkCount)
        val exported = FloatArray(4)
        assertEquals(0, tracker.exportPointVertices(exported))
    }

    @Test
    fun `room data initializes with clean trajectory header and empty keyframes`() {
        val roomData = DatasetWriter.RoomData(2)
        assertEquals("scan_2", roomData.folderName)
        assertEquals(0, roomData.keyframes.size)
        assertTrue(roomData.trajectory.toString().startsWith(DatasetFormat.TRAJECTORY_HEADER))
    }
}
