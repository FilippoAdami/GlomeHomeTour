package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FeatureParallaxTrackerTest {

    @Test
    fun `initial sighting records point as unverified amber candidate`() {
        val tracker = FeatureParallaxTracker()
        val points = floatArrayOf(1.0f, 0.0f, 2.0f, 0.9f)
        val ids = intArrayOf(101)

        tracker.update(points, ids, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)

        assertEquals(0, tracker.verifiedLandmarkCount)
        assertEquals(0f, tracker.coverageFraction, 1e-4f)

        val exported = FloatArray(4)
        val count = tracker.exportPointVertices(exported)
        assertEquals(1, count)
        assertEquals(1.0f, exported[0], 1e-4f)
        assertEquals(0.0f, exported[1], 1e-4f)
        assertEquals(2.0f, exported[2], 1e-4f)
        assertEquals(0.0f, exported[3], 1e-4f) // 0 = Amber candidate
    }

    @Test
    fun `stationary rotation does not promote candidate to permanent landmark`() {
        val tracker = FeatureParallaxTracker()
        val points = floatArrayOf(0.0f, 0.0f, 2.0f, 0.9f)
        val ids = intArrayOf(42)

        tracker.update(points, ids, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        tracker.update(points, ids, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        assertEquals(0, tracker.verifiedLandmarkCount)
    }

    @Test
    fun `verified landmark becomes permanent and is never deleted when out of frame`() {
        val tracker = FeatureParallaxTracker()
        // Feature on wall at true physical position (0, 0, 2)
        val misplacedPoints = floatArrayOf(0.0f, 0.0f, 3.5f, 0.9f)
        val ids = intArrayOf(42)

        // View 1 at (0, 0, 0)
        tracker.update(misplacedPoints, ids, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        assertEquals(0, tracker.verifiedLandmarkCount)

        // View 2 at (0.8, 0, 0) -> Triangulates and promotes to permanent verified landmark
        val currentObservation = floatArrayOf(0.0f, 0.0f, 2.0f, 0.9f)
        tracker.update(currentObservation, ids, 1, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        assertEquals(1, tracker.verifiedLandmarkCount)
        assertTrue(tracker.coverageFraction > 0f)

        // View 3: Point 42 goes out of view entirely (empty point cloud for 10 seconds)
        tracker.update(FloatArray(0), IntArray(0), 0, cameraX = 2.0f, cameraY = 0f, cameraZ = 0f, timestampNs = 12_000_000_000L)

        // Landmark is PERMANENT: count and progress remain solid and non-decreasing!
        assertEquals(1, tracker.verifiedLandmarkCount)
        val exported = FloatArray(4)
        val count = tracker.exportPointVertices(exported)
        assertEquals(1, count)
        assertEquals(1.0f, exported[3], 1e-4f) // Green landmark still rendered in 3D world space
    }

    @Test
    fun `two ray triangulation math computes exact intersection`() {
        val tracker = FeatureParallaxTracker()
        val invSqrt5 = 1f / kotlin.math.sqrt(5f)
        val result = tracker.triangulateTwoRays(
            p1x = 0f, p1y = 0f, p1z = 0f, d1x = 0f, d1y = 0f, d1z = 1f,
            p2x = 1f, p2y = 0f, p2z = 0f, d2x = -1f * invSqrt5, d2y = 0f, d2z = 2f * invSqrt5
        )

        assertNotNull(result)
        assertEquals(0f, result!!.x, 1e-3f)
        assertEquals(0f, result.y, 1e-3f)
        assertEquals(2f, result.z, 1e-3f)
        assertTrue(result.residual < 1e-3f)
    }
}
