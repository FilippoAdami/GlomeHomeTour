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

    /**
     * Regression: the landmark hash table was sized `maxLandmarks * 2` = 40000 while the probe
     * masks with `size - 1`. 39999 is not a power of two, so the AND reached only 1024 distinct
     * buckets and `(idx + 1) and mask` cycled among them instead of finding an empty one --
     * landmarkSlotForId spun forever on the GL thread once a scan passed ~800 landmarks (ANR).
     */
    @Test(timeout = 30_000)
    fun `scan past the old 1024 bucket hash ceiling still terminates`() {
        val tracker = FeatureParallaxTracker()
        val side = 35 // 1225 points, comfortably past the 1024 reachable buckets
        val n = side * side
        val points = FloatArray(n * 4)
        val ids = IntArray(n)
        for (row in 0 until side) {
            for (col in 0 until side) {
                val i = row * side + col
                points[i * 4] = (col - side / 2) * 0.065f // > MIN_LANDMARK_SPACING_M apart
                points[i * 4 + 1] = (row - side / 2) * 0.065f
                points[i * 4 + 2] = 2.5f
                points[i * 4 + 3] = 0.9f
                ids[i] = i + 1
            }
        }

        // Frame 1 registers candidates; frame 2 from a 1.2 m baseline gives even the worst-placed
        // grid corner ~18 deg of parallax, so every point triangulates and promotes.
        tracker.update(points, ids, n, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        tracker.update(points, ids, n, cameraX = 1.2f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        assertEquals(n, tracker.verifiedLandmarkCount)

        // Every id must be findable again through the hash lookup (matchById -> landmarkSlotForId).
        val cur = Array(3) { FloatArray(n) }
        val lm = Array(3) { FloatArray(n) }
        assertEquals(n, tracker.matchById(ids, points, n, cur[0], cur[1], cur[2], lm[0], lm[1], lm[2]))
    }

    @Test
    fun `hash table sizes are powers of two with headroom`() {
        for (entries in intArrayOf(1, 3, 500, 4096, 20000)) {
            val size = FeatureParallaxTracker.tableSizeFor(entries)
            assertTrue("$entries -> $size", size >= entries * 2 && (size and (size - 1)) == 0)
        }
    }

    @Test
    fun `wobble detection detects high jitter and unanchors tracking`() {
        val tracker = FeatureParallaxTracker()
        assertTrue(tracker.isTrackingAnchored)

        val ids = intArrayOf(1, 2, 3, 4)
        val initialPoints = floatArrayOf(
            -0.5f, 0f, 2.0f, 0.9f,
            -0.2f, 0f, 2.0f, 0.9f,
             0.2f, 0f, 2.0f, 0.9f,
             0.5f, 0f, 2.0f, 0.9f
        )
        // Frame 1
        tracker.update(initialPoints, ids, 4, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        assertTrue(tracker.isTrackingAnchored)

        // Frame 2: candidate positions violently wobble by 0.2 meters
        val wobbledPoints = floatArrayOf(
            -0.5f, 0f, 2.2f, 0.9f,
            -0.2f, 0f, 1.8f, 0.9f,
             0.2f, 0f, 2.2f, 0.9f,
             0.5f, 0f, 1.8f, 0.9f
        )
        tracker.update(wobbledPoints, ids, 4, cameraX = 0.01f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        // Smoothed jitter exceeds 0.035m, tracking is not anchored
        org.junit.Assert.assertFalse(tracker.isTrackingAnchored)
    }

    @Test
    fun `high triangulation residual rejects candidate promotion`() {
        val tracker = FeatureParallaxTracker()
        // Two rays that pass each other with a large miss distance (> 0.09m)
        val triangulated = tracker.triangulateTwoRays(
            p1x = 0f, p1y = 0f, p1z = 0f, d1x = 0f, d1y = 0f, d1z = 1f,
            p2x = 1f, p2y = 0.3f, p2z = 0f, d2x = -0.4472f, d2y = 0f, d2z = 0.8944f
        )
        // Since vertical miss is 0.3m > MAX_RESIDUAL_M (0.09m), triangulation must fail
        org.junit.Assert.assertNull(triangulated)
    }

    @Test
    fun `distant outlier candidate is rejected from promotion`() {
        val tracker = FeatureParallaxTracker()
        // Populate 10 valid landmarks around (0, 0, 2)
        val n = 10
        val points = FloatArray(n * 4)
        val ids = IntArray(n)
        for (i in 0 until n) {
            points[i * 4] = (i - 5) * 0.1f
            points[i * 4 + 1] = 0f
            points[i * 4 + 2] = 2.0f
            points[i * 4 + 3] = 0.9f
            ids[i] = i + 1
        }
        tracker.update(points, ids, n, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        tracker.update(points, ids, n, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)
        assertEquals(n, tracker.verifiedLandmarkCount)

        // Now introduce an outlier candidate at (20.0, 0, 20.0) -> far outside cluster and camera limit
        val outlierPoint = floatArrayOf(20.0f, 0f, 20.0f, 0.9f)
        val outlierId = intArrayOf(999)
        tracker.update(outlierPoint, outlierId, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 3_000L)
        tracker.update(outlierPoint, outlierId, 1, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 4_000L)

        // Landmark count remains 10; outlier candidate was not promoted
        assertEquals(10, tracker.verifiedLandmarkCount)
    }

    @Test
    fun `normal culling culls points when camera moves to opposite side`() {
        val tracker = FeatureParallaxTracker()
        // Feature on North wall at (0, 0, 5) seen from camera at (0, 0, 0)
        val initialObservation = floatArrayOf(0.0f, 0.0f, 5.0f, 0.9f)
        val ids = intArrayOf(10)

        // View 1 at (0, 0, 0)
        tracker.update(initialObservation, ids, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        // View 2 at (1.0, 0, 0) -> Triangulates and promotes with observation normal pointing south towards (0, 0, 0)
        tracker.update(initialObservation, ids, 1, cameraX = 1.0f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)
        assertEquals(1, tracker.verifiedLandmarkCount)

        val exported = FloatArray(4)

        // Camera on front side of wall at (0, 0, 2): Landmark must be visible!
        val frontCount = tracker.exportPointVertices(
            exported, nowNs = 2_000L,
            camX = 0f, camY = 0f, camZ = 2f,
            enableNormalCulling = true
        )
        assertEquals(1, frontCount)

        // Camera moved to opposite side of the wall at (0, 0, 10): Landmark must be culled!
        val backCount = tracker.exportPointVertices(
            exported, nowNs = 2_000L,
            camX = 0f, camY = 0f, camZ = 10f,
            enableNormalCulling = true
        )
        assertEquals(0, backCount)
    }

    @Test
    fun `outdoor distance promotion allows distant points beyond 6 meters`() {
        val tracker = FeatureParallaxTracker()
        // Feature on building facade at 8.0 meters away
        val p1 = floatArrayOf(0.0f, 0.0f, 8.0f, 0.9f)
        val ids = intArrayOf(77)

        // Baseline of 1.5 meters for distant feature triangulation
        tracker.update(p1, ids, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        tracker.update(p1, ids, 1, cameraX = 1.5f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        // Must be promoted as permanent landmark despite being > 6.0m
        assertEquals(1, tracker.verifiedLandmarkCount)
    }
}
