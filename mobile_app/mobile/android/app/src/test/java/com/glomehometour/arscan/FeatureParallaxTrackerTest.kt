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

        // Frame 2: candidate positions violently wobble by 0.5 meters
        val wobbledPoints = floatArrayOf(
            -0.5f, 0f, 2.5f, 0.9f,
            -0.2f, 0f, 1.5f, 0.9f,
             0.2f, 0f, 2.5f, 0.9f,
             0.5f, 0f, 1.5f, 0.9f
        )
        tracker.update(wobbledPoints, ids, 4, cameraX = 0.01f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        // Smoothed jitter exceeds WOBBLE_THRESHOLD_M (0.10m), tracking is not anchored
        org.junit.Assert.assertFalse(tracker.isTrackingAnchored)
    }

    @Test
    fun `high triangulation residual rejects candidate promotion`() {
        val tracker = FeatureParallaxTracker()
        // Two rays that pass each other with a large miss distance (> 0.25m)
        val triangulated = tracker.triangulateTwoRays(
            p1x = 0f, p1y = 0f, p1z = 0f, d1x = 0f, d1y = 0f, d1z = 1f,
            p2x = 1f, p2y = 0.35f, p2z = 0f, d2x = -0.4472f, d2y = 0f, d2z = 0.8944f
        )
        // Since vertical miss is 0.35m > MAX_RESIDUAL_M (0.25m), triangulation must fail
        org.junit.Assert.assertNull(triangulated)
    }

    @Test
    fun `computeDimensions computes bounding extents and floor area`() {
        val tracker = FeatureParallaxTracker()
        val f1 = floatArrayOf(1.0f, 0.0f, 2.0f, 0.9f)
        val f2 = floatArrayOf(-2.0f, 1.5f, 5.0f, 0.9f)
        tracker.update(f1, intArrayOf(1), 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        tracker.update(f1, intArrayOf(1), 1, cameraX = 0.5f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        tracker.update(f2, intArrayOf(2), 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 3_000L)
        tracker.update(f2, intArrayOf(2), 1, cameraX = 1.2f, cameraY = 0f, cameraZ = 0f, timestampNs = 4_000L)
        assertEquals(2, tracker.verifiedLandmarkCount)

        val dims = tracker.computeDimensions(
            camMinX = -1.0f, camMaxX = 3.0f,
            camMinY = 0.0f, camMaxY = 1.8f,
            camMinZ = -1.0f, camMaxZ = 4.0f
        )
        assertEquals(5.0f, dims.widthM, 0.05f)
        assertEquals(1.8f, dims.heightM, 0.05f)
        assertEquals(6.0f, dims.lengthM, 0.05f)
        assertEquals(30.0f, dims.floorAreaSqm, 0.5f)
    }

    @Test
    fun `dynamic target landmark updates scale target and adjust coverage fraction`() {
        val tracker = FeatureParallaxTracker()
        // Base target for smallest rooms starts at 500 points
        assertEquals(500, tracker.targetLandmarkCount)

        // Cannot decrease below current target
        tracker.updateTargetLandmarks(300)
        assertEquals(500, tracker.targetLandmarkCount)

        // Can scale upward for 35 sqm room to 5000
        tracker.updateTargetLandmarks(5000)
        assertEquals(5000, tracker.targetLandmarkCount)

        // Can scale upward further for larger room
        tracker.updateTargetLandmarks(7500)
        assertEquals(7500, tracker.targetLandmarkCount)

        // Reset restores base target of 500
        tracker.resetTargetLandmarks()
        assertEquals(500, tracker.targetLandmarkCount)
    }

    @Test
    fun `room area heuristic computes 150 pts per m2 base with 500 min, 300 good, and 750 excellent`() {
        // Small room (2.0 sqm) -> clamped to 500
        val smallAreaSqm = 2.0f
        val smallTarget = kotlin.math.round(smallAreaSqm * 150f).toInt()
            .coerceIn(FeatureParallaxTracker.MIN_ROOM_LANDMARKS, 30000)
        assertEquals(500, smallTarget)

        // 35 sqm room -> 35 * 150 = 5,250 base
        val room35Sqm = 35.0f
        val target35 = kotlin.math.round(room35Sqm * 150f).toInt()
            .coerceIn(FeatureParallaxTracker.MIN_ROOM_LANDMARKS, 30000)
        assertEquals(5250, target35)

        // Quality milestones for 35 sqm room:
        // Good quality = 2x target = 300 pts/m² = 10,500
        assertEquals(10500, target35 * 2)
        // Excellent quality = 5x target = 750 pts/m² = 26,250
        assertEquals(26250, target35 * 5)
    }

    @Test
    fun `2x3m room with window outliers correctly computes 6sqm floor area and 900 points target`() {
        val tracker = FeatureParallaxTracker()
        // Simulate a 2x3m room oriented along axes, camera walking inside [0.2, 1.8] x [0.2, 2.8]
        val camMinX = 0.2f; val camMaxX = 1.8f
        val camMinY = 0.0f; val camMaxY = 1.5f
        val camMinZ = 0.2f; val camMaxZ = 2.8f

        // Add 50 room landmarks spread across [0.0, 2.0] in X and [0.0, 3.0] in Z
        val roomPoints = FloatArray(50 * 4)
        val roomIds = IntArray(50)
        for (i in 0 until 50) {
            roomPoints[i * 4] = (i % 5) * 0.5f // 0.0 to 2.0m
            roomPoints[i * 4 + 1] = 1.0f
            roomPoints[i * 4 + 2] = (i / 5) * 0.3f // 0.0 to 2.7m
            roomPoints[i * 4 + 3] = 0.9f
            roomIds[i] = i + 1
        }
        tracker.update(roomPoints, roomIds, 50, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        tracker.update(roomPoints, roomIds, 50, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        // Add 5 distant window outlier points at (8.0, 1.0, 10.0) -> far outside the room (>3.0m)
        val outlierPoints = FloatArray(5 * 4)
        val outlierIds = IntArray(5)
        for (i in 0 until 5) {
            outlierPoints[i * 4] = 8.0f + i * 0.2f
            outlierPoints[i * 4 + 1] = 1.0f
            outlierPoints[i * 4 + 2] = 10.0f + i * 0.2f
            outlierPoints[i * 4 + 3] = 0.9f
            outlierIds[i] = 100 + i
        }
        tracker.update(outlierPoints, outlierIds, 5, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 3_000L)
        tracker.update(outlierPoints, outlierIds, 5, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 4_000L)

        // Now compute dimensions
        val dims = tracker.computeDimensions(camMinX, camMaxX, camMinY, camMaxY, camMinZ, camMaxZ)

        // Verify window outliers (>3.0m) were rejected and room area is ~6.0 sqm (NOT 80 sqm!)
        assertTrue("Width should be ~2m, was ${dims.widthM}", dims.widthM in 1.8f..2.5f)
        assertTrue("Length should be ~3m, was ${dims.lengthM}", dims.lengthM in 2.6f..3.5f)
        assertTrue("Floor area should be ~6.0 sqm, was ${dims.floorAreaSqm}", dims.floorAreaSqm in 5.0f..8.0f)

        // Target calculation at 150 pts/m² should be ~900 points (<1200 points, NOT >3000!)
        val computedTarget = kotlin.math.round(dims.floorAreaSqm * 150f).toInt()
            .coerceIn(FeatureParallaxTracker.MIN_ROOM_LANDMARKS, 30000)
        assertTrue("Target for 2x3m room should be < 1200 points, was $computedTarget", computedTarget < 1200)

        // Adaptive target updates can set target to 900 even if previously higher
        tracker.setTargetLandmarks(5000)
        assertEquals(5000, tracker.targetLandmarkCount)
        tracker.setTargetLandmarks(computedTarget)
        assertEquals(computedTarget, tracker.targetLandmarkCount)
    }

    @Test
    fun `getRecentLandmarkData returns only landmarks seen within time window or radius`() {
        val tracker = FeatureParallaxTracker()
        val p1 = floatArrayOf(0.0f, 0.0f, 2.0f, 0.9f)
        val p2 = floatArrayOf(5.0f, 0.0f, 2.0f, 0.9f)
        val id1 = intArrayOf(1)
        val id2 = intArrayOf(2)

        // Landmark 1 seen at t = 1s, promoted at t = 2s
        tracker.update(p1, id1, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000_000_000L)
        tracker.update(p1, id1, 1, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000_000_000L)

        // Landmark 2 seen at t = 10s, promoted at t = 11s (camera near p2 at x=5, moves by 0.8m)
        tracker.update(p2, id2, 1, cameraX = 5.0f, cameraY = 0f, cameraZ = 0f, timestampNs = 10_000_000_000L)
        tracker.update(p2, id2, 1, cameraX = 5.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 11_000_000_000L)

        assertEquals(2, tracker.verifiedLandmarkCount)

        val outX = FloatArray(10)
        val outY = FloatArray(10)
        val outZ = FloatArray(10)

        // Query landmarks seen since 8s (only landmark 2 should qualify by time)
        // With camera far away from p1 and p2:
        val recentCount = tracker.getRecentLandmarkData(
            outX, outY, outZ,
            sinceNs = 8_000_000_000L,
            camX = 100f, camY = 100f, camZ = 100f,
            maxDistM = 2.0f
        )
        assertEquals(1, recentCount)
        assertEquals(tracker.getLandmarkData(FloatArray(10), FloatArray(10), FloatArray(10)), 2)

        // Spatial fallback: query at t = 15s with sinceNs = 14s (no points by time),
        // but camera near p1 at (0, 0, 0) within 3m
        val spatialCount = tracker.getRecentLandmarkData(
            outX, outY, outZ,
            sinceNs = 14_000_000_000L,
            camX = 0f, camY = 0f, camZ = 0f,
            maxDistM = 3.0f
        )
        assertEquals(1, spatialCount)
        assertEquals(0.0f, outX[0], 0.05f)
        assertEquals(2.0f, outZ[0], 0.05f)
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

    @Test
    fun `updateRunningMedianDistance computes median and sets dynamicMaxDistanceM to 2_5x median`() {
        val tracker = FeatureParallaxTracker()
        // 5 points at depths 1.5m, 2.0m, 2.5m, 3.0m, 3.5m
        val depths = floatArrayOf(1.5f, 2.0f, 2.5f, 3.0f, 3.5f)
        val points = FloatArray(depths.size * 4)
        val ids = IntArray(depths.size)
        for (i in depths.indices) {
            points[i * 4] = 0f
            points[i * 4 + 1] = 0f
            points[i * 4 + 2] = depths[i]
            points[i * 4 + 3] = 0.9f
            ids[i] = 100 + i
        }

        tracker.update(points, ids, depths.size, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        tracker.update(points, ids, depths.size, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        assertEquals(5, tracker.verifiedLandmarkCount)

        val maxDist = tracker.updateRunningMedianDistance(camX = 0f, camY = 0f, camZ = 0f)
        // Median depth is 2.5m
        assertEquals(2.5f, tracker.runningMedianDistanceM, 0.05f)
        // Max distance is 2.5 * 2.5 = 6.25m
        assertEquals(6.25f, maxDist, 0.1f)
        assertEquals(6.25f, tracker.dynamicMaxDistanceM, 0.1f)
    }

    @Test
    fun `pruneOutliers deletes distant outliers beyond dynamicMaxDistanceM`() {
        val tracker = FeatureParallaxTracker()
        // 5 normal points in room at 2.0m depth
        val normalPoints = FloatArray(5 * 4)
        val normalIds = IntArray(5) { it + 1 }
        for (i in 0 until 5) {
            normalPoints[i * 4] = (i - 2) * 0.3f
            normalPoints[i * 4 + 1] = 0f
            normalPoints[i * 4 + 2] = 2.0f
            normalPoints[i * 4 + 3] = 0.9f
        }
        tracker.update(normalPoints, normalIds, 5, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        tracker.update(normalPoints, normalIds, 5, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)

        // 1 outlier landmark at 10.0m (e.g. through window before calibration)
        val outlierPoint = floatArrayOf(0f, 0f, 10.0f, 0.9f)
        val outlierId = intArrayOf(999)
        tracker.update(outlierPoint, outlierId, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 3_000L)
        tracker.update(outlierPoint, outlierId, 1, cameraX = 1.5f, cameraY = 0f, cameraZ = 0f, timestampNs = 4_000L)

        assertEquals(6, tracker.verifiedLandmarkCount)

        // Calibrate dynamic median: median of [2.0, 2.02, 2.02, 2.09, 2.09, 10.0] is 2.088m
        tracker.updateRunningMedianDistance(camX = 0f, camY = 0f, camZ = 0f)
        // Dynamic max distance becomes 2.088 * 2.5 = 5.22m
        assertEquals(5.22f, tracker.dynamicMaxDistanceM, 0.1f)

        // Prune outliers relative to camera trajectory [0..1, 0..0, 0..0]
        val pruned = tracker.pruneOutliers(camMinX = 0f, camMaxX = 1.0f, camMinZ = 0f, camMaxZ = 0f)
        assertEquals(1, pruned)
        assertEquals(5, tracker.verifiedLandmarkCount)

        // Verify remaining landmarks are the 5 inliers
        val outX = FloatArray(10)
        val outY = FloatArray(10)
        val outZ = FloatArray(10)
        val count = tracker.getLandmarkData(outX, outY, outZ)
        assertEquals(5, count)
        for (i in 0 until count) {
            assertTrue(outZ[i] <= 5.0f)
        }
    }

    @Test
    fun `allowNewLandmarks flag suspends candidate promotion during fast motion`() {
        val tracker = FeatureParallaxTracker()
        val p = floatArrayOf(0f, 0f, 2.0f, 0.9f)
        val id = intArrayOf(55)

        // Sighting 1: registers candidate
        tracker.update(p, id, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L, allowNewLandmarks = true)
        assertEquals(0, tracker.verifiedLandmarkCount)

        // Sighting 2 with fast motion (allowNewLandmarks = false): large baseline of 0.8m but must NOT promote
        tracker.update(p, id, 1, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L, allowNewLandmarks = false)
        assertEquals(0, tracker.verifiedLandmarkCount)

        // Sighting 3 with stabilized motion (allowNewLandmarks = true): now triangulates and promotes
        tracker.update(p, id, 1, cameraX = 0.8f, cameraY = 0f, cameraZ = 0f, timestampNs = 3_000L, allowNewLandmarks = true)
        assertEquals(1, tracker.verifiedLandmarkCount)
    }

    @Test
    fun `on the fly candidate rejection drops points beyond dynamicMaxDistanceM`() {
        val tracker = FeatureParallaxTracker()
        // Calibrate tracker with 5 points at 1.5m -> dynamicMaxDistance becomes 1.5 * 2.5 = 3.75m
        val pts = FloatArray(5 * 4)
        val ids = IntArray(5) { it + 1 }
        for (i in 0 until 5) {
            pts[i * 4] = (i - 2) * 0.2f
            pts[i * 4 + 1] = 0f
            pts[i * 4 + 2] = 1.5f
            pts[i * 4 + 3] = 0.9f
        }
        tracker.update(pts, ids, 5, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 1_000L)
        tracker.update(pts, ids, 5, cameraX = 0.5f, cameraY = 0f, cameraZ = 0f, timestampNs = 2_000L)
        tracker.updateRunningMedianDistance(camX = 0f, camY = 0f, camZ = 0f)
        assertEquals(3.75f, tracker.dynamicMaxDistanceM, 0.05f)

        // Now attempt to introduce a distant point at 7.0m (> 3.75m)
        val distant = floatArrayOf(0f, 0f, 7.0f, 0.9f)
        val distantId = intArrayOf(888)
        tracker.update(distant, distantId, 1, cameraX = 0f, cameraY = 0f, cameraZ = 0f, timestampNs = 3_000L)

        // Verify it was rejected on the fly and never added to totalTrackedCount (verified + candidates)
        // Verified count should still be 5, and candidateCount should be 0
        assertEquals(5, tracker.verifiedLandmarkCount)
        assertEquals(5, tracker.totalTrackedCount)
    }
}
