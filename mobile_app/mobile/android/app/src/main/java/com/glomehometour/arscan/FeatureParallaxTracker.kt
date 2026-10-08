package com.glomehometour.arscan

import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Solid Keyframe Landmarking & Multi-View Triangulation Engine.
 *
 * Designed to provide:
 * 1. Persistent 3D Landmark Memory (Persistent Keyframe Points):
 *    - Once a physical feature point is verified via multi-view parallax (Emerald Green),
 *      it is NEVER deleted when it goes out of frame. It remains permanently anchored in the
 *      3D spatial map of the room.
 *    - Unverified single-view candidate points (Amber) are kept while in view for candidate matching,
 *      but only verified landmarks form the persistent room landmark cloud.
 *
 * 2. High-Quality Feature Filtering & Spatial Uniformity:
 *    - Caps persistent landmarks to a clean, highly distinctive set (e.g. 300 - 500 solid landmarks)
 *      with minimum spatial separation (0.15 m) to prevent dense clusters on a single edge.
 *
 * 3. Monotonically Non-Decreasing Completion Progress:
 *    - Progress is evaluated against a fixed room landmark target (e.g. 150 - 200 solid verified points),
 *      so the progress bar strictly monotonically rises as new areas of the room are discovered and verified.
 *
 * 4. Closed-Form Two-Ray Geometric Triangulation:
 *    - Snaps initial inaccurate monocular depth estimations to the true geometric ray intersection.
 */
class FeatureParallaxTracker(
    private val maxLandmarks: Int = 20000,
    private val maxCandidates: Int = 4096,
) {
    // ---- PERSISTENT VERIFIED LANDMARKS (NEVER DELETED ONCE GREEN) ----
    private val landmarkIds = IntArray(maxLandmarks)
    private val landmarkX = FloatArray(maxLandmarks)
    private val landmarkY = FloatArray(maxLandmarks)
    private val landmarkZ = FloatArray(maxLandmarks)
    private val landmarkNormX = FloatArray(maxLandmarks)
    private val landmarkNormY = FloatArray(maxLandmarks)
    private val landmarkNormZ = FloatArray(maxLandmarks)
    private val landmarkPromoteTimeNs = LongArray(maxLandmarks)
    var verifiedLandmarkCount: Int = 0
        private set

    // ---- UNVERIFIED CANDIDATE POOL (TEMPORARY UNTIL PARALLAX VERIFIED) ----
    private val candIds = IntArray(maxCandidates)
    private val candX = FloatArray(maxCandidates)
    private val candY = FloatArray(maxCandidates)
    private val candZ = FloatArray(maxCandidates)
    private val candRayX = FloatArray(maxCandidates)
    private val candRayY = FloatArray(maxCandidates)
    private val candRayZ = FloatArray(maxCandidates)
    private val candCamX = FloatArray(maxCandidates)
    private val candCamY = FloatArray(maxCandidates)
    private val candCamZ = FloatArray(maxCandidates)
    private val candLastSeenNs = LongArray(maxCandidates)
    private val candStableCount = IntArray(maxCandidates)
    private var candidateCount: Int = 0
    private var smoothedJitterM: Float = 0f

    // Both maps below probe with `idx = (idx + 1) and mask`, which is only a valid wrap when the
    // table size is a power of two. A raw `count * 2` is not: for 20000 landmarks the mask 39999
    // has 10 bits set, so the AND reaches only 2^10 = 1024 distinct buckets and the probe walk
    // cycles among them instead of ever reaching an EMPTY_KEY -- lookups spun forever on the GL
    // thread at ~800 landmarks (ANR). Rounding up keeps the table at >= 2x the entry count, so it
    // also can never fill, which is what bounds the insert loops in addPermanentLandmark/
    // allocateCandSlot.

    // Fast open-addressed hash map for active candidates: pointId -> slot
    private val candHashKeys = IntArray(tableSizeFor(maxCandidates)) { EMPTY_KEY }
    private val candHashSlots = IntArray(tableSizeFor(maxCandidates)) { -1 }

    // Fast open-addressed hash map for verified landmarks: pointId -> slot
    private val landmarkHashKeys = IntArray(tableSizeFor(maxLandmarks)) { EMPTY_KEY }
    private val landmarkHashSlots = IntArray(tableSizeFor(maxLandmarks)) { -1 }

    // Target verified landmarks to reach 100% room coverage
    val targetLandmarkCount: Int = TARGET_ROOM_LANDMARKS

    /** Monotonically increasing coverage progress 0.0f .. 1.0f */
    val coverageFraction: Float
        get() = (verifiedLandmarkCount.toFloat() / targetLandmarkCount).coerceIn(0f, 1f)

    val verifiedFraction: Float
        get() = coverageFraction

    val totalTrackedCount: Int
        get() = verifiedLandmarkCount + candidateCount

    val verifiedCount: Int
        get() = verifiedLandmarkCount

    /**
     * Returns true when visual feature points are solidly anchored in 3D space with low jitter.
     * Returns false when points are wobbling (e.g. scanning a flat poster or low-parallax surface).
     */
    val isTrackingAnchored: Boolean
        get() = smoothedJitterM < WOBBLE_THRESHOLD_M

    /**
     * Integrates an ARCore PointCloud into the tracker.
     */
    fun update(
        pointBuf: FloatArray,
        idBuf: IntArray,
        count: Int,
        cameraX: Float,
        cameraY: Float,
        cameraZ: Float,
        timestampNs: Long,
    ) {
        val n = minOf(count, pointBuf.size / 4, idBuf.size)
        var totalCandidateDisplacement = 0f
        var trackedCandidateCount = 0

        for (i in 0 until n) {
            val px = pointBuf[i * 4]
            val py = pointBuf[i * 4 + 1]
            val pz = pointBuf[i * 4 + 2]
            val id = idBuf[i]
            if (id < 0) continue

            // 1. If this ID is already a persistent verified landmark, it is already permanent
            if (isLandmark(id)) continue

            // 2. Check in candidate pool
            val slot = findCandSlot(id)
            if (slot >= 0) {
                // Existing candidate: update last seen
                candLastSeenNs[slot] = timestampNs

                val cdx = px - candX[slot]
                val cdy = py - candY[slot]
                val cdz = pz - candZ[slot]
                val disp = sqrt(cdx * cdx + cdy * cdy + cdz * cdz)

                totalCandidateDisplacement += disp
                trackedCandidateCount++

                if (disp < 0.04f) {
                    candStableCount[slot]++
                } else {
                    candStableCount[slot] = 0
                }

                val dx = cameraX - px
                val dy = cameraY - py
                val dz = cameraZ - pz
                val dist = sqrt(dx * dx + dy * dy + dz * dz)
                if (dist <= 1e-4f) continue

                val rx = dx / dist
                val ry = dy / dist
                val rz = dz / dist

                val r0x = candRayX[slot]; val r0y = candRayY[slot]; val r0z = candRayZ[slot]
                val cosTheta = (r0x * rx + r0y * ry + r0z * rz).coerceIn(-1f, 1f)
                val angleDeg = Math.toDegrees(acos(cosTheta.toDouble())).toFloat()

                val c1x = candCamX[slot]; val c1y = candCamY[slot]; val c1z = candCamZ[slot]
                val tdx = cameraX - c1x
                val tdy = cameraY - c1y
                val tdz = cameraZ - c1z
                val transDist = sqrt(tdx * tdx + tdy * tdy + tdz * tdz)

                if (angleDeg >= MIN_PARALLAX_DEG && transDist >= MIN_TRANSLATION_M) {
                    // Two-ray geometric triangulation to correct initial placement
                    val u1x = -r0x; val u1y = -r0y; val u1z = -r0z
                    val u2x = -rx; val u2y = -ry; val u2z = -rz

                    val triangulated = triangulateTwoRays(
                        c1x, c1y, c1z, u1x, u1y, u1z,
                        cameraX, cameraY, cameraZ, u2x, u2y, u2z
                    )

                    // If ray triangulation failed (e.g. ill-conditioned rays, excessive residual):
                    // DO NOT PROMOTE! Keep candidate and continue observing.
                    if (triangulated == null) {
                        candX[slot] = px
                        candY[slot] = py
                        candZ[slot] = pz
                        continue
                    }

                    val targetX = triangulated.x
                    val targetY = triangulated.y
                    val targetZ = triangulated.z

                    // Near plane safeguard (>= 0.3m); no arbitrary upper ceiling for large exterior spaces
                    val dCamX = cameraX - targetX
                    val dCamY = cameraY - targetY
                    val dCamZ = cameraZ - targetZ
                    val distFromCam = sqrt(dCamX * dCamX + dCamY * dCamY + dCamZ * dCamZ)
                    val validDist = distFromCam >= 0.3f

                    val normX = if (distFromCam > 1e-4f) dCamX / distFromCam else 0f
                    val normY = if (distFromCam > 1e-4f) dCamY / distFromCam else 0f
                    val normZ = if (distFromCam > 1e-4f) dCamZ / distFromCam else 1f

                    // Promote candidate to permanent verified landmark if space permits and spatially distinct
                    if (isTrackingAnchored && validDist && canPromoteLandmark(targetX, targetY, targetZ)) {
                        addPermanentLandmark(id, targetX, targetY, targetZ, normX, normY, normZ, timestampNs)
                    } else {
                        // Keep current estimate if close to other landmarks
                        candX[slot] = targetX
                        candY[slot] = targetY
                        candZ[slot] = targetZ
                    }
                } else {
                    // Update latest consensus position from ARCore
                    candX[slot] = px
                    candY[slot] = py
                    candZ[slot] = pz
                }
            } else {
                // New candidate: allocate slot
                val newSlot = allocateCandSlot(id) ?: continue
                candIds[newSlot] = id
                candX[newSlot] = px
                candY[newSlot] = py
                candZ[newSlot] = pz
                candCamX[newSlot] = cameraX
                candCamY[newSlot] = cameraY
                candCamZ[newSlot] = cameraZ
                candStableCount[newSlot] = 0

                val dx = cameraX - px; val dy = cameraY - py; val dz = cameraZ - pz
                val dist = sqrt(dx * dx + dy * dy + dz * dz)
                if (dist > 1e-4f) {
                    candRayX[newSlot] = dx / dist
                    candRayY[newSlot] = dy / dist
                    candRayZ[newSlot] = dz / dist
                } else {
                    candRayX[newSlot] = 0f; candRayY[newSlot] = 0f; candRayZ[newSlot] = 1f
                }
                candLastSeenNs[newSlot] = timestampNs
                candidateCount++
            }
        }

        if (trackedCandidateCount >= 4) {
            val frameJitter = totalCandidateDisplacement / trackedCandidateCount
            smoothedJitterM = smoothedJitterM * 0.75f + frameJitter * 0.25f
        }

        // Clean out unverified candidates that left the field of view
        pruneStaleCandidates(timestampNs)
    }

    /**
     * Checks if a new landmark can be added (spatial separation filter).
     * Prevents adding hundreds of points on the exact same corner.
     */
    private fun canPromoteLandmark(x: Float, y: Float, z: Float): Boolean {
        if (verifiedLandmarkCount >= maxLandmarks) return false
        // Quick distance check against existing landmarks
        val minDistSq = MIN_LANDMARK_SPACING_M * MIN_LANDMARK_SPACING_M
        for (i in 0 until verifiedLandmarkCount) {
            val dx = landmarkX[i] - x
            val dy = landmarkY[i] - y
            val dz = landmarkZ[i] - z
            if (dx * dx + dy * dy + dz * dz < minDistSq) {
                return false // Too close to an existing solid landmark
            }
        }

        // Outlier prevention: reject points that lie far outside the established room cluster
        if (verifiedLandmarkCount >= 10) {
            var sumX = 0f; var sumZ = 0f
            for (i in 0 until verifiedLandmarkCount) {
                sumX += landmarkX[i]
                sumZ += landmarkZ[i]
            }
            val cx = sumX / verifiedLandmarkCount
            val cz = sumZ / verifiedLandmarkCount
            var maxDistSq = 0f
            for (i in 0 until verifiedLandmarkCount) {
                val dx = landmarkX[i] - cx
                val dz = landmarkZ[i] - cz
                val d2 = dx * dx + dz * dz
                if (d2 > maxDistSq) maxDistSq = d2
            }
            val clusterRadius = sqrt(maxDistSq)
            val dFromCenter = sqrt((x - cx) * (x - cx) + (z - cz) * (z - cz))
            // Dynamic adaptive cluster boundary (scales naturally with room or exterior size)
            val maxAllowed = 2.5f * clusterRadius + 2.0f
            if (dFromCenter > maxAllowed) {
                return false // Outlier rejected
            }
        }

        return true
    }

    private fun addPermanentLandmark(
        id: Int,
        x: Float, y: Float, z: Float,
        normX: Float, normY: Float, normZ: Float,
        timestampNs: Long = 0L,
    ) {
        val slot = verifiedLandmarkCount
        if (slot >= maxLandmarks) return
        landmarkIds[slot] = id
        landmarkX[slot] = x
        landmarkY[slot] = y
        landmarkZ[slot] = z
        landmarkNormX[slot] = normX
        landmarkNormY[slot] = normY
        landmarkNormZ[slot] = normZ
        landmarkPromoteTimeNs[slot] = timestampNs
        verifiedLandmarkCount++

        // Insert into landmark hash map
        val mask = landmarkHashKeys.size - 1
        var idx = (id * 0x45d9f3b) and mask
        while (landmarkHashKeys[idx] != EMPTY_KEY) {
            idx = (idx + 1) and mask
        }
        landmarkHashKeys[idx] = id
        landmarkHashSlots[idx] = slot
    }

    private fun isLandmark(id: Int): Boolean = landmarkSlotForId(id) >= 0

    private fun landmarkSlotForId(id: Int): Int {
        val mask = landmarkHashKeys.size - 1
        var idx = (id * 0x45d9f3b) and mask
        while (true) {
            val k = landmarkHashKeys[idx]
            if (k == id) return landmarkHashSlots[idx]
            if (k == EMPTY_KEY) return -1
            idx = (idx + 1) and mask
        }
    }

    /**
     * Free, unambiguous correspondences for relocalization: points in the current frame whose
     * ARCore ID already belongs to a verified landmark (ARCore can preserve IDs across a brief
     * tracking interruption, though not a full reset). Far cheaper and far less error-prone than
     * blind triangle-congruence matching, so it's always worth checking first.
     * Returns the number of matched pairs written into the out arrays.
     */
    fun matchById(
        pointIds: IntArray,
        points: FloatArray,
        numPoints: Int,
        outCurrentX: FloatArray, outCurrentY: FloatArray, outCurrentZ: FloatArray,
        outLandmarkX: FloatArray, outLandmarkY: FloatArray, outLandmarkZ: FloatArray,
    ): Int {
        var count = 0
        val maxOut = minOf(outCurrentX.size, outLandmarkX.size)
        for (i in 0 until numPoints) {
            if (count >= maxOut) break
            val slot = landmarkSlotForId(pointIds[i])
            if (slot < 0) continue
            outCurrentX[count] = points[i * 4]
            outCurrentY[count] = points[i * 4 + 1]
            outCurrentZ[count] = points[i * 4 + 2]
            outLandmarkX[count] = landmarkX[slot]
            outLandmarkY[count] = landmarkY[slot]
            outLandmarkZ[count] = landmarkZ[slot]
            count++
        }
        return count
    }

    /**
     * Solves for closest point between two 3D rays.
     */
    fun triangulateTwoRays(
        p1x: Float, p1y: Float, p1z: Float, d1x: Float, d1y: Float, d1z: Float,
        p2x: Float, p2y: Float, p2z: Float, d2x: Float, d2y: Float, d2z: Float,
    ): TriangulatedPoint? {
        val wx = p1x - p2x; val wy = p1y - p2y; val wz = p1z - p2z

        val a = d1x * d1x + d1y * d1y + d1z * d1z
        val b = d1x * d2x + d1y * d2y + d1z * d2z
        val c = d2x * d2x + d2y * d2y + d2z * d2z
        val d = d1x * wx + d1y * wy + d1z * wz
        val e = d2x * wx + d2y * wy + d2z * wz

        val denom = a * c - b * b
        if (denom < 1e-4f) return null

        val t1 = (b * e - c * d) / denom
        val t2 = (a * e - b * d) / denom

        if (t1 < 0.2f || t1 > 35.0f || t2 < 0.2f || t2 > 35.0f) return null

        val pt1x = p1x + t1 * d1x; val pt1y = p1y + t1 * d1y; val pt1z = p1z + t1 * d1z
        val pt2x = p2x + t2 * d2x; val pt2y = p2y + t2 * d2y; val pt2z = p2z + t2 * d2z

        val rx = pt1x - pt2x; val ry = pt1y - pt2y; val rz = pt1z - pt2z
        val residual = sqrt(rx * rx + ry * ry + rz * rz)

        if (residual > MAX_RESIDUAL_M) return null

        return TriangulatedPoint(
            x = 0.5f * (pt1x + pt2x),
            y = 0.5f * (pt1y + pt2y),
            z = 0.5f * (pt1z + pt2z),
            residual = residual
        )
    }

    data class TriangulatedPoint(val x: Float, val y: Float, val z: Float, val residual: Float)

    /**
     * Prunes unverified candidates that haven't been seen in the current viewport for > 1.2s.
     * Note: Permanent landmarks are never in candidate pool, so they are NEVER pruned!
     */
    private fun pruneStaleCandidates(nowNs: Long) {
        if (candidateCount < 30) return

        var writeSlot = 0
        for (readSlot in 0 until candidateCount) {
            val id = candIds[readSlot]
            // If it became a landmark, drop from candidates
            if (isLandmark(id)) continue

            val ageNs = nowNs - candLastSeenNs[readSlot]
            if (ageNs <= CANDIDATE_STALE_TIMEOUT_NS) {
                if (writeSlot != readSlot) {
                    copyCandSlot(from = readSlot, to = writeSlot)
                }
                writeSlot++
            }
        }

        if (writeSlot != candidateCount) {
            candidateCount = writeSlot
            rebuildCandHashMap()
        }
    }

    private fun copyCandSlot(from: Int, to: Int) {
        candIds[to] = candIds[from]
        candX[to] = candX[from]
        candY[to] = candY[from]
        candZ[to] = candZ[from]
        candRayX[to] = candRayX[from]
        candRayY[to] = candRayY[from]
        candRayZ[to] = candRayZ[from]
        candCamX[to] = candCamX[from]
        candCamY[to] = candCamY[from]
        candCamZ[to] = candCamZ[from]
        candLastSeenNs[to] = candLastSeenNs[from]
        candStableCount[to] = candStableCount[from]
    }

    private fun rebuildCandHashMap() {
        candHashKeys.fill(EMPTY_KEY)
        candHashSlots.fill(-1)
        val mask = candHashKeys.size - 1
        for (slot in 0 until candidateCount) {
            val id = candIds[slot]
            var idx = (id * 0x45d9f3b) and mask
            while (candHashKeys[idx] != EMPTY_KEY) {
                idx = (idx + 1) and mask
            }
            candHashKeys[idx] = id
            candHashSlots[idx] = slot
        }
    }

    /**
     * Exports point sprite vertex data: [x, y, z, verified] (4 floats per point).
     * 1. First exports all permanent solid Emerald Green landmarks (permanent history).
     * 2. Then exports currently visible Amber candidate points.
     */
    fun exportPointVertices(
        out: FloatArray,
        nowNs: Long = 0L,
        camX: Float = 0f,
        camY: Float = 0f,
        camZ: Float = 0f,
        enableNormalCulling: Boolean = false,
    ): Int {
        var count = 0
        val maxVerts = out.size / 4

        // 1. Permanent Verified Landmarks (Emerald Gem / Sparkling 4-Pointed Star)
        // Sparkle lasts 1.2 seconds upon promotion
        val sparkleDurationNs = 1_200_000_000L
        for (i in 0 until verifiedLandmarkCount) {
            if (count >= maxVerts) break

            // Observation ray / normal culling for object-centric / exterior scanning:
            // Culls points facing away from camera (opposite wall / back of house)
            if (enableNormalCulling) {
                val vx = camX - landmarkX[i]
                val vy = camY - landmarkY[i]
                val vz = camZ - landmarkZ[i]
                val dot = vx * landmarkNormX[i] + vy * landmarkNormY[i] + vz * landmarkNormZ[i]
                if (dot < -0.05f) continue
            }

            out[count * 4] = landmarkX[i]
            out[count * 4 + 1] = landmarkY[i]
            out[count * 4 + 2] = landmarkZ[i]
            val ageNs = nowNs - landmarkPromoteTimeNs[i]
            val sparkleVal = if (ageNs in 0L..sparkleDurationNs) {
                1.0f + (1.0f - ageNs.toFloat() / sparkleDurationNs) // 1.0 .. 2.0 (sparkle boost)
            } else {
                1.0f // Steady Emerald Gem
            }
            out[count * 4 + 3] = sparkleVal
            count++
        }

        // 2. Currently Visible Candidates (Amber = 0.0f)
        for (i in 0 until candidateCount) {
            if (count >= maxVerts) break
            val id = candIds[i]
            if (isLandmark(id)) continue
            out[count * 4] = candX[i]
            out[count * 4 + 1] = candY[i]
            out[count * 4 + 2] = candZ[i]
            out[count * 4 + 3] = 0.0f // Amber
            count++
        }

        return count
    }

    /**
     * Realigns all permanent landmarks using the estimated rigid transformation.
     * Called when tracking re-anchors with an offset and repointing at an existing
     * object detects a coordinate drift.
     */
    fun transformLandmarks(transform: RelocalizationRecovery.RigidTransform) {
        val out = FloatArray(3)
        for (i in 0 until verifiedLandmarkCount) {
            transform.transformPoint(landmarkX[i], landmarkY[i], landmarkZ[i], out)
            landmarkX[i] = out[0]
            landmarkY[i] = out[1]
            landmarkZ[i] = out[2]

            transform.rotateVector(landmarkNormX[i], landmarkNormY[i], landmarkNormZ[i], out)
            landmarkNormX[i] = out[0]
            landmarkNormY[i] = out[1]
            landmarkNormZ[i] = out[2]
        }
    }

    fun getLandmarkData(outX: FloatArray, outY: FloatArray, outZ: FloatArray): Int {
        val count = minOf(verifiedLandmarkCount, outX.size, outY.size, outZ.size)
        System.arraycopy(landmarkX, 0, outX, 0, count)
        System.arraycopy(landmarkY, 0, outY, 0, count)
        System.arraycopy(landmarkZ, 0, outZ, 0, count)
        return count
    }

    fun reset() {
        verifiedLandmarkCount = 0
        candidateCount = 0
        candHashKeys.fill(EMPTY_KEY)
        candHashSlots.fill(-1)
        landmarkHashKeys.fill(EMPTY_KEY)
        candStableCount.fill(0)
        smoothedJitterM = 0f
    }

    private fun findCandSlot(id: Int): Int {
        val mask = candHashKeys.size - 1
        var idx = (id * 0x45d9f3b) and mask
        while (true) {
            val k = candHashKeys[idx]
            if (k == id) return candHashSlots[idx]
            if (k == EMPTY_KEY) return -1
            idx = (idx + 1) and mask
        }
    }

    private fun allocateCandSlot(id: Int): Int? {
        if (candidateCount >= maxCandidates) return null
        val slot = candidateCount
        val mask = candHashKeys.size - 1
        var idx = (id * 0x45d9f3b) and mask
        while (candHashKeys[idx] != EMPTY_KEY && candHashKeys[idx] != id) {
            idx = (idx + 1) and mask
        }
        candHashKeys[idx] = id
        candHashSlots[idx] = slot
        return slot
    }

    companion object {
        private const val EMPTY_KEY = -1

        /** Smallest power of two >= 2 * [entries]; see the hash map fields for why it must be one. */
        internal fun tableSizeFor(entries: Int): Int = Integer.highestOneBit(entries * 2 - 1) * 2
        /** Target count of solid verified landmarks across a typical room to reach 100% (200 solid points). */
        const val TARGET_ROOM_LANDMARKS = 500
        /** Minimum spatial distance between solid permanent landmarks (0.12 m). */
        const val MIN_LANDMARK_SPACING_M = 0.06f
        /** Minimum viewing angle baseline to trigger ray triangulation (projet.md §2.3). */
        const val MIN_PARALLAX_DEG = 8.0f
        /** Minimum physical camera translation baseline. */
        const val MIN_TRANSLATION_M = 0.20f
        /** Max geometric distance between two rays at closest approach (tightened to 9cm). */
        const val MAX_RESIDUAL_M = 0.09f
        /** Eviction threshold for unverified candidate points when out of frame (1.2 seconds). */
        const val CANDIDATE_STALE_TIMEOUT_NS = 1_200_000_000L
        /** Maximum candidate frame-to-frame jitter threshold to be considered solidly anchored. */
        const val WOBBLE_THRESHOLD_M = 0.035f
    }
}
