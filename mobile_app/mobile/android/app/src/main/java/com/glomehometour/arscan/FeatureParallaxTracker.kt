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
    private val maxLandmarks: Int = 35000,
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
    private val landmarkLastSeenNs = LongArray(maxLandmarks)
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
    // table size is a power of two.
    // Fast open-addressed hash map for active candidates: pointId -> slot
    private val candHashKeys = IntArray(tableSizeFor(maxCandidates)) { EMPTY_KEY }
    private val candHashSlots = IntArray(tableSizeFor(maxCandidates)) { -1 }

    // Fast open-addressed hash map for verified landmarks: pointId -> slot
    private val landmarkHashKeys = IntArray(tableSizeFor(maxLandmarks)) { EMPTY_KEY }
    private val landmarkHashSlots = IntArray(tableSizeFor(maxLandmarks)) { -1 }

    // Fast open-addressed 3D spatial hash grid for O(1) landmark spacing check
    private val gridTableSize = tableSizeFor(maxLandmarks * 2)
    private val gridCellKeys = LongArray(gridTableSize) { EMPTY_GRID_KEY }
    private val gridLandmarkSlots = IntArray(gridTableSize) { -1 }

    // Target verified landmarks to reach 100% room coverage
    var targetLandmarkCount: Int = TARGET_ROOM_LANDMARKS
        private set

    private var hasComputedMedian: Boolean = false

    /** Dynamic running average median distance of features from camera (metres). */
    var runningMedianDistanceM: Float = 2.0f
        private set

    /** Max allowed distance for inliers (2.5x running average median distance). Defaults to 12m until calibrated. */
    var dynamicMaxDistanceM: Float = 12.0f
        private set

    /** Monotonically increasing coverage progress 0.0f .. 1.0f */
    val coverageFraction: Float
        get() = (verifiedLandmarkCount.toFloat() / targetLandmarkCount).coerceIn(0f, 1f)

    val verifiedFraction: Float
        get() = coverageFraction

    val activeCandidateCount: Int
        get() {
            var c = 0
            for (i in 0 until candidateCount) {
                if (!isLandmark(candIds[i])) c++
            }
            return c
        }

    val totalTrackedCount: Int
        get() = verifiedLandmarkCount + activeCandidateCount

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
        allowNewLandmarks: Boolean = true,
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

            // 1. If this ID is already a persistent verified landmark, update its last-seen timestamp
            val lmSlot = landmarkSlotForId(id)
            if (lmSlot >= 0) {
                landmarkLastSeenNs[lmSlot] = timestampNs
                continue
            }

            // 2. Check in candidate pool
            val slot = findCandSlot(id)
            if (slot >= 0) {
                // If candidate is outside dynamic max distance, evict it immediately on the fly
                val dx0 = cameraX - px; val dy0 = cameraY - py; val dz0 = cameraZ - pz
                val d0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
                if (d0 > dynamicMaxDistanceM) {
                    candLastSeenNs[slot] = 0L
                    continue
                }

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
                    if (!allowNewLandmarks) {
                        // Fast motion or recovering tracking: keep candidate position estimate but suspend landmark promotion
                        candX[slot] = px
                        candY[slot] = py
                        candZ[slot] = pz
                        continue
                    }

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

                    // Near plane safeguard (>= 0.3m) and dynamic room envelope ceiling (<= dynamicMaxDistanceM)
                    val dCamX = cameraX - targetX
                    val dCamY = cameraY - targetY
                    val dCamZ = cameraZ - targetZ
                    val distFromCam = sqrt(dCamX * dCamX + dCamY * dCamY + dCamZ * dCamZ)
                    val validDist = distFromCam in 0.3f..dynamicMaxDistanceM

                    val normX = if (distFromCam > 1e-4f) dCamX / distFromCam else 0f
                    val normY = if (distFromCam > 1e-4f) dCamY / distFromCam else 0f
                    val normZ = if (distFromCam > 1e-4f) dCamZ / distFromCam else 1f

                    // Promote candidate to permanent verified landmark if space permits and spatially distinct
                    if (validDist && canPromoteLandmark(targetX, targetY, targetZ)) {
                        addPermanentLandmark(id, targetX, targetY, targetZ, normX, normY, normZ, timestampNs)
                    } else {
                        if (distFromCam > dynamicMaxDistanceM) {
                            // Outlier beyond dynamic max distance: evict immediately on the fly
                            candLastSeenNs[slot] = 0L
                        } else {
                            // Keep current estimate if close to other landmarks
                            candX[slot] = targetX
                            candY[slot] = targetY
                            candZ[slot] = targetZ
                        }
                    }
                } else {
                    // Update latest consensus position from ARCore
                    candX[slot] = px
                    candY[slot] = py
                    candZ[slot] = pz
                }
            } else {
                if (!allowNewLandmarks) continue

                val dx = cameraX - px; val dy = cameraY - py; val dz = cameraZ - pz
                val dist = sqrt(dx * dx + dy * dy + dz * dz)
                if (dist <= 1e-4f || dist > dynamicMaxDistanceM) continue

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

                candRayX[newSlot] = dx / dist
                candRayY[newSlot] = dy / dist
                candRayZ[newSlot] = dz / dist
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

    private fun cellCoord(v: Float): Int = kotlin.math.floor(v / MIN_LANDMARK_SPACING_M).toInt()

    private fun cellKey(cx: Int, cy: Int, cz: Int): Long {
        return ((cx.toLong() and 0x1FFFFFL) shl 42) or
               ((cy.toLong() and 0x1FFFFFL) shl 21) or
               (cz.toLong() and 0x1FFFFFL)
    }

    private fun hashCellKey(key: Long): Int {
        val h = (key xor (key ushr 32)) * 0x45d9f3bL
        return (h xor (h ushr 32)).toInt()
    }

    private fun findGridSlot(key: Long): Int {
        val mask = gridTableSize - 1
        var idx = hashCellKey(key) and mask
        while (gridCellKeys[idx] != EMPTY_GRID_KEY) {
            if (gridCellKeys[idx] == key) return gridLandmarkSlots[idx]
            idx = (idx + 1) and mask
        }
        return -1
    }

    private fun insertGridCell(key: Long, slot: Int) {
        val mask = gridTableSize - 1
        var idx = hashCellKey(key) and mask
        while (gridCellKeys[idx] != EMPTY_GRID_KEY && gridCellKeys[idx] != key) {
            idx = (idx + 1) and mask
        }
        gridCellKeys[idx] = key
        gridLandmarkSlots[idx] = slot
    }

    private fun rebuildSpatialGrid() {
        gridCellKeys.fill(EMPTY_GRID_KEY)
        gridLandmarkSlots.fill(-1)
        for (i in 0 until verifiedLandmarkCount) {
            val key = cellKey(cellCoord(landmarkX[i]), cellCoord(landmarkY[i]), cellCoord(landmarkZ[i]))
            insertGridCell(key, i)
        }
    }

    /**
     * Checks if a new landmark can be added (spatial separation filter).
     * Probes 27 neighboring cells in the spatial hash grid in constant O(1) time.
     */
    private fun canPromoteLandmark(x: Float, y: Float, z: Float): Boolean {
        if (verifiedLandmarkCount >= maxLandmarks) return false
        val cx = cellCoord(x)
        val cy = cellCoord(y)
        val cz = cellCoord(z)
        val minDistSq = MIN_LANDMARK_SPACING_M * MIN_LANDMARK_SPACING_M

        for (dx in -1..1) {
            for (dy in -1..1) {
                for (dz in -1..1) {
                    val key = cellKey(cx + dx, cy + dy, cz + dz)
                    val slot = findGridSlot(key)
                    if (slot >= 0) {
                        val lx = landmarkX[slot]
                        val ly = landmarkY[slot]
                        val lz = landmarkZ[slot]
                        val distSq = (lx - x) * (lx - x) + (ly - y) * (ly - y) + (lz - z) * (lz - z)
                        if (distSq < minDistSq) {
                            return false // Too close to an existing solid landmark
                        }
                    }
                }
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
        landmarkLastSeenNs[slot] = timestampNs
        verifiedLandmarkCount++

        // Insert into landmark hash map
        val mask = landmarkHashKeys.size - 1
        var idx = (id * 0x45d9f3b) and mask
        while (landmarkHashKeys[idx] != EMPTY_KEY) {
            idx = (idx + 1) and mask
        }
        landmarkHashKeys[idx] = id
        landmarkHashSlots[idx] = slot

        // Insert into spatial grid
        val gridKey = cellKey(cellCoord(x), cellCoord(y), cellCoord(z))
        insertGridCell(gridKey, slot)
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
        rebuildSpatialGrid()
    }

    fun getLandmarkData(outX: FloatArray, outY: FloatArray, outZ: FloatArray): Int {
        val count = minOf(verifiedLandmarkCount, outX.size, outY.size, outZ.size)
        System.arraycopy(landmarkX, 0, outX, 0, count)
        System.arraycopy(landmarkY, 0, outY, 0, count)
        System.arraycopy(landmarkZ, 0, outZ, 0, count)
        return count
    }

    /**
     * Extracts coordinates of landmarks that were actively seen recently (since [sinceNs])
     * or are located within [maxDistM] of the camera reference point ([camX], [camY], [camZ]).
     * Crucial for relocalization recovery: focuses alignment search on the physical space
     * the user was just scanning right before tracking loss.
     */
    fun getRecentLandmarkData(
        outX: FloatArray, outY: FloatArray, outZ: FloatArray,
        sinceNs: Long,
        camX: Float, camY: Float, camZ: Float,
        maxDistM: Float = 3.5f,
        fallbackToAll: Boolean = true,
    ): Int {
        val maxOut = minOf(outX.size, outY.size, outZ.size)
        val maxDistSq = maxDistM * maxDistM
        var count = 0
        for (i in 0 until verifiedLandmarkCount) {
            if (count >= maxOut) break
            val seenRecently = landmarkLastSeenNs[i] >= sinceNs
            val dx = landmarkX[i] - camX
            val dy = landmarkY[i] - camY
            val dz = landmarkZ[i] - camZ
            val dSq = dx * dx + dy * dy + dz * dz
            if (seenRecently || dSq <= maxDistSq) {
                outX[count] = landmarkX[i]
                outY[count] = landmarkY[i]
                outZ[count] = landmarkZ[i]
                count++
            }
        }
        if (fallbackToAll && count < 6 && verifiedLandmarkCount >= 12) {
            return getLandmarkData(outX, outY, outZ)
        }
        return count
    }

    data class ScanDimensions(
        val widthM: Float,
        val lengthM: Float,
        val heightM: Float,
        val floorAreaSqm: Float,
    )

    /**
     * Dynamically computes the bounding dimensions (width, length, height, floor area)
     * of the scanned space using robust outlier rejection and orientation-aligned
     * bounding box (OBB) fitting across camera trajectory extents and inlier landmarks.
     */
    fun computeDimensions(
        camMinX: Float = Float.MAX_VALUE, camMaxX: Float = -Float.MAX_VALUE,
        camMinY: Float = Float.MAX_VALUE, camMaxY: Float = -Float.MAX_VALUE,
        camMinZ: Float = Float.MAX_VALUE, camMaxZ: Float = -Float.MAX_VALUE,
        maxDistM: Float = dynamicMaxDistanceM,
    ): ScanDimensions {
        val hasCam = camMinX <= camMaxX && camMinZ <= camMaxZ
        var minY = if (hasCam) camMinY else Float.MAX_VALUE
        var maxY = if (hasCam) camMaxY else -Float.MAX_VALUE

        // 1. Filter landmarks within dynamic max distance of camera trajectory bounding box
        val maxPoints = verifiedLandmarkCount + (if (hasCam) 4 else 0)
        if (maxPoints == 0) {
            return ScanDimensions(0f, 0f, 0f, 0f)
        }

        val inlierX = FloatArray(maxPoints)
        val inlierZ = FloatArray(maxPoints)
        var inlierCount = 0

        if (hasCam) {
            inlierX[inlierCount] = camMinX; inlierZ[inlierCount++] = camMinZ
            inlierX[inlierCount] = camMaxX; inlierZ[inlierCount++] = camMinZ
            inlierX[inlierCount] = camMinX; inlierZ[inlierCount++] = camMaxZ
            inlierX[inlierCount] = camMaxX; inlierZ[inlierCount++] = camMaxZ
        }

        val maxDistSq = maxDistM * maxDistM
        for (i in 0 until verifiedLandmarkCount) {
            val x = landmarkX[i]
            val y = landmarkY[i]
            val z = landmarkZ[i]

            if (hasCam) {
                val dx = maxOf(0f, maxOf(camMinX - x, x - camMaxX))
                val dz = maxOf(0f, maxOf(camMinZ - z, z - camMaxZ))
                if (dx * dx + dz * dz > maxDistSq) {
                    continue // Landmark is outside the active room envelope (window, hallway, reflection)
                }
            }

            inlierX[inlierCount] = x
            inlierZ[inlierCount++] = z
            if (y < minY) minY = y
            if (y > maxY) maxY = y
        }

        if (inlierCount == 0) {
            return ScanDimensions(0f, 0f, 0f, 0f)
        }

        // 2. Compute 2D Oriented Bounding Box (OBB) across test angles (0° to 85° in 5° steps)
        // to find the tightest enclosing rectangle aligned with room walls.
        var bestArea = Float.MAX_VALUE
        var bestWidth = 0f
        var bestLength = 0f

        val projU = FloatArray(inlierCount)
        val projV = FloatArray(inlierCount)

        // Test 18 angles: 0, 5, 10, ..., 85 degrees
        for (step in 0 until 18) {
            val angleRad = Math.toRadians((step * 5).toDouble()).toFloat()
            val cosA = kotlin.math.cos(angleRad)
            val sinA = kotlin.math.sin(angleRad)

            for (i in 0 until inlierCount) {
                val px = inlierX[i]
                val pz = inlierZ[i]
                projU[i] = px * cosA + pz * sinA
                projV[i] = -px * sinA + pz * cosA
            }

            val spanU: Float
            val spanV: Float

            if (inlierCount >= 20) {
                projU.sort()
                projV.sort()
                // 2nd to 98th percentile to discard stray boundary points
                val lowIdx = (inlierCount * 0.02f).toInt().coerceIn(0, inlierCount - 1)
                val highIdx = (inlierCount * 0.98f).toInt().coerceIn(0, inlierCount - 1)
                spanU = if (highIdx > lowIdx) (projU[highIdx] - projU[lowIdx]) / 0.96f else 0f
                spanV = if (highIdx > lowIdx) (projV[highIdx] - projV[lowIdx]) / 0.96f else 0f
            } else {
                var minU = Float.MAX_VALUE; var maxU = -Float.MAX_VALUE
                var minV = Float.MAX_VALUE; var maxV = -Float.MAX_VALUE
                for (i in 0 until inlierCount) {
                    val u = projU[i]; val v = projV[i]
                    if (u < minU) minU = u
                    if (u > maxU) maxU = u
                    if (v < minV) minV = v
                    if (v > maxV) maxV = v
                }
                spanU = if (minU < maxU) maxU - minU else 0f
                spanV = if (minV < maxV) maxV - minV else 0f
            }

            val area = spanU * spanV
            if (area > 0f && area < bestArea) {
                bestArea = area
                bestWidth = minOf(spanU, spanV)
                bestLength = maxOf(spanU, spanV)
            }
        }

        val height = if (minY < maxY) maxY - minY else 0f
        return ScanDimensions(
            widthM = bestWidth,
            lengthM = bestLength,
            heightM = height,
            floorAreaSqm = if (bestArea != Float.MAX_VALUE) bestArea else 0f
        )
    }

    /**
     * Updates the running average median distance of observed features from the camera,
     * and sets the dynamic max inlier distance to 2.5x that median.
     */
    fun updateRunningMedianDistance(camX: Float, camY: Float, camZ: Float): Float {
        val totalPoints = verifiedLandmarkCount + candidateCount
        if (totalPoints == 0) return dynamicMaxDistanceM

        val dists = FloatArray(totalPoints)
        var count = 0
        for (i in 0 until verifiedLandmarkCount) {
            val dx = landmarkX[i] - camX
            val dy = landmarkY[i] - camY
            val dz = landmarkZ[i] - camZ
            dists[count++] = sqrt(dx * dx + dy * dy + dz * dz)
        }
        for (i in 0 until candidateCount) {
            val id = candIds[i]
            if (isLandmark(id)) continue
            val dx = candX[i] - camX
            val dy = candY[i] - camY
            val dz = candZ[i] - camZ
            dists[count++] = sqrt(dx * dx + dy * dy + dz * dz)
        }

        if (count >= 5) {
            dists.sort(0, count)
            val currentMedian = dists[count / 2]
            val clampedMedian = currentMedian.coerceIn(1.2f, 20.0f)
            runningMedianDistanceM = if (hasComputedMedian) {
                runningMedianDistanceM * 0.7f + clampedMedian * 0.3f
            } else {
                hasComputedMedian = true
                clampedMedian
            }
            dynamicMaxDistanceM = runningMedianDistanceM * 2.5f
        }

        return dynamicMaxDistanceM
    }

    /**
     * Prunes verified landmarks that are beyond the dynamic room envelope.
     * A landmark is an outlier if its distance from the camera trajectory bounding box
     * exceeds dynamicMaxDistanceM.
     * Returns the number of pruned landmarks.
     */
    fun pruneOutliers(
        camMinX: Float, camMaxX: Float,
        camMinY: Float = Float.MAX_VALUE, camMaxY: Float = -Float.MAX_VALUE,
        camMinZ: Float, camMaxZ: Float,
    ): Int {
        val hasCam = camMinX <= camMaxX && camMinZ <= camMaxZ
        if (!hasCam || verifiedLandmarkCount == 0) return 0

        val maxDistSq = dynamicMaxDistanceM * dynamicMaxDistanceM
        val hasY = camMinY <= camMaxY
        var writeIdx = 0
        var pruned = 0

        for (readIdx in 0 until verifiedLandmarkCount) {
            val x = landmarkX[readIdx]
            val y = landmarkY[readIdx]
            val z = landmarkZ[readIdx]

            val dx = maxOf(0f, maxOf(camMinX - x, x - camMaxX))
            val dy = if (hasY) maxOf(0f, maxOf(camMinY - y, y - camMaxY)) else 0f
            val dz = maxOf(0f, maxOf(camMinZ - z, z - camMaxZ))
            val distSq = dx * dx + dy * dy + dz * dz

            if (distSq > maxDistSq) {
                pruned++
                continue
            }

            if (writeIdx != readIdx) {
                landmarkIds[writeIdx] = landmarkIds[readIdx]
                landmarkX[writeIdx] = x
                landmarkY[writeIdx] = y
                landmarkZ[writeIdx] = z
                landmarkNormX[writeIdx] = landmarkNormX[readIdx]
                landmarkNormY[writeIdx] = landmarkNormY[readIdx]
                landmarkNormZ[writeIdx] = landmarkNormZ[readIdx]
                landmarkPromoteTimeNs[writeIdx] = landmarkPromoteTimeNs[readIdx]
                landmarkLastSeenNs[writeIdx] = landmarkLastSeenNs[readIdx]
            }
            writeIdx++
        }

        if (pruned > 0) {
            verifiedLandmarkCount = writeIdx
            landmarkHashKeys.fill(EMPTY_KEY)
            landmarkHashSlots.fill(-1)
            val mask = landmarkHashKeys.size - 1
            for (i in 0 until verifiedLandmarkCount) {
                val id = landmarkIds[i]
                var idx = (id * 0x45d9f3b) and mask
                while (landmarkHashKeys[idx] != EMPTY_KEY) {
                    idx = (idx + 1) and mask
                }
                landmarkHashKeys[idx] = id
                landmarkHashSlots[idx] = i
            }
            rebuildSpatialGrid()
        }

        return pruned
    }

    fun setTargetLandmarks(newTarget: Int) {
        targetLandmarkCount = newTarget.coerceIn(MIN_ROOM_LANDMARKS, 30000)
    }

    fun updateTargetLandmarks(newTarget: Int) {
        setTargetLandmarks(newTarget)
    }

    fun resetTargetLandmarks(baseTarget: Int = TARGET_ROOM_LANDMARKS) {
        targetLandmarkCount = baseTarget
    }

    fun reset() {
        verifiedLandmarkCount = 0
        candidateCount = 0
        candHashKeys.fill(EMPTY_KEY)
        candHashSlots.fill(-1)
        landmarkHashKeys.fill(EMPTY_KEY)
        gridCellKeys.fill(EMPTY_GRID_KEY)
        gridLandmarkSlots.fill(-1)
        candStableCount.fill(0)
        smoothedJitterM = 0f
        targetLandmarkCount = TARGET_ROOM_LANDMARKS
        hasComputedMedian = false
        runningMedianDistanceM = 2.0f
        dynamicMaxDistanceM = 12.0f
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
        private const val EMPTY_GRID_KEY = Long.MIN_VALUE

        /** Smallest power of two >= 2 * [entries]; see the hash map fields for why it must be one. */
        internal fun tableSizeFor(entries: Int): Int = Integer.highestOneBit(entries * 2 - 1) * 2
        /** Minimum base target count of solid verified landmarks for the smallest rooms / scan start (500 points). */
        const val MIN_ROOM_LANDMARKS = 500
        const val TARGET_ROOM_LANDMARKS = MIN_ROOM_LANDMARKS
        /** Minimum spatial distance between solid permanent landmarks (0.025 m / 2.5 cm). */
        const val MIN_LANDMARK_SPACING_M = 0.025f
        /** Minimum viewing angle baseline to trigger ray triangulation (projet.md §2.3). */
        const val MIN_PARALLAX_DEG = 8.0f
        /** Minimum physical camera translation baseline. */
        const val MIN_TRANSLATION_M = 0.20f
        /** Max geometric distance between two rays at closest approach (restored from 7945df5). */
        const val MAX_RESIDUAL_M = 0.25f
        /** Eviction threshold for unverified candidate points when out of frame (1.2 seconds). */
        const val CANDIDATE_STALE_TIMEOUT_NS = 1_200_000_000L
        /** Maximum candidate frame-to-frame jitter threshold to be considered solidly anchored. */
        const val WOBBLE_THRESHOLD_M = 0.10f
    }
}
