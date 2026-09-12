package com.glomehometour.arscan

import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Sparse 10 cm occupancy grid with per-voxel parallax bookkeeping (SPEC §2.3).
 *
 * Deliberately an open-addressed hash over parallel primitive arrays rather than a
 * HashMap<Long, Voxel>: free-space carving inserts on the order of 10^5 voxels per scan and
 * touches tens of thousands per second, and boxing that many Longs is a GC problem, not a style
 * preference. It's the one hand-rolled data structure in the app and the reason it exists.
 *
 * Three stored states -- FREE, OCCUPIED, OCCLUDED; absent means unknown (README §5). OCCLUDED is
 * not "unknown": it's volume actively behind a known surface from every observation so far (the
 * far side of a kitchen island), which is exactly the "walk around this obstacle" guidance signal
 * (§6's Occlusion Guidance Arrows) -- a plain "never seen" voxel could be open floor two rooms
 * away, and pointing the operator there instead is the wrong instruction. `SPEC.md` §1.2i dropped
 * this as redundant with frontier detection; it isn't -- frontier finds unexplored space bounded
 * by what's already mapped, OCCLUDED finds space specifically hidden behind a surface, and they
 * point the operator in different directions in the kitchen-island case.
 *
 * Pure Kotlin, no Android imports, so it's unit-testable on the JVM (see VoxelGridTest).
 */
class VoxelGrid(
    val voxelSizeM: Float = 0.10f,
    initialCapacity: Int = 1 shl 17,
) {

    private var keys = LongArray(initialCapacity) { EMPTY }
    private var states = ByteArray(initialCapacity)
    /** First observation bearing per occupied voxel (voxel -> camera, unit), 3 floats per slot. */
    private var bearings = FloatArray(initialCapacity * 3)
    /** Smallest cosine seen between the first bearing and any later one, i.e. the widest
     * parallax angle so far. Starts at 1 (0 degrees). */
    private var widestCos = FloatArray(initialCapacity) { 1f }
    private var entries = 0

    var occupiedCount = 0
        private set
    var freeCount = 0
        private set
    var occludedCount = 0
        private set
    var verifiedCount = 0
        private set

    /** Set once the grid refuses to grow further; surfaced in coverage_summary.json so a scan
     * that silently stopped mapping is distinguishable from one that saw nothing. */
    var full = false
        private set

    val capacity: Int get() = keys.size

    /** Parallax-verified fraction of mapped surface -- the number the progress bar shows. */
    fun coverageFraction(): Float =
        if (occupiedCount == 0) 0f else verifiedCount.toFloat() / occupiedCount

    /**
     * Folds one depth/feature sample into the grid: carve free space along the viewing ray,
     * mark the hit voxel occupied, and update that voxel's parallax record.
     *
     * Carving stops CARVE_BACKOFF voxels short of the hit so the surface itself is never
     * scribbled over by its own ray, and FREE never overwrites OCCUPIED (occupied wins) -- the
     * standard cheap alternative to log-odds fusion. Ceiling: a genuinely wrong occupied voxel
     * (a depth outlier) can therefore never be cleared; upgrade to a hit/miss counter per voxel
     * if speckle turns out to matter in the field.
     */
    fun integrate(cx: Float, cy: Float, cz: Float, px: Float, py: Float, pz: Float) {
        val dx = px - cx
        val dy = py - cy
        val dz = pz - cz
        val dist = sqrt(dx * dx + dy * dy + dz * dz)
        if (dist < MIN_RANGE_M || dist > MAX_RANGE_M) return

        val ux = dx / dist
        val uy = dy / dist
        val uz = dz / dist

        // Fixed half-voxel stepping instead of an exact Amanatides-Woo DDA: at this step size the
        // only voxels it can miss are diagonal grazes, and a missed *free* voxel costs nothing
        // beyond a slightly conservative frontier. ponytail: half-voxel march, swap in a real DDA
        // if free-space carving ever shows up in a profile.
        val step = voxelSizeM * 0.5f
        val stop = dist - voxelSizeM * CARVE_BACKOFF
        var t = voxelSizeM // skip the voxel the camera itself sits in
        while (t < stop) {
            markFree(cx + ux * t, cy + uy * t, cz + uz * t)
            t += step
        }

        // Bearing is voxel -> camera: the direction this surface was looked at *from*.
        markOccupied(px, py, pz, -ux, -uy, -uz)

        // Behind the hit: actively hidden from this viewpoint, not merely unmapped (README §5).
        // Capped at OCCLUSION_DEPTH_M rather than run to MAX_RANGE_M -- a wall behind a wall is
        // still just unknown to this app, and an unbounded carve would flood the grid with
        // OCCLUDED volume no guidance logic ever needs to reach.
        val occludedStop = dist + OCCLUSION_DEPTH_M
        var b = dist + voxelSizeM * CARVE_BACKOFF
        while (b < occludedStop) {
            markOccluded(cx + ux * b, cy + uy * b, cz + uz * b)
            b += step
        }
    }

    private fun markFree(x: Float, y: Float, z: Float) {
        val slot = slotFor(keyOf(x, y, z)) ?: return
        when (states[slot]) {
            UNSET -> { states[slot] = FREE; freeCount++ }
            // Promoted: a later ray passed clean through what a previous viewpoint could only
            // infer was hidden, so it's resolved to FREE (README §5).
            OCCLUDED -> { states[slot] = FREE; occludedCount--; freeCount++ }
            else -> Unit // FREE stays FREE, OCCUPIED is never overwritten by a carve.
        }
    }

    private fun markOccupied(x: Float, y: Float, z: Float, bx: Float, by: Float, bz: Float) {
        val slot = slotFor(keyOf(x, y, z)) ?: return
        when (states[slot]) {
            OCCUPIED -> {
                val cos = (bearings[slot * 3] * bx + bearings[slot * 3 + 1] * by + bearings[slot * 3 + 2] * bz)
                    .coerceIn(-1f, 1f)
                if (cos < widestCos[slot]) {
                    val wasVerified = widestCos[slot] <= PARALLAX_COS
                    widestCos[slot] = cos
                    if (!wasVerified && cos <= PARALLAX_COS) verifiedCount++
                }
            }
            else -> {
                if (states[slot] == FREE) freeCount--
                if (states[slot] == OCCLUDED) occludedCount--
                states[slot] = OCCUPIED
                occupiedCount++
                bearings[slot * 3] = bx
                bearings[slot * 3 + 1] = by
                bearings[slot * 3 + 2] = bz
                widestCos[slot] = 1f
            }
        }
    }

    /** Never overwrites FREE or OCCUPIED -- occlusion is only ever a claim about volume nothing
     * has resolved yet, and a resolved voxel outranks it regardless of which came first. */
    private fun markOccluded(x: Float, y: Float, z: Float) {
        val slot = slotFor(keyOf(x, y, z)) ?: return
        if (states[slot] == UNSET) {
            states[slot] = OCCLUDED
            occludedCount++
        }
    }

    /** State of the voxel containing a world point: UNSET, FREE, OCCUPIED or OCCLUDED. */
    fun stateAt(x: Float, y: Float, z: Float): Byte {
        val slot = find(keyOf(x, y, z))
        return if (slot < 0) UNSET else states[slot]
    }

    /** Widest parallax angle recorded for the voxel containing a point, in degrees; 0 if it
     * isn't an occupied voxel. */
    fun parallaxDegAt(x: Float, y: Float, z: Float): Float {
        val slot = find(keyOf(x, y, z))
        if (slot < 0 || states[slot] != OCCUPIED) return 0f
        return Math.toDegrees(kotlin.math.acos(widestCos[slot].toDouble())).toFloat()
    }

    /**
     * Coverage mesh overlay (README §5/§6, Phase_1.md item 5): a triangle soup covering every
     * OCCUPIED voxel's *exposed* faces -- naive per-voxel cube-face culling, the standard cheap
     * voxel mesher (a face is skipped when its neighbour is also OCCUPIED, so interior faces
     * between two solid voxels are never emitted). Four floats per vertex (x, y, z, progress
     * 0..1 toward parallax-verified), replacing the flat point cloud this used to publish.
     *
     * ponytail: no greedy face merging, so a flat wall is hundreds of unit quads instead of one
     * big one -- more vertices than a real mesher would emit, but still bounded and cheap at this
     * resolution/scan size. Upgrade to greedy meshing if the vertex budget ever gets tight.
     */
    fun meshTriangles(maxTriangles: Int, floorY: Float? = null): FloatArray {
        val maxVerts = maxTriangles * 2
        var out = FloatArray(minOf(maxVerts, INITIAL_MESH_VERTS) * 4)
        var n = 0
        val h = voxelSizeM * 0.5f

        for (slot in keys.indices) {
            if (states[slot] != OCCUPIED) continue
            if (n >= maxVerts) break
            val key = keys[slot]
            val ix = unpackX(key)
            val iy = unpackY(key)
            val iz = unpackZ(key)
            val cy = centre(iy)

            // Surface-only filter: keep floor surface plane and vertical wall contours, discard floating mid-air artifacts
            val isFloor = if (floorY != null) {
                (cy - floorY) in -0.35f..0.35f
            } else true

            val isWall = if (floorY != null) {
                (cy - floorY) in 0.35f..2.40f && hasVerticalNeighbour(ix, iy, iz)
            } else hasSurfaceManifold(ix, iy, iz)

            if (!isFloor && !isWall) continue

            val cx = centre(ix)
            val cz = centre(iz)

            for (face in QUAD_FACES) {
                if (n + 10 > maxVerts) break
                val isHoriz = face.dy != 0
                if (isHoriz && !isFloor) continue
                if (!isHoriz && !isWall) continue

                val nSlot = find(pack(ix + face.dx, iy + face.dy, iz + face.dz))
                if (nSlot >= 0 && states[nSlot] == OCCUPIED) continue // interior face, hidden

                if ((n + 10) * 4 > out.size) out = out.copyOf(minOf(out.size * 2, maxVerts * 4))
                val planeType = if (isHoriz) 0.0f else 1.0f // 0 = Horizontal (White), 1 = Vertical (Blue)

                // Emit 5 lines (10 vertices) for the 2 triangles of the quad
                for (idx in LINE_INDICES) {
                    val vx = face.corners[idx * 3]
                    val vy = face.corners[idx * 3 + 1]
                    val vz = face.corners[idx * 3 + 2]
                    out[n * 4] = cx + vx * h
                    out[n * 4 + 1] = cy + vy * h
                    out[n * 4 + 2] = cz + vz * h
                    out[n * 4 + 3] = planeType
                    n++
                }
            }
        }
        return out.copyOf(n * 4)
    }

    private class QuadFace(val dx: Int, val dy: Int, val dz: Int, val corners: FloatArray)

    private fun hasVerticalNeighbour(ix: Int, iy: Int, iz: Int): Boolean {
        for (dy in -1..1 step 2) {
            val nSlot = find(pack(ix, iy + dy, iz))
            if (nSlot >= 0 && states[nSlot] == OCCUPIED) return true
        }
        return false
    }

    /**
     * Filters out isolated floating mid-air voxels/cubes. A voxel belongs to a physical surface
     * (wall, floor, object surface) if it has at least 1 occupied 26-neighbor forming a surface/mesh.
     */
    private fun hasSurfaceManifold(ix: Int, iy: Int, iz: Int): Boolean {
        for (dx in -1..1) {
            for (dy in -1..1) {
                for (dz in -1..1) {
                    if (dx == 0 && dy == 0 && dz == 0) continue
                    val nSlot = find(pack(ix + dx, iy + dy, iz + dz))
                    if (nSlot >= 0 && states[nSlot] == OCCUPIED) return true
                }
            }
        }
        return false
    }

    private class Face(val dx: Int, val dy: Int, val dz: Int, val verts: FloatArray)

    data class Floorplan2D(
        val floorPoints: FloatArray, // [x0, z0, x1, z1, ...]
        val floorCount: Int,
        val wallPoints: FloatArray,  // [x0, z0, x1, z1, ...]
        val wallCount: Int,
        val minX: Float,
        val maxX: Float,
        val minZ: Float,
        val maxZ: Float,
    )

    /**
     * Extracts a top-down 2D floorplan projection of the mapped space:
     * - [floorPoints]: horizontal 2D footprint cells near floor height
     * - [wallPoints]: vertical structural wall slices between 0.8m and 1.6m above floor
     * - [minX], [maxX], [minZ], [maxZ]: real-time metric bounding box for dynamic auto-fit
     */
    fun floorplan2D(floorY: Float? = null): Floorplan2D {
        val baseFloorY = floorY ?: 0f
        var floor = FloatArray(minOf((occupiedCount + freeCount) * 2, 8192))
        var wall = FloatArray(minOf(occupiedCount * 2, 4096))
        var fCount = 0
        var wCount = 0

        var minX = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var minZ = Float.MAX_VALUE
        var maxZ = -Float.MAX_VALUE

        for (slot in keys.indices) {
            val state = states[slot]
            if (state != OCCUPIED && state != FREE) continue
            val key = keys[slot]
            val ix = unpackX(key)
            val iy = unpackY(key)
            val iz = unpackZ(key)
            val y = centre(iy)
            val x = centre(ix)
            val z = centre(iz)
            val dy = y - baseFloorY

            if (state == OCCUPIED) {
                // Chest-height wall slice (0.8m to 1.6m above floor)
                if (dy in 0.80f..1.60f) {
                    if (wCount * 2 + 2 > wall.size) wall = wall.copyOf(maxOf(wall.size * 2, 64))
                    wall[wCount * 2] = x
                    wall[wCount * 2 + 1] = z
                    wCount++
                }
                // Floor footprint
                if (dy in -0.30f..0.35f) {
                    if (fCount * 2 + 2 > floor.size) floor = floor.copyOf(maxOf(floor.size * 2, 64))
                    floor[fCount * 2] = x
                    floor[fCount * 2 + 1] = z
                    fCount++
                }
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (z < minZ) minZ = z
                if (z > maxZ) maxZ = z
            } else if (state == FREE) {
                // Free floor walking path
                if (dy in -0.30f..0.40f) {
                    if (fCount * 2 + 2 > floor.size) floor = floor.copyOf(maxOf(floor.size * 2, 64))
                    floor[fCount * 2] = x
                    floor[fCount * 2 + 1] = z
                    fCount++
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (z < minZ) minZ = z
                    if (z > maxZ) maxZ = z
                }
            }
        }

        if (minX > maxX) {
            minX = -1.5f; maxX = 1.5f
            minZ = -1.5f; maxZ = 1.5f
        }

        return Floorplan2D(
            floor, fCount,
            wall, wCount,
            minX, maxX, minZ, maxZ,
        )
    }

    /**
     * Guidance target: the centroid of the frontier cluster nearest the operator (SPEC §2.3).
     *
     * A frontier voxel is FREE with at least one unknown 6-neighbour -- the boundary between what
     * has been seen and what hasn't. That boundary is self-limiting in a way a raw "unobserved
     * voxel" count is not: unknown volume *behind* a wall or a kitchen island is walled off by
     * occupied voxels and never becomes frontier, so the operator is only ever sent toward space
     * they could actually walk into and see.
     *
     * Nearest cluster rather than global centroid: averaging every frontier in the flat lands the
     * target in the middle of a wall, pointing at nothing.
     */
    fun frontierTarget(
        camX: Float, camY: Float, camZ: Float,
        minY: Float, maxY: Float,
        out: FloatArray,
    ): Boolean {
        var candidates = FloatArray(1024)
        var n = 0
        var bestDistSq = Float.MAX_VALUE
        var bestIdx = -1

        for (slot in keys.indices) {
            if (states[slot] != FREE) continue
            val key = keys[slot]
            val ix = unpackX(key)
            val iy = unpackY(key)
            val iz = unpackZ(key)
            val y = centre(iy)
            if (y < minY || y > maxY) continue
            val x = centre(ix)
            val z = centre(iz)
            val dsq = (x - camX) * (x - camX) + (y - camY) * (y - camY) + (z - camZ) * (z - camZ)
            if (dsq > FRONTIER_RADIUS_M * FRONTIER_RADIUS_M) continue
            // Free space immediately around the operator always has unknown neighbours just
            // outside the field of view, so without a floor the "nearest frontier" is forever
            // 30 cm from the lens: a marker filling the screen and an arrow that spins on the
            // spot. Guidance has to point somewhere worth walking to.
            if (dsq < FRONTIER_MIN_DISTANCE_M * FRONTIER_MIN_DISTANCE_M) continue
            if (!hasUnknownNeighbour(ix, iy, iz)) continue

            if (n * 3 + 3 > candidates.size) candidates = candidates.copyOf(candidates.size * 2)
            candidates[n * 3] = x
            candidates[n * 3 + 1] = y
            candidates[n * 3 + 2] = z
            if (dsq < bestDistSq) {
                bestDistSq = dsq
                bestIdx = n
            }
            n++
        }
        if (bestIdx < 0) return false

        val nx = candidates[bestIdx * 3]
        val ny = candidates[bestIdx * 3 + 1]
        val nz = candidates[bestIdx * 3 + 2]
        var sx = 0f
        var sy = 0f
        var sz = 0f
        var count = 0
        for (i in 0 until n) {
            val dx = candidates[i * 3] - nx
            val dy = candidates[i * 3 + 1] - ny
            val dz = candidates[i * 3 + 2] - nz
            if (dx * dx + dy * dy + dz * dz > CLUSTER_RADIUS_M * CLUSTER_RADIUS_M) continue
            sx += candidates[i * 3]
            sy += candidates[i * 3 + 1]
            sz += candidates[i * 3 + 2]
            count++
        }
        out[0] = sx / count
        out[1] = sy / count
        out[2] = sz / count
        return true
    }

    private fun hasUnknownNeighbour(ix: Int, iy: Int, iz: Int): Boolean =
        find(pack(ix + 1, iy, iz)) < 0 || find(pack(ix - 1, iy, iz)) < 0 ||
            find(pack(ix, iy + 1, iz)) < 0 || find(pack(ix, iy - 1, iz)) < 0 ||
            find(pack(ix, iy, iz + 1)) < 0 || find(pack(ix, iy, iz - 1)) < 0

    /**
     * Guidance target: the centroid of the OCCLUDED cluster nearest the operator that borders a
     * known OCCUPIED surface (README §5/§6's "walk around this obstacle" arrow) -- the kitchen-
     * island case, distinct from [frontierTarget]'s unexplored-space signal (Phase_1.md §5/§6):
     * frontier points at space nobody has looked at yet, this points at space specifically hidden
     * *behind* something already mapped, which is a different instruction to give the operator.
     *
     * Structurally identical to frontierTarget (nearest-cluster-not-global-centroid, same height
     * band and minimum-distance floor) but walking OCCLUDED voxels adjacent to OCCUPIED ones
     * instead of FREE voxels adjacent to unknown ones.
     */
    fun occlusionTarget(
        camX: Float, camY: Float, camZ: Float,
        minY: Float, maxY: Float,
        out: FloatArray,
    ): Boolean {
        var candidates = FloatArray(1024)
        var n = 0
        var bestDistSq = Float.MAX_VALUE
        var bestIdx = -1

        for (slot in keys.indices) {
            if (states[slot] != OCCLUDED) continue
            val key = keys[slot]
            val ix = unpackX(key)
            val iy = unpackY(key)
            val iz = unpackZ(key)
            val y = centre(iy)
            if (y < minY || y > maxY) continue
            val x = centre(ix)
            val z = centre(iz)
            val dsq = (x - camX) * (x - camX) + (y - camY) * (y - camY) + (z - camZ) * (z - camZ)
            if (dsq > FRONTIER_RADIUS_M * FRONTIER_RADIUS_M) continue
            if (dsq < FRONTIER_MIN_DISTANCE_M * FRONTIER_MIN_DISTANCE_M) continue
            if (!hasOccupiedNeighbour(ix, iy, iz)) continue

            if (n * 3 + 3 > candidates.size) candidates = candidates.copyOf(candidates.size * 2)
            candidates[n * 3] = x
            candidates[n * 3 + 1] = y
            candidates[n * 3 + 2] = z
            if (dsq < bestDistSq) {
                bestDistSq = dsq
                bestIdx = n
            }
            n++
        }
        if (bestIdx < 0) return false

        val nx = candidates[bestIdx * 3]
        val ny = candidates[bestIdx * 3 + 1]
        val nz = candidates[bestIdx * 3 + 2]
        var sx = 0f
        var sy = 0f
        var sz = 0f
        var count = 0
        for (i in 0 until n) {
            val dx = candidates[i * 3] - nx
            val dy = candidates[i * 3 + 1] - ny
            val dz = candidates[i * 3 + 2] - nz
            if (dx * dx + dy * dy + dz * dz > CLUSTER_RADIUS_M * CLUSTER_RADIUS_M) continue
            sx += candidates[i * 3]
            sy += candidates[i * 3 + 1]
            sz += candidates[i * 3 + 2]
            count++
        }
        out[0] = sx / count
        out[1] = sy / count
        out[2] = sz / count
        return true
    }

    private fun hasOccupiedNeighbour(ix: Int, iy: Int, iz: Int): Boolean {
        val n1 = find(pack(ix + 1, iy, iz)); if (n1 >= 0 && states[n1] == OCCUPIED) return true
        val n2 = find(pack(ix - 1, iy, iz)); if (n2 >= 0 && states[n2] == OCCUPIED) return true
        val n3 = find(pack(ix, iy + 1, iz)); if (n3 >= 0 && states[n3] == OCCUPIED) return true
        val n4 = find(pack(ix, iy - 1, iz)); if (n4 >= 0 && states[n4] == OCCUPIED) return true
        val n5 = find(pack(ix, iy, iz + 1)); if (n5 >= 0 && states[n5] == OCCUPIED) return true
        val n6 = find(pack(ix, iy, iz - 1)); if (n6 >= 0 && states[n6] == OCCUPIED) return true
        return false
    }

    // ---- hash table ----

    /** Existing or newly created slot for a key; null once the grid is full. */
    private fun slotFor(key: Long): Int? {
        val existing = find(key)
        if (existing >= 0) return existing
        if (entries + 1 > (keys.size * LOAD_FACTOR).toInt()) {
            if (!grow()) return null
        }
        var i = index(key)
        while (keys[i] != EMPTY) i = (i + 1) and (keys.size - 1)
        keys[i] = key
        entries++
        return i
    }

    private fun find(key: Long): Int {
        var i = index(key)
        while (true) {
            val k = keys[i]
            if (k == key) return i
            if (k == EMPTY) return -1
            i = (i + 1) and (keys.size - 1)
        }
    }

    private fun grow(): Boolean {
        val newCap = keys.size * 2
        if (newCap > MAX_CAPACITY) {
            full = true
            return false
        }
        val oldKeys = keys
        val oldStates = states
        val oldBearings = bearings
        val oldCos = widestCos
        keys = LongArray(newCap) { EMPTY }
        states = ByteArray(newCap)
        bearings = FloatArray(newCap * 3)
        widestCos = FloatArray(newCap) { 1f }
        for (slot in oldKeys.indices) {
            val key = oldKeys[slot]
            if (key == EMPTY) continue
            var i = index(key)
            while (keys[i] != EMPTY) i = (i + 1) and (newCap - 1)
            keys[i] = key
            states[i] = oldStates[slot]
            bearings[i * 3] = oldBearings[slot * 3]
            bearings[i * 3 + 1] = oldBearings[slot * 3 + 1]
            bearings[i * 3 + 2] = oldBearings[slot * 3 + 2]
            widestCos[i] = oldCos[slot]
        }
        return true
    }

    private fun index(key: Long): Int {
        // splitmix64 finalizer: voxel keys are three small integers bit-packed, so their low bits
        // are highly structured and a plain mask would cluster every neighbouring voxel together.
        var h = key
        h = (h xor (h ushr 30)) * -0x40a7b892e31b1a47L
        h = (h xor (h ushr 27)) * -0x6b2fb644ecceee15L
        h = h xor (h ushr 31)
        return (h.toInt()) and (keys.size - 1)
    }

    private fun keyOf(x: Float, y: Float, z: Float): Long =
        pack(cell(x), cell(y), cell(z))

    private fun cell(v: Float): Int = floor(v / voxelSizeM).toInt()

    private fun centre(i: Int): Float = (i + 0.5f) * voxelSizeM

    companion object {
        const val UNSET: Byte = 0
        const val FREE: Byte = 1
        const val OCCUPIED: Byte = 2
        const val OCCLUDED: Byte = 3

        /** SPEC §2.3: two views at least this far apart make a surface adequately captured. */
        const val PARALLAX_MIN_DEG = 25f
        val PARALLAX_COS = kotlin.math.cos(Math.toRadians(PARALLAX_MIN_DEG.toDouble())).toFloat()

        /** Depth samples outside this range are dropped: closer is the operator's own hand,
         * further is where ARCore's motion-stereo depth stops being trustworthy indoors. */
        const val MIN_RANGE_M = 0.3f
        const val MAX_RANGE_M = 5.0f
        /** Voxels of clearance left uncarved in front of a hit, so a ray can't erase the surface
         * it just measured (or its neighbours, given depth noise). */
        const val CARVE_BACKOFF = 1.5f

        /** How far behind a hit surface to carve OCCLUDED, in metres -- roughly a kitchen-island
         * depth. Calibration knob: deep enough to cover a real obstacle, shallow enough that a
         * ray grazing a doorway doesn't paint the room beyond it OCCLUDED. */
        const val OCCLUSION_DEPTH_M = 1.0f

        /** Frontier search radius around the operator, and the radius of the cluster averaged
         * into the guidance target. Both calibration knobs -- tune in a real furnished room. */
        const val FRONTIER_RADIUS_M = 6f
        const val FRONTIER_MIN_DISTANCE_M = 1.5f
        const val CLUSTER_RADIUS_M = 1f

        private const val INITIAL_MESH_VERTS = 4096

        /** Five line segments (10 indices) forming two wireframe triangles for any quad. */
        private val LINE_INDICES = intArrayOf(0, 1,  1, 2,  2, 3,  3, 0,  0, 2)

        /** The six quad faces of a unit cube with their 4 corners in CCW order. */
        private val QUAD_FACES = arrayOf(
            QuadFace(1, 0, 0, floatArrayOf(1f, -1f, -1f,  1f, 1f, -1f,  1f, 1f, 1f,  1f, -1f, 1f)),
            QuadFace(-1, 0, 0, floatArrayOf(-1f, -1f, 1f,  -1f, 1f, 1f,  -1f, 1f, -1f,  -1f, -1f, -1f)),
            QuadFace(0, 1, 0, floatArrayOf(-1f, 1f, -1f,  -1f, 1f, 1f,  1f, 1f, 1f,  1f, 1f, -1f)),
            QuadFace(0, -1, 0, floatArrayOf(-1f, -1f, 1f,  -1f, -1f, -1f,  1f, -1f, -1f,  1f, -1f, 1f)),
            QuadFace(0, 0, 1, floatArrayOf(-1f, -1f, 1f,  1f, -1f, 1f,  1f, 1f, 1f,  -1f, 1f, 1f)),
            QuadFace(0, 0, -1, floatArrayOf(1f, -1f, -1f,  -1f, -1f, -1f,  -1f, 1f, -1f,  1f, 1f, -1f)),
        )

        private const val EMPTY = Long.MIN_VALUE
        private const val LOAD_FACTOR = 0.7f
        /** 2M slots ~= 50 MB of grid. Past that the 350 MB budget (SPEC A2) is at risk and the
         * right fix is a coarser free-space grid, not a bigger table -- see SPEC §1.3.3. */
        private const val MAX_CAPACITY = 1 shl 21

        // 21 bits each, signed: +/-1M voxels = +/-100 km at 10 cm. Bit 63 stays clear of EMPTY.
        private const val MASK = 0x1FFFFFL

        fun pack(ix: Int, iy: Int, iz: Int): Long =
            ((ix.toLong() and MASK) shl 42) or ((iy.toLong() and MASK) shl 21) or (iz.toLong() and MASK)

        fun unpackX(key: Long): Int = sign21(((key ushr 42) and MASK).toInt())
        fun unpackY(key: Long): Int = sign21(((key ushr 21) and MASK).toInt())
        fun unpackZ(key: Long): Int = sign21((key and MASK).toInt())

        private fun sign21(v: Int): Int = if (v >= (1 shl 20)) v - (1 shl 21) else v

        /** Angle between two unit bearings, degrees -- the parallax formula from projet.md §2.3. */
        fun parallaxDeg(ax: Float, ay: Float, az: Float, bx: Float, by: Float, bz: Float): Float {
            val dot = (ax * bx + ay * by + az * bz).coerceIn(-1f, 1f)
            return Math.toDegrees(kotlin.math.acos(dot.toDouble())).toFloat()
        }
    }
}
