package com.glomehometour.arscan

import android.os.Handler
import android.os.HandlerThread
import java.util.HashMap
import java.util.HashSet
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.acos
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Pixel -> world unprojection for ARCore's depth image.
 *
 * Its own object because it is the single easiest thing in this app to get quietly wrong: a sign
 * flip on v or a transposed quaternion rotation still produces a plausible-looking cloud that is
 * mirrored about the camera, and the only symptom is that coverage never converges. Pure math,
 * unit-tested (see CoverageWorkerTest).
 *
 * Conventions: image (u, v) has its origin top-left with v pointing down; ARCore camera space is
 * +X right, +Y up, -Z forward; depth is distance along -Z, not ray length.
 */
object Unproject {

    /** Rotates v by unit quaternion q = (x, y, z, w), in place into out. */
    fun rotate(q: FloatArray, vx: Float, vy: Float, vz: Float, out: FloatArray) {
        val qx = q[0]; val qy = q[1]; val qz = q[2]; val qw = q[3]
        // t = 2 * (q_vec x v); v' = v + qw * t + q_vec x t
        val tx = 2f * (qy * vz - qz * vy)
        val ty = 2f * (qz * vx - qx * vz)
        val tz = 2f * (qx * vy - qy * vx)
        out[0] = vx + qw * tx + (qy * tz - qz * ty)
        out[1] = vy + qw * ty + (qz * tx - qx * tz)
        out[2] = vz + qw * tz + (qx * ty - qy * tx)
    }

    /** World-space point for a depth sample, given the camera's translation t and rotation q. */
    fun toWorld(
        u: Float, v: Float, depthM: Float,
        fx: Float, fy: Float, cx: Float, cy: Float,
        t: FloatArray, q: FloatArray, out: FloatArray,
    ) {
        rotate(q, (u - cx) / fx * depthM, -(v - cy) / fy * depthM, -depthM, out)
        out[0] += t[0]
        out[1] += t[1]
        out[2] += t[2]
    }
}

/**
 * Runs the voxel grid off the render thread (projet.md §4: 10-15 Hz voxel worker under a 30 Hz
 * render loop) and publishes back three plain snapshots the other threads can read without a
 * lock: the overlay point array, the guidance target, and the counters the HUD shows.
 *
 * Back-pressure is a dropped submission, not a queue: if the worker is still busy when the next
 * depth frame arrives, that frame is skipped. Depth frames are highly redundant between
 * consecutive poses, so skipping one costs nothing, while letting them pile up would put the
 * grid minutes behind the operator.
 */
class CoverageWorker(voxelSizeM: Float = DEFAULT_VOXEL_SIZE_M) {

    class Stats(
        val occupied: Int,
        val verified: Int,
        val free: Int,
        val occluded: Int,
        val coverage: Float,
        val gridFull: Boolean,
        val lastIntegrationMs: Float,
    )

    val grid = VoxelGrid(voxelSizeM)

    @Volatile var snapshot: FloatArray? = null
        private set
    @Volatile var target: FloatArray? = null
        private set
    /** Occlusion-centroid guidance (README §5/§6, Phase_1.md item 4): distinct signal from
     * [target]'s frontier arrow -- "something's hidden behind this" vs. "you haven't looked over
     * there yet". Both may be non-null at once; the UI must show them as separate arrows. */
    @Volatile var occlusionTarget: FloatArray? = null
        private set
    @Volatile var floorplan: VoxelGrid.Floorplan2D? = null
        private set
    /** Sparse coverage levels keyed only by columns containing retained feature landmarks. */
    @Volatile var landmarkCoverage: Map<Long, Byte> = emptyMap()
        private set
    @Volatile var stats = Stats(0, 0, 0, 0, 0f, false, 0f)
        private set

    private val thread = HandlerThread("voxel-worker").apply { start() }
    private val handler = Handler(thread.looper)
    private val busy = AtomicBoolean(false)
    private val coverageBusy = AtomicBoolean(false)
    private val coverageX = FloatArray(MAX_COVERAGE_LANDMARKS)
    private val coverageY = FloatArray(MAX_COVERAGE_LANDMARKS)
    private val coverageZ = FloatArray(MAX_COVERAGE_LANDMARKS)
    private val coverageT = FloatArray(3)
    private val coverageQ = FloatArray(4)
    private val columns = HashMap<Long, ColumnStats>()
    private val visibleColumns = HashSet<Long>()
    private val columnViews = HashMap<Long, ColumnViewEvidence>()
    private val columnQuality = HashMap<Long, Byte>()
    private val inverseCoverageQ = FloatArray(4)
    private val cameraPoint = FloatArray(3)

    private val world = FloatArray(3)
    private val targetScratch = FloatArray(3)
    private val occlusionScratch = FloatArray(3)
    private var lastSnapshotNanos = 0L
    private var lastFrontierNanos = 0L

    private class ColumnStats(val centreX: Float, val centreZ: Float) {
        var pointCount = 0
        var minY = Float.MAX_VALUE
        var secondMinY = Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        var secondMaxY = -Float.MAX_VALUE
    }

    private class ColumnViewEvidence {
        var views = 0
        var cameraX = 0f
        var cameraY = 0f
        var cameraZ = 0f
        var bearingX = 0f
        var bearingY = 0f
        var bearingZ = 0f
    }

    /** GL thread. Depth path. Returns false if the worker was busy and the frame was skipped. */
    fun submitDepth(depth: ArScanRenderer.DepthFrame, t: FloatArray, q: FloatArray, floorY: Float): Boolean {
        if (!busy.compareAndSet(false, true)) return false
        handler.post {
            try {
                val start = System.nanoTime()
                integrateDepth(depth, t, q)
                publish(t, floorY, start)
            } finally {
                busy.set(false)
            }
        }
        return true
    }

    private var lastSubmitNanos = 0L
    private var lastSubmitX = 0f
    private var lastSubmitY = 0f
    private var lastSubmitZ = 0f

    /** GL thread. Feature-point fallback for devices without the Depth API; ARCore's cloud is
     * already in world space, so there's nothing to unproject. */
    fun submitPoints(points: FloatArray, numPoints: Int, t: FloatArray, floorY: Float): Boolean {
        val now = System.nanoTime()
        val dx = t[0] - lastSubmitX
        val dy = t[1] - lastSubmitY
        val dz = t[2] - lastSubmitZ
        val moved = (dx * dx + dy * dy + dz * dz) >= 0.001f // ~3cm motion
        // When stationary, throttle to 3.3 Hz (300ms) to save CPU/battery; when moving, allow full 15 Hz
        if (!moved && (now - lastSubmitNanos) < 300_000_000L) {
            return false
        }
        if (!busy.compareAndSet(false, true)) return false
        lastSubmitNanos = now
        lastSubmitX = t[0]; lastSubmitY = t[1]; lastSubmitZ = t[2]
        handler.post {
            try {
                val start = System.nanoTime()
                for (i in 0 until numPoints) {
                    val base = i * 4
                    if (points[base + 3] < MIN_POINT_CONFIDENCE) continue
                    grid.integrate(t[0], t[1], t[2], points[base], points[base + 1], points[base + 2])
                }
                publish(t, floorY, start)
            } finally {
                busy.set(false)
            }
        }
        return true
    }

    /**
     * Recomputes colors for feature-backed minimap columns from an accepted camera view.
     * Coordinate copies keep the render thread's reusable landmark buffers out of the worker.
     */
    fun submitLandmarkCoverage(
        xs: FloatArray, ys: FloatArray, zs: FloatArray, count: Int,
        cameraT: FloatArray, cameraQ: FloatArray,
        imageWidth: Int, imageHeight: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        floorY: Float,
    ): Boolean {
        val n = minOf(count, xs.size, ys.size, zs.size, coverageX.size)
        if (n == 0 || !coverageBusy.compareAndSet(false, true)) return false

        System.arraycopy(xs, 0, coverageX, 0, n)
        System.arraycopy(ys, 0, coverageY, 0, n)
        System.arraycopy(zs, 0, coverageZ, 0, n)
        System.arraycopy(cameraT, 0, coverageT, 0, 3)
        System.arraycopy(cameraQ, 0, coverageQ, 0, 4)
        val queued = handler.post {
            try {
                computeLandmarkCoverage(n, imageWidth, imageHeight, fx, fy, cx, cy, floorY)
            } finally {
                coverageBusy.set(false)
            }
        }
        if (!queued) coverageBusy.set(false)
        return queued
    }

    /** Starts fresh view history at a room boundary while leaving occupancy guidance intact. */
    fun resetLandmarkCoverage() {
        landmarkCoverage = emptyMap()
        handler.post {
            columnViews.clear()
            columnQuality.clear()
            columns.clear()
            visibleColumns.clear()
            landmarkCoverage = emptyMap()
        }
    }

    fun shutdown() {
        thread.quitSafely()
    }

    // ---- worker thread ----

    private fun integrateDepth(d: ArScanRenderer.DepthFrame, t: FloatArray, q: FloatArray) {
        var v = 0
        while (v < d.height) {
            var u = 0
            while (u < d.width) {
                val raw = d.millimetres[v * d.width + u].toInt() and DEPTH_MASK
                if (raw != 0) {
                    val metres = raw / 1000f
                    if (metres in VoxelGrid.MIN_RANGE_M..VoxelGrid.MAX_RANGE_M) {
                        Unproject.toWorld(u.toFloat(), v.toFloat(), metres, d.fx, d.fy, d.cx, d.cy, t, q, world)
                        grid.integrate(t[0], t[1], t[2], world[0], world[1], world[2])
                    }
                }
                u += DEPTH_STRIDE
            }
            v += DEPTH_STRIDE
        }
    }

    private fun computeLandmarkCoverage(
        count: Int,
        imageWidth: Int, imageHeight: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        floorY: Float,
    ) {
        for (column in columns.values) {
            column.pointCount = 0
            column.minY = Float.MAX_VALUE
            column.secondMinY = Float.MAX_VALUE
            column.maxY = -Float.MAX_VALUE
            column.secondMaxY = -Float.MAX_VALUE
        }
        visibleColumns.clear()
        inverseCoverageQ[0] = coverageQ[0]
        inverseCoverageQ[1] = coverageQ[1]
        inverseCoverageQ[2] = coverageQ[2]
        inverseCoverageQ[3] = -coverageQ[3]
        val canProject = imageWidth > 0 && imageHeight > 0 && fx > 0f && fy > 0f

        for (i in 0 until count) {
            val x = coverageX[i]; val y = coverageY[i]; val z = coverageZ[i]
            if (!x.isFinite() || !y.isFinite() || !z.isFinite()) continue
            if (x !in -60f..60f || z !in -60f..60f) continue
            val ix = floor(x / COVERAGE_COLUMN_SIZE_M).toInt()
            val iz = floor(z / COVERAGE_COLUMN_SIZE_M).toInt()
            val key = coverageColumnKey(ix, iz)
            val column = columns[key] ?: ColumnStats(
                (ix + 0.5f) * COVERAGE_COLUMN_SIZE_M,
                (iz + 0.5f) * COVERAGE_COLUMN_SIZE_M,
            ).also {
                columns[key] = it
            }
            column.pointCount++
            if (y < column.minY) {
                column.secondMinY = column.minY
                column.minY = y
            } else if (y < column.secondMinY) {
                column.secondMinY = y
            }
            if (y > column.maxY) {
                column.secondMaxY = column.maxY
                column.maxY = y
            } else if (y > column.secondMaxY) {
                column.secondMaxY = y
            }
            // A retained feature inside an accepted camera frustum anchors view evidence to this column.
            if (!canProject) continue
            val dx = x - coverageT[0]; val dy = y - coverageT[1]; val dz = z - coverageT[2]
            val distanceSq = dx * dx + dy * dy + dz * dz
            if (distanceSq !in MIN_FRUSTUM_DISTANCE_SQ..MAX_FRUSTUM_DISTANCE_SQ) continue
            Unproject.rotate(inverseCoverageQ, dx, dy, dz, cameraPoint)
            val depth = -cameraPoint[2]
            if (depth !in MIN_FRUSTUM_RANGE_M..MAX_FRUSTUM_RANGE_M) continue
            val u = fx * cameraPoint[0] / depth + cx
            val v = cy - fy * cameraPoint[1] / depth
            if (u in 0f..imageWidth.toFloat() && v in 0f..imageHeight.toFloat()) visibleColumns.add(key)
        }

        val staleColumns = columns.entries.iterator()
        while (staleColumns.hasNext()) {
            val entry = staleColumns.next()
            if (entry.value.pointCount == 0) {
                staleColumns.remove()
                columnViews.remove(entry.key)
                columnQuality.remove(entry.key)
            }
        }

        for (key in visibleColumns) {
            val column = columns[key] ?: continue
            val evidence = columnViews.getOrPut(key) { ColumnViewEvidence() }
            updateColumnView(evidence, column, coverageT, floorY)
            columnQuality[key] = coverageLevel(column, evidence.views)
        }

        // Publish a detached snapshot; the UI reads it while this worker continues updating its maps.
        landmarkCoverage = HashMap(columnQuality)
    }

    private fun updateColumnView(
        evidence: ColumnViewEvidence,
        column: ColumnStats,
        cameraT: FloatArray,
        floorY: Float,
    ) {
        val bx = cameraT[0] - column.centreX
        val by = cameraT[1] - (floorY + VIEW_REFERENCE_HEIGHT_M)
        val bz = cameraT[2] - column.centreZ
        val length = sqrt(bx * bx + by * by + bz * bz)
        if (length <= 1e-4f) return
        val nx = bx / length; val ny = by / length; val nz = bz / length

        if (evidence.views == 0) {
            evidence.views = 1
            evidence.cameraX = cameraT[0]; evidence.cameraY = cameraT[1]; evidence.cameraZ = cameraT[2]
            evidence.bearingX = nx; evidence.bearingY = ny; evidence.bearingZ = nz
            return
        }

        val dx = cameraT[0] - evidence.cameraX
        val dy = cameraT[1] - evidence.cameraY
        val dz = cameraT[2] - evidence.cameraZ
        val moved = sqrt(dx * dx + dy * dy + dz * dz) >= MIN_VIEW_TRANSLATION_M
        val dot = (nx * evidence.bearingX + ny * evidence.bearingY + nz * evidence.bearingZ).coerceIn(-1f, 1f)
        val angleDeg = Math.toDegrees(acos(dot.toDouble())).toFloat()
        if (moved && angleDeg >= MIN_VIEW_ANGLE_DEG && evidence.views < MAX_DISTINCT_VIEWS) {
            evidence.views++
            evidence.cameraX = cameraT[0]; evidence.cameraY = cameraT[1]; evidence.cameraZ = cameraT[2]
            evidence.bearingX = nx; evidence.bearingY = ny; evidence.bearingZ = nz
        }
    }

    private fun coverageLevel(column: ColumnStats, views: Int): Byte {
        // Trim one isolated high/low point when supported; keep sparse floor-to-ceiling pairs intact.
        val span = when {
            column.pointCount >= 5 -> column.secondMaxY - column.secondMinY
            column.pointCount >= 2 -> column.maxY - column.minY
            else -> 0f
        }
        val verticalSpan = span.coerceAtLeast(0f).coerceAtMost(MAX_VERTICAL_SPAN_M)
        // shortcut: fixed indoor-room thresholds; tune with representative scans if room heights vary widely.
        return when {
            views >= 6 && column.pointCount >= 12 && verticalSpan >= 2.0f -> COVERAGE_LIGHT_BLUE
            verticalSpan >= 2.0f -> COVERAGE_GREEN
            views >= 2 || (views >= 1 && verticalSpan >= 0.45f) -> COVERAGE_YELLOW
            else -> COVERAGE_ORANGE
        }
    }

    private fun publish(t: FloatArray, floorY: Float, startNanos: Long) {
        val now = System.nanoTime()
        stats = Stats(
            occupied = grid.occupiedCount,
            verified = grid.verifiedCount,
            free = grid.freeCount,
            occluded = grid.occludedCount,
            coverage = grid.coverageFraction(),
            gridFull = grid.full,
            lastIntegrationMs = (now - startNanos) / 1e6f,
        )
        // Both of these walk the whole table, so they run on their own slower clocks than
        // integration does.
        if (now - lastSnapshotNanos > SNAPSHOT_INTERVAL_NS) {
            lastSnapshotNanos = now
            // Mesh generation happens here, on the worker's own ~3 Hz clock, not the 60 Hz render
            // loop (Phase_1.md §10 item 5) -- the GL thread only ever re-uploads whatever this
            // publishes, same pattern the point cloud it replaces already used.
            snapshot = grid.meshTriangles(ArScanRenderer.MAX_OVERLAY_TRIANGLES, floorY)
            floorplan = grid.floorplan2D(floorY)
        }
        if (now - lastFrontierNanos > FRONTIER_INTERVAL_NS) {
            lastFrontierNanos = now
            target = if (grid.frontierTarget(
                    t[0], t[1], t[2],
                    floorY + FRONTIER_MIN_HEIGHT_M, floorY + FRONTIER_MAX_HEIGHT_M,
                    targetScratch,
                )
            ) targetScratch.copyOf() else null
            occlusionTarget = if (grid.occlusionTarget(
                    t[0], t[1], t[2],
                    floorY + FRONTIER_MIN_HEIGHT_M, floorY + FRONTIER_MAX_HEIGHT_M,
                    occlusionScratch,
                )
            ) occlusionScratch.copyOf() else null
        }
    }

    companion object {
        /** SPEC §2.3's grid pitch, for a real depth sensor. The monocular fallback is given a
         * coarser one: its depth carries several percent of scale error, and at 10 cm the same
         * wall lands in a different voxel each pass, so nothing is ever seen twice and coverage
         * never leaves 0%. Calibration knob -- tighten it if the model's error drops. */
        const val DEFAULT_VOXEL_SIZE_M = 0.10f
        const val MONO_VOXEL_SIZE_M = 0.15f

        /** Every 4th pixel of a ~160x120 depth image: ~1200 rays per update, which is dense
         * enough to fill 10 cm voxels and cheap enough to stay inside the worker's budget. */
        const val DEPTH_STRIDE = 4
        /** DEPTH16: low 13 bits are millimetres, the top 3 are confidence. */
        const val DEPTH_MASK = 0x1FFF
        /** ARCore point-cloud confidence below this is noise rather than geometry. */
        const val MIN_POINT_CONFIDENCE = 0.25f

        private const val SNAPSHOT_INTERVAL_NS = 300_000_000L
        private const val FRONTIER_INTERVAL_NS = 1_000_000_000L

        /** Height band the guidance search runs in, above the floor estimate: below knee height
         * is furniture bases nobody needs a panorama of, above head height is ceiling. */
        const val FRONTIER_MIN_HEIGHT_M = 0.3f
        const val FRONTIER_MAX_HEIGHT_M = 2.0f

        const val COVERAGE_ORANGE: Byte = 0
        const val COVERAGE_YELLOW: Byte = 1
        const val COVERAGE_GREEN: Byte = 2
        const val COVERAGE_LIGHT_BLUE: Byte = 3

        const val COVERAGE_COLUMN_SIZE_M = 0.5f
        private const val MAX_COVERAGE_LANDMARKS = 35000
        private const val MIN_FRUSTUM_DISTANCE_SQ = 0.09f
        private const val MAX_FRUSTUM_DISTANCE_SQ = 64f
        private const val MIN_FRUSTUM_RANGE_M = 0.3f
        private const val MAX_FRUSTUM_RANGE_M = 8.0f
        private const val VIEW_REFERENCE_HEIGHT_M = 1.2f
        private const val MIN_VIEW_TRANSLATION_M = 0.20f
        private const val MIN_VIEW_ANGLE_DEG = 8.0f
        private const val MAX_DISTINCT_VIEWS = 8
        private const val MAX_VERTICAL_SPAN_M = 3.0f

        fun coverageColumnKey(x: Float, z: Float): Long = coverageColumnKey(
            floor(x / COVERAGE_COLUMN_SIZE_M).toInt(), floor(z / COVERAGE_COLUMN_SIZE_M).toInt(),
        )

        private fun coverageColumnKey(ix: Int, iz: Int): Long =
            (ix.toLong() shl 32) or (iz.toLong() and 0xffffffffL)
    }
}
