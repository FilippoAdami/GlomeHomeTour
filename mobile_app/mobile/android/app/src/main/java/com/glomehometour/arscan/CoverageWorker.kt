package com.glomehometour.arscan

import android.os.Handler
import android.os.HandlerThread
import java.util.concurrent.atomic.AtomicBoolean

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
    @Volatile var stats = Stats(0, 0, 0, 0, 0f, false, 0f)
        private set

    private val thread = HandlerThread("voxel-worker").apply { start() }
    private val handler = Handler(thread.looper)
    private val busy = AtomicBoolean(false)

    private val world = FloatArray(3)
    private val targetScratch = FloatArray(3)
    private val occlusionScratch = FloatArray(3)
    private var lastSnapshotNanos = 0L
    private var lastFrontierNanos = 0L

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
    }
}
