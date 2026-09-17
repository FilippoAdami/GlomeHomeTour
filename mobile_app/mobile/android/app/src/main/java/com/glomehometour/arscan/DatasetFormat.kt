package com.glomehometour.arscan

import kotlin.math.atan

/**
 * Serialisation for the export package (SPEC §2.5). Pure string/array math, no Android and no IO,
 * so the formats are unit-testable -- a silently transposed pose matrix would otherwise only show
 * up as a failed reconstruction hours later on the backend.
 *
 * Hand-rolled JSON rather than a serialisation dependency: two object shapes, both write-only.
 */
object DatasetFormat {

    /**
     * One exported keyframe. `matrix` is 16 floats, row-major camera-to-world.
     *
     * Intrinsics ride along per frame as well as globally: autofocus moves the focal length
     * mid-scan (see ArScanRenderer's focusMode note), and instant-ngp/nerfstudio both honour
     * per-frame fl_x/fl_y/cx/cy, so the dataset stays correct instead of averaging the breathing
     * into one wrong number.
     */
    class Keyframe(
        val fileName: String,
        val matrix: FloatArray,
        val timestampNs: Long,
        val fx: Float, val fy: Float, val cx: Float, val cy: Float,
        val focusDistanceDiopters: Float = 0f,
        val afState: Int = 0,
        val afMode: Int = 0,
        val focalLengthMm: Float = 0f,
        /** Magnetic compass heading in degrees [0, 360), clockwise from magnetic north, of the
         * device at capture time -- not corrected for declination (no location fix available).
         * Lets the backend rotate the otherwise-arbitrary ARCore world yaw to a real-world
         * orientation. Null if the magnetometer hadn't produced a reading yet. */
        val compassHeadingDeg: Float? = null,
    )

    /**
     * Camera-to-world matrix from an ARCore pose, row-major, in the convention transforms.json
     * expects (+X right, +Y up, -Z forward).
     *
     * No basis change is applied and none is needed: ARCore's camera pose already uses the
     * OpenGL/NeRF convention. The COLMAP-style y-down/z-forward flip that most conversion scripts
     * carry would actually *introduce* an error here.
     */
    fun cameraToWorld(tx: Float, ty: Float, tz: Float, qx: Float, qy: Float, qz: Float, qw: Float): FloatArray {
        val xx = qx * qx; val yy = qy * qy; val zz = qz * qz
        val xy = qx * qy; val xz = qx * qz; val yz = qy * qz
        val wx = qw * qx; val wy = qw * qy; val wz = qw * qz
        return floatArrayOf(
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), tx,
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), ty,
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), tz,
            0f, 0f, 0f, 1f,
        )
    }

    /** NeRF/instant-ngp transforms.json over the accepted keyframes. */
    fun transformsJson(
        fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int,
        frames: List<Keyframe>,
        k1: Float? = null, k2: Float? = null,
    ): String = buildString {
        append("{\n")
        append("  \"schema_version\": \"1.0.0\",\n")
        append("  \"camera_model\": \"OPENCV\",\n")
        append("  \"fl_x\": ${f(fx)},\n  \"fl_y\": ${f(fy)},\n")
        append("  \"cx\": ${f(cx)},\n  \"cy\": ${f(cy)},\n")
        append("  \"w\": $width,\n  \"h\": $height,\n")
        append("  \"camera_angle_x\": ${f(2.0 * atan(width / (2.0 * fx)))},\n")
        // k1/k2 are Android's LENS_DISTORTION kappa_0/kappa_1 (rational model), used as an
        // approximate seed for OpenCV's plumb-bob model -- not an exact undistortion. p1/p2
        // stay 0: Android's model has no tangential term. Null on API <28/no characteristic.
        append("  \"k1\": ${f(k1 ?: 0f)},\n  \"k2\": ${f(k2 ?: 0f)},\n  \"p1\": 0.0,\n  \"p2\": 0.0,\n")
        append("  \"frames\": [\n")
        frames.forEachIndexed { i, frame ->
            append("    {\"file_path\": \"images/${frame.fileName}\", \"timestamp_ns\": ${frame.timestampNs}, ")
            append("\"fl_x\": ${f(frame.fx)}, \"fl_y\": ${f(frame.fy)}, ")
            append("\"cx\": ${f(frame.cx)}, \"cy\": ${f(frame.cy)}, ")
            append("\"compass_heading_deg\": ${frame.compassHeadingDeg?.let { f(it) } ?: "null"}, ")
            append("\"transform_matrix\": [")
            for (r in 0 until 4) {
                append("[")
                for (c in 0 until 4) {
                    append(f(frame.matrix[r * 4 + c]))
                    if (c < 3) append(", ")
                }
                append("]")
                if (r < 3) append(", ")
            }
            append("]}")
            if (i < frames.size - 1) append(",")
            append("\n")
        }
        append("  ]\n}\n")
    }

    /** Companion focus metadata file recording camera autofocus metrics per keyframe. */
    fun focusMetadataJson(frames: List<Keyframe>): String = buildString {
        append("{\n  \"frames\": [\n")
        frames.forEachIndexed { i, frame ->
            val distM = if (frame.focusDistanceDiopters > 1e-4f) 1.0f / frame.focusDistanceDiopters else 0f
            append("    {\"file_path\": \"images/${frame.fileName}\", ")
            append("\"timestamp_ns\": ${frame.timestampNs}, ")
            append("\"focus_distance_diopters\": ${f(frame.focusDistanceDiopters)}, ")
            append("\"focus_distance_m\": ${f(distM)}, ")
            append("\"af_state\": ${frame.afState}, ")
            append("\"af_mode\": ${frame.afMode}, ")
            append("\"focal_length_mm\": ${f(frame.focalLengthMm)}}")
            if (i < frames.size - 1) append(",")
            append("\n")
        }
        append("  ]\n}\n")
    }

    const val TRAJECTORY_HEADER = "timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking,exported"

    /** One row of trajectory.csv. Written for *every* tracked frame, exported or not -- SPEC A5:
     * the backend's pose prior needs the unbroken trajectory, not just the frames that survived
     * the illumination gate. */
    fun trajectoryLine(
        timestampNs: Long, tx: Float, ty: Float, tz: Float,
        qx: Float, qy: Float, qz: Float, qw: Float,
        tracking: String, exported: Boolean,
    ): String = "$timestampNs,${f(tx)},${f(ty)},${f(tz)},${f(qx)},${f(qy)},${f(qz)},${f(qw)},$tracking,${if (exported) 1 else 0}"

    fun summaryJson(
        durationSeconds: Float,
        coverageFraction: Float,
        occupiedVoxels: Int,
        verifiedVoxels: Int,
        freeVoxels: Int,
        occludedVoxels: Int,
        voxelSizeM: Float,
        gridFull: Boolean,
        framesSeen: Long,
        framesExported: Int,
        droppedDark: Long,
        droppedBlown: Long,
        droppedTransition: Long,
        droppedMotion: Long,
        droppedQueue: Int,
        trackingLosses: Int,
        depthSource: String,
    ): String = buildString {
        append("{\n")
        append("  \"schema_version\": \"1.0.0\",\n")
        append("  \"duration_s\": ${f(durationSeconds)},\n")
        append("  \"coverage_fraction\": ${f(coverageFraction)},\n")
        append("  \"voxel_size_m\": ${f(voxelSizeM)},\n")
        append("  \"parallax_min_deg\": ${f(VoxelGrid.PARALLAX_MIN_DEG)},\n")
        append("  \"voxels\": {\"occupied\": $occupiedVoxels, \"parallax_verified\": $verifiedVoxels, ")
        append("\"free\": $freeVoxels, \"occluded\": $occludedVoxels, \"grid_full\": $gridFull},\n")
        append("  \"frames\": {\"seen\": $framesSeen, \"exported\": $framesExported},\n")
        append("  \"dropped\": {\"dark\": $droppedDark, \"blown\": $droppedBlown, ")
        append("\"illumination_transition\": $droppedTransition, \"motion\": $droppedMotion, ")
        append("\"writer_queue\": $droppedQueue},\n")
        append("  \"tracking_loss_events\": $trackingLosses,\n")
        append("  \"depth_source\": \"$depthSource\"\n")
        append("}\n")
    }

    /** Fixed 6 decimals, locale-independent: "%.6f" follows the device locale and writes commas
     * as decimal separators on half of Europe, which produces JSON no parser will read. */
    private fun f(v: Float): String = f(v.toDouble())

    private fun f(v: Double): String = String.format(java.util.Locale.US, "%.6f", v)
}
