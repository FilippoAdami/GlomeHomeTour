package com.glomehometour.capture

import android.util.Log
import com.google.ar.core.Pose

/**
 * Anchors ZipDepth's per-frame affine-invariant disparity to real metric distance using
 * ARCore's tracked sparse point cloud, so "value X" means the same real distance in every
 * frame — including after the camera moves away and comes back — instead of drifting with
 * each frame's own arbitrary internal scale (which is all the old percentile-based
 * normalization could ever guarantee).
 *
 * Model: disparity ~= alpha * (1/distance) + beta (ZipDepth's own affine-invariant-inverse-
 * depth convention). Fit alpha, beta per frame via least squares against ARCore points that
 * land inside the model's input crop, then convert two fixed real-world bounds (near/far
 * meters) into this frame's raw-disparity units — those become the color-scale bounds.
 */
object DepthCalibration {

    data class Result(val nearRawBound: Float, val farRawBound: Float, val pointsUsed: Int)

    // ponytail: fixed guidance range, not scene-adaptive — widen if far walls get clipped flat.
    private const val nearMeters = 0.3f
    private const val farMeters = 8f
    private const val minPoints = 6
    private const val minConfidence = 0.5f

    fun fit(
        disparity: FloatArray,
        modelSize: Int,
        cameraPose: Pose,
        points: FloatArray, // x,y,z,confidence quads in ARCore world space
        numPoints: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        sensorWidth: Int,
        sensorHeight: Int,
        rotationDegrees: Int,
    ): Result? {
        val poseInv = cameraPose.inverse()
        val pIn = FloatArray(3)

        var sumX = 0f
        var sumY = 0f
        var sumXY = 0f
        var sumXX = 0f
        var n = 0

        // ponytail: debug-only stage counters — remove once calib stops rejecting every frame.
        var passConfidence = 0
        var passDepthBounds = 0
        var passSensorBounds = 0
        var passCrop = 0

        for (i in 0 until numPoints) {
            val base = i * 4
            if (points[base + 3] < minConfidence) continue
            passConfidence++
            pIn[0] = points[base]
            pIn[1] = points[base + 1]
            pIn[2] = points[base + 2]
            val pc = poseInv.transformPoint(pIn)
            val depthFwd = -pc[2] // ARCore camera space: -Z is forward
            if (depthFwd < 0.1f || depthFwd > 15f) continue
            passDepthBounds++

            val u = fx * (pc[0] / depthFwd) + cx
            val v = cy - fy * (pc[1] / depthFwd)
            if (u < 0f || u >= sensorWidth || v < 0f || v >= sensorHeight) continue
            passSensorBounds++

            val mapped = rotateCropScale(u, v, sensorWidth, sensorHeight, rotationDegrees, modelSize) ?: continue
            passCrop++
            val ix = mapped.first.toInt().coerceIn(0, modelSize - 1)
            val iy = mapped.second.toInt().coerceIn(0, modelSize - 1)

            val x = 1f / depthFwd
            val y = disparity[iy * modelSize + ix]
            sumX += x
            sumY += y
            sumXY += x * y
            sumXX += x * x
            n++
        }

        if (n < minPoints) {
            Log.d(
                "DepthCalib",
                "reject n<minPoints: total=$numPoints conf=$passConfidence depth=$passDepthBounds " +
                    "sensor=$passSensorBounds crop=$passCrop n=$n",
            )
            return null
        }

        val denom = n * sumXX - sumX * sumX
        if (kotlin.math.abs(denom) < 1e-6f) {
            Log.d("DepthCalib", "reject denom~0: n=$n")
            return null
        }
        val alpha = (n * sumXY - sumX * sumY) / denom
        val beta = (sumY - alpha * sumX) / n
        if (alpha <= 0f) {
            Log.d("DepthCalib", "reject alpha<=0: n=$n alpha=$alpha beta=$beta")
            return null
        }

        val nearRaw = alpha / nearMeters + beta
        val farRaw = alpha / farMeters + beta
        if (!nearRaw.isFinite() || !farRaw.isFinite() || nearRaw <= farRaw) {
            Log.d("DepthCalib", "reject bounds: n=$n alpha=$alpha beta=$beta near=$nearRaw far=$farRaw")
            return null
        }

        Log.d("DepthCalib", "accept: n=$n alpha=$alpha beta=$beta near=$nearRaw far=$farRaw")
        return Result(nearRaw, farRaw, n)
    }

    /** Mirrors MainActivity.cropToModelInput's postRotate + center-crop-square + scale. */
    private fun rotateCropScale(
        u: Float,
        v: Float,
        w: Int,
        h: Int,
        rotationDegrees: Int,
        modelSize: Int,
    ): Pair<Float, Float>? {
        var rw = w.toFloat()
        var rh = h.toFloat()
        var rx = u
        var ry = v
        when (((rotationDegrees % 360) + 360) % 360) {
            90 -> { rw = h.toFloat(); rh = w.toFloat(); rx = h - v; ry = u }
            180 -> { rx = w - u; ry = h - v }
            270 -> { rw = h.toFloat(); rh = w.toFloat(); rx = v; ry = w - u }
        }
        val side = minOf(rw, rh)
        val offX = (rw - side) / 2f
        val offY = (rh - side) / 2f
        val cx0 = rx - offX
        val cy0 = ry - offY
        if (cx0 < 0f || cx0 >= side || cy0 < 0f || cy0 >= side) return null
        val scale = modelSize / side
        return Pair(cx0 * scale, cy0 * scale)
    }
}
