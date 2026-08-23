package com.glomehometour.capture

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import com.google.ar.core.Pose
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/** Draws the latest depth map as a near=red, far=blue heatmap. */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var depthBitmap: Bitmap? = null
    private var smoothed: FloatArray? = null
    private var smoothedLow = Float.NaN
    private var smoothedHigh = Float.NaN
    private var lastUpdateAtMs = 0L
    private var prevPose: Pose? = null

    // Time constants, not fixed per-frame alphas: alpha = 1 - exp(-dt/tau) so smoothing behaves
    // the same in wall-clock terms regardless of the capture gate's Hz or any frame-to-frame
    // jitter in actual elapsed time (a fixed alpha implicitly assumes a constant dt, which broke
    // when we moved the gate from ~1.5Hz to 5Hz).
    //
    // A single fixed tau can never be right: this EMA blends raw values in *fixed pixel
    // coordinates*, with no idea the camera moved. A tau big enough to kill flicker on a still
    // camera (pixel (x,y) really is the same real point frame to frame) is exactly what makes
    // shapes "drag"/smear when the camera pans (pixel (x,y) is now a *different* real point, and
    // we're blending it with the old one anyway). So tau has to shrink toward near-zero smoothing
    // while the camera is moving, and only relax to heavy smoothing once it's actually still.
    // Real ARCore pose delta between frames drives this (see motionFactor below).
    private val valueTauStillMs = 2000f
    private val valueTauMovingMs = 80f
    // Same story for the color-scale bounds — a stale scale while panning reads as smearing too.
    private val rangeTauStillMs = 6000f
    private val rangeTauMovingMs = 300f

    // ponytail: motion thresholds — combined score is meters-of-translation + radians-of-rotation
    // between consecutive poses; past this, treat the camera as "fully moving" (min tau). Tune
    // down if drag is still visible on slow pans, up if normal hand tremor is triggering it.
    private val motionFullScore = 0.04f

    // ZipDepth outputs affine-invariant inverse depth (near = high value); raw disparity
    // compresses far distances into a razor-thin band near the low end of the range (disparity
    // ~ 1/distance), which is what made 2-3m out look uniformly "flat". Log-remapping the
    // normalized disparity un-compresses that band, trading some near-field resolution for
    // usable far-field contrast.
    // ponytail: fraction of the near..far range treated as "at infinity" — raise if far still
    // looks flat, lower if near content feels too compressed into one color band.
    private val logFarEpsilon = 0.03f
    private val logNormalizer = -ln(logFarEpsilon)

    /**
     * [calibratedBounds], when non-null, is (nearRawValue, farRawValue) from
     * [DepthCalibration] — anchors the color scale to fixed real-world meters instead of this
     * frame's own percentile stats, so the same real distance always maps to the same color,
     * across frames and even after the camera moves away and back. Falls back to the old
     * percentile-based scale when ARCore isn't tracking or too few calibration points landed
     * in-frame (e.g. a blank wall).
     */
    fun updateDepth(
        depth: FloatArray,
        size: Int,
        edges: BooleanArray? = null,
        calibratedBounds: Pair<Float, Float>? = null,
        pose: Pose? = null,
    ) {
        val now = SystemClock.elapsedRealtime()
        val dtMs = (now - lastUpdateAtMs).toFloat()
        lastUpdateAtMs = now

        // No pose info (ARCore not tracking) -> assume mid-motion rather than risk smearing.
        val motionFactor = if (pose != null && prevPose != null) {
            (poseDelta(pose, prevPose!!) / motionFullScore).coerceIn(0f, 1f)
        } else {
            0.5f
        }
        prevPose = pose
        val valueTauMs = valueTauStillMs + (valueTauMovingMs - valueTauStillMs) * motionFactor
        val rangeTauMs = rangeTauStillMs + (rangeTauMovingMs - rangeTauStillMs) * motionFactor

        val prev = smoothed
        val blended = if (prev != null && prev.size == depth.size) {
            val valueAlpha = 1f - exp(-dtMs / valueTauMs)
            FloatArray(depth.size) { i -> valueAlpha * depth[i] + (1 - valueAlpha) * prev[i] }
        } else {
            depth.copyOf()
        }
        smoothed = blended

        val low: Float
        val high: Float
        if (calibratedBounds != null) {
            low = calibratedBounds.second
            high = calibratedBounds.first
        } else {
            val sorted = blended.copyOf().also { it.sort() }
            low = sorted[(sorted.size * 0.05f).toInt()]
            high = sorted[(sorted.size * 0.95f).toInt()]
        }
        if (smoothedLow.isNaN()) {
            smoothedLow = low
            smoothedHigh = high
        } else {
            val rangeAlpha = 1f - exp(-dtMs / rangeTauMs)
            smoothedLow += rangeAlpha * (low - smoothedLow)
            smoothedHigh += rangeAlpha * (high - smoothedHigh)
        }
        val range = (smoothedHigh - smoothedLow).coerceAtLeast(1e-6f)

        val colors = IntArray(blended.size)
        for (i in blended.indices) {
            if (edges != null && edges[i]) {
                colors[i] = Color.WHITE
                continue
            }
            val tLin = ((blended[i] - smoothedLow) / range).coerceIn(0f, 1f) // 0 = far, 1 = near
            val disparity = tLin * (1f - logFarEpsilon) + logFarEpsilon
            val t = (-ln(disparity) / logNormalizer).coerceIn(0f, 1f) // 0 = near, 1 = far
            colors[i] = Color.HSVToColor(floatArrayOf(t * 240f, 1f, 1f))
        }
        depthBitmap = Bitmap.createBitmap(colors, size, size, Bitmap.Config.ARGB_8888)
        post { invalidate() }
    }

    /** Combined translation (m) + rotation (rad) magnitude between two ARCore poses. */
    private fun poseDelta(a: Pose, b: Pose): Float {
        val ta = a.translation
        val tb = b.translation
        val dx = ta[0] - tb[0]
        val dy = ta[1] - tb[1]
        val dz = ta[2] - tb[2]
        val transMeters = sqrt(dx * dx + dy * dy + dz * dz)

        val qa = a.rotationQuaternion
        val qb = b.rotationQuaternion
        val dot = (qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]).coerceIn(-1f, 1f)
        val rotRad = 2f * acos(abs(dot))

        return transMeters + rotRad
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bmp = depthBitmap ?: return
        canvas.drawBitmap(bmp, Rect(0, 0, bmp.width, bmp.height), Rect(0, 0, width, height), null)
    }
}
