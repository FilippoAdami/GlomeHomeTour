package com.glomehometour.arscan

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

/**
 * README section 6 step 3's "on-screen level indicator": the pre-flight tilt guide.
 *
 * A degree readout ("38 deg downward, aim for 30-45") makes the operator do the comparison; a
 * gauge with the target band drawn on it makes the answer positional -- put the dot in the green
 * zone -- which is readable at a glance while holding a phone at chest height. Same reason a
 * spirit level isn't a number.
 *
 * Scale is 0 (level) to [MAX_DEG] (straight down); [tiltDeg] is clamped, so a phone pointing
 * upward parks the thumb at the left end rather than disappearing.
 */
class TiltGaugeView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    /** Downward tilt in degrees, positive is down (MainActivity.tiltDownDeg). */
    var tiltDeg: Float = 0f
        set(value) {
            val clamped = value.coerceIn(0f, MAX_DEG)
            if (kotlin.math.abs(clamped - field) < 0.25f) return  // sub-pixel churn, skip the redraw
            field = clamped
            invalidate()
        }

    val inBand: Boolean get() = tiltDeg >= BAND_MIN_DEG && tiltDeg <= BAND_MAX_DEG

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.track)
    }
    private val bandPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.action_ready)
    }
    private val thumbPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.text_primary)
    }
    private val thumbShadow = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x66000000 }
    private val rect = RectF()

    override fun onDraw(canvas: Canvas) {
        val h = height.toFloat()
        val thumbR = h * 0.5f - 1f
        // Inset by the thumb radius so the thumb stays fully on-screen at both ends.
        val left = thumbR
        val right = width - thumbR
        val span = right - left
        val trackH = h * 0.28f
        val cy = h * 0.5f

        rect.set(left, cy - trackH / 2f, right, cy + trackH / 2f)
        canvas.drawRoundRect(rect, trackH, trackH, trackPaint)

        rect.set(
            left + span * (BAND_MIN_DEG / MAX_DEG), cy - trackH / 2f,
            left + span * (BAND_MAX_DEG / MAX_DEG), cy + trackH / 2f,
        )
        canvas.drawRoundRect(rect, trackH, trackH, bandPaint)

        val cx = left + span * (tiltDeg / MAX_DEG)
        canvas.drawCircle(cx, cy + 1f, thumbR, thumbShadow)
        canvas.drawCircle(cx, cy, thumbR - 1f, thumbPaint)
    }

    companion object {
        /** README section 6 step 3's target band (5-20 degrees for environment capture). Calibration knobs. */
        const val BAND_MIN_DEG = 5f
        const val BAND_MAX_DEG = 20f
        const val MAX_DEG = 90f
    }
}
