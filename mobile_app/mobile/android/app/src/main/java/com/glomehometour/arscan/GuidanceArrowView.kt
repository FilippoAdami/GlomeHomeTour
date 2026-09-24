package com.glomehometour.arscan

import android.content.Context
import android.graphics.Canvas
import android.graphics.CornerPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

/**
 * The 2D half of the guidance cue (SPEC section 2.4): an arrow pointing at the guidance target,
 * rotated by CaptureActivity via the standard View.rotation property -- so this class only ever has to
 * draw an arrow pointing up, and the bearing math stays in the renderer that already has the MVP.
 *
 * The 3D marker alone isn't enough: the thing the operator most needs to be told about is a
 * region that is, by definition, not currently on screen.
 *
 * One view class, two colours: reused for both the frontier arrow (amber, default) and the
 * occlusion arrow (set via [fillColor] in code -- see CaptureActivity) so the two guidance mechanics
 * README/Phase_1.md section 6 requires stay visually distinct without a second copy of this class.
 */
class GuidanceArrowView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    var fillColor: Int = 0xE6F59E0B.toInt()
        set(value) { field = value; fill.color = value; invalidate() }

    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xE6F59E0B.toInt() }
    /** Drawn as an offset copy of the path rather than Paint.setShadowLayer: shadow layers on
     * filled paths are unreliable under a hardware-accelerated canvas, an offset copy never is. */
    private val shadow = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x59000000 }
    private val path = Path()

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        // Rounded corners on the fill, not a black keyline: the outline read as a sticker over the
        // camera feed, where a soft-cornered solid shape reads as a HUD element.
        val effect = CornerPathEffect(w * 0.09f)
        fill.pathEffect = effect
        shadow.pathEffect = effect
        buildPath(w.toFloat(), h.toFloat())
    }

    private fun buildPath(w: Float, h: Float) {
        path.reset()
        path.moveTo(w * 0.50f, h * 0.06f)
        path.lineTo(w * 0.94f, h * 0.52f)
        path.lineTo(w * 0.66f, h * 0.52f)
        path.lineTo(w * 0.66f, h * 0.94f)
        path.lineTo(w * 0.34f, h * 0.94f)
        path.lineTo(w * 0.34f, h * 0.52f)
        path.lineTo(w * 0.06f, h * 0.52f)
        path.close()
    }

    override fun onDraw(canvas: Canvas) {
        val dy = height * 0.025f
        canvas.save()
        canvas.translate(0f, dy)
        canvas.drawPath(path, shadow)
        canvas.restore()
        canvas.drawPath(path, fill)
    }
}
