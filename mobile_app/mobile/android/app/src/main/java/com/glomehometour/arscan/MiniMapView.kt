package com.glomehometour.arscan

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Real-time 2D Bird's-Eye Mini-Map for spatial walkthrough navigation.
 *
 * Locked to Cardinal North-South-East-West coordinates established at session start
 * (-Z is North / Up, +X is East / Right, +Z is South / Down, -X is West / Left).
 *
 * Features:
 * 1. Live user position cursor with forward camera FOV vision cone.
 * 2. Scanned floor footprint and chest-height structural wall contours.
 * 3. Starting doorway anchor marker for intuitive loop-closure return.
 * 4. Smooth dynamic auto-fit bounding box that expands as new rooms are explored.
 */
class MiniMapView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var userX: Float = 0f
    private var userZ: Float = 0f
    private var userYawDeg: Float = 0f
    private var startX: Float = 0f
    private var startZ: Float = 0f
    private var hasStart: Boolean = false
    private var targetX: Float? = null
    private var targetZ: Float? = null

    private var floorplan: VoxelGrid.Floorplan2D? = null

    // Breadcrumb trail points (x, z)
    private val trailX = FloatArray(1000)
    private val trailZ = FloatArray(1000)
    private var trailCount = 0
    private var lastTrailX = 0f
    private var lastTrailZ = 0f

    // Smooth bounding box interpolation for jitter-free auto-zooming
    private var smoothMinX = -2f
    private var smoothMaxX = 2f
    private var smoothMinZ = -2f
    private var smoothMaxZ = 2f
    private var initializedBounds = false

    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xD80F172A.toInt() // Slate 900 translucent
        style = Paint.Style.FILL
    }
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x4094A3B8.toInt() // Slate 400 subtle border
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * resources.displayMetrics.density
    }
    private val floorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x5510B981.toInt() // Emerald 500 translucent floor fill
        style = Paint.Style.FILL
    }
    private val wallPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFF1F5F9.toInt() // Slate 100 crisp wall contour
        style = Paint.Style.FILL
    }
    private val startPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFF59E0B.toInt() // Amber 500 start marker
        style = Paint.Style.FILL
    }
    private val startRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x80F59E0B.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * resources.displayMetrics.density
    }
    private val userConePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x5038BDF8.toInt() // Sky 400 FOV cone with vibrant glow
        style = Paint.Style.FILL
    }
    private val trailPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x8038BDF8.toInt() // Sky 400 breadcrumb path
        style = Paint.Style.STROKE
        strokeWidth = 2f * resources.displayMetrics.density
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val userPulsePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x3038BDF8.toInt() // Subtle radiant pulse around user
        style = Paint.Style.FILL
    }
    private val userDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF0284C7.toInt() // Sky 600
        style = Paint.Style.FILL
    }
    private val userCenterPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        style = Paint.Style.FILL
    }
    private val targetPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFBBF24.toInt() // Amber 400 frontier target
        style = Paint.Style.FILL
    }
    private val northTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x8094A3B8.toInt()
        textSize = 10f * resources.displayMetrics.scaledDensity
        textAlign = Paint.Align.CENTER
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    private val containerRect = RectF()
    private val conePath = Path()

    fun updateUser(x: Float, z: Float, yawDeg: Float) {
        userX = x
        userZ = z
        userYawDeg = yawDeg
        val dx = x - lastTrailX
        val dz = z - lastTrailZ
        if (trailCount == 0 || (dx * dx + dz * dz) >= 0.04f) { // Every 20cm of motion
            if (trailCount < trailX.size) {
                trailX[trailCount] = x
                trailZ[trailCount] = z
                trailCount++
                lastTrailX = x
                lastTrailZ = z
            }
        }
        invalidate()
    }

    fun resetTrail() {
        trailCount = 0
        lastTrailX = 0f
        lastTrailZ = 0f
    }

    fun setStart(x: Float, z: Float) {
        startX = x
        startZ = z
        hasStart = true
        invalidate()
    }

    fun setTarget(x: Float?, z: Float?) {
        targetX = x
        targetZ = z
        invalidate()
    }

    fun updateFloorplan(fp: VoxelGrid.Floorplan2D?) {
        floorplan = fp
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        val density = resources.displayMetrics.density
        val cornerRadius = 16f * density
        val pad = 14f * density

        // Draw background container
        containerRect.set(0f, 0f, w, h)
        canvas.drawRoundRect(containerRect, cornerRadius, cornerRadius, bgPaint)
        canvas.drawRoundRect(containerRect, cornerRadius, cornerRadius, borderPaint)

        // Cardinal North 'N' indicator at top-center
        canvas.drawText("N", w * 0.5f, 13f * density, northTextPaint)

        // Clip to rounded container
        canvas.save()
        val clipPath = Path().apply { addRoundRect(containerRect, cornerRadius, cornerRadius, Path.Direction.CW) }
        canvas.clipPath(clipPath)

        // Calculate dynamic bounding box
        val fp = floorplan
        val targetMinX = minOf(fp?.minX ?: -1.5f, userX - 1.2f, if (hasStart) startX - 1f else -1.5f)
        val targetMaxX = maxOf(fp?.maxX ?: 1.5f, userX + 1.2f, if (hasStart) startX + 1f else 1.5f)
        val targetMinZ = minOf(fp?.minZ ?: -1.5f, userZ - 1.2f, if (hasStart) startZ - 1f else -1.5f)
        val targetMaxZ = maxOf(fp?.maxZ ?: 1.5f, userZ + 1.2f, if (hasStart) startZ + 1f else 1.5f)

        if (!initializedBounds) {
            smoothMinX = targetMinX
            smoothMaxX = targetMaxX
            smoothMinZ = targetMinZ
            smoothMaxZ = targetMaxZ
            initializedBounds = true
        } else {
            smoothMinX += (targetMinX - smoothMinX) * 0.12f
            smoothMaxX += (targetMaxX - smoothMaxX) * 0.12f
            smoothMinZ += (targetMinZ - smoothMinZ) * 0.12f
            smoothMaxZ += (targetMaxZ - smoothMaxZ) * 0.12f
        }

        val spanX = maxOf(smoothMaxX - smoothMinX, 2.5f)
        val spanZ = maxOf(smoothMaxZ - smoothMinZ, 2.5f)
        val scale = minOf((w - 2 * pad) / spanX, (h - 2 * pad) / spanZ)
        val cX = (smoothMinX + smoothMaxX) * 0.5f
        val cZ = (smoothMinZ + smoothMaxZ) * 0.5f
        val midX = w * 0.5f
        val midY = h * 0.5f

        fun toScreenX(worldX: Float): Float = midX + (worldX - cX) * scale
        // -Z is North (Up on screen), so invert Z offset
        fun toScreenY(worldZ: Float): Float = midY + (worldZ - cZ) * scale

        val cellSize = maxOf(scale * 0.15f, 2.5f * density)

        // 1. Draw floor footprint
        if (fp != null && fp.floorCount > 0) {
            val pts = fp.floorPoints
            val count = fp.floorCount
            for (i in 0 until count) {
                val sx = toScreenX(pts[i * 2])
                val sy = toScreenY(pts[i * 2 + 1])
                canvas.drawRect(sx - cellSize * 0.5f, sy - cellSize * 0.5f, sx + cellSize * 0.5f, sy + cellSize * 0.5f, floorPaint)
            }
        }

        // 2. Draw wall contours
        if (fp != null && fp.wallCount > 0) {
            val pts = fp.wallPoints
            val count = fp.wallCount
            val wallSize = maxOf(cellSize * 1.1f, 3.5f * density)
            for (i in 0 until count) {
                val sx = toScreenX(pts[i * 2])
                val sy = toScreenY(pts[i * 2 + 1])
                canvas.drawRect(sx - wallSize * 0.5f, sy - wallSize * 0.5f, sx + wallSize * 0.5f, sy + wallSize * 0.5f, wallPaint)
            }
        }

        // 3. Draw start position marker (doorway)
        if (hasStart) {
            val sx = toScreenX(startX)
            val sy = toScreenY(startZ)
            canvas.drawCircle(sx, sy, 5f * density, startPaint)
            canvas.drawCircle(sx, sy, 8f * density, startRingPaint)
        }

        // 4. Draw frontier target
        val tx = targetX
        val tz = targetZ
        if (tx != null && tz != null) {
            val tsx = toScreenX(tx)
            val tsy = toScreenY(tz)
            canvas.drawCircle(tsx, tsy, 4f * density, targetPaint)
        }

        // 5. Draw breadcrumb walking trail
        if (trailCount > 1) {
            val trailPath = Path()
            trailPath.moveTo(toScreenX(trailX[0]), toScreenY(trailZ[0]))
            for (i in 1 until trailCount) {
                trailPath.lineTo(toScreenX(trailX[i]), toScreenY(trailZ[i]))
            }
            canvas.drawPath(trailPath, trailPaint)
        }

        // 6. Draw live user position & heading vision cone
        val ux = toScreenX(userX)
        val uy = toScreenY(userZ)
        canvas.drawCircle(ux, uy, 12f * density, userPulsePaint)

        val coneLength = 22f * density
        val halfFovRad = Math.toRadians(35.0).toFloat() // ~70 deg FOV
        // In screen coords, heading 0 is facing North (-Z in world, -Y on screen)
        val headingRad = Math.toRadians((userYawDeg - 90.0)).toFloat()

        val leftRad = headingRad - halfFovRad
        val rightRad = headingRad + halfFovRad

        conePath.reset()
        conePath.moveTo(ux, uy)
        conePath.lineTo(ux + kotlin.math.cos(leftRad) * coneLength, uy + kotlin.math.sin(leftRad) * coneLength)
        conePath.arcTo(
            RectF(ux - coneLength, uy - coneLength, ux + coneLength, uy + coneLength),
            Math.toDegrees(leftRad.toDouble()).toFloat(),
            Math.toDegrees((rightRad - leftRad).toDouble()).toFloat(),
            false
        )
        conePath.close()
        canvas.drawPath(conePath, userConePaint)

        // User central dot
        canvas.drawCircle(ux, uy, 5f * density, userDotPaint)
        canvas.drawCircle(ux, uy, 2f * density, userCenterPaint)

        canvas.restore()
    }
}
