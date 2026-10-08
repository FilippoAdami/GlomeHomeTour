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
 * Fixed-axis magnetically aligned (North Up) floor plan:
 * (-Z is Magnetic North / Up, +X is Magnetic East / Right).
 *
 * Pure, uncluttered visual elements:
 * 1. Starting doorway anchor position (loop-closure reference).
 * 2. Real-time online 2D landmark points (Emerald Green #10B981) populated dynamically as objects are orbited.
 * 3. Cardinal North "N" indicator at the top center.
 * 4. Fixed physical screen-size user navigation arrow indicating live position and heading.
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

    // Magnetic North offset: angle to rotate ARCore coordinates so that -Z aligns with Magnetic North
    private var northOffsetDeg: Float = 0f
    private var cosNorth: Float = 1f
    private var sinNorth: Float = 0f

    // Multi-room doorway anchors across all visited rooms: (x, z)
    private val doorwayX = FloatArray(32)
    private val doorwayZ = FloatArray(32)
    private var doorwayCount = 0

    // Live active room landmarks (updated online at 60 Hz during scanning)
    private val activeLandmarksX = FloatArray(2500)
    private val activeLandmarksZ = FloatArray(2500)
    private var activeLandmarkCount = 0
    private var activeMinMagX = Float.MAX_VALUE
    private var activeMaxMagX = -Float.MAX_VALUE
    private var activeMinMagZ = Float.MAX_VALUE
    private var activeMaxMagZ = -Float.MAX_VALUE

    // Persistent archived landmarks across previous rooms
    private val archivedLandmarksX = FloatArray(20000)
    private val archivedLandmarksZ = FloatArray(20000)
    private var archivedLandmarkCount = 0
    private var archivedMinMagX = Float.MAX_VALUE
    private var archivedMaxMagX = -Float.MAX_VALUE
    private var archivedMinMagZ = Float.MAX_VALUE
    private var archivedMaxMagZ = -Float.MAX_VALUE

    // Scratch buffers for real-time statistical outlier detection and removal (zero-allocation)
    private val scratchCandX = FloatArray(2500)
    private val scratchCandZ = FloatArray(2500)
    private val scratchSortX = FloatArray(2500)
    private val scratchSortZ = FloatArray(2500)

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
    private val activeLandmarkPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xF034D399.toInt() // Vibrant Emerald 400 (matches 3D emerald gems in AR view)
        style = Paint.Style.FILL
    }
    private val archivedLandmarkPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xB3059669.toInt() // Softer Emerald 600 for previous rooms
        style = Paint.Style.FILL
    }
    private val startPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFF59E0B.toInt() // Amber 500 active doorway marker
        style = Paint.Style.FILL
    }
    private val startRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x80F59E0B.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * resources.displayMetrics.density
    }
    private val prevDoorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF34D399.toInt() // Emerald 400 completed doorway marker
        style = Paint.Style.FILL
    }
    private val prevDoorRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x8034D399.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * resources.displayMetrics.density
    }
    private val northTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x8094A3B8.toInt()
        textSize = 10f * resources.displayMetrics.scaledDensity
        textAlign = Paint.Align.CENTER
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val userHaloPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x3010B981.toInt() // Subtle emerald glow around user
        style = Paint.Style.FILL
    }
    private val userArrowFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt() // Crisp white navigation arrow
        style = Paint.Style.FILL
    }
    private val userArrowOutlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF0F172A.toInt() // Slate 900 high-contrast edge
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * resources.displayMetrics.density
        strokeJoin = Paint.Join.ROUND
    }

    private val containerRect = RectF()
    private val arrowPath = Path()

    /**
     * Set the magnetic north offset angle in degrees.
     * When applied, -Z in rotated space points strictly towards Magnetic North.
     */
    fun setNorthOffsetDeg(deg: Float) {
        northOffsetDeg = deg
        val rad = Math.toRadians(deg.toDouble()).toFloat()
        cosNorth = kotlin.math.cos(rad)
        sinNorth = kotlin.math.sin(rad)
        recomputeAllBounds()
        invalidate()
    }

    // Convert ARCore world coordinates (x, z) to magnetically aligned coordinates (magX, magZ)
    private fun toMagX(x: Float, z: Float): Float = x * cosNorth - z * sinNorth
    private fun toMagZ(x: Float, z: Float): Float = x * sinNorth + z * cosNorth

    private fun recomputeAllBounds() {
        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minZ = Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
        for (i in 0 until activeLandmarkCount) {
            val mx = toMagX(activeLandmarksX[i], activeLandmarksZ[i])
            val mz = toMagZ(activeLandmarksX[i], activeLandmarksZ[i])
            if (mx < minX) minX = mx
            if (mx > maxX) maxX = mx
            if (mz < minZ) minZ = mz
            if (mz > maxZ) maxZ = mz
        }
        if (activeLandmarkCount > 0) {
            activeMinMagX = minX; activeMaxMagX = maxX
            activeMinMagZ = minZ; activeMaxMagZ = maxZ
        }

        minX = Float.MAX_VALUE; maxX = -Float.MAX_VALUE
        minZ = Float.MAX_VALUE; maxZ = -Float.MAX_VALUE
        for (i in 0 until archivedLandmarkCount) {
            val mx = toMagX(archivedLandmarksX[i], archivedLandmarksZ[i])
            val mz = toMagZ(archivedLandmarksX[i], archivedLandmarksZ[i])
            if (mx < minX) minX = mx
            if (mx > maxX) maxX = mx
            if (mz < minZ) minZ = mz
            if (mz > maxZ) maxZ = mz
        }
        if (archivedLandmarkCount > 0) {
            archivedMinMagX = minX; archivedMaxMagX = maxX
            archivedMinMagZ = minZ; archivedMaxMagZ = maxZ
        }
    }

    fun updateUser(x: Float, z: Float, yawDeg: Float) {
        userX = x
        userZ = z
        userYawDeg = yawDeg
        invalidate()
    }

    fun resetTrail() {}

    fun setStart(x: Float, z: Float) {
        startX = x
        startZ = z
        hasStart = true
        addDoorway(x, z)
        invalidate()
    }

    fun addDoorway(x: Float, z: Float) {
        startX = x
        startZ = z
        hasStart = true
        if (doorwayCount < doorwayX.size) {
            doorwayX[doorwayCount] = x
            doorwayZ[doorwayCount] = z
            doorwayCount++
        }
        invalidate()
    }

    fun resetDoorways() {
        doorwayCount = 0
        hasStart = false
        activeLandmarkCount = 0
        archivedLandmarkCount = 0
        activeMinMagX = Float.MAX_VALUE; activeMaxMagX = -Float.MAX_VALUE
        activeMinMagZ = Float.MAX_VALUE; activeMaxMagZ = -Float.MAX_VALUE
        archivedMinMagX = Float.MAX_VALUE; archivedMaxMagX = -Float.MAX_VALUE
        archivedMinMagZ = Float.MAX_VALUE; archivedMaxMagZ = -Float.MAX_VALUE
        initializedBounds = false
        invalidate()
    }

    /**
     * Real-time Statistical Outlier Removal (SOR) and proximity filtering.
     * Prevents distant/errant points from collapsing the auto-scale bounding box.
     */
    private fun filterOutliers(
        inX: FloatArray,
        inZ: FloatArray,
        inCount: Int,
        outX: FloatArray,
        outZ: FloatArray,
    ): Int {
        if (inCount <= 0) return 0
        val maxOut = minOf(outX.size, outZ.size, scratchCandX.size)

        // Phase 1: Finite coordinate verification (adaptive per-room scale, no hardcoded proximity cut)
        var candidateCount = 0
        for (i in 0 until inCount) {
            val x = inX[i]; val z = inZ[i]
            if (x.isNaN() || z.isNaN() || x.isInfinite() || z.isInfinite()) continue

            if (candidateCount < scratchCandX.size) {
                scratchCandX[candidateCount] = x
                scratchCandZ[candidateCount] = z
                candidateCount++
            }
        }

        if (candidateCount < 8) {
            val n = minOf(candidateCount, maxOut)
            System.arraycopy(scratchCandX, 0, outX, 0, n)
            System.arraycopy(scratchCandZ, 0, outZ, 0, n)
            return n
        }

        // Phase 2: Statistical Outlier Removal via Median Absolute Deviation (MAD)
        System.arraycopy(scratchCandX, 0, scratchSortX, 0, candidateCount)
        scratchSortX.sort(0, candidateCount)
        val medX = scratchSortX[candidateCount / 2]

        System.arraycopy(scratchCandZ, 0, scratchSortZ, 0, candidateCount)
        scratchSortZ.sort(0, candidateCount)
        val medZ = scratchSortZ[candidateCount / 2]

        for (i in 0 until candidateCount) {
            scratchSortX[i] = kotlin.math.abs(scratchCandX[i] - medX)
            scratchSortZ[i] = kotlin.math.abs(scratchCandZ[i] - medZ)
        }
        scratchSortX.sort(0, candidateCount)
        scratchSortZ.sort(0, candidateCount)
        val madX = scratchSortX[candidateCount / 2]
        val madZ = scratchSortZ[candidateCount / 2]

        val cutoffX = maxOf(3.2f * 1.4826f * madX, 2.0f)
        val cutoffZ = maxOf(3.2f * 1.4826f * madZ, 2.0f)

        var inlierCount = 0
        for (i in 0 until candidateCount) {
            val x = scratchCandX[i]
            val z = scratchCandZ[i]
            if (kotlin.math.abs(x - medX) <= cutoffX && kotlin.math.abs(z - medZ) <= cutoffZ) {
                if (inlierCount < maxOut) {
                    outX[inlierCount] = x
                    outZ[inlierCount] = z
                    inlierCount++
                }
            }
        }

        return inlierCount
    }

    /**
     * Update active room landmark points live online during scanning with outlier removal.
     */
    fun updateActiveLandmarks(xs: FloatArray, zs: FloatArray, count: Int) {
        val inliers = filterOutliers(xs, zs, count, activeLandmarksX, activeLandmarksZ)
        activeLandmarkCount = inliers
        if (inliers > 0) {
            var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
            var minZ = Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
            for (i in 0 until inliers) {
                val mx = toMagX(activeLandmarksX[i], activeLandmarksZ[i])
                val mz = toMagZ(activeLandmarksX[i], activeLandmarksZ[i])
                if (mx < minX) minX = mx
                if (mx > maxX) maxX = mx
                if (mz < minZ) minZ = mz
                if (mz > maxZ) maxZ = mz
            }
            activeMinMagX = minX; activeMaxMagX = maxX
            activeMinMagZ = minZ; activeMaxMagZ = maxZ
        } else {
            activeMinMagX = Float.MAX_VALUE; activeMaxMagX = -Float.MAX_VALUE
            activeMinMagZ = Float.MAX_VALUE; activeMaxMagZ = -Float.MAX_VALUE
        }
        invalidate()
    }

    /**
     * Commit active landmarks to persistent multi-room archive (called when moving to next room).
     */
    fun archiveActiveLandmarks() {
        val toCopy = minOf(activeLandmarkCount, archivedLandmarksX.size - archivedLandmarkCount)
        if (toCopy > 0) {
            System.arraycopy(activeLandmarksX, 0, archivedLandmarksX, archivedLandmarkCount, toCopy)
            System.arraycopy(activeLandmarksZ, 0, archivedLandmarksZ, archivedLandmarkCount, toCopy)
            archivedLandmarkCount += toCopy
            if (activeMinMagX < archivedMinMagX) archivedMinMagX = activeMinMagX
            if (activeMaxMagX > archivedMaxMagX) archivedMaxMagX = activeMaxMagX
            if (activeMinMagZ < archivedMinMagZ) archivedMinMagZ = activeMinMagZ
            if (activeMaxMagZ > archivedMaxMagZ) archivedMaxMagZ = activeMaxMagZ
        }
        activeLandmarkCount = 0
        activeMinMagX = Float.MAX_VALUE; activeMaxMagX = -Float.MAX_VALUE
        activeMinMagZ = Float.MAX_VALUE; activeMaxMagZ = -Float.MAX_VALUE
        invalidate()
    }

    fun addProjectedLandmarks(xs: FloatArray, zs: FloatArray, count: Int) {
        val filtered = filterOutliers(xs, zs, count, scratchCandX, scratchCandZ)
        val toCopy = minOf(filtered, archivedLandmarksX.size - archivedLandmarkCount)
        if (toCopy > 0) {
            System.arraycopy(scratchCandX, 0, archivedLandmarksX, archivedLandmarkCount, toCopy)
            System.arraycopy(scratchCandZ, 0, archivedLandmarksZ, archivedLandmarkCount, toCopy)
            for (i in 0 until toCopy) {
                val mx = toMagX(scratchCandX[i], scratchCandZ[i])
                val mz = toMagZ(scratchCandX[i], scratchCandZ[i])
                if (mx < archivedMinMagX) archivedMinMagX = mx
                if (mx > archivedMaxMagX) archivedMaxMagX = mx
                if (mz < archivedMinMagZ) archivedMinMagZ = mz
                if (mz > archivedMaxMagZ) archivedMaxMagZ = mz
            }
            archivedLandmarkCount += toCopy
            invalidate()
        }
    }

    @Suppress("UNUSED_PARAMETER")
    fun setTarget(x: Float?, z: Float?) {}

    @Suppress("UNUSED_PARAMETER")
    fun updateFloorplan(fp: VoxelGrid.Floorplan2D?) {}

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

        // Cardinal North 'N' indicator at top-center (since map is strictly North Up)
        canvas.drawText("N", w * 0.5f, 13f * density, northTextPaint)

        // Clip to rounded container
        canvas.save()
        val clipPath = Path().apply { addRoundRect(containerRect, cornerRadius, cornerRadius, Path.Direction.CW) }
        canvas.clipPath(clipPath)

        // Calculate dynamic bounding box in magnetically-aligned coordinates
        val uMagX = toMagX(userX, userZ)
        val uMagZ = toMagZ(userX, userZ)

        var targetMinX = uMagX - 1.2f
        var targetMaxX = uMagX + 1.2f
        var targetMinZ = uMagZ - 1.2f
        var targetMaxZ = uMagZ + 1.2f

        if (hasStart) {
            val sMagX = toMagX(startX, startZ)
            val sMagZ = toMagZ(startX, startZ)
            targetMinX = minOf(targetMinX, sMagX - 0.8f)
            targetMaxX = maxOf(targetMaxX, sMagX + 0.8f)
            targetMinZ = minOf(targetMinZ, sMagZ - 0.8f)
            targetMaxZ = maxOf(targetMaxZ, sMagZ + 0.8f)
        }

        for (i in 0 until doorwayCount) {
            val dMagX = toMagX(doorwayX[i], doorwayZ[i])
            val dMagZ = toMagZ(doorwayX[i], doorwayZ[i])
            targetMinX = minOf(targetMinX, dMagX - 0.8f)
            targetMaxX = maxOf(targetMaxX, dMagX + 0.8f)
            targetMinZ = minOf(targetMinZ, dMagZ - 0.8f)
            targetMaxZ = maxOf(targetMaxZ, dMagZ + 0.8f)
        }

        if (activeLandmarkCount > 0 && activeMinMagX <= activeMaxMagX) {
            targetMinX = minOf(targetMinX, activeMinMagX - 0.5f)
            targetMaxX = maxOf(targetMaxX, activeMaxMagX + 0.5f)
            targetMinZ = minOf(targetMinZ, activeMinMagZ - 0.5f)
            targetMaxZ = maxOf(targetMaxZ, activeMaxMagZ + 0.5f)
        }

        if (archivedLandmarkCount > 0 && archivedMinMagX <= archivedMaxMagX) {
            targetMinX = minOf(targetMinX, archivedMinMagX - 0.5f)
            targetMaxX = maxOf(targetMaxX, archivedMaxMagX + 0.5f)
            targetMinZ = minOf(targetMinZ, archivedMinMagZ - 0.5f)
            targetMaxZ = maxOf(targetMaxZ, archivedMaxMagZ + 0.5f)
        }

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

        fun toScreenX(magX: Float): Float = midX + (magX - cX) * scale
        // -Z is North (Up on screen), so positive Z is South (Down on screen)
        fun toScreenY(magZ: Float): Float = midY + (magZ - cZ) * scale

        // Point radius scales smoothly with overall scene extent, bounded by a safe minimum size
        val minPtRadius = 1.8f * density
        val ptRadius = maxOf(scale * 0.05f, minPtRadius)

        // 1. Draw archived landmarks from previous rooms (softer emerald green)
        for (i in 0 until archivedLandmarkCount) {
            val mx = toMagX(archivedLandmarksX[i], archivedLandmarksZ[i])
            val mz = toMagZ(archivedLandmarksX[i], archivedLandmarksZ[i])
            canvas.drawCircle(toScreenX(mx), toScreenY(mz), ptRadius * 0.85f, archivedLandmarkPaint)
        }

        // 2. Draw active landmarks for current room live online (vibrant emerald green)
        for (i in 0 until activeLandmarkCount) {
            val mx = toMagX(activeLandmarksX[i], activeLandmarksZ[i])
            val mz = toMagZ(activeLandmarksX[i], activeLandmarksZ[i])
            canvas.drawCircle(toScreenX(mx), toScreenY(mz), ptRadius, activeLandmarkPaint)
        }

        // 3. Draw doorway anchors (starting point position and previous doorways)
        for (i in 0 until doorwayCount) {
            val mx = toMagX(doorwayX[i], doorwayZ[i])
            val mz = toMagZ(doorwayX[i], doorwayZ[i])
            val sx = toScreenX(mx)
            val sy = toScreenY(mz)
            val isActive = (i == doorwayCount - 1)
            val paint = if (isActive) startPaint else prevDoorPaint
            val ring = if (isActive) startRingPaint else prevDoorRingPaint
            canvas.drawCircle(sx, sy, 5f * density, paint)
            canvas.drawCircle(sx, sy, 8f * density, ring)
        }

        // 4. Draw live user position & fixed-size navigation arrow
        val ux = toScreenX(uMagX)
        val uy = toScreenY(uMagZ)

        // Subtle radiant halo around user
        canvas.drawCircle(ux, uy, 11f * density, userHaloPaint)

        // Fixed physical screen-size user navigation arrow
        val arrowLength = 16f * density
        val arrowWidth = 13f * density
        arrowPath.reset()
        arrowPath.moveTo(0f, -arrowLength * 0.60f) // tip pointing North (Up)
        arrowPath.lineTo(arrowWidth * 0.5f, arrowLength * 0.40f) // right wing
        arrowPath.lineTo(0f, arrowLength * 0.15f) // inner rear notch
        arrowPath.lineTo(-arrowWidth * 0.5f, arrowLength * 0.40f) // left wing
        arrowPath.close()

        val userHeadingMag = (userYawDeg + northOffsetDeg + 360f) % 360f

        canvas.save()
        canvas.translate(ux, uy)
        canvas.rotate(userHeadingMag)
        canvas.drawPath(arrowPath, userArrowFillPaint)
        canvas.drawPath(arrowPath, userArrowOutlinePaint)
        canvas.restore()

        canvas.restore()
    }
}
