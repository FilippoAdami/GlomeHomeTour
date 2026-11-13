package com.glomehometour.arscan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import android.util.LruCache
import java.io.File
import java.io.InputStream
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Generates and caches high-performance 2D orthographic/isometric snapshots of 3D point clouds
 * for scan cards.
 *
 * Uses:
 * 1. [ScanPointCloudAccumulator] to parse binary PLY vertices with automatic stride sampling.
 * 2. Isometric rotation (35° azimuth, 26° elevation) looking into the reconstructed room.
 * 3. Turbo-style height colormap (Floor: Deep Blue -> Cyan -> Emerald Green -> Warm Yellow -> Coral Red).
 * 4. Ground/camera trajectory overlay in subtle warm gold.
 * 5. Two-tier caching: in-memory [LruCache] for instant 60 FPS scrolling + private disk cache.
 */
object PointCloudPreviewGenerator {

    private const val TAG = "PCPreviewGenerator"
    private val memoryCache = LruCache<String, Bitmap>(48)
    private data class Snapshot(val bitmap: Bitmap, val cacheable: Boolean)

    fun getInMemory(sessionName: String, scanFolder: String = "scan_0", width: Int = 216, height: Int = 216): Bitmap? {
        val cacheKey = "${sessionName}_${scanFolder}_pathclip_sor_v3_${width}x${height}"
        val bmp = memoryCache.get(cacheKey)
        return if (bmp != null && !bmp.isRecycled) bmp else null
    }

    fun getOrGeneratePreview(
        context: Context,
        sessionName: String,
        scanFolder: String = "scan_0",
        width: Int = 216,
        height: Int = 216,
    ): Bitmap {
        val cacheKey = "${sessionName}_${scanFolder}_pathclip_sor_v3_${width}x${height}"

        // 1. In-memory LRU cache
        val inMem = memoryCache.get(cacheKey)
        if (inMem != null && !inMem.isRecycled) {
            return inMem
        }

        // 2. Disk cache in app cache directory
        val diskFile = File(context.cacheDir, "pc_thumb_pathclip_sor_v3_${sessionName}_${scanFolder}.png")
        if (diskFile.exists() && diskFile.length() > 0) {
            try {
                val diskBmp = android.graphics.BitmapFactory.decodeFile(diskFile.absolutePath)
                if (diskBmp != null) {
                    memoryCache.put(cacheKey, diskBmp)
                    return diskBmp
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to decode disk-cached pointcloud preview for $sessionName", e)
            }
        }

        // 3. Generate snapshot
        val generated = generateSnapshot(context, sessionName, scanFolder, width, height)
        if (!generated.cacheable) return generated.bitmap
        memoryCache.put(cacheKey, generated.bitmap)

        // Save to disk cache asynchronously
        Thread {
            try {
                diskFile.outputStream().use { out ->
                    generated.bitmap.compress(Bitmap.CompressFormat.PNG, 90, out)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to write disk cache for $sessionName", e)
            }
        }.start()

        return generated.bitmap
    }

    private fun generateSnapshot(
        context: Context,
        sessionName: String,
        scanFolder: String,
        width: Int,
        height: Int,
    ): Snapshot {
        var plyStream: InputStream? = null
        try {
            plyStream = ScanProcessor.openPointCloudStream(context, sessionName, scanFolder)
            if (plyStream == null) {
                return Snapshot(generatePlaceholderBitmap(width, height, "Cloud Pending"), false)
            }

            val acc = ScanPointCloudAccumulator(maxPoints = 25000)
            val loadedCount = acc.readBinaryPly(plyStream)
            if (loadedCount <= 0) {
                return Snapshot(generatePlaceholderBitmap(width, height, "No 3D Points"), false)
            }

            // Load camera trajectory points if available
            val rawTrajectory = loadTrajectoryPoints(context, sessionName, scanFolder)
            for (i in 0 until rawTrajectory.size / 3) {
                acc.addTrajectoryPoint(rawTrajectory[i * 3], rawTrajectory[i * 3 + 1], rawTrajectory[i * 3 + 2])
            }
            acc.clipToTrajectory()
            if (acc.pointCount == 0) {
                return Snapshot(generatePlaceholderBitmap(width, height, "No 3D Points"), false)
            }
            val trajPoints = acc.trajectoryCoords.copyOf(acc.trajectoryCount * 3)

            val center = FloatArray(3)
            acc.getCenter(center)
            val cx = center[0]
            val cy = center[1]
            val cz = center[2]

            // Isometric view angles: 35 deg azimuth, 26 deg elevation
            val azimuthRad = 0.610865f // 35 deg
            val elevRad = 0.453786f    // 26 deg
            val cosTheta = cos(azimuthRad)
            val sinTheta = sin(azimuthRad)
            val cosPhi = cos(elevRad)
            val sinPhi = sin(elevRad)

            val count = acc.pointCount
            val projCoords = FloatArray(count * 2)
            var minPX = Float.MAX_VALUE
            var maxPX = -Float.MAX_VALUE
            var minPY = Float.MAX_VALUE
            var maxPY = -Float.MAX_VALUE

            for (i in 0 until count) {
                val px = acc.pointCoords[i * 3 + 0] - cx
                val py = acc.pointCoords[i * 3 + 1] - cy
                val pz = acc.pointCoords[i * 3 + 2] - cz

                // Rotate around Y (azimuth)
                val x1 = px * cosTheta - pz * sinTheta
                val z1 = px * sinTheta + pz * cosTheta
                val y1 = py

                // Tilt around horizontal axis (elevation)
                val x2 = x1
                val y2 = y1 * cosPhi - z1 * sinPhi

                projCoords[i * 2 + 0] = x2
                projCoords[i * 2 + 1] = y2

                if (x2 < minPX) minPX = x2
                if (x2 > maxPX) maxPX = x2
                if (y2 < minPY) minPY = y2
                if (y2 > maxPY) maxPY = y2
            }

            val spanX = max(maxPX - minPX, 0.001f)
            val spanY = max(maxPY - minPY, 0.001f)
            val padding = width * 0.12f
            val targetW = width - 2f * padding
            val targetH = height - 2f * padding
            val scale = min(targetW / spanX, targetH / spanY)
            val midPX = (minPX + maxPX) * 0.5f
            val midPY = (minPY + maxPY) * 0.5f

            for (i in 0 until count) {
                val x2 = projCoords[i * 2 + 0]
                val y2 = projCoords[i * 2 + 1]
                projCoords[i * 2 + 0] = width * 0.5f + (x2 - midPX) * scale
                projCoords[i * 2 + 1] = height * 0.5f - (y2 - midPY) * scale
            }

            val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bmp)
            // Futuristic dark slate background matching app theme
            canvas.drawColor(0xFF0B101B.toInt())

            // Height normalization & binning into 16 Turbo color buckets
            val hRange = max(acc.maxY - acc.minY, 0.001f)
            val numBins = 16
            val binCounts = IntArray(numBins)

            for (i in 0 until count) {
                val t = ((acc.pointCoords[i * 3 + 1] - acc.minY) / hRange).coerceIn(0f, 1f)
                val bin = (t * (numBins - 0.001f)).toInt()
                binCounts[bin]++
            }

            val binCoords = Array(numBins) { b -> FloatArray(binCounts[b] * 2) }
            val binIndices = IntArray(numBins)

            for (i in 0 until count) {
                val t = ((acc.pointCoords[i * 3 + 1] - acc.minY) / hRange).coerceIn(0f, 1f)
                val bin = (t * (numBins - 0.001f)).toInt()
                val idx = binIndices[bin]++
                binCoords[bin][idx * 2 + 0] = projCoords[i * 2 + 0]
                binCoords[bin][idx * 2 + 1] = projCoords[i * 2 + 1]
            }

            val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                strokeWidth = max(2.5f, width / 70f)
                strokeCap = Paint.Cap.ROUND
            }

            for (bin in 0 until numBins) {
                if (binCounts[bin] > 0) {
                    val t = (bin + 0.5f) / numBins.toFloat()
                    pointPaint.color = heightColor(t)
                    canvas.drawPoints(binCoords[bin], 0, binCounts[bin] * 2, pointPaint)
                }
            }

            // Draw camera trajectory if available
            if (trajPoints.isNotEmpty()) {
                val trajPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = 0xBBF1C40F.toInt() // Subtle warm gold trajectory
                    strokeWidth = max(2.0f, width / 90f)
                    style = Paint.Style.STROKE
                    strokeCap = Paint.Cap.ROUND
                    strokeJoin = Paint.Join.ROUND
                }

                var lastSX = Float.NaN
                var lastSY = Float.NaN
                val numTraj = trajPoints.size / 3
                for (i in 0 until numTraj) {
                    val px = trajPoints[i * 3 + 0] - cx
                    val py = trajPoints[i * 3 + 1] - cy
                    val pz = trajPoints[i * 3 + 2] - cz

                    val x1 = px * cosTheta - pz * sinTheta
                    val z1 = px * sinTheta + pz * cosTheta
                    val y1 = py

                    val x2 = x1
                    val y2 = y1 * cosPhi - z1 * sinPhi

                    val sx = width * 0.5f + (x2 - midPX) * scale
                    val sy = height * 0.5f - (y2 - midPY) * scale

                    if (!lastSX.isNaN()) {
                        canvas.drawLine(lastSX, lastSY, sx, sy, trajPaint)
                    }
                    lastSX = sx
                    lastSY = sy
                }
            }

            return Snapshot(bmp, true)
        } catch (e: Exception) {
            Log.e(TAG, "Error generating point cloud snapshot for $sessionName", e)
            return Snapshot(generatePlaceholderBitmap(width, height, "Preview Error"), false)
        } finally {
            try { plyStream?.close() } catch (_: Exception) {}
        }
    }

    private fun loadTrajectoryPoints(context: Context, sessionName: String, scanFolder: String): FloatArray {
        try {
            val csv = ScanProcessor.readTrajectoryCsv(context, sessionName, scanFolder) ?: return FloatArray(0)
            return trajectoryPointsFromCsv(csv)
        } catch (_: Exception) {}
        return FloatArray(0)
    }

    /**
     * Turbo colormap identical to [PointCloudRenderer.FRAGMENT_SHADER].
     */
    private fun heightColor(t: Float): Int {
        val ct = t.coerceIn(0f, 1f)
        val r: Float
        val g: Float
        val b: Float
        when {
            ct < 0.25f -> {
                val u = ct / 0.25f
                r = 0.20f + (0.00f - 0.20f) * u
                g = 0.20f + (0.75f - 0.20f) * u
                b = 0.85f + (0.90f - 0.85f) * u
            }
            ct < 0.50f -> {
                val u = (ct - 0.25f) / 0.25f
                r = 0.00f + (0.12f - 0.00f) * u
                g = 0.75f + (0.85f - 0.75f) * u
                b = 0.90f + (0.40f - 0.90f) * u
            }
            ct < 0.75f -> {
                val u = (ct - 0.50f) / 0.25f
                r = 0.12f + (0.95f - 0.12f) * u
                g = 0.85f + (0.85f - 0.85f) * u
                b = 0.40f + (0.20f - 0.40f) * u
            }
            else -> {
                val u = (ct - 0.75f) / 0.25f
                r = 0.95f + (0.95f - 0.95f) * u
                g = 0.85f + (0.35f - 0.85f) * u
                b = 0.20f + (0.25f - 0.20f) * u
            }
        }
        return Color.argb(255, (r * 255).toInt(), (g * 255).toInt(), (b * 255).toInt())
    }

    private fun generatePlaceholderBitmap(width: Int, height: Int, message: String): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(0xFF0F172A.toInt())

        val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0x2238BDF8.toInt()
            strokeWidth = 1.2f
        }
        for (step in 1..4) {
            val y = height * (0.35f + step * 0.12f)
            canvas.drawLine(width * 0.1f, y, width * 0.9f, y, gridPaint)
        }

        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF38BDF8.toInt()
            textSize = width * 0.11f
            textAlign = Paint.Align.CENTER
            isFakeBoldText = true
        }
        canvas.drawText("☁️ 3D", width * 0.5f, height * 0.46f, textPaint)

        textPaint.textSize = width * 0.07f
        textPaint.color = 0xFF94A3B8.toInt()
        textPaint.isFakeBoldText = false
        canvas.drawText(message, width * 0.5f, height * 0.68f, textPaint)

        return bmp
    }
}
