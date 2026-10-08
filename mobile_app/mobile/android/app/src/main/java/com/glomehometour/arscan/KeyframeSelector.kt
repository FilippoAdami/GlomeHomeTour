package com.glomehometour.arscan

import kotlin.math.*

/**
 * Android mobile port of select_keyframes_arcore.py and floor_bands.py.
 *
 * Estimates physical room floor area from the ARCore camera trajectory using 2D trajectory dilation
 * (0.8m standoff buffer), derives an optimal keyframe budget N = 50 + (min..max) * A_floor, and
 * selects a non-redundant, well-spaced keyframe stream prior to SfM/COLMAP.
 */
class KeyframeSelector(
    val minTranslationM: Float = 0.08f,
    val minRotationDeg: Float = 4.0f,
    val keyframesPerM2Lo: Float = 8.0f,
    val keyframesPerM2Hi: Float = 11.0f,
    val baseKeyframes: Float = 50.0f,
    val standoffM: Float = 0.8f,
    val minRoomAreaM2: Float = 9.0f,
) {

    data class PoseFrame(
        val index: Int,
        val filePath: String,
        val c2w: FloatArray, // 4x4 row-major matrix
        val sharpnessScore: Float = 25.0f,
    )

    data class ExtentResult(
        val totalFloorAreaM2: Float,
        val floorCount: Int,
        val cameraSpanX: Float,
        val cameraSpanZ: Float,
        val targetBudget: Int,
        val budgetLo: Int,
        val budgetHi: Int,
    )

    data class SelectionResult(
        val selectedIndices: List<Int>,
        val selectedFrames: List<PoseFrame>,
        val extent: ExtentResult,
        val originalCount: Int,
        val selectedCount: Int,
        val sharpnessSwappedCount: Int,
    )

    fun selectKeyframes(frames: List<PoseFrame>, targetCount: Int? = null): SelectionResult {
        val nTotal = frames.size
        if (nTotal <= 1) {
            val emptyExtent = ExtentResult(minRoomAreaM2, 1, 0f, 0f, nTotal, nTotal, nTotal)
            return SelectionResult(
                selectedIndices = frames.indices.toList(),
                selectedFrames = frames,
                extent = emptyExtent,
                originalCount = nTotal,
                selectedCount = nTotal,
                sharpnessSwappedCount = 0,
            )
        }

        val centers = Array(nTotal) { i ->
            floatArrayOf(frames[i].c2w[3], frames[i].c2w[7], frames[i].c2w[11])
        }
        val rots = Array(nTotal) { i ->
            floatArrayOf(
                frames[i].c2w[0], frames[i].c2w[1], frames[i].c2w[2],
                frames[i].c2w[4], frames[i].c2w[5], frames[i].c2w[6],
                frames[i].c2w[8], frames[i].c2w[9], frames[i].c2w[10],
            )
        }

        // Camera look direction is -Z in OpenGL c2w: -rots[i][2, 5, 8]. Pitch angle in degrees:
        val pitches = FloatArray(nTotal) { i ->
            val lookY = -rots[i][5]
            Math.toDegrees(asin(lookY.coerceIn(-1.0f, 1.0f).toDouble())).toFloat()
        }

        // 1. Room extent & budget
        val extent = estimateRoomExtent(centers, nTotal)
        val computedTarget = extent.targetBudget.coerceIn(2, nTotal)
        val target = (targetCount ?: computedTarget).coerceIn(2, nTotal)

        if (targetCount == null && nTotal <= target) {
            return SelectionResult(
                selectedIndices = frames.indices.toList(),
                selectedFrames = frames,
                extent = extent,
                originalCount = nTotal,
                selectedCount = nTotal,
                sharpnessSwappedCount = 0,
            )
        }

        // 2. Sequential walk with minimum baseline, angular change & gap floor
        val selected = mutableListOf(0)
        val maxWalkGap = 8
        for (i in 1 until nTotal - 1) {
            val last = selected.last()
            val dist = distance3D(centers[i], centers[last])
            val ang = rotationAngleDeg(rots[i], rots[last])
            val gap = i - last

            val isUp = pitches[i] > 2.0f
            val isBlurry = frames[i].sharpnessScore < 10.0f

            if (gap >= maxWalkGap || ang >= 15.0f || dist >= 0.35f) {
                selected.add(i)
            } else if (!isBlurry) {
                if ((dist >= minTranslationM || ang >= minRotationDeg) || (isUp && gap >= 2)) {
                    selected.add(i)
                }
            }
        }
        if (selected.last() != nTotal - 1) {
            selected.add(nTotal - 1)
        }

        // 3. Top up by bisecting largest gaps if below target
        while (selected.size < target && selected.size < nTotal) {
            var maxGap = 0
            var maxGapIdx = -1
            for (k in 0 until selected.size - 1) {
                val g = selected[k + 1] - selected[k]
                if (g > maxGap) {
                    maxGap = g
                    maxGapIdx = k
                }
            }
            if (maxGap <= 1 || maxGapIdx == -1) break
            val mid = (selected[maxGapIdx] + selected[maxGapIdx + 1]) / 2
            selected.add(maxGapIdx + 1, mid)
        }

        // 4. Greedy SE(3) pruning if exceeding target
        val maxPruneGap = 6
        val maxPruneRotDeg = 20.0f
        val maxPruneTransM = 0.65f

        while (selected.size > target && selected.size > 2) {
            var minCost = Float.POSITIVE_INFINITY
            var dropIdxInSelected = -1

            for (k in 0 until selected.size - 2) {
                val prev = selected[k]
                val curr = selected[k + 1]
                val next = selected[k + 2]

                val resultingGap = next - prev
                val stepTrans = distance3D(centers[next], centers[prev])
                val stepRot = rotationAngleDeg(rots[next], rots[prev])

                if (resultingGap > maxPruneGap || stepRot > maxPruneRotDeg || stepTrans > maxPruneTransM) {
                    continue
                }

                var combined = stepTrans + 0.02f * stepRot
                if (pitches[curr] > 2.0f) {
                    combined += 5.0f // protect upward-looking frames
                }
                val sVal = frames[curr].sharpnessScore
                if (sVal < 15.0f) {
                    combined -= (15.0f - sVal) * 0.1f // prioritize pruning blurry frames
                }

                if (combined < minCost) {
                    minCost = combined
                    dropIdxInSelected = k + 1
                }
            }

            if (dropIdxInSelected == -1 || minCost == Float.POSITIVE_INFINITY) {
                break
            }
            selected.removeAt(dropIdxInSelected)
        }

        // 5. Sharpness-aware neighbor swapping
        var swappedCount = 0
        for (pass in 0 until 2) {
            var anySwap = false
            for (i in 1 until selected.size - 1) {
                val curr = selected[i]
                val prev = selected[i - 1]
                val next = selected[i + 1]

                var bestCand = curr
                var bestSharpness = frames[curr].sharpnessScore

                val searchStart = max(prev + 1, curr - 5)
                val searchEnd = min(next - 1, curr + 5)

                for (cand in searchStart..searchEnd) {
                    if (cand == curr) continue
                    val dist = distance3D(centers[curr], centers[cand])
                    val ang = rotationAngleDeg(rots[curr], rots[cand])

                    if (dist <= 0.18f && ang <= 8.0f) {
                        val candSharpness = frames[cand].sharpnessScore
                        if (candSharpness > bestSharpness * 1.30f && candSharpness > frames[curr].sharpnessScore + 4.0f) {
                            bestSharpness = candSharpness
                            bestCand = cand
                        }
                    }
                }

                if (bestCand != curr) {
                    selected[i] = bestCand
                    swappedCount++
                    anySwap = true
                }
            }
            if (!anySwap) break
        }

        val selectedFramesList = selected.map { frames[it] }
        return SelectionResult(
            selectedIndices = selected,
            selectedFrames = selectedFramesList,
            extent = extent,
            originalCount = nTotal,
            selectedCount = selected.size,
            sharpnessSwappedCount = swappedCount,
        )
    }

    private fun estimateRoomExtent(centers: Array<FloatArray>, totalFrames: Int): ExtentResult {
        val yCoords = centers.map { it[1] }.sorted()
        val pLoY = yCoords[(yCoords.size * 0.01).toInt()]

        // Floor count using 2.8m rise and 15 points threshold
        var floorCount = 1
        var floorNum = 2
        while (true) {
            val thresh = 2.8f * (floorNum - 1) + 0.2f
            val countAbove = centers.count { (it[1] - pLoY) > thresh }
            if (countAbove > 15) {
                floorCount = floorNum
                floorNum++
            } else {
                break
            }
        }

        // Partition centers into floors
        val floorIndices = IntArray(centers.size) { i ->
            val h = centers[i][1] - pLoY
            (h / 2.5f).toInt().coerceIn(0, floorCount - 1)
        }

        var totalFloorArea = 0f
        var maxSpanX = 0f
        var maxSpanZ = 0f

        for (f in 0 until floorCount) {
            val floorPts = centers.indices.filter { floorIndices[it] == f }.map { centers[it] }
            if (floorPts.isEmpty()) continue

            val xs = floorPts.map { it[0] }.sorted()
            val zs = floorPts.map { it[2] }.sorted()

            val spanX = if (xs.size >= 5) {
                xs[(xs.size * 0.99).toInt()] - xs[(xs.size * 0.01).toInt()]
            } else {
                xs.last() - xs.first()
            }
            val spanZ = if (zs.size >= 5) {
                zs[(zs.size * 0.99).toInt()] - zs[(zs.size * 0.01).toInt()]
            } else {
                zs.last() - zs.first()
            }

            maxSpanX = max(maxSpanX, spanX)
            maxSpanZ = max(maxSpanZ, spanZ)

            val area = computeDilatedArea(floorPts.map { floatArrayOf(it[0], it[2]) }, standoffM)
            totalFloorArea += max(area, minRoomAreaM2)
        }

        totalFloorArea = max(totalFloorArea, minRoomAreaM2)
        val budgetLo = ceil(baseKeyframes + keyframesPerM2Lo * totalFloorArea).toInt().coerceAtMost(totalFrames)
        val budgetHi = ceil(baseKeyframes + keyframesPerM2Hi * totalFloorArea).toInt().coerceIn(budgetLo, totalFrames)
        val target = ((budgetLo + budgetHi) / 2).coerceIn(2, totalFrames)

        return ExtentResult(
            totalFloorAreaM2 = totalFloorArea,
            floorCount = floorCount,
            cameraSpanX = maxSpanX,
            cameraSpanZ = maxSpanZ,
            targetBudget = target,
            budgetLo = budgetLo,
            budgetHi = budgetHi,
        )
    }

    /**
     * Approximates the 2D dilated area (LineString.buffer) using a fine 2D occupancy grid.
     */
    fun computeDilatedArea(pts2D: List<FloatArray>, radius: Float): Float {
        if (pts2D.isEmpty()) return 0f
        if (pts2D.size == 1) return (PI * radius * radius).toFloat()

        var minX = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var minZ = Float.POSITIVE_INFINITY
        var maxZ = Float.NEGATIVE_INFINITY

        for (p in pts2D) {
            minX = min(minX, p[0])
            maxX = max(maxX, p[0])
            minZ = min(minZ, p[1])
            maxZ = max(maxZ, p[1])
        }

        val pad = radius + 0.1f
        minX -= pad
        maxX += pad
        minZ -= pad
        maxZ += pad

        val cellSize = 0.15f // 15 cm grid cells
        val gridW = max(1, ceil((maxX - minX) / cellSize).toInt())
        val gridH = max(1, ceil((maxZ - minZ) / cellSize).toInt())

        // BitSet for occupied cells
        val occupied = java.util.BitSet(gridW * gridH)
        val r2 = radius * radius

        for (i in 0 until pts2D.size - 1) {
            val ax = pts2D[i][0]
            val az = pts2D[i][1]
            val bx = pts2D[i + 1][0]
            val bz = pts2D[i + 1][1]

            val segMinX = min(ax, bx) - radius
            val segMaxX = max(ax, bx) + radius
            val segMinZ = min(az, bz) - radius
            val segMaxZ = max(az, bz) + radius

            val startGx = max(0, ((segMinX - minX) / cellSize).toInt())
            val endGx = min(gridW - 1, ((segMaxX - minX) / cellSize).toInt())
            val startGz = max(0, ((segMinZ - minZ) / cellSize).toInt())
            val endGz = min(gridH - 1, ((segMaxZ - minZ) / cellSize).toInt())

            val abx = bx - ax
            val abz = bz - az
            val abLen2 = abx * abx + abz * abz

            for (gz in startGz..endGz) {
                val cz = minZ + (gz + 0.5f) * cellSize
                val rowOff = gz * gridW
                for (gx in startGx..endGx) {
                    val cellIdx = rowOff + gx
                    if (occupied.get(cellIdx)) continue

                    val cx = minX + (gx + 0.5f) * cellSize
                    val distSq = if (abLen2 <= 1e-7f) {
                        val dx = cx - ax
                        val dz = cz - az
                        dx * dx + dz * dz
                    } else {
                        val t = (((cx - ax) * abx + (cz - az) * abz) / abLen2).coerceIn(0.0f, 1.0f)
                        val projX = ax + t * abx
                        val projZ = az + t * abz
                        val dx = cx - projX
                        val dz = cz - projZ
                        dx * dx + dz * dz
                    }

                    if (distSq <= r2) {
                        occupied.set(cellIdx)
                    }
                }
            }
        }

        return occupied.cardinality() * (cellSize * cellSize)
    }

    companion object {
        fun distance3D(a: FloatArray, b: FloatArray): Float {
            val dx = a[0] - b[0]
            val dy = a[1] - b[1]
            val dz = a[2] - b[2]
            return sqrt(dx * dx + dy * dy + dz * dz)
        }

        fun rotationAngleDeg(rA: FloatArray, rB: FloatArray): Float {
            // R_rel = rA * rB^T (since rB is orthogonal, rB^T is its inverse)
            // trace(R_rel) = sum_{i,j} rA[i,j] * rB[i,j]
            var tr = 0f
            for (i in 0 until 9) tr += rA[i] * rB[i]
            val cosTheta = ((tr - 1.0f) / 2.0f).coerceIn(-1.0f, 1.0f)
            return Math.toDegrees(acos(cosTheta.toDouble())).toFloat()
        }
    }
}
