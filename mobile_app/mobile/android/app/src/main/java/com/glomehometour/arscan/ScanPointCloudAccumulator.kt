package com.glomehometour.arscan

import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Arrays
import java.util.Comparator
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * High-performance, lock-free spatial accumulator for ARCore feature points and camera trajectory.
 *
 * Designed for mobile efficiency:
 * 1. 3.5 cm spatial hash grid (open-addressed) drops duplicate points in the same voxel in O(1) time.
 * 2. Pre-allocated primitive float arrays prevent GC allocations.
 * 3. Fast binary little-endian PLY serialization and parsing.
 * 4. Trajectory breadcrumb sampling every ~10 cm.
 */
class ScanPointCloudAccumulator(
    val maxPoints: Int = MAX_POINTS_DEFAULT,
    val voxelSizeM: Float = VOXEL_SIZE_DEFAULT,
) {
    companion object {
        const val MAX_POINTS_DEFAULT = 60000
        const val VOXEL_SIZE_DEFAULT = 0.035f // 3.5 cm
        const val MAX_TRAJECTORY_POINTS = 8192
        const val MIN_TRAJECTORY_STEP_M = 0.08f // 8 cm
        const val MAX_TRAJECTORY_STEP_M = 0.5f
        const val MAX_POINT_DISTANCE_FROM_PATH_M = 6f

        private const val EMPTY_GRID_KEY = Long.MIN_VALUE
        private const val SOR_NEIGHBOR_COUNT = 20
        private const val SOR_STD_DEVIATION_MULTIPLIER = 2f
    }

    // Point coordinates [x, y, z] packed
    val pointCoords = FloatArray(maxPoints * 3)
    var pointCount: Int = 0
        private set

    // Trajectory coordinates [x, y, z] packed
    val trajectoryCoords = FloatArray(MAX_TRAJECTORY_POINTS * 3)
    var trajectoryCount: Int = 0
        private set

    private var lastTrajX = Float.NaN
    private var lastTrajY = Float.NaN
    private var lastTrajZ = Float.NaN

    // Spatial hash table for O(1) duplicate suppression
    private val tableSize = tableSizeFor(maxPoints * 2)
    private val gridKeys = LongArray(tableSize) { EMPTY_GRID_KEY }

    // Dynamic bounding box
    var minY = Float.MAX_VALUE; private set
    var maxY = -Float.MAX_VALUE; private set
    var minX = Float.MAX_VALUE; private set
    var maxX = -Float.MAX_VALUE; private set
    var minZ = Float.MAX_VALUE; private set
    var maxZ = -Float.MAX_VALUE; private set

    private fun tableSizeFor(cap: Int): Int {
        var n = cap - 1
        n = n or (n ushr 1)
        n = n or (n ushr 2)
        n = n or (n ushr 4)
        n = n or (n ushr 8)
        n = n or (n ushr 16)
        return max(64, n + 1)
    }

    private fun cellCoord(v: Float): Int = floor(v / voxelSizeM).toInt()

    private fun cellKey(cx: Int, cy: Int, cz: Int): Long {
        return ((cx.toLong() and 0x1FFFFFL) shl 42) or
               ((cy.toLong() and 0x1FFFFFL) shl 21) or
               (cz.toLong() and 0x1FFFFFL)
    }

    private fun hashCellKey(key: Long): Int {
        val h = (key xor (key ushr 32)) * 0x45d9f3bL
        return (h xor (h ushr 32)).toInt()
    }

    fun reset() {
        pointCount = 0
        trajectoryCount = 0
        gridKeys.fill(EMPTY_GRID_KEY)
        lastTrajX = Float.NaN
        lastTrajY = Float.NaN
        lastTrajZ = Float.NaN
        minY = Float.MAX_VALUE; maxY = -Float.MAX_VALUE
        minX = Float.MAX_VALUE; maxX = -Float.MAX_VALUE
        minZ = Float.MAX_VALUE; maxZ = -Float.MAX_VALUE
    }

    /**
     * Adds an ARCore feature point (x, y, z) in world space.
     * Returns true if point was added (unique voxel), false if rejected or table full.
     */
    fun addPoint(x: Float, y: Float, z: Float): Boolean {
        if (!x.isFinite() || !y.isFinite() || !z.isFinite()) return false
        if (pointCount >= maxPoints) return false

        val cx = cellCoord(x)
        val cy = cellCoord(y)
        val cz = cellCoord(z)
        val key = cellKey(cx, cy, cz)

        val mask = tableSize - 1
        var idx = hashCellKey(key) and mask
        while (gridKeys[idx] != EMPTY_GRID_KEY) {
            if (gridKeys[idx] == key) {
                return false // Already occupied voxel
            }
            idx = (idx + 1) and mask
        }

        gridKeys[idx] = key
        val off = pointCount * 3
        pointCoords[off] = x
        pointCoords[off + 1] = y
        pointCoords[off + 2] = z
        pointCount++

        if (x < minX) minX = x; if (x > maxX) maxX = x
        if (y < minY) minY = y; if (y > maxY) maxY = y
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z

        return true
    }

    /**
     * Bulk integration of raw float buffer [x0, y0, z0, conf0, x1, y1, z1, conf1...]
     */
    fun addPointsFromBuffer(buffer: FloatArray, count: Int, minConfidence: Float = 0.20f): Int {
        val n = min(count, buffer.size / 4)
        var added = 0
        for (i in 0 until n) {
            val base = i * 4
            val conf = buffer[base + 3]
            if (conf >= minConfidence) {
                if (addPoint(buffer[base], buffer[base + 1], buffer[base + 2])) {
                    added++
                }
            }
        }
        return added
    }

    /**
     * Records camera trajectory breadcrumbs if moved >= [MIN_TRAJECTORY_STEP_M].
     */
    fun addTrajectoryPoint(x: Float, y: Float, z: Float): Boolean {
        if (!x.isFinite() || !y.isFinite() || !z.isFinite()) return false
        if (trajectoryCount >= MAX_TRAJECTORY_POINTS) return false

        if (!lastTrajX.isNaN()) {
            val dx = x - lastTrajX
            val dy = y - lastTrajY
            val dz = z - lastTrajZ
            val distSq = dx * dx + dy * dy + dz * dz
            if (distSq < MIN_TRAJECTORY_STEP_M * MIN_TRAJECTORY_STEP_M) return false
            if (distSq > MAX_TRAJECTORY_STEP_M * MAX_TRAJECTORY_STEP_M) {
                reanchorTrajectory(x, y, z)
                return false
            }
        }

        val off = trajectoryCount * 3
        trajectoryCoords[off] = x
        trajectoryCoords[off + 1] = y
        trajectoryCoords[off + 2] = z
        trajectoryCount++
        lastTrajX = x
        lastTrajY = y
        lastTrajZ = z
        return true
    }

    fun reanchorTrajectory(x: Float, y: Float, z: Float) {
        if (!x.isFinite() || !y.isFinite() || !z.isFinite()) return
        lastTrajX = x
        lastTrajY = y
        lastTrajZ = z
    }

    /** Clips to the camera path, then removes statistical outliers. */
    fun clipToTrajectory(maxDistanceM: Float = MAX_POINT_DISTANCE_FROM_PATH_M): Int {
        if (pointCount == 0 || (trajectoryCount > 0 && (!maxDistanceM.isFinite() || maxDistanceM < 0f))) return 0

        var removed = 0
        if (trajectoryCount > 0) removed += clipOutsideTrajectory(maxDistanceM)
        removed += removeStatisticalOutliers()
        if (removed > 0) rebuildSpatialState()
        return removed
    }

    private fun clipOutsideTrajectory(maxDistanceM: Float): Int {

        val indices = Array(trajectoryCount) { it }
        val left = IntArray(trajectoryCount) { -1 }
        val right = IntArray(trajectoryCount) { -1 }
        val points = IntArray(trajectoryCount)
        val nodeAxis = IntArray(trajectoryCount)
        var sortAxis = 0
        val comparator = Comparator<Int> { a, b ->
            trajectoryCoords[a * 3 + sortAxis].compareTo(trajectoryCoords[b * 3 + sortAxis])
        }
        var nextNode = 0
        fun buildTree(start: Int, end: Int, depth: Int): Int {
            if (start >= end) return -1
            sortAxis = depth % 3
            Arrays.sort(indices, start, end, comparator)
            val middle = (start + end) ushr 1
            val node = nextNode++
            points[node] = indices[middle]
            nodeAxis[node] = sortAxis
            left[node] = buildTree(start, middle, depth + 1)
            right[node] = buildTree(middle + 1, end, depth + 1)
            return node
        }
        val root = buildTree(0, trajectoryCount, 0)
        val maxDistanceSq = maxDistanceM * maxDistanceM
        fun isNearPath(node: Int, x: Float, y: Float, z: Float): Boolean {
            if (node < 0) return false
            val pathOff = points[node] * 3
            val dx = x - trajectoryCoords[pathOff]
            val dy = y - trajectoryCoords[pathOff + 1]
            val dz = z - trajectoryCoords[pathOff + 2]
            if (dx * dx + dy * dy + dz * dz <= maxDistanceSq) return true

            val axis = (nodeAxis[node]).toInt()
            val delta = when (axis) {
                0 -> x - trajectoryCoords[pathOff]
                1 -> y - trajectoryCoords[pathOff + 1]
                else -> z - trajectoryCoords[pathOff + 2]
            }
            val near = if (delta < 0f) left[node] else right[node]
            val far = if (delta < 0f) right[node] else left[node]
            return isNearPath(near, x, y, z) ||
                (delta * delta <= maxDistanceSq && isNearPath(far, x, y, z))
        }

        var write = 0
        val originalCount = pointCount
        for (read in 0 until originalCount) {
            val off = read * 3
            val x = pointCoords[off]; val y = pointCoords[off + 1]; val z = pointCoords[off + 2]
            if (!isNearPath(root, x, y, z)) continue
            val targetOff = write * 3
            pointCoords[targetOff] = x
            pointCoords[targetOff + 1] = y
            pointCoords[targetOff + 2] = z
            write++
        }

        val removed = originalCount - write
        pointCount = write
        return removed
    }

    private fun removeStatisticalOutliers(): Int {
        val count = pointCount
        // shortcut: clouds with <=20 points lack 20 neighbors; lower the count if small-cloud cleanup is needed.
        if (count <= SOR_NEIGHBOR_COUNT) return 0

        val indices = Array(count) { it }
        val left = IntArray(count) { -1 }
        val right = IntArray(count) { -1 }
        val points = IntArray(count)
        val nodeAxis = IntArray(count)
        var sortAxis = 0
        val comparator = Comparator<Int> { a, b ->
            pointCoords[a * 3 + sortAxis].compareTo(pointCoords[b * 3 + sortAxis])
        }
        var nextNode = 0
        fun buildTree(start: Int, end: Int, depth: Int): Int {
            if (start >= end) return -1
            sortAxis = depth % 3
            Arrays.sort(indices, start, end, comparator)
            val middle = (start + end) ushr 1
            val node = nextNode++
            points[node] = indices[middle]
            nodeAxis[node] = sortAxis
            left[node] = buildTree(start, middle, depth + 1)
            right[node] = buildTree(middle + 1, end, depth + 1)
            return node
        }
        val root = buildTree(0, count, 0)
        val meanDistances = FloatArray(count)
        val nearestDistancesSq = FloatArray(SOR_NEIGHBOR_COUNT)
        var queryIndex = -1
        var heapSize = 0

        fun offer(distanceSq: Float) {
            if (heapSize < SOR_NEIGHBOR_COUNT) {
                var position = heapSize++
                nearestDistancesSq[position] = distanceSq
                while (position > 0) {
                    val parent = (position - 1) ushr 1
                    if (nearestDistancesSq[parent] >= nearestDistancesSq[position]) break
                    val swap = nearestDistancesSq[parent]
                    nearestDistancesSq[parent] = nearestDistancesSq[position]
                    nearestDistancesSq[position] = swap
                    position = parent
                }
            } else if (distanceSq < nearestDistancesSq[0]) {
                nearestDistancesSq[0] = distanceSq
                var position = 0
                while (true) {
                    val leftChild = position * 2 + 1
                    if (leftChild >= heapSize) break
                    val rightChild = leftChild + 1
                    val fartherChild = if (rightChild < heapSize && nearestDistancesSq[rightChild] > nearestDistancesSq[leftChild]) rightChild else leftChild
                    if (nearestDistancesSq[position] >= nearestDistancesSq[fartherChild]) break
                    val swap = nearestDistancesSq[position]
                    nearestDistancesSq[position] = nearestDistancesSq[fartherChild]
                    nearestDistancesSq[fartherChild] = swap
                    position = fartherChild
                }
            }
        }

        fun search(node: Int) {
            if (node < 0) return
            val pointIndex = points[node]
            val queryOff = queryIndex * 3
            val pointOff = pointIndex * 3
            if (pointIndex != queryIndex) {
                val dx = pointCoords[queryOff] - pointCoords[pointOff]
                val dy = pointCoords[queryOff + 1] - pointCoords[pointOff + 1]
                val dz = pointCoords[queryOff + 2] - pointCoords[pointOff + 2]
                offer(dx * dx + dy * dy + dz * dz)
            }

            val axis = nodeAxis[node]
            val delta = pointCoords[queryOff + axis] - pointCoords[pointOff + axis]
            val near = if (delta < 0f) left[node] else right[node]
            val far = if (delta < 0f) right[node] else left[node]
            search(near)
            if (heapSize < SOR_NEIGHBOR_COUNT || delta * delta < nearestDistancesSq[0]) search(far)
        }

        for (i in 0 until count) {
            queryIndex = i
            heapSize = 0
            search(root)
            var sum = 0.0
            for (j in 0 until SOR_NEIGHBOR_COUNT) sum += sqrt(nearestDistancesSq[j].toDouble())
            meanDistances[i] = (sum / SOR_NEIGHBOR_COUNT).toFloat()
        }

        var total = 0.0
        for (distance in meanDistances) total += distance
        val mean = total / count
        var squaredDeviationTotal = 0.0
        for (distance in meanDistances) {
            val deviation = distance - mean
            squaredDeviationTotal += deviation * deviation
        }
        val threshold = mean + SOR_STD_DEVIATION_MULTIPLIER * sqrt(squaredDeviationTotal / count)

        var write = 0
        for (read in 0 until count) {
            if (meanDistances[read] > threshold) continue
            val sourceOff = read * 3
            val targetOff = write * 3
            pointCoords[targetOff] = pointCoords[sourceOff]
            pointCoords[targetOff + 1] = pointCoords[sourceOff + 1]
            pointCoords[targetOff + 2] = pointCoords[sourceOff + 2]
            write++
        }
        pointCount = write
        return count - write
    }

    private fun rebuildSpatialState() {
        gridKeys.fill(EMPTY_GRID_KEY)
        minX = Float.MAX_VALUE; maxX = -Float.MAX_VALUE
        minY = Float.MAX_VALUE; maxY = -Float.MAX_VALUE
        minZ = Float.MAX_VALUE; maxZ = -Float.MAX_VALUE
        val mask = tableSize - 1
        for (i in 0 until pointCount) {
            val off = i * 3
            val x = pointCoords[off]; val y = pointCoords[off + 1]; val z = pointCoords[off + 2]
            val key = cellKey(cellCoord(x), cellCoord(y), cellCoord(z))
            var idx = hashCellKey(key) and mask
            while (gridKeys[idx] != EMPTY_GRID_KEY) idx = (idx + 1) and mask
            gridKeys[idx] = key
            if (x < minX) minX = x; if (x > maxX) maxX = x
            if (y < minY) minY = y; if (y > maxY) maxY = y
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
        }
    }

    fun getBounds(outBounds: FloatArray) {
        outBounds[0] = if (minX != Float.MAX_VALUE) minX else 0f
        outBounds[1] = if (maxX != -Float.MAX_VALUE) maxX else 0f
        outBounds[2] = if (minY != Float.MAX_VALUE) minY else 0f
        outBounds[3] = if (maxY != -Float.MAX_VALUE) maxY else 0f
        outBounds[4] = if (minZ != Float.MAX_VALUE) minZ else 0f
        outBounds[5] = if (maxZ != -Float.MAX_VALUE) maxZ else 0f
    }

    fun getCenter(outCenter: FloatArray) {
        val b = FloatArray(6)
        getBounds(b)
        outCenter[0] = (b[0] + b[1]) * 0.5f
        outCenter[1] = (b[2] + b[3]) * 0.5f
        outCenter[2] = (b[4] + b[5]) * 0.5f
    }

    /**
     * Serializes points to a standard binary little-endian PLY file.
     */
    fun writeBinaryPly(out: OutputStream) {
        val count = pointCount
        val header = "ply\n" +
                "format binary_little_endian 1.0\n" +
                "element vertex $count\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "end_header\n"
        out.write(header.toByteArray(Charsets.US_ASCII))

        val byteBuffer = ByteBuffer.allocate(count * 12).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until count * 3) {
            byteBuffer.putFloat(pointCoords[i])
        }
        out.write(byteBuffer.array())
        out.flush()
    }

    /**
     * Deserializes points from a binary little-endian PLY input stream.
     * Supports arbitrary vertex property layouts (e.g. XYZ, XYZ+RGBA) and uniform strided sampling.
     */
    fun readBinaryPly(inputStream: InputStream): Int {
        val allBytes = inputStream.readBytes()
        if (allBytes.isEmpty()) return 0
        val searchLimit = kotlin.math.min(allBytes.size, 4096)
        val headerStr = String(allBytes, 0, searchLimit, Charsets.US_ASCII)

        val endHeaderIdx = headerStr.indexOf("end_header")
        if (endHeaderIdx < 0) return 0
        val afterEnd = headerStr.indexOf('\n', endHeaderIdx)
        val headerSize = if (afterEnd >= 0) afterEnd + 1 else endHeaderIdx + 10

        val vertexRegex = Regex("""element vertex (\d+)""")
        val match = vertexRegex.find(headerStr) ?: return 0
        val numVerts = match.groupValues[1].toIntOrNull() ?: return 0
        if (numVerts <= 0) return 0

        // Parse vertex properties to determine byte stride and (x, y, z) offsets
        val lines = headerStr.substring(0, endHeaderIdx).lines()
        var inVertexElement = false
        var stride = 0
        var xOffset = -1
        var yOffset = -1
        var zOffset = -1

        for (line in lines) {
            val trimmed = line.trim()
            if (trimmed.startsWith("element vertex")) {
                inVertexElement = true
                continue
            } else if (trimmed.startsWith("element ")) {
                inVertexElement = false
                continue
            }
            if (inVertexElement && trimmed.startsWith("property ")) {
                val parts = trimmed.split(Regex("""\s+"""))
                if (parts.size >= 3) {
                    val type = parts[1].lowercase()
                    val name = parts[2].lowercase()
                    val size = when (type) {
                        "char", "uchar", "int8", "uint8" -> 1
                        "short", "ushort", "int16", "uint16" -> 2
                        "int", "uint", "int32", "uint32", "float", "float32" -> 4
                        "double", "float64" -> 8
                        else -> 4
                    }
                    if (name == "x") xOffset = stride
                    if (name == "y") yOffset = stride
                    if (name == "z") zOffset = stride
                    stride += size
                }
            }
        }

        if (stride <= 0) stride = 12
        if (xOffset < 0) xOffset = 0
        if (yOffset < 0) yOffset = 4
        if (zOffset < 0) zOffset = 8

        reset()
        val availableVerts = (allBytes.size - headerSize) / stride
        val totalValidVerts = min(numVerts, availableVerts)
        if (totalValidVerts <= 0) return 0

        val step = if (totalValidVerts > maxPoints) (totalValidVerts + maxPoints - 1) / maxPoints else 1
        val bb = ByteBuffer.wrap(allBytes).order(ByteOrder.LITTLE_ENDIAN)

        var i = 0
        while (i < totalValidVerts && pointCount < maxPoints) {
            val vOffset = headerSize + i * stride
            val x = bb.getFloat(vOffset + xOffset)
            val y = bb.getFloat(vOffset + yOffset)
            val z = bb.getFloat(vOffset + zOffset)
            addPoint(x, y, z)
            i += step
        }
        return pointCount
    }
}
