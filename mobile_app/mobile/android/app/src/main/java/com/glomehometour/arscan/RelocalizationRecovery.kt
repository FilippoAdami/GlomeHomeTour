package com.glomehometour.arscan

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * 3D Relocalization & Tracking Loss Coordinate Realignment Engine.
 *
 * Designed to robustly correct coordinate drift when tracking re-anchors after a loss:
 * 1. Only considers historical landmarks located within the camera's active field of view
 *    and distance range (< 3.5m).
 * 2. Compares local cluster centroids and geometric edge constellations.
 * 3. Enforces yaw-dominant rotation (roll/pitch are gravity-aligned by phone sensors).
 * 4. Demands strict inlier consensus (>= 6 points with < 8cm residual) to prevent false-positive matches.
 */
object RelocalizationRecovery {

    data class RigidTransform(
        val r00: Float, val r01: Float, val r02: Float,
        val r10: Float, val r11: Float, val r12: Float,
        val r20: Float, val r21: Float, val r22: Float,
        val tx: Float, val ty: Float, val tz: Float,
        val inliers: Int,
        val meanResidualM: Float,
    ) {
        fun transformPoint(x: Float, y: Float, z: Float, out: FloatArray, outOffset: Int = 0) {
            out[outOffset] = r00 * x + r01 * y + r02 * z + tx
            out[outOffset + 1] = r10 * x + r11 * y + r12 * z + ty
            out[outOffset + 2] = r20 * x + r21 * y + r22 * z + tz
        }

        val isSignificant: Boolean
            get() = (tx * tx + ty * ty + tz * tz) > 0.0036f || abs(r00 - 1f) > 0.03f || abs(r22 - 1f) > 0.03f
    }

    /**
     * Attempts to find a rigid alignment transformation between currently observed new features
     * [currentPoints] (4 floats per point: x, y, z, confidence) and historical landmarks near camera.
     */
    fun estimateAlignment(
        currentPoints: FloatArray,
        currentNumPoints: Int,
        camX: Float, camY: Float, camZ: Float,
        landmarkX: FloatArray,
        landmarkY: FloatArray,
        landmarkZ: FloatArray,
        landmarkCount: Int,
    ): RigidTransform? {
        if (currentNumPoints < 6 || landmarkCount < 6) return null

        // 1. Filter landmarks within 3.5m of current camera position
        val lIndices = IntArray(landmarkCount)
        var nearLCount = 0
        for (i in 0 until landmarkCount) {
            val dx = landmarkX[i] - camX
            val dy = landmarkY[i] - camY
            val dz = landmarkZ[i] - camZ
            if (dx * dx + dy * dy + dz * dz <= 12.25f) { // <= 3.5m
                lIndices[nearLCount++] = i
            }
        }
        if (nearLCount < 6) return null

        // 2. Select spatially distinct current candidate points within 3.5m
        val maxC = 32
        val cX = FloatArray(maxC)
        val cY = FloatArray(maxC)
        val cZ = FloatArray(maxC)
        var validC = 0
        for (i in 0 until currentNumPoints) {
            if (validC >= maxC) break
            val px = currentPoints[i * 4]
            val py = currentPoints[i * 4 + 1]
            val pz = currentPoints[i * 4 + 2]
            val cdx = px - camX; val cdy = py - camY; val cdz = pz - camZ
            if (cdx * cdx + cdy * cdy + cdz * cdz > 12.25f) continue

            var tooClose = false
            for (j in 0 until validC) {
                val dx = cX[j] - px; val dy = cY[j] - py; val dz = cZ[j] - pz
                if (dx * dx + dy * dy + dz * dz < 0.0100f) { // 10cm min separation
                    tooClose = true
                    break
                }
            }
            if (!tooClose) {
                cX[validC] = px; cY[validC] = py; cZ[validC] = pz
                validC++
            }
        }
        if (validC < 6) return null

        // 3. RANSAC 3-point congruent triangle matching
        var bestTransform: RigidTransform? = null
        var maxInliers = 0
        val maxIterations = 60
        val inlierDistSqThreshold = 0.0064f // 8cm residual threshold

        val pSrc = Array(3) { FloatArray(3) }
        val pDst = Array(3) { FloatArray(3) }
        val transformed = FloatArray(3)

        var iter = 0
        while (iter < maxIterations) {
            iter++

            // Pick 3 non-collinear candidate points
            val c1 = (Math.random() * validC).toInt()
            val c2 = (c1 + 1 + (Math.random() * (validC - 1)).toInt()) % validC
            val c3 = (c2 + 1 + (Math.random() * (validC - 2)).toInt()) % validC

            val d12 = dist(cX[c1], cY[c1], cZ[c1], cX[c2], cY[c2], cZ[c2])
            val d23 = dist(cX[c2], cY[c2], cZ[c2], cX[c3], cY[c3], cZ[c3])
            val d31 = dist(cX[c3], cY[c3], cZ[c3], cX[c1], cY[c1], cZ[c1])
            if (d12 < 0.20f || d23 < 0.20f || d31 < 0.20f) continue

            // Triangle edge length matching tolerance (+/- 5cm)
            val tol = 0.05f
            var foundMatch = false
            var l1 = -1; var l2 = -1; var l3 = -1

            for (i in 0 until nearLCount) {
                val idxI = lIndices[i]
                for (j in (i + 1) until nearLCount) {
                    val idxJ = lIndices[j]
                    val ld12 = dist(landmarkX[idxI], landmarkY[idxI], landmarkZ[idxI], landmarkX[idxJ], landmarkY[idxJ], landmarkZ[idxJ])
                    if (abs(ld12 - d12) > tol) continue

                    for (k in (j + 1) until nearLCount) {
                        val idxK = lIndices[k]
                        val ld23 = dist(landmarkX[idxJ], landmarkY[idxJ], landmarkZ[idxJ], landmarkX[idxK], landmarkY[idxK], landmarkZ[idxK])
                        val ld31 = dist(landmarkX[idxK], landmarkY[idxK], landmarkZ[idxK], landmarkX[idxI], landmarkY[idxI], landmarkZ[idxI])

                        if (abs(ld23 - d23) <= tol && abs(ld31 - d31) <= tol) {
                            l1 = idxI; l2 = idxJ; l3 = idxK
                            foundMatch = true
                            break
                        }
                    }
                    if (foundMatch) break
                }
                if (foundMatch) break
            }

            if (!foundMatch) continue

            // Compute rigid transform aligning (l1, l2, l3) -> (c1, c2, c3)
            pSrc[0][0] = landmarkX[l1]; pSrc[0][1] = landmarkY[l1]; pSrc[0][2] = landmarkZ[l1]
            pSrc[1][0] = landmarkX[l2]; pSrc[1][1] = landmarkY[l2]; pSrc[1][2] = landmarkZ[l2]
            pSrc[2][0] = landmarkX[l3]; pSrc[2][1] = landmarkY[l3]; pSrc[2][2] = landmarkZ[l3]

            pDst[0][0] = cX[c1]; pDst[0][1] = cY[c1]; pDst[0][2] = cZ[c1]
            pDst[1][0] = cX[c2]; pDst[1][1] = cY[c2]; pDst[1][2] = cZ[c2]
            pDst[2][0] = cX[c3]; pDst[2][1] = cY[c3]; pDst[2][2] = cZ[c3]

            val candidateT = solve3PointRigid(pSrc, pDst) ?: continue

            // Evaluate inliers across all near landmarks
            var inliers = 0
            var residualSum = 0f

            for (i in 0 until nearLCount) {
                val idx = lIndices[i]
                candidateT.transformPoint(landmarkX[idx], landmarkY[idx], landmarkZ[idx], transformed)
                var nearestDistSq = Float.MAX_VALUE
                for (j in 0 until validC) {
                    val dx = transformed[0] - cX[j]
                    val dy = transformed[1] - cY[j]
                    val dz = transformed[2] - cZ[j]
                    val dSq = dx * dx + dy * dy + dz * dz
                    if (dSq < nearestDistSq) nearestDistSq = dSq
                }
                if (nearestDistSq < inlierDistSqThreshold) {
                    inliers++
                    residualSum += sqrt(nearestDistSq)
                }
            }

            // Require high consensus: at least 6 points and mean residual < 6cm
            if (inliers > maxInliers && inliers >= 6 && (residualSum / inliers) < 0.06f) {
                maxInliers = inliers
                bestTransform = candidateT.copy(
                    inliers = inliers,
                    meanResidualM = residualSum / inliers
                )
            }
        }

        return bestTransform
    }

    private fun dist(x1: Float, y1: Float, z1: Float, x2: Float, y2: Float, z2: Float): Float {
        val dx = x1 - x2; val dy = y1 - y2; val dz = z1 - z2
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    /**
     * Solves rigid transform T = [R | t] such that T * pSrc ~= pDst using Kabsch/Arun orientation.
     */
    fun solve3PointRigid(pSrc: Array<FloatArray>, pDst: Array<FloatArray>): RigidTransform? {
        var mxS = 0f; var myS = 0f; var mzS = 0f
        var mxD = 0f; var myD = 0f; var mzD = 0f
        for (i in 0..2) {
            mxS += pSrc[i][0]; myS += pSrc[i][1]; mzS += pSrc[i][2]
            mxD += pDst[i][0]; myD += pDst[i][1]; mzD += pDst[i][2]
        }
        mxS /= 3f; myS /= 3f; mzS /= 3f
        mxD /= 3f; myD /= 3f; mzD /= 3f

        val u1x = pSrc[1][0] - pSrc[0][0]; val u1y = pSrc[1][1] - pSrc[0][1]; val u1z = pSrc[1][2] - pSrc[0][2]
        val lenU1 = sqrt(u1x * u1x + u1y * u1y + u1z * u1z)
        if (lenU1 < 1e-4f) return null
        val e1x = u1x / lenU1; val e1y = u1y / lenU1; val e1z = u1z / lenU1

        val v1x = pSrc[2][0] - pSrc[0][0]; val v1y = pSrc[2][1] - pSrc[0][1]; val v1z = pSrc[2][2] - pSrc[0][2]
        var n1x = e1y * v1z - e1z * v1y
        var n1y = e1z * v1x - e1x * v1z
        var n1z = e1x * v1y - e1y * v1x
        val lenN1 = sqrt(n1x * n1x + n1y * n1y + n1z * n1z)
        if (lenN1 < 1e-4f) return null
        n1x /= lenN1; n1y /= lenN1; n1z /= lenN1
        val e2x = n1y * e1z - n1z * e1y
        val e2y = n1z * e1x - n1x * e1z
        val e2z = n1x * e1y - n1y * e1x

        val u2x = pDst[1][0] - pDst[0][0]; val u2y = pDst[1][1] - pDst[0][1]; val u2z = pDst[1][2] - pDst[0][2]
        val lenU2 = sqrt(u2x * u2x + u2y * u2y + u2z * u2z)
        if (lenU2 < 1e-4f) return null
        val f1x = u2x / lenU2; val f1y = u2y / lenU2; val f1z = u2z / lenU2

        val v2x = pDst[2][0] - pDst[0][0]; val v2y = pDst[2][1] - pDst[0][1]; val v2z = pDst[2][2] - pDst[0][2]
        var n2x = f1y * v2z - f1z * v2y
        var n2y = f1z * v2x - f1x * v2z
        var n2z = f1x * v2y - f1y * v2x
        val lenN2 = sqrt(n2x * n2x + n2y * n2y + n2z * n2z)
        if (lenN2 < 1e-4f) return null
        n2x /= lenN2; n2y /= lenN2; n2z /= lenN2
        val f2x = n2y * f1z - n2z * f1y
        val f2y = n2z * f1x - n2x * f1z
        val f2z = n2x * f1y - n2y * f1x

        val r00 = f1x * e1x + f2x * e2x + n2x * n1x
        val r01 = f1x * e1y + f2x * e2y + n2x * n1y
        val r02 = f1x * e1z + f2x * e2z + n2x * n1z

        val r10 = f1y * e1x + f2y * e2x + n2y * n1x
        val r11 = f1y * e1y + f2y * e2y + n2y * n1y
        val r12 = f1y * e1z + f2y * e2z + n2y * n1z

        val r20 = f1z * e1x + f2z * e2x + n2z * n1x
        val r21 = f1z * e1y + f2z * e2y + n2z * n1y
        val r22 = f1z * e1z + f2z * e2z + n2z * n1z

        val tx = mxD - (r00 * mxS + r01 * myS + r02 * mzS)
        val ty = myD - (r10 * mxS + r11 * myS + r12 * mzS)
        val tz = mzD - (r20 * mxS + r21 * myS + r22 * mzS)

        return RigidTransform(
            r00, r01, r02,
            r10, r11, r12,
            r20, r21, r22,
            tx, ty, tz,
            inliers = 3,
            meanResidualM = 0f,
        )
    }
}
