package com.glomehometour.arscan

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.media.Image
import android.os.Handler
import android.os.HandlerThread
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Monocular depth as the geometry source when the ARCore Depth API isn't available.
 *
 * Measured on the target device (Redmi Note 10S, "rosemary"): isDepthModeSupported returns false
 * for both AUTOMATIC and RAW_DEPTH_ONLY, and the feature point cloud returns 0 points for whole
 * seconds at a time. Neither can fill a voxel grid, which is why coverage sat at 0% with nothing
 * to draw. This is the fallback the project architecture always called for -- a deliberately
 * small on-device depth model, guidance only, never exported.
 *
 * The model, the ONNX wiring and the ARCore-anchored metric fit are lifted from
 * mobile_depth_map/, where they were validated on this same phone.
 */

/**
 * Sensor pixels <-> model input pixels for a rotate-to-upright, centre-crop-to-square, scale-to-
 * SIZE pipeline.
 *
 * Both directions live here because the guidance loop needs both and they must agree exactly:
 * the metric fit maps ARCore points *into* the model image, and the depth resample maps model
 * pixels *back* to sensor rays. A mismatch between the two is invisible on screen and shifts the
 * whole map by a fixed offset.
 */
object ModelCrop {

    /** Sensor (u, v) -> model (mx, my). False when the pixel falls outside the square crop. */
    fun toModel(u: Float, v: Float, w: Int, h: Int, rot: Int, size: Int, out: FloatArray): Boolean {
        val rw: Float
        val rh: Float
        val rx: Float
        val ry: Float
        when (normalise(rot)) {
            90 -> { rw = h.toFloat(); rh = w.toFloat(); rx = h - v; ry = u }
            180 -> { rw = w.toFloat(); rh = h.toFloat(); rx = w - u; ry = h - v }
            270 -> { rw = h.toFloat(); rh = w.toFloat(); rx = v; ry = w - u }
            else -> { rw = w.toFloat(); rh = h.toFloat(); rx = u; ry = v }
        }
        val side = minOf(rw, rh)
        val cx = rx - (rw - side) / 2f
        val cy = ry - (rh - side) / 2f
        if (cx < 0f || cx >= side || cy < 0f || cy >= side) return false
        out[0] = cx * size / side
        out[1] = cy * size / side
        return true
    }

    /** Model (mx, my) -> sensor (u, v). */
    fun toSensor(mx: Float, my: Float, w: Int, h: Int, rot: Int, size: Int, out: FloatArray) {
        val r = normalise(rot)
        val rw = if (r == 90 || r == 270) h.toFloat() else w.toFloat()
        val rh = if (r == 90 || r == 270) w.toFloat() else h.toFloat()
        val side = minOf(rw, rh)
        val rx = mx * side / size + (rw - side) / 2f
        val ry = my * side / size + (rh - side) / 2f
        when (r) {
            90 -> { out[0] = ry; out[1] = h - rx }
            180 -> { out[0] = w - rx; out[1] = h - ry }
            270 -> { out[0] = w - ry; out[1] = rx }
            else -> { out[0] = rx; out[1] = ry }
        }
    }

    /**
     * The same mapping as toSensor, as an affine [u0, v0, du/dmx, dv/dmx, du/dmy, dv/dmy].
     * toSensor is affine in (mx, my) for every rotation, so sampling the crop can step the
     * sensor coordinate instead of recomputing the mapping 65k times per frame.
     */
    fun affine(w: Int, h: Int, rot: Int, size: Int, out: FloatArray) {
        val o = FloatArray(2)
        val ax = FloatArray(2)
        val ay = FloatArray(2)
        toSensor(0.5f, 0.5f, w, h, rot, size, o)
        toSensor(1.5f, 0.5f, w, h, rot, size, ax)
        toSensor(0.5f, 1.5f, w, h, rot, size, ay)
        out[0] = o[0]; out[1] = o[1]
        out[2] = ax[0] - o[0]; out[3] = ax[1] - o[1]
        out[4] = ay[0] - o[0]; out[5] = ay[1] - o[1]
    }

    private fun normalise(rot: Int) = ((rot % 360) + 360) % 360
}

/**
 * Model input straight out of the YUV planes: for each of the 256x256 input pixels, read the one
 * sensor pixel it comes from.
 *
 * Nearest-neighbour on purpose. The alternative -- full-frame YUV to Bitmap, rotate, crop, scale,
 * as mobile_depth_map did -- converts 2 megapixels to throw away 97% of them, on the GL thread,
 * for an input the model then blurs anyway.
 */
object YuvCrop {

    /** Scratch for the crop affine. sample() is called from the GL thread only. */
    private val map = FloatArray(6)

    fun sample(image: Image, rot: Int, size: Int, out: IntArray) {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val y = yPlane.buffer
        val u = uPlane.buffer
        val v = vPlane.buffer
        val w = image.width
        val h = image.height
        ModelCrop.affine(w, h, rot, size, map)
        for (my in 0 until size) {
            var su = map[0] + my * map[4]
            var sv = map[1] + my * map[5]
            for (mx in 0 until size) {
                val sx = su.toInt().coerceIn(0, w - 1)
                val sy = sv.toInt().coerceIn(0, h - 1)
                su += map[2]
                sv += map[3]
                val luma = y.get(sy * yPlane.rowStride + sx * yPlane.pixelStride).toInt() and 0xFF
                val uvIndex = (sy / 2) * uPlane.rowStride + (sx / 2) * uPlane.pixelStride
                val cb = (u.get(uvIndex).toInt() and 0xFF) - 128
                val cr = (v.get(uvIndex).toInt() and 0xFF) - 128
                val r = (luma + 1.402f * cr).toInt().coerceIn(0, 255)
                val g = (luma - 0.344f * cb - 0.714f * cr).toInt().coerceIn(0, 255)
                val b = (luma + 1.772f * cb).toInt().coerceIn(0, 255)
                out[my * size + mx] = (r shl 16) or (g shl 8) or b
            }
        }
    }
}

/**
 * ZipDepth base, 256x256, ONNX Runtime CPU EP. Output is affine-invariant inverse depth: near =
 * high value, and the affine part is per-frame arbitrary, which is what DepthScale pins down.
 */
class MonoDepthEngine(context: Context) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val input = FloatBuffer.allocate(3 * SIZE * SIZE)

    @Volatile var latencyMs: Long = 0
        private set

    init {
        val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            // 2, not 4: on the 8-core G95 four inference threads starve the render and camera
            // threads and the preview visibly stutters. Inference gets ~30% slower; it runs at
            // 10 Hz behind a drop-if-busy gate, so that costs nothing the operator can see.
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(2))
        }
        session = env.createSession(bytes, options)
    }

    /** Blocking, ~200 ms on the target device. Returns disparity, SIZE*SIZE, row-major. */
    fun run(pixels: IntArray): FloatArray {
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            input.put(i, ((p shr 16) and 0xFF) / 255f)
            input.put(plane + i, ((p shr 8) and 0xFF) / 255f)
            input.put(2 * plane + i, (p and 0xFF) / 255f)
        }
        input.rewind()
        val started = System.nanoTime()
        OnnxTensor.createTensor(env, input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { tensor ->
            session.run(mapOf("image" to tensor)).use { outputs ->
                @Suppress("UNCHECKED_CAST")
                val raw = (outputs[0].value as Array<Array<Array<FloatArray>>>)[0][0]
                val out = FloatArray(plane)
                for (row in 0 until SIZE) System.arraycopy(raw[row], 0, out, row * SIZE, SIZE)
                latencyMs = (System.nanoTime() - started) / 1_000_000
                return out
            }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val SIZE = 256
        private const val MODEL_ASSET = "zipdepth_base_256x256.onnx"
    }
}

/**
 * Anchors the model's affine-invariant disparity to metres against ARCore's tracked feature
 * points: disparity ~= alpha / distance + beta, least-squares over the points that land inside
 * the model crop.
 *
 * Without this the map would breathe -- each frame's own arbitrary scale would put the same wall
 * at a different distance, and no voxel would ever be seen twice.
 */
object DepthScale {

    /** Returns (alpha, beta), or null when the frame doesn't support a trustworthy fit. */
    fun fit(
        disparity: FloatArray, size: Int,
        t: FloatArray, q: FloatArray,
        points: FloatArray, numPoints: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        sensorWidth: Int, sensorHeight: Int, rot: Int,
    ): FloatArray? {
        // Conjugate rotates world -> camera; the pose's own quaternion goes the other way.
        val inv = floatArrayOf(-q[0], -q[1], -q[2], q[3])
        val cam = FloatArray(3)
        val uv = FloatArray(2)
        var sumX = 0f
        var sumY = 0f
        var sumXY = 0f
        var sumXX = 0f
        var n = 0

        for (i in 0 until numPoints) {
            val base = i * 4
            if (points[base + 3] < MIN_CONFIDENCE) continue
            Unproject.rotate(inv, points[base] - t[0], points[base + 1] - t[1], points[base + 2] - t[2], cam)
            val distance = -cam[2] // ARCore camera space looks down -Z
            if (distance < MIN_FIT_DISTANCE_M || distance > MAX_FIT_DISTANCE_M) continue
            val u = fx * (cam[0] / distance) + cx
            val v = cy - fy * (cam[1] / distance)
            if (u < 0f || u >= sensorWidth || v < 0f || v >= sensorHeight) continue
            if (!ModelCrop.toModel(u, v, sensorWidth, sensorHeight, rot, size, uv)) continue
            val mx = uv[0].toInt().coerceIn(0, size - 1)
            val my = uv[1].toInt().coerceIn(0, size - 1)

            val x = 1f / distance
            val y = disparity[my * size + mx]
            sumX += x; sumY += y; sumXY += x * y; sumXX += x * x
            n++
        }
        if (n < MIN_POINTS) return null

        val denom = n * sumXX - sumX * sumX
        if (kotlin.math.abs(denom) < 1e-6f) return null
        val alpha = (n * sumXY - sumX * sumY) / denom
        val beta = (sumY - alpha * sumX) / n
        // alpha <= 0 means the fit decided nearer surfaces have *lower* disparity, i.e. it latched
        // onto noise; a negative scale would turn the whole map inside out.
        if (alpha <= 0f || !alpha.isFinite() || !beta.isFinite()) return null
        return floatArrayOf(alpha, beta)
    }

    /**
     * Model disparity -> a sensor-aligned metric depth image, the same contract the ARCore depth
     * path hands the voxel worker (so nothing downstream has to know which source it came from).
     * Pixels outside the square crop, and distances outside the trusted band, stay 0 = unknown.
     */
    fun toDepthFrame(
        disparity: FloatArray, size: Int, alpha: Float, beta: Float,
        sensorWidth: Int, sensorHeight: Int, rot: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
        outWidth: Int, outHeight: Int,
    ): ArScanRenderer.DepthFrame {
        val mm = ShortArray(outWidth * outHeight)
        val uv = FloatArray(2)
        val sx = sensorWidth.toFloat() / outWidth
        val sy = sensorHeight.toFloat() / outHeight
        for (row in 0 until outHeight) {
            for (col in 0 until outWidth) {
                val u = (col + 0.5f) * sx
                val v = (row + 0.5f) * sy
                if (!ModelCrop.toModel(u, v, sensorWidth, sensorHeight, rot, size, uv)) continue
                val d = disparity[uv[1].toInt().coerceIn(0, size - 1) * size + uv[0].toInt().coerceIn(0, size - 1)]
                val metres = alpha / (d - beta)
                if (metres >= VoxelGrid.MIN_RANGE_M && metres <= VoxelGrid.MAX_RANGE_M) {
                    mm[row * outWidth + col] = (metres * 1000f).toInt().toShort()
                }
            }
        }
        val scaleX = outWidth.toFloat() / sensorWidth
        val scaleY = outHeight.toFloat() / sensorHeight
        return ArScanRenderer.DepthFrame(
            mm, outWidth, outHeight,
            fx * scaleX, fy * scaleY, cx * scaleX, cy * scaleY,
        )
    }

    const val MIN_POINTS = 4
    private const val MIN_CONFIDENCE = 0.25f
    private const val MIN_FIT_DISTANCE_M = 0.2f
    private const val MAX_FIT_DISTANCE_M = 15f
}

/**
 * Inference thread: samples nothing, decides nothing, just turns a submitted frame into a depth
 * image and hands it on. Drop-if-busy like the voxel worker -- inference is ~4 Hz against a 30 Hz
 * loop, so most frames are simply not submitted.
 *
 * The metric fit is held across frames: ARCore publishes 0 feature points whenever the operator
 * stands still, and a scan that lost its scale every time the operator paused would be useless.
 * Stale scale is far better than no scale; it is refreshed the moment points come back.
 */
class MonoDepthWorker(
    context: Context,
    private val onDepth: (ArScanRenderer.DepthFrame, FloatArray, FloatArray) -> Unit,
) {

    /** ROTATION_DEGREES: the model wants an upright image and the activity is portrait-locked, so
     * the sensor's landscape frame is always a fixed quarter turn away. Same constant, same
     * reason, as mobile_depth_map's camera source. */
    private val rotation = 90

    private val thread = HandlerThread("mono-depth").apply { start() }
    private val handler = Handler(thread.looper)
    private val busy = AtomicBoolean(false)
    private val pixels = IntArray(MonoDepthEngine.SIZE * MonoDepthEngine.SIZE)

    @Volatile private var engine: MonoDepthEngine? = null
    @Volatile var ready = false
        private set
    @Volatile var calibrated = false
        private set
    @Volatile var pointsUsed = 0
        private set
    @Volatile var latencyMs = 0L
        private set
    private var alpha = 0f
    private var beta = 0f

    init {
        // 24 MB of model off the caller's thread; submit() no-ops until it lands.
        handler.post {
            engine = MonoDepthEngine(context)
            ready = true
        }
    }

    val isBusy: Boolean get() = busy.get()

    /**
     * Caller thread (GL): reads the camera image into the model input buffer, then hands off.
     * Returns false when the previous frame is still in flight, which is the throttle.
     */
    fun submit(
        image: Image,
        t: FloatArray, q: FloatArray,
        points: FloatArray?, numPoints: Int,
        fx: Float, fy: Float, cx: Float, cy: Float,
    ): Boolean {
        val e = engine ?: return false
        if (!busy.compareAndSet(false, true)) return false
        val width = image.width
        val height = image.height
        YuvCrop.sample(image, rotation, MonoDepthEngine.SIZE, pixels)
        val tCopy = t.copyOf()
        val qCopy = q.copyOf()
        val pointsCopy = points?.copyOf(numPoints * 4)
        handler.post {
            try {
                val disparity = e.run(pixels)
                latencyMs = e.latencyMs
                val fit = if (pointsCopy != null && numPoints >= DepthScale.MIN_POINTS) {
                    DepthScale.fit(
                        disparity, MonoDepthEngine.SIZE, tCopy, qCopy, pointsCopy, numPoints,
                        fx, fy, cx, cy, width, height, rotation,
                    )
                } else null
                if (fit != null) {
                    // Smoothed: a single frame's fit rides on however many points ARCore happened
                    // to publish, and an unsmoothed jump moves every wall at once.
                    if (calibrated) {
                        alpha += (fit[0] - alpha) * FIT_ALPHA
                        beta += (fit[1] - beta) * FIT_ALPHA
                    } else {
                        alpha = fit[0]
                        beta = fit[1]
                    }
                    calibrated = true
                    pointsUsed = numPoints
                }
                if (calibrated) {
                    onDepth(
                        DepthScale.toDepthFrame(
                            disparity, MonoDepthEngine.SIZE, alpha, beta,
                            width, height, rotation, fx, fy, cx, cy,
                            DEPTH_WIDTH, DEPTH_HEIGHT,
                        ),
                        tCopy, qCopy,
                    )
                }
            } catch (e: Exception) {
                android.util.Log.w("ArScan", "mono depth failed", e)
            } finally {
                busy.set(false)
            }
        }
        return true
    }

    fun shutdown() {
        handler.post { engine?.close() }
        thread.quitSafely()
    }

    companion object {
        /** Output depth image, sensor-aligned. Same order of magnitude as ARCore's own depth
         * image; at the voxel worker's stride that is ~2.3k rays per integration. Calibration
         * knob: raise it if coverage grows too slowly, lower it if integration time creeps up. */
        const val DEPTH_WIDTH = 256
        const val DEPTH_HEIGHT = 144
        private const val FIT_ALPHA = 0.3f
    }
}
