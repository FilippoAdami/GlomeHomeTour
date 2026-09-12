package com.glomehometour.arscan

import android.content.ContentValues
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicInteger

/** Camera planes -> NV21, split into the cheap half (GL thread) and the per-pixel half
 * (writer thread). */
object Nv21 {

    fun sizeOf(width: Int, height: Int): Int = width * height * 3 / 2

    /**
     * Caller (GL) thread half: bulk row copies out of the camera planes and nothing else.
     * The Image is recycled the moment we return, so the pixels have to be taken here, but
     * taking them as memcpys costs ~2 ms where the per-pixel interleave costs tens.
     * Reads absolutely (duplicate + rewind), so it can't disturb any other reader of the plane.
     */
    fun copyPlanes(
        y: ByteBuffer, u: ByteBuffer, v: ByteBuffer,
        width: Int, height: Int, yRowStride: Int,
        dstY: ByteArray, dstU: ByteArray, dstV: ByteArray,
    ) {
        val yb = y.duplicate()
        for (row in 0 until height) {
            yb.position(row * yRowStride)
            yb.get(dstY, row * width, width)
        }
        for ((src, dst) in listOf(u to dstU, v to dstV)) {
            val b = src.duplicate()
            b.rewind()
            b.get(dst, 0, minOf(dst.size, b.remaining()))
        }
    }

    /**
     * Writer thread half: YUV_420_888 -> NV21, the one format Android's own JPEG encoder
     * (YuvImage) accepts.
     *
     * Kept as a pure function on plain arrays so it can be unit-tested (see DatasetWriterTest):
     * the U/V plane layout varies per device -- semi-planar with pixelStride 2 on most, fully
     * planar on some -- and getting the interleave backwards produces images that look fine in
     * luma and have their colours swapped, which is exactly the sort of thing nobody notices
     * until the backend renders a blue sofa.
     */
    fun interleave(
        y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int, uvRowStride: Int, uvPixelStride: Int,
        out: ByteArray,
    ) {
        System.arraycopy(y, 0, out, 0, width * height)
        // NV21 is V then U, interleaved, at quarter resolution.
        var o = width * height
        for (row in 0 until height / 2) {
            val base = row * uvRowStride
            for (col in 0 until width / 2) {
                val i = base + col * uvPixelStride
                out[o++] = v[i]
                out[o++] = u[i]
            }
        }
    }
}

/**
 * Writes the export package (SPEC §2.5) off the render thread.
 *
 * JPEG encoding is the expensive part and happens on a single writer thread with a bounded
 * queue; a backed-up queue drops the keyframe and counts it rather than stalling ARCore's frame
 * loop, because a dropped keyframe costs a little coverage while a stalled loop costs tracking.
 *
 * Output goes to MediaStore under Documents/ so the dataset is visible over MTP and in the Files
 * app without any storage permission. Whether MediaStore accepts image/jpeg under Documents/ is
 * a policy that has moved between Android versions, so it's probed once at startup and the whole
 * session falls back to the app's external files dir (adb-pullable) if the probe fails.
 */
class DatasetWriter(private val context: Context, val sessionName: String) {

    private val thread = HandlerThread("dataset-writer").apply { start() }
    private val handler = Handler(thread.looper)
    private val queued = AtomicInteger()

    val written = AtomicInteger()
    val droppedQueue = AtomicInteger()

    /** Touched on the GL thread only, until finish() hands them to the writer thread. */
    private val keyframes = ArrayList<DatasetFormat.Keyframe>()
    private val trajectory = StringBuilder(1 shl 16).apply { append(DatasetFormat.TRAJECTORY_HEADER).append('\n') }

    /** Writer-thread scratch, allocated once per session. See addKeyframe. */
    private var planeY: ByteArray? = null
    private var planeU: ByteArray? = null
    private var planeV: ByteArray? = null
    private var nv21: ByteArray? = null

    @Volatile private var useMediaStore = true
    private val relativeRoot = "Documents/$ALBUM/$sessionName"

    val keyframeCount: Int get() = keyframes.size

    /** Human-readable destination for the HUD; only meaningful after the probe has run. */
    val destination: String
        get() = if (useMediaStore) relativeRoot else File(context.getExternalFilesDir(null), sessionName).absolutePath

    init {
        handler.post { probeSink() }
    }

    /** GL thread. Every tracked frame, exported or not (SPEC A5). */
    fun addPose(
        timestampNs: Long, tx: Float, ty: Float, tz: Float,
        qx: Float, qy: Float, qz: Float, qw: Float,
        tracking: String, exported: Boolean,
    ) {
        trajectory.append(
            DatasetFormat.trajectoryLine(timestampNs, tx, ty, tz, qx, qy, qz, qw, tracking, exported)
        ).append('\n')
    }

    /**
     * GL thread. Queues one keyframe for JPEG encoding and records its pose entry.
     * Returns false if the queue was full and the frame was dropped.
     *
     * The scratch buffers are reused across calls, which is only safe because MAX_QUEUED is 1:
     * a frame is refused while the previous one is still being encoded, so the writer thread is
     * never reading a buffer this thread is refilling.
     */
    fun addKeyframe(
        image: Image, timestampNs: Long, matrix: FloatArray,
        fx: Float, fy: Float, cx: Float, cy: Float,
        focusDistanceDiopters: Float = 0f,
        afState: Int = 0,
        afMode: Int = 0,
        focalLengthMm: Float = 0f,
    ): Boolean {
        if (queued.get() >= MAX_QUEUED) {
            droppedQueue.incrementAndGet()
            return false
        }
        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val uvBytes = uPlane.rowStride * height / 2
        if (planeY?.size != width * height) planeY = ByteArray(width * height)
        if (planeU?.size != uvBytes) { planeU = ByteArray(uvBytes); planeV = ByteArray(uvBytes) }
        if (nv21?.size != Nv21.sizeOf(width, height)) nv21 = ByteArray(Nv21.sizeOf(width, height))
        val y = planeY!!; val u = planeU!!; val v = planeV!!; val out = nv21!!
        Nv21.copyPlanes(
            yPlane.buffer, uPlane.buffer, vPlane.buffer,
            width, height, yPlane.rowStride, y, u, v,
        )
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val name = "frame_%05d.jpg".format(keyframes.size)
        keyframes.add(
            DatasetFormat.Keyframe(
                name, matrix, timestampNs, fx, fy, cx, cy,
                focusDistanceDiopters, afState, afMode, focalLengthMm,
            )
        )
        queued.incrementAndGet()
        handler.post {
            try {
                Nv21.interleave(y, u, v, width, height, uvRowStride, uvPixelStride, out)
                openOutput("images", name, "image/jpeg")?.use { o ->
                    YuvImage(out, ImageFormat.NV21, width, height, null)
                        .compressToJpeg(Rect(0, 0, width, height), JPEG_QUALITY, o)
                }
                written.incrementAndGet()
            } catch (e: Exception) {
                Log.w(TAG, "keyframe write failed", e)
            } finally {
                queued.decrementAndGet()
            }
        }
        return true
    }

    /**
     * GL thread. Serialises the manifests and flushes them once the keyframe queue drains (the
     * writer is a single ordered handler thread, so posting last is enough to be last).
     * onDone fires on the writer thread.
     */
    fun finish(
        fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int,
        summaryJson: String, onDone: (String) -> Unit,
    ) {
        val transforms = DatasetFormat.transformsJson(fx, fy, cx, cy, width, height, keyframes)
        val csv = trajectory.toString()
        val focusJson = DatasetFormat.focusMetadataJson(keyframes)
        handler.post {
            writeText("transforms.json", "application/json", transforms)
            writeText("trajectory.csv", "text/csv", csv)
            writeText("coverage_summary.json", "application/json", summaryJson)
            writeText("focus_metadata.json", "application/json", focusJson)
            onDone(destination)
        }
    }

    fun shutdown() {
        thread.quitSafely()
    }

    // ---- writer thread ----

    private fun probeSink() {
        if (!useMediaStore) return
        try {
            val uri = insert("images", "probe.jpg", "image/jpeg") ?: throw IllegalStateException("insert returned null")
            context.contentResolver.delete(uri, null, null)
        } catch (e: Exception) {
            Log.w(TAG, "MediaStore rejected the dataset layout, falling back to app storage", e)
            useMediaStore = false
        }
    }

    private fun writeText(name: String, mime: String, body: String) {
        try {
            openOutput(null, name, mime)?.use { it.write(body.toByteArray()) }
        } catch (e: Exception) {
            Log.w(TAG, "manifest write failed: $name", e)
        }
    }

    private fun openOutput(dir: String?, name: String, mime: String): OutputStream? {
        if (!useMediaStore) {
            val target = File(File(context.getExternalFilesDir(null), sessionName), dir ?: "")
            target.mkdirs()
            return FileOutputStream(File(target, name))
        }
        val uri = insert(dir, name, mime) ?: return null
        val stream = context.contentResolver.openOutputStream(uri) ?: return null
        // IS_PENDING keeps half-written files hidden from other apps; clearing it is what
        // publishes them (same lesson as mobile_sphere_capture's GalleryOutput).
        return object : OutputStream() {
            override fun write(b: Int) = stream.write(b)
            override fun write(b: ByteArray, off: Int, len: Int) = stream.write(b, off, len)
            override fun close() {
                stream.close()
                context.contentResolver.update(
                    uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null,
                )
            }
        }
    }

    private fun insert(dir: String?, name: String, mime: String): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, mime)
            put(MediaStore.MediaColumns.RELATIVE_PATH, if (dir == null) relativeRoot else "$relativeRoot/$dir")
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        return context.contentResolver.insert(
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values,
        )
    }

    companion object {
        private const val TAG = "ArScan"
        private const val ALBUM = "GlomeHomeTour"
        private const val JPEG_QUALITY = 92
        /** Deep enough to ride out an encode that runs long, shallow enough that a sustained
         * backlog is dropped now rather than eaten as latency and RAM. */
        /** 1, not 4: the keyframe buffers are reused, so a queued frame must be finished before
         * the next one may be taken. Keyframes are capped at 5 Hz and a 1080p encode is ~40 ms,
         * so the queue is rarely the thing that drops a frame; droppedQueue says if it is. */
        private const val MAX_QUEUED = 1
    }
}
