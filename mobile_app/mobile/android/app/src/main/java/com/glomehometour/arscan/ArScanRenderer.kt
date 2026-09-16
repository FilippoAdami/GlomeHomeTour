package com.glomehometour.arscan

import android.app.Activity
import android.content.Context
import android.graphics.ImageFormat
import android.media.Image
import android.media.ImageReader
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import java.util.concurrent.atomic.AtomicReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * GLSurfaceView renderer that owns the ARCore session, same structural pattern as
 * mobile_sphere_capture's ArCoreRenderer. What differs here: a camera config is *chosen* rather
 * than defaulted (SPEC §1.2a -- the CPU image stream is what gets exported, and its default is
 * 640x480 on most devices), the Depth API drives coverage instead of the feature point cloud
 * where it's available, and the overlay is a world-anchored voxel point cloud plus a guidance
 * marker rather than a bearing shell.
 *
 * Everything ARCore-facing stays on the GL thread. The voxel grid is deliberately *not* touched
 * here: it runs on its own worker (see MainActivity) and publishes back a plain FloatArray.
 */
class ArScanRenderer(
    private val context: Context,
    private val onFrame: (Sample?) -> Unit,
    private val onError: (Exception) -> Unit,
    /** Pre-flight WB/ISO metering needs the raw camera image but not a pose, so unlike [Sample]
     * it isn't gated on ARCore having reached TRACKING -- a still-black frame is exactly the case
     * metering exists to fix, and TRACKING can depend on the scene no longer being too dark to
     * find features in. Caller must close() the image. */
    private val onUntrackedImage: (Image) -> Unit = {},
) : GLSurfaceView.Renderer {

    /** Copied out of ARCore's depth image so the worker thread can use it after the frame is
     * released. Intrinsics are the camera's, rescaled to the depth image's own dimensions. */
    class DepthFrame(
        val millimetres: ShortArray,
        val width: Int,
        val height: Int,
        val fx: Float, val fy: Float, val cx: Float, val cy: Float,
    )

    class Sample(
        val pose: Pose,
        val timestampNs: Long,
        /** Non-null only when captureImage was set for this frame. Caller must close(). */
        val image: Image?,
        val depth: DepthFrame?,
        /** ARCore point cloud (x, y, z, confidence). */
        val points: FloatArray?,
        val pointIds: IntArray?,
        val numPoints: Int,
    )

    @Volatile var trackingState: String = "?"
        private set

    /** Set by MainActivity for the *next* frame. Image acquisition is what the photometric gate
     * and the keyframe writer both feed on; depth is throttled to the voxel worker's rate. */
    @Volatile var captureImage: Boolean = false
    @Volatile var wantDepth: Boolean = false

    /** Published by the voxel worker: occupied voxel centres as x, y, z, parallax-progress. */
    @Volatile var voxelPoints: FloatArray? = null

    /** World-space frontier guidance target, or null for none -- "you haven't looked over there
     * yet" (README/Phase_1.md §5/§6). */
    @Volatile var guidanceTarget: FloatArray? = null

    /** Screen bearing to the guidance target, degrees clockwise from screen-up, and whether it's
     * already near the centre of view. Read by the UI thread for the 2D arrow. */
    @Volatile var targetBearingDeg: Float = 0f
    @Volatile var targetCentred: Boolean = false

    /** World-space occlusion-centroid guidance target, or null for none -- "something's hidden
     * behind this" (README/Phase_1.md §5/§6's kitchen-island case). Deliberately a separate field
     * from [guidanceTarget]: the two are different signals and the UI shows them as two arrows. */
    @Volatile var occlusionTarget: FloatArray? = null
    @Volatile var occlusionBearingDeg: Float = 0f
    @Volatile var occlusionCentred: Boolean = false

    @Volatile var depthSupported = false
        private set
    @Volatile var imageWidth = 0
        private set
    @Volatile var imageHeight = 0
        private set
    /** Intrinsics of the CPU image stream, refreshed every frame (focus changes move fl_x). */
    @Volatile var focalX = 0f
        private set
    @Volatile var focalY = 0f
        private set
    @Volatile var principalX = 0f
        private set
    @Volatile var principalY = 0f
        private set

    private var session: Session? = null
    private var cameraPipeline: CameraPipeline? = null
    /** ARCore's own CPU image surface (frame.acquireCameraImage()) never delivers a frame on
     * some devices (confirmed on a Redmi Note 10S: 0/152 in a clean capture window) even though
     * the same capture session's GPU preview texture works fine -- a HAL/stream quirk, not a
     * tracking or exposure problem. This reader rides as an extra surface on the same repeating
     * request and is what every acquireCameraImage() call site below now reads from instead. */
    private var ownImageReader: ImageReader? = null
    private var ownImageReaderThread: HandlerThread? = null
    private var ownImageReaderSurfaces: List<Surface> = emptyList()
    private val pendingOwnImage = AtomicReference<Image?>(null)
    /** False the moment a device can't offer a 60 fps camera config -- surfaced on the
     * diagnostic HUD rather than silently retargeting at 30 (Phase_1.md §10 item 1). */
    @Volatile var sustained60Fps = true

    /** See [resumeSession]: false between session creation/pause and the camera actually opening. */
    @Volatile private var sessionResumed = false
        private set
    private var cameraTextureId = 0
    private var bgProgram = 0
    private var quadVertices: FloatBuffer? = null
    private var quadTexCoords: FloatBuffer? = null
    private var quadTexCoordsTransformed: FloatBuffer? = null
    private var viewportWidth = 0
    private var viewportHeight = 0

    private var meshProgram = 0
    private var pointProgram = 0
    private var soloProgram = 0
    private var meshBuffer: FloatBuffer? = null
    private var pointBuffer: FloatBuffer? = null
    private var meshVertexCount = 0
    private var uploadedMesh: FloatArray? = null
    private var markerBuffer: FloatBuffer? = null

    /** Point sprite buffer holding [x, y, z, verified] (4 floats per vertex). */
    @Volatile var pointVertices: FloatArray? = null
    @Volatile var pointVertexCount: Int = 0

    private val projMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val vpMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    private val mvpMatrix = FloatArray(16)
    private val clip = FloatArray(4)
    private val worldPoint = FloatArray(4)

    fun pause() {
        sessionResumed = false
        cameraPipeline?.stop()
        session?.pause()
    }

    fun resume() {
        val s = session ?: return
        cameraPipeline?.start(ownImageReaderSurfaces) { resumeSession(s) }
    }

    /**
     * ARCore can only be resumed once the shared camera is actually open, which CameraPipeline
     * reports back asynchronously -- so there is a window (hundreds of ms at startup, and again
     * after every pause) where `session` exists but is not resumed, and `session.update()` throws
     * SessionPausedException on every single frame. [sessionResumed] is what [onDrawFrame] skips
     * on: without it that window reported one exception per frame to `onError`.
     */
    private fun resumeSession(s: Session) {
        try {
            s.resume()
            sessionResumed = true
        } catch (e: Exception) {
            onError(e)
        }
    }

    /** Pre-flight WB lock (README §6 step 2): manual color-correction gains derived from the
     * live preview, applied to the running capture request. */
    fun lockWhiteBalance(gains: android.hardware.camera2.params.RggbChannelVector) {
        cameraPipeline?.whiteBalanceGains = gains
    }

    /** Pre-flight ambient-light metering (README §4/§6): ISO and shutter jointly adjusted live
     * while the operator is still in pre-flight, then simply stop being called once scanning
     * starts -- the last values set are what stay locked for the session. */
    fun adjustExposure(iso: Int, shutterNanos: Long) {
        cameraPipeline?.isoSensitivity = iso
        cameraPipeline?.shutterNanos = shutterNanos
    }

    val pipeline: CameraPipeline? get() = cameraPipeline

    fun triggerAutoFocus(normX: Float = 0.5f, normY: Float = 0.5f) {
        cameraPipeline?.triggerAutoFocus(normX, normY)
    }

    fun destroy() {
        cameraPipeline?.shutdown()
        cameraPipeline = null
        pendingOwnImage.getAndSet(null)?.close()
        ownImageReader?.close()
        ownImageReader = null
        ownImageReaderThread?.quitSafely()
        ownImageReaderThread = null
        session?.close()
        session = null
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        cameraTextureId = createExternalTexture()
        bgProgram = createProgram(BG_VERTEX_SHADER, BG_FRAGMENT_SHADER)
        meshProgram = createProgram(MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER)
        pointProgram = createProgram(POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER)
        soloProgram = createProgram(SOLO_VERTEX_SHADER, SOLO_FRAGMENT_SHADER)
        quadVertices = directFloatBuffer(QUAD_COORDS)
        quadTexCoords = directFloatBuffer(QUAD_TEXCOORDS)
        quadTexCoordsTransformed = directFloatBuffer(FloatArray(QUAD_TEXCOORDS.size))
        meshBuffer = ByteBuffer.allocateDirect(MAX_OVERLAY_TRIANGLES * 3 * 4 * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        pointBuffer = ByteBuffer.allocateDirect(MAX_POINT_SPRITES * 4 * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        markerBuffer = directFloatBuffer(octahedronEdges())

        try {
            // SHARED_CAMERA: ARCore no longer opens Camera2 itself. The app owns the capture
            // session and puts a manual CaptureRequest on the sensor (CameraPipeline), so README
            // §4's shutter/ISO/WB/stabilization/distortion overrides are actually enforced --
            // ARCore's own auto camera path has no hook for any of them.
            val s = Session(context, EnumSet.of(Session.Feature.SHARED_CAMERA))
            selectCameraConfig(s)
            depthSupported = s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
            val cfg = Config(s).apply {
                // AUTO, not FIXED: fixed focus on a phone lens is focus-at-infinity, and indoor
                // surfaces at ~1 m sit inside the hyperfocal distance, i.e. visibly soft in
                // exactly the frames the reconstruction cares most about. The cost is that
                // focus breathing moves fl_x mid-scan, which is why intrinsics are recorded
                // per keyframe as well as globally (DatasetFormat.transformsJson).
                focusMode = Config.FocusMode.AUTO
                planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                lightEstimationMode = Config.LightEstimationMode.DISABLED
                depthMode = if (depthSupported) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
            }
            s.configure(cfg)
            s.setCameraTextureName(cameraTextureId)
            session = s
            if (viewportWidth > 0) s.setDisplayGeometry(displayRotation, viewportWidth, viewportHeight)

            val imageSize = s.cameraConfig.imageSize
            val readerThread = HandlerThread("own-image-reader").apply { start() }
            ownImageReaderThread = readerThread
            val reader = ImageReader.newInstance(imageSize.width, imageSize.height, ImageFormat.YUV_420_888, 2)
            reader.setOnImageAvailableListener({ r ->
                val img = try { r.acquireLatestImage() } catch (e: Exception) { null }
                if (img != null) pendingOwnImage.getAndSet(img)?.close()
            }, Handler(readerThread.looper))
            ownImageReader = reader
            ownImageReaderSurfaces = listOf(reader.surface)

            val pipeline = CameraPipeline(context, s, onError)
            cameraPipeline = pipeline
            pipeline.start(ownImageReaderSurfaces) { resumeSession(s) }
        } catch (e: Exception) {
            onError(e)
        }
    }

    /**
     * Picks the largest CPU image stream at README §4's 60 fps sensor readout rate, capped at
     * 1080p, on the widest-FOV back camera ARCore is actually willing to bind to. Falls back
     * toward 30 fps only if a device genuinely offers no 60 fps config -- a hardware-tier gap to
     * surface (via onError, and `sustainedFps` on the diagnostic HUD), not a reason to quietly
     * retarget the whole pipeline at 30.
     *
     * ARCore's default config optimises for tracking, not for the frames we export: on most
     * devices that's a 640x480 CPU image regardless of how large the GPU texture is. Bigger CPU
     * streams cost frame rate, hence the cap.
     *
     * Ultra-wide auto-select: on-device logging (`rosemary`, 2026-08-30) disproved the earlier
     * assumption here that ARCore's default camera config was already the ultra-wide-equivalent
     * lens -- `getSupportedCameraConfigs()` only ever returned `cameraId "0"`, the primary
     * 68 deg-FOV sensor, never the device's true ultra-wide (122 deg FOV, `cameraId "2"` on that
     * device, confirmed via `dumpsys media.camera`). That sensor turned out to be invisible to
     * `CameraManager.getCameraIdList()` for third-party apps altogether -- a Xiaomi/MediaTek OEM
     * restriction, not something this code can route around: ARCore's `SharedCamera` binds one
     * physical device for both pose and image, so there is no way to point the export image at a
     * sensor ARCore itself never lists, without decoupling the two and losing pose-image
     * correspondence (extrinsic offset between an untracked second sensor and the tracked one) --
     * a correctness regression for the reconstruction pipeline, not just more code. What this can
     * do, and now does, is stop assuming and instead pick whichever camera ID among ARCore's own
     * offered configs actually has the widest field of view, so a device that *does* expose its
     * ultra-wide to ARCore (multiple `cameraId`s show up in `configs`) gets it automatically.
     */
    private fun selectCameraConfig(s: Session) {
        try {
            val allConfigs = s.supportedCameraConfigs
            android.util.Log.i("AR-Scan", "=== ARCore supportedCameraConfigs (total: ${allConfigs.size}) ===")
            for (c in allConfigs) {
                android.util.Log.i("AR-Scan", "  config: cameraId=${c.cameraId}, fps=${c.fpsRange}, imgSize=${c.imageSize.width}x${c.imageSize.height}, texSize=${c.textureSize.width}x${c.textureSize.height}, facing=${c.facingDirection}, depth=${c.depthSensorUsage}")
            }
            val manager = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager
            android.util.Log.i("AR-Scan", "=== CameraManager.cameraIdList: ${manager.cameraIdList.joinToString()} ===")
            for (id in manager.cameraIdList) {
                try {
                    val chars = manager.getCameraCharacteristics(id)
                    val facing = chars.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING)
                    val focals = chars.get(android.hardware.camera2.CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.joinToString()
                    val phys = chars.get(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    val zoomRange = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
                        chars.get(android.hardware.camera2.CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)
                    } else null
                    val physicalIds = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
                        chars.physicalCameraIds
                    } else emptySet()
                    android.util.Log.i("AR-Scan", "  Camera ID $id: facing=$facing, focals=$focals, physSize=$phys, zoomRange=$zoomRange, physicalIds=$physicalIds")
                } catch (e: Exception) {
                    android.util.Log.w("AR-Scan", "  Camera ID $id error: $e")
                }
            }

            val target60 = CameraConfigFilter(s).setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_60))
            var configs = s.getSupportedCameraConfigs(target60)
            if (configs.isEmpty()) {
                sustained60Fps = false
                configs = s.getSupportedCameraConfigs(
                    CameraConfigFilter(s).setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30)),
                )
            } else {
                sustained60Fps = true
            }
            val wideId = widestFovCameraId(configs.map { it.cameraId }.distinct())
            android.util.Log.i("AR-Scan", "Selected wideId: $wideId, sustained60Fps: $sustained60Fps")
            val candidates = if (wideId != null) configs.filter { it.cameraId == wideId } else configs
            val best = candidates
                .maxByOrNull { it.imageSize.width.toLong() * it.imageSize.height }
            if (best != null) {
                android.util.Log.i("AR-Scan", "Setting s.cameraConfig = cameraId:${best.cameraId} size:${best.imageSize.width}x${best.imageSize.height}")
                s.cameraConfig = best
            }
        } catch (e: Exception) {
            // A device that offers no config matching the filter keeps ARCore's default -- a
            // smaller export image, not a broken session.
            android.util.Log.e("AR-Scan", "selectCameraConfig error", e)
            onError(e)
        }
    }

    /** Widest-FOV back-facing camera ID among the ones ARCore actually offered a config for.
     * FOV from real `CameraCharacteristics` (focal length vs. sensor width), not an assumption --
     * see `selectCameraConfig`'s comment for why the previous "ARCore already defaults to
     * ultra-wide" belief didn't hold up on-device. Returns null (keep ARCore's own pick) when
     * there's only one ID to choose from or characteristics can't be read for any of them. */
    private fun widestFovCameraId(ids: List<String>): String? {
        if (ids.size < 2) return null
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as android.hardware.camera2.CameraManager
        return ids.mapNotNull { id ->
            try {
                val chars = manager.getCameraCharacteristics(id)
                if (chars.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING) !=
                    android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK
                ) return@mapNotNull null
                val focalLength = chars.get(android.hardware.camera2.CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    ?.minOrNull() ?: return@mapNotNull null
                val sensorWidth = chars.get(android.hardware.camera2.CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    ?.width ?: return@mapNotNull null
                val fovDeg = 2.0 * Math.toDegrees(kotlin.math.atan((sensorWidth / (2f * focalLength)).toDouble()))
                android.util.Log.i("AR-Scan", "widestFovCameraId candidate: id=$id, fovDeg=$fovDeg")
                id to fovDeg
            } catch (e: Exception) {
                null
            }
        }.maxByOrNull { it.second }?.first
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewportWidth = width
        viewportHeight = height
        session?.setDisplayGeometry(displayRotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        val s = session ?: return
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        if (viewportWidth == 0 || viewportHeight == 0) return
        if (!sessionResumed) return  // see resumeSession

        try {
            val frame = s.update()
            val camera = frame.camera

            frame.transformDisplayUvCoords(quadTexCoords, quadTexCoordsTransformed)
            drawCameraBackground()

            trackingState = if (camera.trackingState == TrackingState.TRACKING) {
                "TRACKING"
            } else {
                "${camera.trackingState}/${camera.trackingFailureReason}"
            }

            val sample = if (camera.trackingState == TrackingState.TRACKING) {
                val intrinsics = camera.imageIntrinsics
                imageWidth = intrinsics.imageDimensions[0]
                imageHeight = intrinsics.imageDimensions[1]
                focalX = intrinsics.focalLength[0]
                focalY = intrinsics.focalLength[1]
                principalX = intrinsics.principalPoint[0]
                principalY = intrinsics.principalPoint[1]

                // ARCore's own CPU image surface never delivers a frame on this hardware (see the
                // ownImageReader comment above); read from our own reader instead.
                val image = if (captureImage) pendingOwnImage.getAndSet(null) else null
                val depth = if (wantDepth && depthSupported) copyDepth(frame) else null
                var points: FloatArray? = null
                var pointIds: IntArray? = null
                var numPoints = 0
                try {
                    frame.acquirePointCloud().use { cloud ->
                        val buf = cloud.points
                        points = FloatArray(buf.remaining())
                        buf.get(points!!)
                        val idsBuf = cloud.ids
                        pointIds = IntArray(idsBuf.remaining())
                        idsBuf.get(pointIds!!)
                        numPoints = points!!.size / 4
                    }
                } catch (e: NotYetAvailableException) {
                    // Nothing this frame; the next one carries the same geometry.
                }
                Sample(camera.pose, frame.timestamp, image, depth, points, pointIds, numPoints)
            } else {
                if (captureImage) {
                    val image = pendingOwnImage.getAndSet(null)
                    if (image != null) {
                        onUntrackedImage(image)
                        image.close()
                    }
                }
                null
            }

            onFrame(sample)

            if (camera.trackingState == TrackingState.TRACKING) {
                camera.getProjectionMatrix(projMatrix, 0, NEAR_M, FAR_M)
                camera.getViewMatrix(viewMatrix, 0)
                Matrix.multiplyMM(vpMatrix, 0, projMatrix, 0, viewMatrix, 0)
                drawPointSprites()
                // Guidance targets are represented on the 2D Mini-Map to keep the AR camera view clear and uncluttered
                targetCentred = true
                occlusionCentred = true
            }
        } catch (e: Exception) {
            onError(e)
        }
    }

    /**
     * ARCore's depth image is aligned with the camera image and shares its field of view, so the
     * camera intrinsics rescale to it directly. Copied rather than held: the Image must be closed
     * before the frame is released, and the consumer is another thread.
     */
    private fun copyDepth(frame: com.google.ar.core.Frame): DepthFrame? = try {
        frame.acquireDepthImage16Bits().use { img ->
            val w = img.width
            val h = img.height
            val plane = img.planes[0]
            val shorts = ShortArray(w * h)
            val sb = plane.buffer.order(ByteOrder.nativeOrder()).asShortBuffer()
            val strideShorts = plane.rowStride / 2
            for (row in 0 until h) {
                sb.position(row * strideShorts)
                sb.get(shorts, row * w, w)
            }
            val scaleX = w.toFloat() / imageWidth
            val scaleY = h.toFloat() / imageHeight
            DepthFrame(shorts, w, h, focalX * scaleX, focalY * scaleY, principalX * scaleX, principalY * scaleY)
        }
    } catch (e: NotYetAvailableException) {
        null
    }

    private fun drawCameraBackground() {
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glUseProgram(bgProgram)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(bgProgram, "sTexture"), 0)

        val positionHandle = GLES20.glGetAttribLocation(bgProgram, "a_Position")
        val texCoordHandle = GLES20.glGetAttribLocation(bgProgram, "a_TexCoord")
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, quadVertices)
        GLES20.glEnableVertexAttribArray(texCoordHandle)
        GLES20.glVertexAttribPointer(texCoordHandle, 2, GLES20.GL_FLOAT, false, 0, quadTexCoordsTransformed)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(positionHandle)
        GLES20.glDisableVertexAttribArray(texCoordHandle)
    }

    /** Coverage mesh overlay: exposed voxel faces, amber (needs another view) to green
     * (parallax-verified) -- README §5/§6's primary visual, replacing the raw point cloud
     * (Phase_1.md §10 item 5). The mesh itself is built by `VoxelGrid.meshTriangles()` on the
     * voxel worker's own cadence (see CoverageWorker.publish); this only re-uploads and draws
     * whatever it last published, so mesh generation never runs on the 60 Hz render loop. */
    /**
     * Renders multi-view parallax tracked sparse feature points as soft circular point sprites:
     * - Amber / Orange Dot: Initial sighting (< 20 deg angular baseline or single view).
     * - Emerald Green Dot: Multi-view parallax verified (>= 20 deg angular baseline + >= 0.35m translation).
     * Zero floaters in mid-air and zero plane estimation artifacts.
     */
    private fun drawPointSprites() {
        val count = pointVertexCount
        val verts = pointVertices ?: return
        val buffer = pointBuffer ?: return
        if (count <= 0) return

        val numDraw = minOf(count, MAX_POINT_SPRITES, verts.size / 4)
        buffer.position(0)
        buffer.put(verts, 0, numDraw * 4)
        buffer.position(0)

        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(pointProgram)
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(pointProgram, "u_MVP"), 1, false, vpMatrix, 0)

        val posHandle = GLES20.glGetAttribLocation(pointProgram, "a_Position")
        val verifiedHandle = GLES20.glGetAttribLocation(pointProgram, "a_Verified")
        buffer.position(0)
        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 3, GLES20.GL_FLOAT, false, 16, buffer)
        buffer.position(3)
        GLES20.glEnableVertexAttribArray(verifiedHandle)
        GLES20.glVertexAttribPointer(verifiedHandle, 1, GLES20.GL_FLOAT, false, 16, buffer)

        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, numDraw)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(verifiedHandle)
        GLES20.glDepthMask(true)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    private fun drawVoxels() {
        val snapshot = voxelPoints ?: return
        val buffer = meshBuffer ?: return
        if (snapshot !== uploadedMesh) {
            meshVertexCount = minOf(snapshot.size / 4, MAX_OVERLAY_TRIANGLES * 3)
            buffer.position(0)
            buffer.put(snapshot, 0, meshVertexCount * 4)
            uploadedMesh = snapshot
        }
        if (meshVertexCount == 0) return

        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(meshProgram)
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(meshProgram, "u_MVP"), 1, false, vpMatrix, 0)

        val posHandle = GLES20.glGetAttribLocation(meshProgram, "a_Position")
        val planeTypeHandle = GLES20.glGetAttribLocation(meshProgram, "a_PlaneType")
        buffer.position(0)
        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 3, GLES20.GL_FLOAT, false, 16, buffer)
        buffer.position(3)
        GLES20.glEnableVertexAttribArray(planeTypeHandle)
        GLES20.glVertexAttribPointer(planeTypeHandle, 1, GLES20.GL_FLOAT, false, 16, buffer)

        GLES20.glLineWidth(2.5f)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, meshVertexCount)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(planeTypeHandle)
        GLES20.glDepthMask(true)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    private val planeMatrix = FloatArray(16)
    private val planeLocalPoint = FloatArray(4)
    private val planeWorldPoint = FloatArray(4)
    private val planeLineBuffer = directFloatBuffer(FloatArray(MAX_PLANE_VERTS * 4))

    /**
     * Renders ARCore's native verified structural planes:
     * - Horizontal floor planes -> White wireframe triangle grid
     * - Vertical wall planes -> Blue wireframe triangle grid
     * Zero floating phantom surfaces in empty air / open hallways.
     */
    private fun drawPlanes(session: Session) {
        val planes = session.getAllTrackables(Plane::class.java)
        if (planes.isEmpty()) return

        var vertexCount = 0
        planeLineBuffer.position(0)

        for (plane in planes) {
            if (plane.trackingState != TrackingState.TRACKING) continue
            if (plane.subsumedBy != null) continue

            val type = plane.type
            val isHorizontal = type == Plane.Type.HORIZONTAL_UPWARD_FACING || type == Plane.Type.HORIZONTAL_DOWNWARD_FACING
            val isVertical = type == Plane.Type.VERTICAL
            if (!isHorizontal && !isVertical) continue

            val planeTypeVal = if (isHorizontal) 0.0f else 1.0f // 0 = Horizontal (White), 1 = Vertical (Blue)
            plane.centerPose.toMatrix(planeMatrix, 0)

            val polygon = plane.polygon
            val numPoints = polygon.limit() / 2
            if (numPoints < 3) continue

            val worldVerts = FloatArray(numPoints * 3)
            for (i in 0 until numPoints) {
                planeLocalPoint[0] = polygon.get(i * 2)
                planeLocalPoint[1] = 0.0f
                planeLocalPoint[2] = polygon.get(i * 2 + 1)
                planeLocalPoint[3] = 1.0f
                Matrix.multiplyMV(planeWorldPoint, 0, planeMatrix, 0, planeLocalPoint, 0)
                worldVerts[i * 3] = planeWorldPoint[0]
                worldVerts[i * 3 + 1] = planeWorldPoint[1]
                worldVerts[i * 3 + 2] = planeWorldPoint[2]
            }

            // 1. Outer boundary wireframe line segments
            for (i in 0 until numPoints) {
                val next = (i + 1) % numPoints
                if (vertexCount + 2 > MAX_PLANE_VERTS) break
                planeLineBuffer.put(worldVerts[i * 3])
                planeLineBuffer.put(worldVerts[i * 3 + 1])
                planeLineBuffer.put(worldVerts[i * 3 + 2])
                planeLineBuffer.put(planeTypeVal)

                planeLineBuffer.put(worldVerts[next * 3])
                planeLineBuffer.put(worldVerts[next * 3 + 1])
                planeLineBuffer.put(worldVerts[next * 3 + 2])
                planeLineBuffer.put(planeTypeVal)
                vertexCount += 2
            }

            // 2. Interior triangle grid fan lines
            for (i in 2 until numPoints - 1) {
                if (vertexCount + 2 > MAX_PLANE_VERTS) break
                planeLineBuffer.put(worldVerts[0])
                planeLineBuffer.put(worldVerts[1])
                planeLineBuffer.put(worldVerts[2])
                planeLineBuffer.put(planeTypeVal)

                planeLineBuffer.put(worldVerts[i * 3])
                planeLineBuffer.put(worldVerts[i * 3 + 1])
                planeLineBuffer.put(worldVerts[i * 3 + 2])
                planeLineBuffer.put(planeTypeVal)
                vertexCount += 2
            }
        }

        if (vertexCount == 0) return

        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(meshProgram)
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(meshProgram, "u_MVP"), 1, false, vpMatrix, 0)

        val posHandle = GLES20.glGetAttribLocation(meshProgram, "a_Position")
        val planeTypeHandle = GLES20.glGetAttribLocation(meshProgram, "a_PlaneType")
        planeLineBuffer.position(0)
        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 3, GLES20.GL_FLOAT, false, 16, planeLineBuffer)
        planeLineBuffer.position(3)
        GLES20.glEnableVertexAttribArray(planeTypeHandle)
        GLES20.glVertexAttribPointer(planeTypeHandle, 1, GLES20.GL_FLOAT, false, 16, planeLineBuffer)

        GLES20.glLineWidth(3.0f)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, vertexCount)

        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisableVertexAttribArray(planeTypeHandle)
        GLES20.glDepthMask(true)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    /**
     * Wireframe diamond at a guidance target, plus the screen bearing the 2D arrow needs. Bearing
     * is computed from the same MVP the marker is drawn with, so the arrow and the marker can't
     * disagree about where the target is. Shared by the frontier marker (amber) and the occlusion
     * marker (a distinct colour, per README/Phase_1.md §6 -- the two arrow mechanics must not be
     * visually conflated). Returns whether the target is centred (true, with bearing untouched,
     * when there's no target at all -- the arrow hides itself).
     */
    private fun drawGuidanceMarker(target: FloatArray?, rgba: FloatArray, onBearing: (Float) -> Unit): Boolean {
        target ?: return true
        worldPoint[0] = target[0]; worldPoint[1] = target[1]; worldPoint[2] = target[2]; worldPoint[3] = 1f
        Matrix.multiplyMV(clip, 0, vpMatrix, 0, worldPoint, 0)
        val behind = clip[3] <= 0f
        val ndcX = if (behind) -clip[0] / -clip[3] else clip[0] / clip[3]
        val ndcY = if (behind) -clip[1] / -clip[3] else clip[1] / clip[3]
        // Screen-up is +Y in NDC; the arrow view rotates clockwise from up.
        onBearing(Math.toDegrees(kotlin.math.atan2(ndcX.toDouble(), ndcY.toDouble())).toFloat())
        val centred = !behind && kotlin.math.abs(ndcX) < CENTRED_NDC && kotlin.math.abs(ndcY) < CENTRED_NDC

        Matrix.setIdentityM(modelMatrix, 0)
        Matrix.translateM(modelMatrix, 0, target[0], target[1], target[2])
        Matrix.scaleM(modelMatrix, 0, MARKER_SIZE_M, MARKER_SIZE_M, MARKER_SIZE_M)
        Matrix.multiplyMM(mvpMatrix, 0, vpMatrix, 0, modelMatrix, 0)

        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glUseProgram(soloProgram)
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(soloProgram, "u_MVP"), 1, false, mvpMatrix, 0)
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(soloProgram, "u_Color"), 1, rgba, 0)
        val posHandle = GLES20.glGetAttribLocation(soloProgram, "a_Position")
        GLES20.glEnableVertexAttribArray(posHandle)
        GLES20.glVertexAttribPointer(posHandle, 3, GLES20.GL_FLOAT, false, 0, markerBuffer)
        GLES20.glLineWidth(4f)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, 24)
        GLES20.glDisableVertexAttribArray(posHandle)
        GLES20.glDisable(GLES20.GL_BLEND)
        return centred
    }

    /** Unit octahedron as its 12 edges, for GL_LINES: a wireframe diamond.
     *
     * It was 8 solid triangles, and in the field that read as an unexplained orange square
     * blotting out the camera feed -- the marker sits close to the operator, has no depth test,
     * and a filled octahedron seen head-on is a square. An outline you can see the room through
     * reads as a target instead of an artefact. */
    private fun octahedronEdges(): FloatArray {
        val p = arrayOf(
            floatArrayOf(1f, 0f, 0f), floatArrayOf(-1f, 0f, 0f),
            floatArrayOf(0f, 1f, 0f), floatArrayOf(0f, -1f, 0f),
            floatArrayOf(0f, 0f, 1f), floatArrayOf(0f, 0f, -1f),
        )
        // Every pair of vertices except the three opposite pairs (0/1, 2/3, 4/5) = 12 edges.
        val out = FloatArray(12 * 6)
        var k = 0
        for (i in 0 until 6) for (j in i + 1 until 6) {
            if (i / 2 == j / 2) continue
            for (e in intArrayOf(i, j)) {
                out[k++] = p[e][0]; out[k++] = p[e][1]; out[k++] = p[e][2]
            }
        }
        return out
    }

    @Suppress("DEPRECATION")
    private val displayRotation: Int
        get() = (context as? Activity)?.windowManager?.defaultDisplay?.rotation ?: Surface.ROTATION_0

    private fun createExternalTexture(): Int {
        val texIds = IntArray(1)
        GLES20.glGenTextures(1, texIds, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texIds[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return texIds[0]
    }

    private fun createProgram(vertexSrc: String, fragmentSrc: String): Int {
        val prog = GLES20.glCreateProgram()
        GLES20.glAttachShader(prog, compileShader(GLES20.GL_VERTEX_SHADER, vertexSrc))
        GLES20.glAttachShader(prog, compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentSrc))
        GLES20.glLinkProgram(prog)
        val status = IntArray(1)
        GLES20.glGetProgramiv(prog, GLES20.GL_LINK_STATUS, status, 0)
        if (status[0] == 0) onError(IllegalStateException("shader link failed: ${GLES20.glGetProgramInfoLog(prog)}"))
        return prog
    }

    private fun compileShader(type: Int, src: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, src)
        GLES20.glCompileShader(shader)
        val status = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) onError(IllegalStateException("shader compile failed: ${GLES20.glGetShaderInfoLog(shader)}"))
        return shader
    }

    private fun directFloatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(values)
            position(0)
        }

    companion object {
        /** Mesh overlay budget, in triangles. A 100 sqm home maps to roughly 15k occupied voxels
         * at 10 cm, each contributing up to 12 exposed-face triangles, so this is comfortable
         * headroom before VoxelGrid.meshTriangles starts truncating. */
        const val MAX_OVERLAY_TRIANGLES = 120_000
        private const val MAX_PLANE_VERTS = 8192
        const val MAX_POINT_SPRITES = 8192

        private const val NEAR_M = 0.1f
        private const val FAR_M = 30f
        private const val MARKER_SIZE_M = 0.30f
        private val MARKER_RGBA = floatArrayOf(0.96f, 0.62f, 0.04f, 0.9f)
        /** Cyan, not amber: the occlusion marker must read as a different instruction on sight
         * (README/Phase_1.md §6). */
        private val OCCLUSION_MARKER_RGBA = floatArrayOf(0.06f, 0.72f, 0.85f, 0.9f)
        /** Half-width of the "you're looking at it" box, in NDC. */
        private const val CENTRED_NDC = 0.35f

        private val QUAD_COORDS = floatArrayOf(-1f, -1f, +1f, -1f, -1f, +1f, +1f, +1f)
        private val QUAD_TEXCOORDS = floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f)

        private const val BG_VERTEX_SHADER = """
            attribute vec4 a_Position;
            attribute vec2 a_TexCoord;
            varying vec2 v_TexCoord;
            void main() {
                gl_Position = a_Position;
                v_TexCoord = a_TexCoord;
            }
        """

        private const val BG_FRAGMENT_SHADER = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 v_TexCoord;
            uniform samplerExternalOES sTexture;
            void main() {
                gl_FragColor = texture2D(sTexture, v_TexCoord);
            }
        """

        private const val POINT_VERTEX_SHADER = """
            uniform mat4 u_MVP;
            attribute vec4 a_Position;
            attribute float a_Verified; // 0.0 = Amber candidate, 1.0 = Emerald Diamond Gem, >1.0 = Sparkling Star burst
            varying float v_Verified;
            void main() {
                gl_Position = u_MVP * a_Position;
                float dist = gl_Position.w;
                // Base point sprite size: Amber (22px), Emerald Diamond (28px), Sparkle Burst (up to 44px)
                float sparkleScale = max(1.0, a_Verified);
                float baseSize = a_Verified < 0.5 ? 20.0 : (26.0 * (1.0 + (sparkleScale - 1.0) * 0.7));
                gl_PointSize = clamp(baseSize / max(dist, 0.35), 8.0, 52.0);
                v_Verified = a_Verified;
            }
        """

        private const val POINT_FRAGMENT_SHADER = """
            precision mediump float;
            varying float v_Verified; // 0.0 = Amber, 1.0 = Emerald Gem, >1.0 = 4-pointed sparkling star
            void main() {
                vec2 p = abs(gl_PointCoord - vec2(0.5)); // Coordinate folded into first quadrant [0, 0.5]
                
                if (v_Verified < 0.5) {
                    // Amber Candidate: Smooth Circular Dot
                    float d = length(p);
                    if (d > 0.5) discard;
                    float alpha = smoothstep(0.5, 0.15, d) * 0.85;
                    gl_FragColor = vec4(0.98, 0.64, 0.08, alpha); // #F59E0B Amber
                } else {
                    // Emerald Gem Landmark: 4-pointed Star / Rhomboid Diamond
                    // Diamond SDF: |x| + |y| <= 0.5
                    float manhattan = p.x + p.y;
                    
                    // 4-pointed star SDF: concavity curve (x^0.6 + y^0.6)
                    float star = pow(p.x, 0.65) + pow(p.y, 0.65);
                    
                    float sparkle = clamp(v_Verified - 1.0, 0.0, 1.0); // 1.0 at birth -> 0.0 after 1.2s
                    
                    // Blend between diamond gem (manhattan) and sharp 4-pointed star during sparkle
                    float distMetric = mix(manhattan * 1.05, star * 0.85, sparkle);
                    
                    if (distMetric > 0.52) discard;
                    
                    float edgeAlpha = smoothstep(0.52, 0.20, distMetric);
                    // Diamond core glow
                    float coreGlow = smoothstep(0.30, 0.0, length(p));
                    
                    // Color transitions: Radiant Emerald Green (#10B981) + Bright White/Cyan Star Center during sparkle
                    vec3 emeraldBase = vec3(0.06, 0.88, 0.48);
                    vec3 starSparkle = vec3(0.75, 1.0, 0.88); // Brilliant sparkling crystalline core
                    
                    vec3 finalRgb = mix(emeraldBase, starSparkle, coreGlow * (0.4 + 0.6 * sparkle));
                    float finalAlpha = clamp(edgeAlpha * 0.95 + coreGlow * sparkle * 0.4, 0.0, 1.0);
                    
                    gl_FragColor = vec4(finalRgb, finalAlpha);
                }
            }
        """

        private const val MESH_VERTEX_SHADER = """
            uniform mat4 u_MVP;
            attribute vec4 a_Position;
            attribute float a_PlaneType;
            varying float v_PlaneType;
            void main() {
                gl_Position = u_MVP * a_Position;
                v_PlaneType = a_PlaneType;
            }
        """

        private const val MESH_FRAGMENT_SHADER = """
            precision mediump float;
            varying float v_PlaneType;
            void main() {
                // v_PlaneType < 0.5 is Horizontal (White lines), >= 0.5 is Vertical (Cyan/Blue lines)
                vec4 white = vec4(1.0, 1.0, 1.0, 0.90);
                vec4 blue = vec4(0.22, 0.74, 0.97, 0.90);
                gl_FragColor = mix(white, blue, step(0.5, v_PlaneType));
            }
        """

        private const val SOLO_VERTEX_SHADER = """
            uniform mat4 u_MVP;
            attribute vec4 a_Position;
            void main() { gl_Position = u_MVP * a_Position; }
        """

        private const val SOLO_FRAGMENT_SHADER = """
            precision mediump float;
            uniform vec4 u_Color;
            void main() { gl_FragColor = u_Color; }
        """
    }
}
