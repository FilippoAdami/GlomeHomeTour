package com.glomehometour.capture

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.SystemClock
import com.google.ar.core.Config
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.ByteArrayOutputStream

/**
 * Headless ARCore session: no GL rendering, just a dummy pbuffer/texture to satisfy ARCore's
 * camera pipeline so we can pull CPU-side camera images + tracked pose + sparse point cloud
 * (needed for [DepthCalibration]). Replaces CameraX, which can't share the camera with ARCore
 * without the more fragile SharedCamera API — this proof-of-concept doesn't need CameraX's
 * separate locked-exposure recording path yet, so full ARCore ownership is the simpler choice.
 */
class ArCoreCameraSource(
    private val context: Context,
    private val targetIntervalMs: Long,
    private val onFrame: (bitmap: Bitmap, rotationDegrees: Int, pose: PoseSample?) -> Unit,
    private val onError: (Exception) -> Unit,
) {
    data class PoseSample(
        val pose: Pose,
        val points: FloatArray,
        val numPoints: Int,
        val fx: Float,
        val fy: Float,
        val cx: Float,
        val cy: Float,
        val sensorWidth: Int,
        val sensorHeight: Int,
    )

    // Debug-only, read by MainActivity for the HUD — not used in the calibration math itself.
    @Volatile var lastTrackingState: String = "?"
        private set

    // ponytail: portrait-locked app, back camera sensor is always landscape-native -> fixed 90.
    private val rotationDegrees = 90

    @Volatile private var running = false
    private var thread: Thread? = null

    fun start() {
        running = true
        thread = Thread({ loop() }, "ArCoreCamera").apply { start() }
    }

    fun stop() {
        running = false
        thread?.join(500)
        thread = null
    }

    private fun loop() {
        var display = EGL14.EGL_NO_DISPLAY
        var eglContext = EGL14.EGL_NO_CONTEXT
        var eglSurface = EGL14.EGL_NO_SURFACE
        var session: Session? = null
        try {
            display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            EGL14.eglInitialize(display, IntArray(2), 0, IntArray(2), 1)
            val configAttribs = intArrayOf(
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8, EGL14.EGL_NONE,
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val numConfigs = IntArray(1)
            EGL14.eglChooseConfig(display, configAttribs, 0, configs, 0, 1, numConfigs, 0)
            val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
            eglContext = EGL14.eglCreateContext(display, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
            val surfaceAttribs = intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE)
            eglSurface = EGL14.eglCreatePbufferSurface(display, configs[0], surfaceAttribs, 0)
            EGL14.eglMakeCurrent(display, eglSurface, eglSurface, eglContext)

            val texIds = IntArray(1)
            GLES20.glGenTextures(1, texIds, 0)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texIds[0])
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

            val s = Session(context)
            val config = Config(s).apply {
                updateMode = Config.UpdateMode.BLOCKING
                focusMode = Config.FocusMode.AUTO
                planeFindingMode = Config.PlaneFindingMode.DISABLED
                lightEstimationMode = Config.LightEstimationMode.DISABLED
            }
            s.configure(config)
            s.setCameraTextureName(texIds[0])
            s.resume()
            session = s

            var lastGateAtMs = 0L
            while (running) {
                val frame = s.update()
                val now = SystemClock.elapsedRealtime()
                if (now - lastGateAtMs < targetIntervalMs) continue
                lastGateAtMs = now

                val image = try {
                    frame.acquireCameraImage()
                } catch (e: NotYetAvailableException) {
                    continue
                }
                val bitmap = try {
                    imageToBitmap(image)
                } finally {
                    image.close()
                }

                val camera = frame.camera
                lastTrackingState = if (camera.trackingState == TrackingState.TRACKING) {
                    "TRACKING"
                } else {
                    "${camera.trackingState}/${camera.trackingFailureReason}"
                }
                val poseSample = if (camera.trackingState == TrackingState.TRACKING) {
                    try {
                        frame.acquirePointCloud().use { cloud ->
                            val buf = cloud.points
                            val floatCount = buf.remaining()
                            lastTrackingState = "TRACKING/${floatCount / 4}pts"
                            val points = FloatArray(floatCount)
                            buf.get(points)
                            val intrinsics = camera.imageIntrinsics
                            PoseSample(
                                pose = camera.pose,
                                points = points,
                                numPoints = floatCount / 4,
                                fx = intrinsics.focalLength[0],
                                fy = intrinsics.focalLength[1],
                                cx = intrinsics.principalPoint[0],
                                cy = intrinsics.principalPoint[1],
                                sensorWidth = intrinsics.imageDimensions[0],
                                sensorHeight = intrinsics.imageDimensions[1],
                            )
                        }
                    } catch (e: NotYetAvailableException) {
                        null
                    }
                } else {
                    null
                }

                onFrame(bitmap, rotationDegrees, poseSample)
            }
        } catch (e: Exception) {
            onError(e)
        } finally {
            session?.pause()
            session?.close()
            if (display != EGL14.EGL_NO_DISPLAY) {
                EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                if (eglSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(display, eglSurface)
                if (eglContext != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(display, eglContext)
                EGL14.eglTerminate(display)
            }
        }
    }

    /**
     * Direct, stride-aware YUV_420_888 -> RGB conversion. The old CameraX path routed through
     * YuvImage.compressToJpeg as a shortcut, but that's a *lossy* JPEG encode+decode every
     * frame — its 8x8 DCT block artifacts were exactly the kind of high-frequency noise Sobel
     * picks up, which is why Canny degraded into speckle on lower-contrast (far) scenes.
     */
    private fun imageToBitmap(image: Image): Bitmap {
        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val yRowStride = yPlane.rowStride
        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride

        val pixels = IntArray(width * height)
        for (row in 0 until height) {
            val yRowStart = row * yRowStride
            val uvRow = row / 2
            val uRowStart = uvRow * uRowStride
            val vRowStart = uvRow * vRowStride
            val rowOut = row * width
            for (col in 0 until width) {
                val y = yBuffer.get(yRowStart + col).toInt() and 0xFF
                val uvCol = col / 2
                val u = (uBuffer.get(uRowStart + uvCol * uPixelStride).toInt() and 0xFF) - 128
                val v = (vBuffer.get(vRowStart + uvCol * vPixelStride).toInt() and 0xFF) - 128

                val r = (y + 1.370705f * v).toInt().coerceIn(0, 255)
                val g = (y - 0.337633f * u - 0.698001f * v).toInt().coerceIn(0, 255)
                val b = (y + 1.732446f * u).toInt().coerceIn(0, 255)

                pixels[rowOut + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
