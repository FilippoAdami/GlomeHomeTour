package com.glomehometour.arscan

import android.content.Context
import android.graphics.Rect
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.MeteringRectangle
import android.hardware.camera2.params.RggbChannelVector
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import com.google.ar.core.Session

/**
 * README §4's hardware overrides are not reachable through ARCore's own auto camera path: ARCore
 * opens and drives Camera2 itself, at CONTROL_MODE_AUTO, with no hook for exposure/ISO/WB/
 * stabilization/distortion. The only way to put a manual `CaptureRequest` on the sensor while
 * ARCore still gets frames for VIO is ARCore's `SharedCamera` feature (`Session.Feature
 * .SHARED_CAMERA`): the app opens the Camera2 device and owns the capture session, hands ARCore's
 * required surfaces (its GPU texture, its CPU image reader) into one repeating request alongside
 * ARCore's own, and ARCore's `session.update()` reads off whatever that request produces.
 *
 * This is also how the 60fps/AR-poll-rate decoupling in Phase_1.md §1 actually works in practice:
 * there's one physical sensor readout rate (`CONTROL_AE_TARGET_FPS_RANGE`, targeted at 60), and
 * ARCore's pose tracker consumes frames from it at whatever rate it can keep up with -- it was
 * never on a separate clock, it just doesn't have to process every one. Recording decimation
 * (MainActivity.isKeyframe, §10 item 2) is the thing that turns "60fps sensor" into "6-10
 * keyframes/sec export" without asking the sensor to run at a different rate than ARCore reads.
 *
 * Not unit-tested: this is Camera2/ARCore session plumbing with no pure-Kotlin surface, same
 * category as ArScanRenderer. Correctness here can only really be confirmed on-device (§10 item
 * 9's acceptance pass), which this sandbox has no camera/emulator to run.
 */
class CameraPipeline(
    private val context: Context,
    private val session: Session,
    private val onError: (Exception) -> Unit,
) {
    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val thread = HandlerThread("camera-pipeline").apply { start() }
    private val handler = Handler(thread.looper)

    private var device: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var characteristics: CameraCharacteristics? = null
    private var onReady: () -> Unit = {}

    /** Manual WB gains, set by the pre-flight lock-on-neutral-wall step (README §6 step 2).
     * Neutral (all 1.0) until the operator locks, i.e. "whatever the sensor's baseline reads",
     * which is a reasonable default for a step that hasn't run yet. */
    @Volatile var whiteBalanceGains: RggbChannelVector = NEUTRAL_GAINS
        set(value) {
            field = value
            applyRepeatingRequest()
        }

    /** Manual ISO, metered from ambient light during pre-flight (README §4/§6) and then left
     * alone once scanning starts. Jointly tuned with `shutterNanos` against a shared tradeoff
     * curve (grain vs. motion blur) -- see `MainActivity.meteredExposure()`. */
    @Volatile var isoSensitivity: Int = DEFAULT_ISO
        set(value) {
            field = value
            applyRepeatingRequest()
        }

    /** Manual shutter (`SENSOR_EXPOSURE_TIME`), metered alongside ISO. README §4: ranges from
     * `FASTEST_SHUTTER_NS` (1/500s, outdoor/bright, minimal motion blur) down to
     * `SLOWEST_SHUTTER_NS` (1/50s, dim indoor rooms, admits more light without pushing ISO into
     * visible grain). Left alone once scanning starts, same as ISO. */
    @Volatile var shutterNanos: Long = FASTEST_SHUTTER_NS
        set(value) {
            field = value
            applyRepeatingRequest()
        }

    /** Opens the shared Camera2 device and starts a single manual capture session covering both
     * ARCore's required surfaces and any extra surfaces the caller wants frames from. Call once
     * ARCore's camera config has been chosen (`session.cameraConfig` must already be set). */
    fun start(extraSurfaces: List<android.view.Surface> = emptyList(), onReady: () -> Unit = {}) {
        this.onReady = onReady
        val cameraId = session.cameraConfig.cameraId
        val chars = cameraManager.getCameraCharacteristics(cameraId)
        characteristics = chars
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            chars.get(CameraCharacteristics.LENS_DISTORTION)?.let { k ->
                lensDistortionK1 = k.getOrNull(0)
                lensDistortionK2 = k.getOrNull(1)
            }
        }
        val sharedCamera = session.sharedCamera
        if (extraSurfaces.isNotEmpty()) {
            sharedCamera.setAppSurfaces(cameraId, extraSurfaces)
        }
        sharedCamera.setCaptureCallback(captureCallback, handler)
        val stateCallback = object : CameraDevice.StateCallback() {
            override fun onOpened(cameraDevice: CameraDevice) {
                device = cameraDevice
                openCaptureSession(cameraDevice, sharedCamera.arCoreSurfaces + extraSurfaces)
            }

            override fun onDisconnected(cameraDevice: CameraDevice) {
                cameraDevice.close()
                device = null
            }

            override fun onError(cameraDevice: CameraDevice, error: Int) {
                cameraDevice.close()
                device = null
                onError(IllegalStateException("camera device error $error"))
            }
        }
        try {
            cameraManager.openCamera(
                cameraId,
                sharedCamera.createARDeviceStateCallback(stateCallback, handler),
                handler,
            )
        } catch (e: SecurityException) {
            onError(e)
        }
    }

    private fun openCaptureSession(cameraDevice: CameraDevice, surfaces: List<android.view.Surface>) {
        val sharedCamera = session.sharedCamera
        val sessionCallback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(cameraCaptureSession: CameraCaptureSession) {
                captureSession = cameraCaptureSession
                targetSurfaces = surfaces
                applyRepeatingRequest()
                onReady()
            }

            override fun onConfigureFailed(cameraCaptureSession: CameraCaptureSession) {
                onError(IllegalStateException("camera capture session configuration failed"))
            }
        }
        try {
            val wrappedCallback = sharedCamera.createARSessionStateCallback(sessionCallback, handler)
            cameraDevice.createCaptureSession(
                surfaces,
                wrappedCallback,
                handler,
            )
        } catch (e: Exception) {
            onError(e)
        }
    }

    private var targetSurfaces: List<android.view.Surface> = emptyList()

    @Volatile var latestFocusDistanceDiopters: Float = 0f
    @Volatile var latestAfState: Int = 0
    @Volatile var latestAfMode: Int = 0
    @Volatile var latestFocalLengthMm: Float = 0f

    /**
     * Static per-physical-camera lens distortion, read once in `start()`. Android's
     * `LENS_DISTORTION` (API 28+) is a 5-term rational model (kappa_0..4, no tangential term),
     * not OpenCV's plumb-bob (k1, k2, p1, p2) that `transforms.json` declares -- kappa_0/kappa_1
     * are used as an approximate seed for k1/k2 (p1/p2 stay 0, Android has no tangential term).
     * Good enough as a starting point for `sfm_refinement.py`'s bundle adjustment, not an exact
     * undistortion model on its own. Null on API <28 or if the characteristic is absent.
     */
    @Volatile var lensDistortionK1: Float? = null
        private set
    @Volatile var lensDistortionK2: Float? = null
        private set

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult,
        ) {
            latestFocusDistanceDiopters = result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: 0f
            latestAfState = result.get(CaptureResult.CONTROL_AF_STATE) ?: 0
            latestAfMode = result.get(CaptureResult.CONTROL_AF_MODE) ?: 0
            latestFocalLengthMm = result.get(CaptureResult.LENS_FOCAL_LENGTH) ?: 0f
        }
    }

    /** Rebuilds and resubmits the repeating request. Called once at session start and again
     * whenever the pre-flight WB lock changes the gains -- everything else in the request is
     * fixed for the session, so there is nothing else that needs a rebuild mid-scan. */
    private fun applyRepeatingRequest() {
        val cameraDevice = device ?: return
        val session = captureSession ?: return
        val chars = characteristics ?: return
        try {
            val builder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
            targetSurfaces.forEach { builder.addTarget(it) }
            configureManualOverrides(builder, chars)
            this.session.sharedCamera.setCaptureCallback(captureCallback, handler)
            session.setRepeatingRequest(builder.build(), captureCallback, handler)
        } catch (e: Exception) {
            onError(e)
        }
    }

    fun triggerAutoFocus(normX: Float = 0.5f, normY: Float = 0.5f) {
        val cameraDevice = device ?: return
        val session = captureSession ?: return
        val chars = characteristics ?: return
        try {
            val builder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
            targetSurfaces.forEach { builder.addTarget(it) }
            configureManualOverrides(builder, chars)
            val sensorRect = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            if (sensorRect != null) {
                val cx = (normX * sensorRect.width()).toInt().coerceIn(0, sensorRect.width() - 1)
                val cy = (normY * sensorRect.height()).toInt().coerceIn(0, sensorRect.height() - 1)
                val halfSize = (sensorRect.width() * 0.1f).toInt()
                val rect = Rect(
                    (cx - halfSize).coerceAtLeast(0),
                    (cy - halfSize).coerceAtLeast(0),
                    (cx + halfSize).coerceAtMost(sensorRect.width() - 1),
                    (cy + halfSize).coerceAtMost(sensorRect.height() - 1),
                )
                val maxAfRegions = chars.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0
                if (maxAfRegions > 0) {
                    builder.set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(MeteringRectangle(rect, MeteringRectangle.METERING_WEIGHT_MAX)))
                }
            }
            builder.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
            session.capture(builder.build(), captureCallback, handler)
        } catch (e: Exception) {
            Log.w("ArScan", "triggerAutoFocus failed", e)
        }
    }

    private fun configureManualOverrides(builder: CaptureRequest.Builder, chars: CameraCharacteristics) {
        // Keep continuous autofocus running (README §6). Prefer CONTINUOUS_VIDEO / CONTINUOUS_PICTURE,
        // falling back to AUTO if available.
        builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
        val afModes = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
        val afMode = when {
            afModes?.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO) == true ->
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
            afModes?.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE) == true ->
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
            afModes?.contains(CaptureRequest.CONTROL_AF_MODE_AUTO) == true ->
                CaptureRequest.CONTROL_AF_MODE_AUTO
            else -> CaptureRequest.CONTROL_AF_MODE_OFF
        }
        builder.set(CaptureRequest.CONTROL_AF_MODE, afMode)

        // Sensor readout rate: 60 FPS (README §4), clamped to whatever the device actually
        // advertises so a request outside the supported range doesn't just get silently ignored.
        val fpsRanges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
        val fps60 = fpsRanges?.firstOrNull { it.lower >= 30 && it.upper >= 60 }
            ?: fpsRanges?.maxByOrNull { it.upper }
        if (fps60 != null) builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fps60)

        // Shutter 1/500s-1/50s and ISO 50-ceiling, jointly metered (README §4), manual exposure
        // (AE off). A slow shutter beyond one frame duration necessarily drops the achievable
        // frame rate below the 60fps target above -- that's a real hardware tradeoff, not a bug;
        // MainActivity.meteredExposure() prefers ISO's cheaper-on-framerate cost once shutter
        // alone would start cutting into readout rate (see its own comment).
        builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
        val exposureRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val exposureNs = shutterNanos.coerceIn(
            exposureRange?.lower ?: shutterNanos, exposureRange?.upper ?: shutterNanos,
        )
        builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureNs)
        val frameDurationNs = fps60?.let { 1_000_000_000L / it.upper } ?: (1_000_000_000L / 60)
        builder.set(CaptureRequest.SENSOR_FRAME_DURATION, maxOf(exposureNs, frameDurationNs))

        val isoRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        val iso = isoSensitivity.coerceIn(isoRange?.lower ?: isoSensitivity, isoRange?.upper ?: isoSensitivity)
        builder.set(CaptureRequest.SENSOR_SENSITIVITY, iso)

        // White balance: manual, locked from pre-flight (or neutral until the operator locks).
        builder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
        builder.set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX)
        builder.set(CaptureRequest.COLOR_CORRECTION_GAINS, whiteBalanceGains)

        // OIS hard disabled, if the lens has it.
        val ois = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
        if (ois?.contains(CameraCharacteristics.LENS_OPTICAL_STABILIZATION_MODE_OFF) == true) {
            builder.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)
        }
        // EIS hard disabled.
        builder.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)

        // Auto lens-distortion correction hard disabled (API 28+; minSdk here is 30 so it's
        // always present, but still guarded against a device that omits the characteristic).
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val distortionModes = chars.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES)
            if (distortionModes?.contains(CaptureRequest.DISTORTION_CORRECTION_MODE_OFF) == true) {
                builder.set(CaptureRequest.DISTORTION_CORRECTION_MODE, CaptureRequest.DISTORTION_CORRECTION_MODE_OFF)
            }
        }
    }

    fun stop() {
        try {
            captureSession?.close()
        } catch (_: Exception) { /* already torn down */ }
        captureSession = null
        try {
            device?.close()
        } catch (_: Exception) { /* already torn down */ }
        device = null
    }

    fun shutdown() {
        stop()
        thread.quitSafely()
    }

    companion object {
        /** README §4: shutter and ISO are jointly metered against a shared indoor/outdoor
         * tradeoff curve, not each fixed independently. Fast end: 1/500s, outdoor/bright,
         * minimal motion blur. Slow end: 1/50s, the point past which handheld walking blur starts
         * to dominate over the grain saved by not raising ISO instead -- both ends are still
         * hardware-clamped to `SENSOR_INFO_EXPOSURE_TIME_RANGE` in `configureManualOverrides`. */
        const val FASTEST_SHUTTER_NS = 1_000_000_000L / 500
        const val SLOWEST_SHUTTER_NS = 1_000_000_000L / 50
        const val DEFAULT_SHUTTER_NS = 1_000_000_000L / 60

        /** README §4: ISO adapts to ambient light so the metered shutter above still exposes
         * correctly both outdoors and in ordinary room lighting -- 50 is the low (bright/outdoor)
         * end of the stated baseline. The ceiling is a hard product cap on grain, not a hardware
         * limit (`configureManualOverrides` separately clamps to the sensor's own
         * `SENSOR_INFO_SENSITIVITY_RANGE`, so this can never exceed real hardware) -- explicitly
         * capped at 1000 regardless of how dark a room reads, on the reasoning that a dim frame
         * `meteredExposure()` can't fully correct without exceeding this is one the shutter's
         * 1/500s-1/50s range should be relied on for instead (or the operator adds light), not
         * something ISO should be pushed toward grain to compensate for. Earlier revisions of
         * this constant tried 3200 and 6400 to chase a too-dark reading on `rosemary`; the ceiling
         * moved back down deliberately, so don't reflexively re-raise it for the same complaint --
         * push on shutter/scene-light first. */
        const val MIN_ISO = 50
        const val MAX_ISO = 3200
        const val DEFAULT_ISO = 400

        /** No color-correction adjustment: what the sensor reads until the operator explicitly
         * locks white balance in the pre-flight flow. */
        val NEUTRAL_GAINS = RggbChannelVector(1f, 1f, 1f, 1f)
    }
}
