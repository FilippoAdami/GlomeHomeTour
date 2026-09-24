package com.glomehometour.arscan

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.res.ColorStateList
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Looper
import android.util.Log
import android.view.HapticFeedbackConstants
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Pose
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.roundToInt
import kotlin.math.roundToLong
import kotlin.math.sqrt

/**
 * Guided spatial video acquisition (see ../SPEC.md).
 *
 * One continuous walk per session. The render thread owns everything ARCore-facing and the whole
 * gating decision for each frame; the voxel grid and the JPEG writer each have their own thread
 * and are fed plain arrays, so neither can stall tracking. Buttons only park a PendingAction for
 * the render thread to pick up on its next tracked frame -- a click handler has no pose, and
 * starting a session needs one.
 */
class CaptureActivity : AppCompatActivity() {

    private enum class State { IDLE, PREFLIGHT, SCANNING, DONE }

    /** Identifies a warning independent of its (sometimes dynamic) text, so the queue below can
     * dedupe by "what kind of issue" rather than by exact string. */
    private enum class WarningKind {
        REALIGNED, TRACKING_LOST, FINDING_POSITION, RELOCALIZE, CALIBRATING_DEPTH,
        TOO_BRIGHT, LIGHT_TRANSITION, TOO_FAST, TURN_SLOWLY,
    }
    private enum class Pending { START, PREFLIGHT_NEXT, FINISH, RESUME }

    private lateinit var root: FrameLayout
    private lateinit var topColumn: View
    private lateinit var bottomColumn: View
    private lateinit var trackingDot: View
    private lateinit var phaseText: TextView
    private lateinit var timerText: TextView
    private lateinit var infoButton: TextView
    private lateinit var memoryButton: TextView
    private lateinit var coverageValue: TextView
    private lateinit var coverageLabel: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var warnBanner: TextView
    private lateinit var miniMapView: MiniMapView
    private lateinit var reticle: View
    private lateinit var frontierGroup: View
    private lateinit var occlusionGroup: View
    private lateinit var arrowView: GuidanceArrowView
    private lateinit var occlusionArrowView: GuidanceArrowView
    private lateinit var diagText: TextView
    private lateinit var tiltCard: View
    private lateinit var tiltValue: TextView
    private lateinit var tiltGauge: TiltGaugeView
    private lateinit var wbCard: View
    private lateinit var wbSlider: SeekBar
    private lateinit var wbTintSlider: SeekBar
    private lateinit var wbResetButton: TextView
    private lateinit var instructionTitle: TextView
    private lateinit var instructionBody: TextView
    private lateinit var primaryButton: Button
    private lateinit var hintText: TextView
    /** The layout's own bottom padding, kept so the inset listener can add the gesture-bar height
     * to it instead of replacing it. */
    private var bottomPadBase = 0

    private var glSurfaceView: GLSurfaceView? = null
    private var renderer: ArScanRenderer? = null
    private var worker: CoverageWorker? = null
    private var writer: DatasetWriter? = null
    /** Only built when the device has no ARCore Depth API -- see MonoDepth.kt. */
    private var mono: MonoDepthWorker? = null
    private var gate = PhotometricGate()
    val parallaxTracker = FeatureParallaxTracker()
    private val pointSpriteExportBuf = FloatArray(ArScanRenderer.MAX_POINT_SPRITES * 4)

    @Volatile private var state = State.IDLE
    @Volatile private var pending: Pending? = null
    private var pendingAtNanos = 0L
    /** A scan parked by backgrounding: ARCore resumes its session on its own, acquisition may
     * only ever restart on a tap (mobile_sphere_capture learned this one the hard way). */
    @Volatile private var interrupted = false

    /** SPEC section 1.3's on-device diagnostics: still one tap away (the info button), no longer
     * permanently in front of an operator who is not the person debugging the capture stack. */
    private var showDiag = false
    /** Warnings buzz once on appearance, not on every 6 Hz repaint. */
    private var lastWarning: String? = null
    private var lastAutoFocusTriggerNanos = 0L

    // ---- warning queue (see pickWarning) ----
    private var currentWarning: Pair<WarningKind, String>? = null
    private var currentWarningShownAtNanos = 0L
    private val warningQueue = ArrayDeque<Pair<WarningKind, String>>()

    /**
     * Debounces the raw per-frame warning conditions into something a human can actually read:
     * some conditions (autofocus settling, a single overexposed frame) clear within a fraction
     * of a second, so showing them level-triggered just flickers. Each distinct kind, once
     * triggered, stays on screen at least MIN_WARNING_DISPLAY_NANOS; anything else triggered
     * meanwhile queues up (one slot per kind, no duplicates) instead of interrupting it.
     */
    private fun pickWarning(nowNanos: Long, active: List<Pair<WarningKind, String>>): String? {
        for (candidate in active) {
            if (currentWarning?.first == candidate.first) continue
            if (warningQueue.any { it.first == candidate.first }) continue
            warningQueue.addLast(candidate)
        }
        if (currentWarning != null && nowNanos - currentWarningShownAtNanos >= MIN_WARNING_DISPLAY_NANOS) {
            currentWarning = null
        }
        if (currentWarning == null && warningQueue.isNotEmpty()) {
            currentWarning = warningQueue.removeFirst()
            currentWarningShownAtNanos = nowNanos
        }
        return currentWarning?.second
    }

    private fun resetWarningQueue() {
        currentWarning = null
        warningQueue.clear()
    }
    /** Hard failures (no permission, no ARCore, dead AR session) used to be written into the debug
     * HUD line, where the next UI tick overwrote them 160 ms later -- or, before tracking ever
     * started, where nobody was looking. They now own the instruction card until resolved. */
    @Volatile private var fatalTitle: String? = null
    @Volatile private var fatalBody: String? = null
    /** Last exception out of the renderer's draw loop; surfaced as a self-clearing notice, see the
     * onError callback in [startTracking]. */
    @Volatile private var sessionErrorAtNanos = 0L

    private var sessionStartNanos = 0L
    private var floorY = 0f
    /** CLAUDE.md's entry-door loop-closure anchor: the pose Finish is gated against. */
    private val startPosition = FloatArray(3)
    private var frameCounter = 0L
    private var framesSeen = 0L

    // Gate accounting, all render thread.
    private var droppedDark = 0L
    private var droppedBlown = 0L
    private var droppedTransition = 0L
    private var droppedMotion = 0L
    private var trackingLosses = 0
    private var wasTracking = false

    // Kinematics (SPEC §2.4), all render thread.
    private val forward = FloatArray(3)
    private var prevForward: FloatArray? = null
    private var prevForwardNanos = 0L
    private var angularRateDegPerSec = 0f
    private var prevPosition: FloatArray? = null
    private var prevPositionNanos = 0L
    private var speedMetresPerSec = 0f

    /**
     * README §6 step 3's level indicator reads gravity, not the ARCore pose. Tilt is a property
     * of how the phone is held, which the accelerometer always knows; the pose is only published
     * while VIO is TRACKING, so the gauge used to sit frozen at 0 deg through the whole of
     * pre-flight on a phone that hadn't tracked yet -- exactly when the operator is being asked
     * to set the angle. Value is the world-up direction's Z in device coordinates, i.e. the
     * downward tilt of the rear camera's optical axis (see [tiltDownDeg]).
     */
    @Volatile private var gravityUpZ = 0f
    private val sensors by lazy { getSystemService(Context.SENSOR_SERVICE) as SensorManager }
    private val gravityListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) {
            val x = e.values[0]; val y = e.values[1]; val z = e.values[2]
            val n = sqrt(x * x + y * y + z * z)
            if (n > 1e-3f) gravityUpZ = z / n
            System.arraycopy(e.values, 0, gravityVec, 0, 3)
            updateCompassHeading()
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    /**
     * Magnetic compass heading (degrees, clockwise from magnetic north) at the most recent
     * keyframe -- not declination-corrected, there's no location fix to derive it from. ARCore's
     * world yaw is otherwise arbitrary (set by wherever tracking happened to start), so this is
     * the only way the backend can lock scene orientation to something real-world.
     */
    @Volatile private var compassHeadingDeg: Float? = null
    private val gravityVec = FloatArray(3)
    private val magneticVec = FloatArray(3)
    private val compassRotationMatrix = FloatArray(9)
    private val compassOrientation = FloatArray(3)
    private val magneticListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) {
            System.arraycopy(e.values, 0, magneticVec, 0, 3)
            updateCompassHeading()
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private fun updateCompassHeading() {
        if (!SensorManager.getRotationMatrix(compassRotationMatrix, null, gravityVec, magneticVec)) return
        // The camera optical axis is -Z in device coordinates (pointing out the back of the phone).
        // Row 2 of compassRotationMatrix (R[6], R[7], R[8]) represents the device +Z axis (screen normal)
        // in world coordinates [East, North, Up].
        // Therefore, the camera optical axis in world coordinates is -[R[6], R[7], R[8]]:
        // - East component = -R[6]
        // - North component = -R[7]
        // Standard Android SensorManager.getOrientation() tracks the top of the phone (+Y device axis).
        // When holding the phone nearly vertical (~90 deg), slight tilts (>90 deg vs <90 deg) cause the top of the phone
        // to flip between tilting forward and backward, producing a 180 deg compass jump (gimbal flip).
        // By deriving heading directly from the camera optical axis (-Z), the horizontal azimuth is continuous,
        // stable, and correctly reflects the camera line of sight regardless of vertical tilt.
        val camEast = -compassRotationMatrix[6]
        val camNorth = -compassRotationMatrix[7]
        val deg = Math.toDegrees(kotlin.math.atan2(camEast.toDouble(), camNorth.toDouble())).toFloat()
        compassHeadingDeg = (deg + 360f) % 360f
    }

    /** README §4/§5: fixed-ratio decimation of the post-illumination-gate stream, not a
     * displacement/angle filter. Counts frames that passed the photometric gate this session. */
    private var gatePassedCounter = 0L

    private val translation = FloatArray(3)
    private val quaternion = FloatArray(4)

    @Volatile private var lastVerdict = PhotometricGate.Verdict.OK
    @Volatile private var finishedPath: String? = null

    /** Which property/room this capture belongs to, supplied by PropertyDetailActivity. Null
     * when the capture screen was opened without one -- the scan is still written normally and
     * shows up under "Unassigned" in the gallery, it just isn't indexed against a property. */
    private var propertyId: String? = null
    private var roomLabel: String? = null
    private val propertyStore by lazy { PropertyStore(this) }

    // Milestone haptic flags to give physical tactile feedback on landmark targets
    private var hapticMilestone200 = false
    private var hapticMilestone350 = false
    private var hapticMilestone500 = false
    private var hapticLoopClosed = false

    // Tracking recovery & relocalization state. The actual point-matching/RANSAC search
    // (RelocalizationRecovery.estimateAlignmentPreferId) runs on its own HandlerThread, never the
    // GL thread -- see two prior ANRs where that search blocked onPause() for 5+ seconds mid-scan.
    private var needsRelocalizationCheck = false
    private var lastRelocalizationNanos = 0L
    private var relocalizedToastUntilNanos = 0L
    private val scratchLandmarkX = FloatArray(20000)
    private val scratchLandmarkY = FloatArray(20000)
    private val scratchLandmarkZ = FloatArray(20000)
    private val relocIdCurX = FloatArray(64)
    private val relocIdCurY = FloatArray(64)
    private val relocIdCurZ = FloatArray(64)
    private val relocIdLandX = FloatArray(64)
    private val relocIdLandY = FloatArray(64)
    private val relocIdLandZ = FloatArray(64)
    private val relocThread = android.os.HandlerThread("relocalization").apply { start() }
    private val relocHandler = android.os.Handler(relocThread.looper)
    private val relocJobRunning = java.util.concurrent.atomic.AtomicBoolean(false)
    private val relocConfirmedTransform =
        java.util.concurrent.atomic.AtomicReference<RelocalizationRecovery.RigidTransform?>(null)
    /** Touched only on [relocThread] -- one candidate must be rediscovered on a second, independent
     * pass before it's trusted enough to snap the landmark cloud (see [relocConfirmedTransform]). */
    private var relocPendingCandidate: RelocalizationRecovery.RigidTransform? = null

    /** Pre-flight calibration flow:
     * 0 = 5-second automatic multi-angle ISO/shutter exposure & auto-WB calibration.
     * 1 = Camera angle check & quick WB manual trim (if user wants to tweak). */
    private var preflightStep = 0
    private var calibrationStartNanos = 0L
    private var lastMeanY = 128f
    private var lastMeanU = 128f
    private var lastMeanV = 128f
    /** The auto white-balance gains (red, green, blue) pre-flight froze: the ISP AWB's own
     * converged estimate, or a grey-world solve on devices that never report one. */
    private val autoWbGains = floatArrayOf(1f, 1f, 1f)
    /** False while the ISP's AWB is still driving; true once pre-flight has frozen its answer and
     * the operator's sliders trim on top of it. Cleared by the reset button. */
    @Volatile private var autoWbLocked = false
    /** Manual trim on top of the auto grey-world proposal (`whiteBalanceGains`), both -1..1.
     * `wbBias` is warm<->cool (red vs blue, `wbSlider`); `wbTintBias` is green<->magenta (green
     * gain, `wbTintSlider`) -- the axis the old red/blue-only heuristic couldn't reach at all, so
     * a green- or magenta-cast room had no slider position that fixed it. Both 0 = trust the auto
     * proposal as-is. */
    @Volatile private var wbBias = 0f
    @Volatile private var wbTintBias = 0f
    private var currentIso = CameraPipeline.DEFAULT_ISO
    private var currentShutterNs = CameraPipeline.DEFAULT_SHUTTER_NS

    private val requestPermissions = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { granted ->
        if (granted[Manifest.permission.CAMERA] == true) startTracking()
        else fatal("Camera access needed", "Allow camera access for Glome AR-Scan in Settings, then reopen the app.")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Operator-set capture thresholds, if any (SettingsActivity); defaults otherwise.
        Tunables.load(this)
        propertyId = intent.getStringExtra(PropertyStore.EXTRA_PROPERTY_ID)
        roomLabel = intent.getStringExtra(PropertyStore.EXTRA_ROOM_LABEL)
        installCameraRaceGuard()
        setContentView(R.layout.activity_main)
        root = findViewById(R.id.root)
        topColumn = findViewById(R.id.topColumn)
        bottomColumn = findViewById(R.id.bottomColumn)
        trackingDot = findViewById(R.id.trackingDot)
        phaseText = findViewById(R.id.phaseText)
        timerText = findViewById(R.id.timerText)
        infoButton = findViewById(R.id.infoButton)
        memoryButton = findViewById(R.id.memoryButton)
        coverageValue = findViewById(R.id.coverageValue)
        coverageLabel = findViewById(R.id.coverageLabel)
        progressBar = findViewById(R.id.progressBar)
        warnBanner = findViewById(R.id.warnBanner)
        miniMapView = findViewById(R.id.miniMapView)
        reticle = findViewById(R.id.reticle)
        frontierGroup = findViewById(R.id.frontierGroup)
        occlusionGroup = findViewById(R.id.occlusionGroup)
        arrowView = findViewById(R.id.arrowView)
        occlusionArrowView = findViewById(R.id.occlusionArrowView)
        occlusionArrowView.fillColor = color(R.color.cyan_fill)
        arrowView.fillColor = color(R.color.amber_fill)
        diagText = findViewById(R.id.diagText)
        tiltCard = findViewById(R.id.tiltCard)
        tiltValue = findViewById(R.id.tiltValue)
        tiltGauge = findViewById(R.id.tiltGauge)
        wbCard = findViewById(R.id.wbCard)
        wbResetButton = findViewById(R.id.wbResetButton)
        instructionTitle = findViewById(R.id.instructionTitle)
        instructionBody = findViewById(R.id.instructionBody)
        primaryButton = findViewById(R.id.primaryButton)
        hintText = findViewById(R.id.hintText)
        bottomPadBase = bottomColumn.paddingBottom
        applyInsets()

        val touchSlopPx = android.view.ViewConfiguration.get(this).scaledTouchSlop
        var afDownX = 0f
        var afDownY = 0f
        root.setOnTouchListener { _, event ->
            when (event.action) {
                android.view.MotionEvent.ACTION_DOWN -> {
                    afDownX = event.x
                    afDownY = event.y
                }
                android.view.MotionEvent.ACTION_UP -> {
                    val moved = kotlin.math.hypot((event.x - afDownX).toDouble(), (event.y - afDownY).toDouble()) > touchSlopPx
                    val now = System.nanoTime()
                    // A tap-to-focus is a deliberate, stationary tap, not a drag, and is
                    // debounced: continuous AF is already running (configureManualOverrides),
                    // so spamming CONTROL_AF_TRIGGER_START on every incidental touch just makes
                    // the lens hunt instead of settle (README §6 relies on continuous AF).
                    if (!moved && event.y > topColumn.bottom && event.y < bottomColumn.top &&
                        now - lastAutoFocusTriggerNanos > AUTO_FOCUS_COOLDOWN_NANOS
                    ) {
                        lastAutoFocusTriggerNanos = now
                        renderer?.triggerAutoFocus(event.x / root.width.toFloat(), event.y / root.height.toFloat())
                    }
                }
            }
            false
        }

        primaryButton.setOnClickListener { view ->
            // The label only catches up on the next 6 Hz repaint (see postUi), so the tap gets its
            // own immediate acknowledgement -- an operator holding a phone at arm's length can
            // feel this and doesn't have to double-tap to find out whether it registered.
            view.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
            pending = when (state) {
                State.IDLE, State.DONE -> Pending.START
                State.PREFLIGHT -> Pending.PREFLIGHT_NEXT
                State.SCANNING -> if (interrupted) Pending.RESUME else Pending.FINISH
            }
            pendingAtNanos = System.nanoTime()
        }

        infoButton.setOnClickListener {
            showDiag = !showDiag
            diagText.setVisible(showDiag)
            infoButton.alpha = if (showDiag) 1f else 0.7f
        }
        memoryButton.setOnClickListener {
            startActivity(Intent(this, ScanGalleryActivity::class.java))
        }
        infoButton.alpha = 0.7f

        // Back to the auto read: drop the operator's trim and re-run the ISP's AWB from scratch,
        // so a reset after walking into a differently-lit room actually re-estimates.
        wbResetButton.setOnClickListener {
            wbBias = 0f
            wbTintBias = 0f
            autoWbLocked = false
            wbSlider.progress = 50
            wbTintSlider.progress = 50
            renderer?.resumeAutoWhiteBalance()
        }

        wbSlider = findViewById(R.id.wbSlider)
        wbSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                wbBias = (progress - 50) / 50f
                if (fromUser) freezeAutoWhiteBalance()
                if (state == State.PREFLIGHT && preflightStep == 1) previewWhiteBalance()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        wbTintSlider = findViewById(R.id.wbTintSlider)
        wbTintSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                wbTintBias = (progress - 50) / 50f
                if (fromUser) freezeAutoWhiteBalance()
                if (state == State.PREFLIGHT && preflightStep == 1) previewWhiteBalance()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) {}
            override fun onStopTrackingTouch(seekBar: SeekBar) {}
        })

        checkArCoreAndProceed()
    }

    private fun color(id: Int) = ContextCompat.getColor(this, id)

    /**
     * Edge-to-edge: the camera preview runs the full height of the display, and the two HUD
     * columns pad themselves out of the status bar and the gesture bar. Previously the coverage
     * readout sat under the status bar clock and the primary button under the navigation bar.
     */
    private fun applyInsets() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            topColumn.updatePadding(top = bars.top)
            bottomColumn.updatePadding(bottom = bottomPadBase + bars.bottom)
            insets
        }
    }

    /** Fade views in rather than popping them, and never restart the animation on a repaint. */
    private fun View.setVisible(visible: Boolean) {
        val target = if (visible) View.VISIBLE else View.GONE
        if (visibility == target) return
        animate().cancel()
        if (visible) {
            alpha = 0f
            visibility = View.VISIBLE
            animate().alpha(1f).setDuration(150).start()
        } else {
            alpha = 1f
            visibility = View.GONE
        }
    }

    /** A failure the operator has to act on outside the app. Owns the instruction card from here
     * on; [updateUi] will not overwrite it. */
    private fun fatal(title: String, body: String) {
        fatalTitle = title
        fatalBody = body
        runOnUiThread { updateUi(null) }
    }

    /**
     * Sideloading via adb/Gradle skips the Play Store compatibility filtering that the
     * manifest's `com.google.ar.core required` tag normally relies on, so an unsupported device
     * can install this app and get nothing but silent tracking failure. `checkAvailability` is
     * the same check Play would have made; run it explicitly before ever touching the camera.
     */
    private fun checkArCoreAndProceed() {
        when (val availability = ArCoreApk.getInstance().checkAvailability(this)) {
            ArCoreApk.Availability.SUPPORTED_INSTALLED -> requestPermissions.launch(arrayOf(Manifest.permission.CAMERA))
            else -> if (availability.isTransient) {
                // Still checking (e.g. no network yet) -- re-poll rather than false-negative.
                root.postDelayed({ checkArCoreAndProceed() }, 200)
            } else if (availability.isSupported) {
                // Device is ARCore-capable but Play Services for AR isn't installed/current;
                // this triggers the Play Store install/update flow and comes back through
                // onResume once it returns.
                try {
                    ArCoreApk.getInstance().requestInstall(this, !arCoreInstallRequested)
                    arCoreInstallRequested = true
                } catch (e: Exception) {
                    fatal("Google Play Services for AR is required", "Install failed: ${e.message}")
                }
            } else {
                fatal(
                    "This phone can't run AR scans",
                    "ARCore isn't supported on this device ($availability).",
                )
            }
        }
    }

    private var arCoreInstallRequested = false

    /**
     * ARCore tears the camera down from its own coroutine threads; a pause landing while the
     * camera is still opening kills the process from a thread we don't own. Inherited verbatim
     * from mobile_sphere_capture, where it was diagnosed -- whatever it was stopping is already
     * stopped, so log and stay alive. Everything else still crashes.
     */
    private fun installCameraRaceGuard() {
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, e ->
            if (thread !== Looper.getMainLooper().thread && isClosedCaptureSessionRace(e)) {
                android.util.Log.w(TAG, "swallowed camera teardown race on ${thread.name}", e)
            } else {
                previous?.uncaughtException(thread, e)
            }
        }
    }

    private fun startTracking() {
        val view = GLSurfaceView(this)
        view.setEGLContextClientVersion(2)
        val r = ArScanRenderer(
            context = this,
            onFrame = { sample -> onFrame(sample) },
            onError = { e ->
                // ArScanRenderer routes *every* exception from its draw loop here, including
                // one-off startup blips, so this must not be treated as fatal: a permanently
                // disabled button on a session that is actually running fine is worse than the
                // silent swallowing it replaced. Persistent failures show a persistent notice
                // because the timestamp keeps being refreshed; a single blip clears itself.
                android.util.Log.w(TAG, "ARCore frame error", e)
                sessionErrorAtNanos = System.nanoTime()
            },
            onUntrackedImage = { image -> onMeteringImage(image) },
        )
        view.setRenderer(r)
        view.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        root.addView(view, 0, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        glSurfaceView = view
        renderer = r
    }

    /**
     * The depth model is built on first use and kept for the process: loading it costs 24 MB and
     * about a second, and a second scan should not pay that again. Null when ARCore's own Depth
     * API is available, which is the better source wherever it exists.
     */
    private fun monoWorker(): MonoDepthWorker? {
        if (renderer?.depthSupported != false) return null
        mono?.let { return it }
        return MonoDepthWorker(applicationContext) { depth, t, q ->
            worker?.submitDepth(depth, t, q, floorY)
        }.also { mono = it }
    }

    @Volatile private var lastKnownPose: Pose? = null

    private fun applyPending(pose: Pose?) {
        val action = pending ?: return
        if ((System.nanoTime() - pendingAtNanos) / 1e6f > PENDING_TTL_MS) {
            pending = null
            return
        }
        val currentPose = pose ?: lastKnownPose
        pending = null
        when (action) {
            Pending.START -> beginPreflight()
            Pending.PREFLIGHT_NEXT -> advancePreflight(currentPose)
            Pending.RESUME -> interrupted = false
            // CLAUDE.md's enforced entry-door loop closure. Coverage stays advisory-only (see
            // COMPLETION_FRACTION's comment: an 85% target may simply be unreachable in some
            // rooms, and a scan the operator can never end is a trap) -- only loop closure is a
            // hard requirement here, since walking back to where you started is always physically
            // possible if you got there in the first place. An unqualified tap is a no-op, not an
            // error; the button label already says why before they tap.
            // CLAUDE.md's on-device pre-flight validation gate before upload: a post-scan
            // completeness/quality check, distinct from the pre-scan setup flow in item 6.
            // Unlike coverage, both conditions mean data that genuinely cannot be reconstructed
            // from (not merely "could be better"), so this blocks like loop closure does.
            Pending.FINISH -> {
                val closed = distance(translation, startPosition) <= LOOP_CLOSURE_RADIUS_M
                val issues = validationIssues(worker?.stats?.gridFull ?: false, writer?.keyframeCount ?: 0)
                if (closed && issues.isEmpty()) finishSession()
            }
        }
    }

    private fun beginPreflight() {
        state = State.PREFLIGHT
        preflightStep = 0
        calibrationStartNanos = System.nanoTime()
        lastMeanY = 128f
        lastMeanU = 128f
        lastMeanV = 128f
        wbBias = 0f
        wbTintBias = 0f
        autoWbLocked = false
        autoWbGains[0] = 1f; autoWbGains[1] = 1f; autoWbGains[2] = 1f
        renderer?.resumeAutoWhiteBalance()
        runOnUiThread { wbSlider.progress = 50; wbTintSlider.progress = 50 }
        // Start from a high-brightness indoor baseline: ISO 800, 1/60s shutter (16.6ms) for bright, clear tracking
        currentIso = 800
        currentShutterNs = 16_666_666L
        renderer?.adjustExposure(currentIso, currentShutterNs)
    }

    /**
     * Pre-flight professional calibration flow (README §6):
     * Step 0: 5-second dynamic multi-angle lighting sweep (auto ISO/shutter metering + initial auto-WB estimate).
     * Step 1: Professional White Balance fine-tuning with reticle & pre-centered auto gains.
     * Step 2: Camera Angle / Horizon guide (5–20° tilt) -> tap Start 3D Capture to begin session.
     */
    private fun advancePreflight(pose: Pose?) {
        when (preflightStep) {
            0 -> {
                // Step 0 -> Step 1 (White Balance fine-tuning with reticle)
                preflightStep = 1
                // The sweep is over: freeze what the ISP's AWB settled on. Centre (= no trim) is
                // that estimate, so the sliders only move if the operator disagrees with it.
                wbBias = 0f
                wbTintBias = 0f
                runOnUiThread { wbSlider.progress = 50; wbTintSlider.progress = 50 }
                freezeAutoWhiteBalance()
            }
            1 -> {
                // Step 1 -> Step 2 (Camera Angle check)
                preflightStep = 2
            }
            else -> {
                // Step 2 -> Begin 3D capture session
                val activePose = pose ?: lastKnownPose ?: Pose.makeTranslation(0f, 0f, 0f)
                beginSession(activePose)
            }
        }
    }

    private fun beginSession(pose: Pose) {
        val name = "scan_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        worker?.shutdown()
        writer?.shutdown()
        worker = CoverageWorker(
            if (renderer?.depthSupported == true) CoverageWorker.DEFAULT_VOXEL_SIZE_M
            else CoverageWorker.MONO_VOXEL_SIZE_M
        )
        writer = DatasetWriter(this, name)
        gate = PhotometricGate()
        // The operator holds the phone at roughly chest height; the floor is that far below the
        // starting pose. Only the frontier height band depends on it, so a 20 cm error costs a
        // slightly wrong search band, not a wrong map. Calibration knob.
        floorY = pose.ty() - PHONE_HEIGHT_M
        pose.getTranslation(startPosition, 0)
        miniMapView.resetTrail()
        miniMapView.setStart(startPosition[0], startPosition[2])
        sessionStartNanos = System.nanoTime()
        framesSeen = 0
        droppedDark = 0; droppedBlown = 0; droppedTransition = 0; droppedMotion = 0
        trackingLosses = 0
        gatePassedCounter = 0L
        finishedPath = null
        interrupted = false
        hapticMilestone200 = false
        hapticMilestone350 = false
        hapticMilestone500 = false
        hapticLoopClosed = false
        needsRelocalizationCheck = false
        lastRelocalizationNanos = 0L
        relocalizedToastUntilNanos = 0L
        parallaxTracker.reset()
        state = State.SCANNING
    }

    private fun finishSession() {
        val w = writer ?: return
        val g = worker
        val r = renderer
        val stats = g?.stats
        val summary = DatasetFormat.summaryJson(
            durationSeconds = (System.nanoTime() - sessionStartNanos) / 1e9f,
            coverageFraction = stats?.coverage ?: 0f,
            occupiedVoxels = stats?.occupied ?: 0,
            verifiedVoxels = stats?.verified ?: 0,
            freeVoxels = stats?.free ?: 0,
            occludedVoxels = stats?.occluded ?: 0,
            voxelSizeM = g?.grid?.voxelSizeM ?: 0f,
            gridFull = stats?.gridFull ?: false,
            framesSeen = framesSeen,
            framesExported = w.keyframeCount,
            droppedDark = droppedDark,
            droppedBlown = droppedBlown,
            droppedTransition = droppedTransition,
            droppedMotion = droppedMotion,
            droppedQueue = w.droppedQueue.get(),
            trackingLosses = trackingLosses,
            depthSource = when {
                r?.depthSupported == true -> "arcore_depth"
                mono?.calibrated == true -> "zipdepth_arcore_scaled"
                else -> "arcore_point_cloud"
            },
        )
        w.finish(
            r?.focalX ?: 0f, r?.focalY ?: 0f, r?.principalX ?: 0f, r?.principalY ?: 0f,
            r?.imageWidth ?: 0, r?.imageHeight ?: 0, summary,
            k1 = r?.pipeline?.lensDistortionK1, k2 = r?.pipeline?.lensDistortionK2,
        ) { path ->
            finishedPath = path
            android.util.Log.i(TAG, "dataset written to $path")
            // Index the finished zip against its property only once the write actually landed,
            // so a row can never point at a zip that isn't there (see the 2026-09-18
            // save-completion race).
            val id = propertyId
            if (id != null) {
                propertyStore.recordRoomScan(
                    propertyId = id,
                    label = roomLabel ?: "Room",
                    sessionName = w.sessionName,
                    coveragePercent = (stats?.coverage ?: 0f) * 100f,
                )
            }
        }
        state = State.DONE
        renderer?.guidanceTarget = null
        renderer?.occlusionTarget = null
    }

    // ---- per-frame (render thread) ----

    /** Diagnostic only: the render thread stalling for seconds (see two ANRs at ~800-900
     * landmarks) has to be caught mid-frame to find which stage is slow -- flags anything over
     * 50ms, since a healthy frame budget at 30fps is ~33ms total. */
    private fun logIfSlow(label: String, startNs: Long, n: Int) {
        val ms = (System.nanoTime() - startNs) / 1_000_000.0
        if (ms > 50.0) Log.w(TAG, "SLOW FRAME: $label took ${"%.1f".format(ms)}ms (n=$n)")
    }

    private fun onFrame(sample: ArScanRenderer.Sample?) {
        frameCounter++
        val scanning = state == State.SCANNING && !interrupted
        // ISO metering (README §4) runs from the moment the camera opens (IDLE, before the
        // operator has tapped anything) through all of pre-flight, so it has the most time to
        // converge before the value gets locked in for the whole scan.
        val meteringActive = state == State.IDLE || state == State.PREFLIGHT
        // Every other frame, not every frame (SPEC §1.3 risk 2): acquiring and reading a 1080p
        // CPU image is the single most expensive thing on the render thread, and 15 Hz is still
        // three times the keyframe cap. The photometric window then spans twice the wall time,
        // which only makes the gate slower to reopen after a light change -- the safe direction.
        renderer?.captureImage = (scanning || meteringActive) && frameCounter % IMAGE_EVERY_N_FRAMES == 0L
        renderer?.wantDepth = scanning && renderer?.depthSupported == true &&
            frameCounter % DEPTH_EVERY_N_FRAMES == 0L

        // Button taps (Start Scan above all) must not depend on ARCore having a TRACKING pose --
        // gating this behind `sample != null` left the UI permanently unresponsive whenever
        // tracking hadn't started yet (e.g. an underexposed room with no visual features).
        applyPending(sample?.pose)

        if (sample == null) {
            if (wasTracking && state == State.SCANNING) {
                trackingLosses++
                needsRelocalizationCheck = true
                // A candidate carried over from a prior recovery cycle must not be allowed to
                // "confirm" a candidate from this new one -- they were never independently
                // rediscovering the same alignment, just coincidentally close.
                relocPendingCandidate = null
            }
            wasTracking = false
            // Poses either side of a tracking gap are unrelated; differencing them would
            // manufacture a huge rate the moment tracking comes back.
            prevForward = null
            prevPosition = null
            postUi(null)
            return
        }
        if (!wasTracking && state == State.SCANNING) {
            needsRelocalizationCheck = true
            relocPendingCandidate = null
        }
        wasTracking = true
        framesSeen++

        val pose = sample.pose
        lastKnownPose = pose
        pose.getTranslation(translation, 0)
        pose.getRotationQuaternion(quaternion, 0)
        updateKinematics()

        // Feed sparse feature points to parallax tracker continuously (live viewport feedback & triangulation)
        if (sample.points != null && sample.pointIds != null && sample.numPoints > 0) {
            val now = System.nanoTime()
            if (state == State.SCANNING && needsRelocalizationCheck && parallaxTracker.verifiedLandmarkCount >= 10 &&
                (now - lastRelocalizationNanos) > 1_000_000_000L && relocJobRunning.compareAndSet(false, true)
            ) {
                lastRelocalizationNanos = now
                // Cheap on the GL thread: bounded array copies, no search. The actual matching
                // happens off-thread in the block below.
                val landmarkCount = parallaxTracker.getLandmarkData(scratchLandmarkX, scratchLandmarkY, scratchLandmarkZ)
                val idMatchCount = parallaxTracker.matchById(
                    sample.pointIds, sample.points, sample.numPoints,
                    relocIdCurX, relocIdCurY, relocIdCurZ, relocIdLandX, relocIdLandY, relocIdLandZ,
                )
                val camX = translation[0]; val camY = translation[1]; val camZ = translation[2]
                val currentPoints = sample.points
                val currentNumPoints = sample.numPoints
                relocHandler.post {
                    try {
                        val transform = RelocalizationRecovery.estimateAlignmentPreferId(
                            idMatchCount, relocIdCurX, relocIdCurY, relocIdCurZ, relocIdLandX, relocIdLandY, relocIdLandZ,
                            currentPoints, currentNumPoints, camX, camY, camZ,
                            scratchLandmarkX, scratchLandmarkY, scratchLandmarkZ, landmarkCount,
                        )
                        val pending = relocPendingCandidate
                        if (transform != null && transform.isSignificant) {
                            if (pending != null && RelocalizationRecovery.isSimilarTransform(pending, transform)) {
                                relocConfirmedTransform.set(transform)
                                relocPendingCandidate = null
                            } else {
                                relocPendingCandidate = transform
                            }
                        } else {
                            relocPendingCandidate = null
                        }
                    } finally {
                        relocJobRunning.set(false)
                    }
                }
            }

            // Cheap poll every frame: applies a transform once it's been rediscovered on two
            // independent background passes (see relocPendingCandidate above).
            val confirmedTransform = relocConfirmedTransform.getAndSet(null)
            if (confirmedTransform != null) {
                parallaxTracker.transformLandmarks(confirmedTransform)
                needsRelocalizationCheck = false
                relocalizedToastUntilNanos = now + 2_500_000_000L // show for 2.5s
                root.post { root.performHapticFeedback(HapticFeedbackConstants.CONFIRM) }
            }

            val updateStartNs = System.nanoTime()
            parallaxTracker.update(
                sample.points, sample.pointIds, sample.numPoints,
                translation[0], translation[1], translation[2],
                sample.timestampNs,
            )
            logIfSlow("parallaxTracker.update", updateStartNs, parallaxTracker.verifiedLandmarkCount)
            val exportStartNs = System.nanoTime()
            val exportedCount = parallaxTracker.exportPointVertices(pointSpriteExportBuf, sample.timestampNs)
            logIfSlow("exportPointVertices", exportStartNs, exportedCount)
            renderer?.pointVertices = pointSpriteExportBuf
            renderer?.pointVertexCount = exportedCount
        }

        if (state != State.SCANNING || interrupted) {
            sample.image?.let { image ->
                onMeteringImage(image)
                image.close()
            }
            postUi(sample)
            return
        }

        // Photometric gate first: it decides whether this frame may be exported at all, and the
        // luma it computes is read straight from the Y plane we already hold.
        val r0 = renderer ?: return
        var exported = false
        sample.image?.let { image ->
            try {
                val plane = image.planes[0]
                val mean = Luma.mean(plane.buffer, image.width, image.height, plane.rowStride, plane.pixelStride)
                val verdict = gate.offer(mean)
                lastVerdict = verdict
                when (verdict) {
                    PhotometricGate.Verdict.BLOWN -> droppedBlown++
                    PhotometricGate.Verdict.TRANSITION -> droppedTransition++
                    else -> Unit
                }
                // Once scan has started, never drop frames for dark areas (user already verified lights)
                if (verdict == PhotometricGate.Verdict.OK || verdict == PhotometricGate.Verdict.DARK) {
                    if (tooFastForCapture()) droppedMotion++
                    else if (isKeyframe()) exported = writeKeyframe(image, sample.timestampNs)
                }
                monoWorker()?.submit(
                    image, translation, quaternion, sample.points, sample.numPoints,
                    r0.focalX, r0.focalY, r0.principalX, r0.principalY,
                )
            } finally {
                image.close()
            }
        }

        writer?.addPose(
            sample.timestampNs, translation[0], translation[1], translation[2],
            quaternion[0], quaternion[1], quaternion[2], quaternion[3],
            renderer?.trackingState ?: "?", exported,
        )



        val w = worker
        if (w != null) {
            val depth = sample.depth
            if (depth != null) {
                w.submitDepth(depth, translation, quaternion, floorY)
            } else if (sample.points != null && sample.numPoints > 0 && mono?.calibrated != true) {
                w.submitPoints(sample.points, sample.numPoints, translation, floorY)
            }
            renderer?.voxelPoints = w.snapshot
            renderer?.guidanceTarget = w.target
            renderer?.occlusionTarget = w.occlusionTarget
        }

        postUi(sample)
    }

    /** Rebuilding the whole HUD 30 times a second is UI-thread work nobody can read that fast;
     * 6 Hz costs a banner up to 160 ms of latency and buys back the posts. Button taps get a
     * haptic acknowledgement on the spot (see the click listener) so nothing waits on this tick to
     * confirm the press landed. */
    private fun postUi(sample: ArScanRenderer.Sample?) {
        if (frameCounter % UI_EVERY_N_FRAMES != 0L) return
        runOnUiThread { updateUi(sample) }
    }

    /**
     * Angular rate of the optical axis and linear speed of the camera, both smoothed. These are
     * the kinematic guards of projet.md §2.4: past them the frame is motion-blurred, and a
     * blurred keyframe poisons the reconstruction far more than a missing one costs it.
     */
    private fun updateKinematics() {
        val now = System.nanoTime()
        // Camera forward straight from the pose quaternion. It was previously derived from
        // Pose.getTransformedAxis, which does not include the translation the code subtracted
        // from it: the result was a non-unit vector that swung with the camera's *position*, and
        // it reported 500 deg/s on a phone lying still -- permanent TURN SLOWER, and 95% of
        // keyframes thrown away as motion-blurred.
        Unproject.rotate(quaternion, 0f, 0f, -1f, forward)
        val prev = prevForward
        val turnDt = (now - prevForwardNanos) / 1e9f
        if (prev != null && turnDt >= MIN_KINEMATIC_DT_S && turnDt <= MAX_KINEMATIC_DT_S) {
            angularRateDegPerSec += (angleBetweenDeg(prev, forward) / turnDt - angularRateDegPerSec) * SMOOTH_ALPHA
        }
        prevForward = forward.copyOf()
        prevForwardNanos = now

        val prevPos = prevPosition
        val moveDt = (now - prevPositionNanos) / 1e9f
        if (prevPos != null && moveDt >= MIN_KINEMATIC_DT_S && moveDt <= MAX_KINEMATIC_DT_S) {
            speedMetresPerSec += (distance(prevPos, translation) / moveDt - speedMetresPerSec) * SMOOTH_ALPHA
        }
        prevPosition = translation.copyOf()
        prevPositionNanos = now
    }

    /** README §6 step 2's live WB preview. U/V (Cb/Cr) planes of a YUV_420_888 image sit at 128
     * for a neutral grey subject regardless of white balance, so their means alone give the red/
     * blue imbalance to correct -- reuses `Luma.mean`'s generic strided-byte-average, it isn't
     * luma-specific despite the name. */
    /** Runs off both the tracked and untracked preview paths (`ArScanRenderer`'s `onFrame` and
     * `onUntrackedImage`) -- metering has to work before ARCore reaches TRACKING, since a too-dark
     * exposure is exactly what can keep it from ever getting there. */
    private fun onMeteringImage(image: android.media.Image) {
        if (state != State.IDLE && state != State.PREFLIGHT) return
        // Auto-meter exposure during IDLE and Step 0
        if (state == State.IDLE || (state == State.PREFLIGHT && preflightStep == 0)) {
            sampleExposurePreview(image)
        }
        // Sample white balance in Step 0 (continuous auto estimate) and Step 1 (reticle sampling)
        if (state == State.PREFLIGHT && (preflightStep == 0 || preflightStep == 1)) {
            sampleWhiteBalancePreview(image)
        }
    }

    private fun sampleWhiteBalancePreview(image: android.media.Image) {
        val y = image.planes[0]
        val u = image.planes[1]
        val v = image.planes[2]
        lastMeanY = sampleCenterPatchMean(y.buffer, image.width, image.height, y.rowStride, y.pixelStride)
        lastMeanU = sampleCenterPatchMean(u.buffer, image.width / 2, image.height / 2, u.rowStride, u.pixelStride)
        lastMeanV = sampleCenterPatchMean(v.buffer, image.width / 2, image.height / 2, v.rowStride, v.pixelStride)
        previewWhiteBalance()
    }

    /**
     * End of the pre-flight sweep: take whatever the ISP's AWB converged on and hold it there for
     * the rest of the scan. Doing the illuminant estimate in the ISP rather than from the reticle
     * patch is the whole point -- a grey-world solve on the patch neutralises the patch, so a
     * wooden floor in the reticle came out grey-blue instead of wooden.
     *
     * Fallback for a device that never reports its AWB gains: the grey-world solve on the last
     * sampled patch, which is at least better than leaving the gains neutral. Nothing is applied
     * to the sensor before this point, so that solve reads an uncorrected frame and is a one-shot,
     * not a loop.
     */
    private fun freezeAutoWhiteBalance() {
        if (autoWbLocked) return
        val isp = renderer?.autoWhiteBalanceGains()
        if (isp != null) {
            autoWbGains[0] = isp.red
            autoWbGains[1] = (isp.greenEven + isp.greenOdd) / 2f
            autoWbGains[2] = isp.blue
        } else {
            whiteBalanceGains(lastMeanY, lastMeanU, lastMeanV).let {
                autoWbGains[0] = it[0]; autoWbGains[1] = it[1]; autoWbGains[2] = it[3]
            }
        }
        Log.i(TAG, "WB frozen: isp=${isp != null} gains=${autoWbGains.joinToString()}")
        autoWbLocked = true
        previewWhiteBalance()
    }

    private fun sampleCenterPatchMean(
        buf: java.nio.ByteBuffer, width: Int, height: Int, rowStride: Int, pixelStride: Int
    ): Float {
        // Sample central 30% x 30% region corresponding to the on-screen reticle
        val minX = (width * 0.35f).toInt()
        val maxX = (width * 0.65f).toInt()
        val minY = (height * 0.35f).toInt()
        val maxY = (height * 0.65f).toInt()
        val rowBytes = ByteArray(maxX - minX)
        var sum = 0L
        var count = 0
        var y = minY
        val step = 4
        while (y < maxY) {
            val rowOffset = y * rowStride + minX * pixelStride
            if (pixelStride == 1 && rowOffset + rowBytes.size <= buf.capacity()) {
                buf.position(rowOffset)
                buf.get(rowBytes)
                for (i in rowBytes.indices step step) {
                    sum += (rowBytes[i].toInt() and 0xFF)
                    count++
                }
            } else {
                var x = minX
                while (x < maxX) {
                    val pos = y * rowStride + x * pixelStride
                    if (pos < buf.capacity()) {
                        sum += (buf.get(pos).toInt() and 0xFF)
                        count++
                    }
                    x += step
                }
            }
            y += step
        }
        return if (count > 0) sum.toFloat() / count else 128f
    }

    /** The frozen auto gains plus the operator's slider trims, applied live to the sensor during
     * pre-flight step 1. No-op until they're frozen -- before that the ISP's AWB owns the sensor
     * and pushing gains at it would only take it out of auto early. */
    private fun previewWhiteBalance() {
        if (!autoWbLocked) return
        val gains = whiteBalanceTrim(autoWbGains, wbBias, wbTintBias)
        renderer?.lockWhiteBalance(
            android.hardware.camera2.params.RggbChannelVector(gains[0], gains[1], gains[2], gains[3])
        )
    }

    /** README §4: shutter and ISO are jointly metered against a shared indoor/outdoor tradeoff
     * curve (see `meteredExposure`), not one fixed and one adaptive. Runs live from the moment
     * the camera opens through pre-flight; whatever it has converged to when the operator taps
     * Start Scan is what stays locked for the rest of the session. */
    private fun sampleExposurePreview(image: android.media.Image) {
        val y = image.planes[0]
        val meanLuma = Luma.mean(y.buffer, image.width, image.height, y.rowStride, y.pixelStride)
        val exposure = meteredExposure(meanLuma, currentIso, currentShutterNs)
        currentIso = exposure.iso
        currentShutterNs = exposure.shutterNs
        renderer?.adjustExposure(currentIso, currentShutterNs)
    }

    private fun tooFastForCapture(): Boolean =
        speedMetresPerSec > MAX_KEYFRAME_SPEED_MS || angularRateDegPerSec > MAX_KEYFRAME_RATE_DEG_S

    /**
     * Keyframe selection (README §4/§5): fixed-ratio decimation of the stream, keeping 1 of
     * every `DECIMATION_STRIDE` frames that already passed the photometric gate. At a 60fps
     * sensor readout this yields 60/DECIMATION_STRIDE keyframes/sec -- 6-10/sec across the
     * README-specified 6-10 stride range. This replaced a displacement/angle-gated cap
     * (`SPEC.md` §1.2c): that was a materially different algorithm (a spatial-baseline filter),
     * not a parameter tweak on this one -- it could suppress a keyframe indefinitely while the
     * operator paused, where decimation always keeps moving through the stream.
     */
    private fun isKeyframe(): Boolean {
        gatePassedCounter++
        return isDecimationKeyframe(gatePassedCounter)
    }

    private fun writeKeyframe(image: android.media.Image, timestampNs: Long): Boolean {
        val w = writer ?: return false
        val r = renderer
        val p = r?.pipeline
        return w.addKeyframe(
            image, timestampNs,
            DatasetFormat.cameraToWorld(
                translation[0], translation[1], translation[2],
                quaternion[0], quaternion[1], quaternion[2], quaternion[3],
            ),
            r?.focalX ?: 0f, r?.focalY ?: 0f, r?.principalX ?: 0f, r?.principalY ?: 0f,
            p?.latestFocusDistanceDiopters ?: 0f,
            p?.latestAfState ?: 0,
            p?.latestAfMode ?: 0,
            p?.latestFocalLengthMm ?: 0f,
            compassHeadingDeg,
        )
    }

    // ---- UI thread ----

    private fun updateUi(sample: ArScanRenderer.Sample?) {
        val r = renderer
        val stats = worker?.stats
        val coverage = stats?.coverage ?: 0f
        val complete = coverage >= Tunables.coverageCompleteFraction
        // CLAUDE.md's enforced entry-door loop closure: the only hard requirement to finish.
        // Coverage stays advisory (see the Pending.FINISH comment in applyPending).
        val distanceToStart = distance(translation, startPosition)
        val loopClosed = distanceToStart <= LOOP_CLOSURE_RADIUS_M
        val issues = validationIssues(stats?.gridFull ?: false, writer?.keyframeCount ?: 0)
        val scanning = state == State.SCANNING && !interrupted
        val tracking = sample != null
        val fatal = fatalTitle != null

        // ---- top column: state ----
        trackingDot.backgroundTintList = ColorStateList.valueOf(
            color(if (tracking) R.color.action_ready else R.color.amber)
        )
        phaseText.text = when (state) {
            State.IDLE -> "READY"
            State.PREFLIGHT -> "SETUP"
            State.SCANNING -> if (interrupted) "PAUSED" else "SCANNING"
            State.DONE -> "SAVED"
        }
        timerText.text = if (state == State.SCANNING) elapsed() else ""
        // Monotonically non-decreasing 3D Landmark Discovery & Multi-View Parallax Progress
        // Progress strictly matches landmark density (%d/500 points):
        val displayCoverage = parallaxTracker.coverageFraction
        val displayPercent = (displayCoverage * 100).toInt().coerceIn(0, 100)
        val isParallaxComplete = displayCoverage >= 0.70f || parallaxTracker.verifiedLandmarkCount >= 350

        val landmarkCount = parallaxTracker.verifiedLandmarkCount
        val densityTag = when {
            landmarkCount >= 350 -> "EXCELLENT · 4K 3DGS"
            landmarkCount >= 200 -> "GOOD COVERAGE"
            else -> "ORBIT OBJECTS"
        }
        coverageValue.text = "$displayPercent%"
        coverageLabel.text = "DENSITY: %d/500 PTS · %s".format(landmarkCount, densityTag)
        progressBar.progress = displayPercent
        progressBar.progressTintList = ColorStateList.valueOf(
            color(if (isParallaxComplete) R.color.action_ready else R.color.text_primary)
        )

        // ---- 2D Bird's-Eye Mini-Map & spatial navigation ----
        val showMiniMap = scanning
        miniMapView.setVisible(showMiniMap)
        if (showMiniMap) {
            val qx = quaternion[0]; val qy = quaternion[1]; val qz = quaternion[2]; val qw = quaternion[3]
            val fwdX = -2f * (qx * qz + qw * qy)
            val fwdZ = -(1f - 2f * (qx * qx + qy * qy))
            val yawDeg = Math.toDegrees(kotlin.math.atan2(fwdZ.toDouble(), fwdX.toDouble())).toFloat()
            miniMapView.updateUser(translation[0], translation[2], yawDeg)
            miniMapView.updateFloorplan(worker?.floorplan)
            val target = worker?.target
            if (target != null) miniMapView.setTarget(target[0], target[2]) else miniMapView.setTarget(null, null)
        }

        // Mini-map serves as the primary real-time spatial navigation tool.
        // Viewport center arrows are suppressed so the camera feed remains 100% clean and clear.
        frontierGroup.setVisible(false)
        occlusionGroup.setVisible(false)

        // ---- warning slot ----
        val nowUi = System.nanoTime()
        val warning: String?
        if (fatal || !scanning) {
            resetWarningQueue()
            warning = null
        } else {
            val active = mutableListOf<Pair<WarningKind, String>>()
            if (nowUi < relocalizedToastUntilNanos) {
                active += WarningKind.REALIGNED to "✓ Re-aligned to Room Objects\nPoints snapped back to physical surfaces"
            }
            if (sample == null) {
                active += if (trackingLosses > 0) {
                    WarningKind.TRACKING_LOST to "Tracking lost\n" + trackingAdvice(r?.trackingState)
                } else {
                    WarningKind.FINDING_POSITION to "Finding position…\n" + trackingAdvice(r?.trackingState)
                }
            }
            if (needsRelocalizationCheck) {
                active += WarningKind.RELOCALIZE to "🔍 Aim at Previously Scanned Objects\nHold steady at familiar corners/furniture to restore alignment"
            }
            // If using the fallback depth worker and no landmarks have been triangulated yet, prompt walk forward.
            // Clears immediately once calibrated OR once parallax tracker has acquired landmarks.
            if (mono?.let { it.ready && !it.calibrated } == true && parallaxTracker.verifiedLandmarkCount < 5) {
                active += WarningKind.CALIBRATING_DEPTH to "Calibrating depth\nWalk forward a couple of steps"
            }
            if (lastVerdict == PhotometricGate.Verdict.BLOWN) {
                active += WarningKind.TOO_BRIGHT to "Too bright\nAim away from the window or lamp"
            }
            if (lastVerdict == PhotometricGate.Verdict.TRANSITION) {
                active += WarningKind.LIGHT_TRANSITION to "Adjusting to the light change…"
            }
            if (speedMetresPerSec > MAX_WALK_SPEED_MS) {
                active += WarningKind.TOO_FAST to "Slow down\nWalk slower for sharp frames"
            }
            if (angularRateDegPerSec > MAX_TURN_RATE_DEG_S) {
                active += WarningKind.TURN_SLOWLY to "Turn more slowly"
            }
            warning = pickWarning(nowUi, active)
        }
        // Buzz once when a warning appears: the operator is looking at the room, not the screen.
        if (warning != null && warning != lastWarning) {
            root.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
        }
        lastWarning = warning
        warnBanner.text = warning ?: ""
        warnBanner.setVisible(warning != null)

        // ---- pre-flight controls ----
        val showReticle = !fatal && state == State.PREFLIGHT && preflightStep == 1
        val showWbCard = !fatal && state == State.PREFLIGHT && preflightStep == 1
        val showTiltCard = !fatal && state == State.PREFLIGHT && preflightStep == 2
        reticle.setVisible(showReticle)
        wbCard.setVisible(showWbCard)
        tiltCard.setVisible(showTiltCard)
        if (showTiltCard) {
            tiltGauge.tiltDeg = tiltDownDeg(-gravityUpZ)
            val tilt = tiltGauge.tiltDeg
            tiltValue.text = if (tiltGauge.inBand) "%.0f° · good".format(tilt) else "%.0f°".format(tilt)
            tiltValue.setTextColor(
                color(if (tiltGauge.inBand) R.color.action_ready else R.color.text_secondary)
            )
        }

        // ---- instruction card, button, hint ----
        val frames = writer?.keyframeCount ?: 0
        var title: String
        var body: String
        var label: String
        var hint: String? = null
        var enabled = true
        when {
            fatal -> {
                title = fatalTitle ?: ""
                body = fatalBody ?: ""
                label = primaryButton.text.toString()
                enabled = false
            }
            state == State.IDLE -> {
                title = "Ready when you are"
                body = "Stand in the doorway you would walk in through — the scan starts and ends there."
                label = "Start scan"
                if (!tracking) hint = trackingAdvice(r?.trackingState)
            }
            state == State.PREFLIGHT && preflightStep == 0 -> {
                val elapsedCalibS = (System.nanoTime() - calibrationStartNanos) / 1e9f
                val remainingS = (5.0f - elapsedCalibS).coerceAtLeast(0f)
                val isCalibDone = elapsedCalibS >= 5.0f
                if (!isCalibDone) {
                    title = "⚡ Calibrating Lighting (%.0fs)".format(remainingS)
                    body = "Slowly pan the phone across the room for 5 seconds. The optimal ISO and shutter speed are being calibrated to maximize brightness."
                    label = "Calibrating (%.0fs)…".format(remainingS)
                    enabled = false
                } else {
                    title = "✓ Lighting Calibrated"
                    body = "Exposure optimized for clear visibility. Tap Next to review color balance."
                    label = "Next: Color Balance"
                    enabled = true
                }
            }
            state == State.PREFLIGHT && preflightStep == 1 -> {
                title = "Step 1 of 2 · Color Balance"
                body = "Colour is balanced for this room's light and now held steady. Check it looks right — nudge the sliders if it doesn't — then tap Lock Color."
                label = "Lock Color"
                enabled = true
            }
            state == State.PREFLIGHT -> {
                title = "Step 2 of 2 · Camera Angle"
                body = "Tilt the phone down until the dot sits in the green band (5–20°). Hold this angle to capture floors and room geometry."
                label = "Start 3D Capture"
                if (!tracking) {
                    hint = trackingAdvice(r?.trackingState)
                }
            }
            interrupted -> {
                title = "Scan paused"
                body = "Nothing is lost. Tap resume and carry on from where you stopped."
                label = "Resume scan"
            }
            state == State.SCANNING -> {
                val count = parallaxTracker.verifiedLandmarkCount
                val minReady = count >= 200
                val idealReady = count >= 350

                // Physical tactile haptic pulses on milestone crossings:
                if (count >= 200 && !hapticMilestone200) {
                    hapticMilestone200 = true
                    root.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
                }
                if (count >= 350 && !hapticMilestone350) {
                    hapticMilestone350 = true
                    root.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
                }
                if (count >= 500 && !hapticMilestone500) {
                    hapticMilestone500 = true
                    root.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
                }
                if (loopClosed && !hapticLoopClosed && minReady) {
                    hapticLoopClosed = true
                    root.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
                } else if (!loopClosed) {
                    hapticLoopClosed = false
                }
                when {
                    idealReady && loopClosed -> {
                        title = "🌟 Ideal 3DGS Quality (4K Splats)!"
                        body = "Superb coverage achieved (%d landmarks)! Tap Finish to save your high-density 3D reconstruction.".format(count)
                    }
                    minReady && loopClosed -> {
                        title = "✓ Minimum Room Coverage (200+ pts)"
                        body = "Ready to finish, or keep orbiting objects to hit 350-500 pts for ultra-sharp 4K splat detail!"
                    }
                    idealReady && !loopClosed -> {
                        title = "🎯 3D Coverage Complete (%d pts)".format(count)
                        body = "Room fully mapped! Walk back to your starting doorway (%.1f m away) to close the loop and finish.".format(distanceToStart)
                    }
                    minReady && !loopClosed -> {
                        title = "🎮 Good Progress (%d pts)".format(count)
                        body = "Base coverage unlocked! Keep orbiting unexplored furniture & corners to reach 350–500 pts, then return to doorway."
                    }
                    else -> {
                        title = "✨ Orbit Objects & Corners"
                        body = "Move dynamically around furniture and points of interest! Watch amber dots pop into emerald green as you orbit (%d/500).".format(count)
                    }
                }
                label = if (idealReady) "Finish scan (High Quality)" else "Finish scan"
                enabled = minReady && loopClosed && issues.isEmpty()
                hint = when {
                    !minReady -> "Capture at least 200 points · currently %d/200".format(count)
                    !loopClosed -> "Return to starting doorway to seal loop · %.1f m away".format(distanceToStart)
                    issues.isNotEmpty() -> issues.joinToString(" · ")
                    else -> null
                }
            }
            else -> {
                if (finishedPath == null) {
                    title = "Saving scan…"
                    body = "$frames frames · compressing and writing to disk. Keep the app open until this finishes."
                    label = "Saving…"
                    enabled = false
                } else {
                    title = "Scan saved"
                    body = "$frames frames · saved to $finishedPath"
                    label = "Start new scan"
                }
            }
        }
        if ((System.nanoTime() - sessionErrorAtNanos) / 1e6f < SESSION_ERROR_NOTICE_MS) {
            hint = "AR session is struggling — hold the phone steady"
        }
        instructionTitle.text = title
        instructionBody.text = body
        instructionBody.setVisible(body.isNotEmpty())
        primaryButton.text = label
        primaryButton.isEnabled = enabled
        primaryButton.backgroundTintList = ColorStateList.valueOf(
            color(
                when {
                    !enabled -> R.color.action_disabled
                    scanning && complete && loopClosed && issues.isEmpty() -> R.color.action_ready
                    else -> R.color.action
                }
            )
        )
        primaryButton.setTextColor(color(if (enabled) R.color.text_primary else R.color.text_faint))
        hintText.text = hint ?: ""
        hintText.setVisible(hint != null)

        if (!showDiag) return
        diagText.text = buildString {
            append("state $state  frames $frames kept / $framesSeen seen  [${r?.trackingState}]\n")
            val m = mono
            append("depth: " + when {
                r?.depthSupported == true -> "ARCore"
                m == null -> "point cloud"
                !m.ready -> "model loading"
                !m.calibrated -> "model, calibrating"
                else -> "model ${m.latencyMs}ms/${m.pointsUsed}pt"
            })
            append("  img: ${r?.imageWidth}x${r?.imageHeight}")
            append("  fl: %.0f\n".format(r?.focalX ?: 0f))
            append("iso $currentIso  1/${shutterFractionDenominator(currentShutterNs)}s")
            append("  wb %.2f/%.2f\n".format(wbBias, wbTintBias))
            val p = r?.pipeline
            val focusDistM = if ((p?.latestFocusDistanceDiopters ?: 0f) > 1e-4f) 1f / p!!.latestFocusDistanceDiopters else 0f
            append("focus: %.2fm (%.2fdpt)  af:%d/%d\n".format(
                focusDistM, p?.latestFocusDistanceDiopters ?: 0f, p?.latestAfMode ?: 0, p?.latestAfState ?: 0
            ))
            append("voxels: ${stats?.occupied ?: 0} occ / ${stats?.verified ?: 0} verified")
            append("  free ${stats?.free ?: 0}  occl ${stats?.occluded ?: 0}${if (stats?.gridFull == true) " FULL" else ""}")
            append("  %.1fms\n".format(stats?.lastIntegrationMs ?: 0f))
            append("luma %.0f  var %.1f  %s".format(gate.lastMean, gate.variance, lastVerdict))
            append("   %.2fm/s  %.0f°/s\n".format(speedMetresPerSec, angularRateDegPerSec))
            append("dropped: dark $droppedDark  blown $droppedBlown  trans $droppedTransition")
            append("  motion $droppedMotion  queue ${writer?.droppedQueue?.get() ?: 0}")
        }
    }

    private fun elapsed(): String {
        val seconds = ((System.nanoTime() - sessionStartNanos) / 1_000_000_000L).coerceAtLeast(0)
        return "%d:%02d".format(seconds / 60, seconds % 60)
    }

    override fun onResume() {
        super.onResume()
        if (arCoreInstallRequested && renderer == null) checkArCoreAndProceed()
        // TYPE_GRAVITY is the fused, already-smoothed one; raw accelerometer points the same way
        // while the phone is roughly still, which is what pre-flight asks for anyway.
        val gravity = sensors.getDefaultSensor(Sensor.TYPE_GRAVITY)
            ?: sensors.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gravity?.let { sensors.registerListener(gravityListener, it, SensorManager.SENSOR_DELAY_UI) }
        sensors.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
            ?.let { sensors.registerListener(magneticListener, it, SensorManager.SENSOR_DELAY_UI) }
        renderer?.resume()
        glSurfaceView?.onResume()
    }

    override fun onPause() {
        super.onPause()
        sensors.unregisterListener(gravityListener)
        sensors.unregisterListener(magneticListener)
        glSurfaceView?.onPause()
        // onPause() above blocks until the GL thread parks, so nothing is inside session.update().
        if (state == State.SCANNING) interrupted = true
        renderer?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        renderer?.destroy()
        mono?.shutdown()
        worker?.shutdown()
        writer?.shutdown()
    }

    companion object {
        private const val TAG = "ArScan"

        /** camera2 refusing a change to a session it already closed -- see installCameraRaceGuard. */
        fun isClosedCaptureSessionRace(e: Throwable): Boolean =
            e is IllegalStateException &&
                e.message?.contains("Session has been closed") == true &&
                e.stackTrace.any { it.className.endsWith("CameraCaptureSessionImpl") }

        /** SPEC §2.3: the finish button goes green here. It stays *tappable* below the threshold
         * on purpose -- a scan the operator cannot end is a trap, and an 85% target in a room
         * with a mirrored wardrobe may simply never be reachable. */
        const val COMPLETION_FRACTION = 0.85f

        /** CLAUDE.md's entry-door loop-closure anchor: how close (metres, XZ+Y combined) the
         * operator's pose needs to be to their starting pose before Finish is allowed to act.
         * Unlike COMPLETION_FRACTION this is a hard gate -- walking back to the door is always
         * achievable if you got there in the first place, so it doesn't carry the same "may be
         * unreachable" trap risk. 1 m is generous against a few cm/min of indoor VIO drift while
         * still meaning "actually back near the door", not "somewhere in the room". Calibration
         * knob. */
        const val LOOP_CLOSURE_RADIUS_M = 1.0f

        /** CLAUDE.md's on-device pre-flight validation gate: below this many kept frames, the
         * dataset is too short to reconstruct anything useful from -- catches an accidental or
         * premature Finish tap seconds into a scan, not a real completeness bar (that's
         * COMPLETION_FRACTION, and stays advisory). Calibration knob. */
        const val MIN_KEYFRAMES_FOR_EXPORT = 30

        /** Depth is integrated at ~10 Hz against a 30 Hz render loop (projet.md §4). */
        const val DEPTH_EVERY_N_FRAMES = 3L
        const val IMAGE_EVERY_N_FRAMES = 2L
        const val UI_EVERY_N_FRAMES = 5L

        /** Operator-facing kinematic limits (projet.md §2.4): warn here. */
        const val MAX_WALK_SPEED_MS = 0.4f
        const val MAX_TURN_RATE_DEG_S = 30f

        /** Minimum gap between tap-to-focus triggers (see root.setOnTouchListener): continuous
         * AF is already running, so this only exists to stop rapid incidental touches from
         * making the lens hunt. Calibration knob. */
        const val AUTO_FOCUS_COOLDOWN_NANOS = 1_200_000_000L

        /** Every operator-facing warning stays up at least this long once triggered, queued
         * behind whichever one is currently showing -- most underlying conditions (autofocus,
         * overexposure, speed) clear in a fraction of a second, which made the banner
         * unreadable before this floor existed. */
        const val MIN_WARNING_DISPLAY_NANOS = 3_000_000_000L
        /** Keyframe rejection limits, deliberately above the warning ones: the banner should
         * appear before frames start being thrown away, otherwise the operator's first signal
         * that anything is wrong is a coverage bar that stopped moving. */
        const val MAX_KEYFRAME_SPEED_MS = 0.55f
        const val MAX_KEYFRAME_RATE_DEG_S = 45f

        /** README §4/§5: keep 1 of every 6-10 gate-passed frames. 8 is the midpoint of that
         * range -- calibration knob, move within [6, 10] rather than outside it. */
        const val DECIMATION_STRIDE = 8L

        /** Frame intervals outside this band are a dropped frame or a tracking gap, not motion. */
        const val MIN_KINEMATIC_DT_S = 0.005f
        const val MAX_KINEMATIC_DT_S = 0.5f

        private const val PHONE_HEIGHT_M = 1.35f
        private const val SMOOTH_ALPHA = 0.3f
        private const val PENDING_TTL_MS = 2500f

        /** How long a renderer exception keeps its notice on screen. Long enough to be readable,
         * short enough that a single blip doesn't linger. Calibration knob. */
        private const val SESSION_ERROR_NOTICE_MS = 1500f

        /** ARCore's TrackingFailureReason as something the operator can act on. The raw enum
         * ("PAUSED/INSUFFICIENT_FEATURES") was previously shown as-is, which tells a real estate
         * agent nothing about what to do with the phone in their hand. */
        fun trackingAdvice(trackingState: String?): String = when {
            trackingState == null -> "Starting the camera…"
            trackingState.contains("INSUFFICIENT_LIGHT") -> "Turn on the room lights"
            trackingState.contains("INSUFFICIENT_FEATURES") -> "Point at furniture or a door frame, not a blank wall"
            trackingState.contains("EXCESSIVE_MOTION") -> "Move the phone more slowly"
            trackingState.contains("CAMERA_UNAVAILABLE") -> "Another app is using the camera"
            trackingState.contains("TRACKING") -> "Tracking"
            else -> "Move the phone slowly to find your position"
        }

        fun distance(a: FloatArray, b: FloatArray): Float {
            val dx = a[0] - b[0]
            val dy = a[1] - b[1]
            val dz = a[2] - b[2]
            return sqrt(dx * dx + dy * dy + dz * dz)
        }

        /** Angle between two orientations, degrees. |dot| because q and -q are the same
         * rotation, and without the abs a perfectly still phone occasionally reports 180. */
        /** Angle between two unit vectors. atan2 of the cross/dot rather than acos of the dot:
         * acos loses all its precision exactly where this is used, a few degrees from zero. */
        fun angleBetweenDeg(a: FloatArray, b: FloatArray): Float {
            val cx = a[1] * b[2] - a[2] * b[1]
            val cy = a[2] * b[0] - a[0] * b[2]
            val cz = a[0] * b[1] - a[1] * b[0]
            val cross = sqrt(cx * cx + cy * cy + cz * cz)
            val dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
            return Math.toDegrees(kotlin.math.atan2(cross.toDouble(), dot.toDouble())).toFloat()
        }

        fun quaternionAngleDeg(a: FloatArray, b: FloatArray): Float {
            val dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]).coerceIn(0f, 1f)
            return Math.toDegrees(2.0 * acos(dot.toDouble())).toFloat()
        }

        /** README §4/§5 decimation: true on the Nth gate-passed frame, 1-indexed so the very
         * first frame of a session isn't always kept regardless of stride. */
        fun isDecimationKeyframe(gatePassedCounter: Long, stride: Long = Tunables.decimationStride): Boolean =
            gatePassedCounter % stride == 0L

        /** README §6 step 3: downward tilt of the camera's forward vector, degrees, positive is
         * down. `forwardY` is the world-space forward vector's Y component (Unproject.rotate's
         * output) -- reading it directly is simpler and cheaper than decomposing the pose
         * quaternion into Euler angles for one axis. */
        fun tiltDownDeg(forwardY: Float): Float =
            Math.toDegrees(-kotlin.math.asin(forwardY.coerceIn(-1f, 1f).toDouble())).toFloat()

        /** Gain per channel (red, green, blue) that makes the sampled patch read as neutral grey:
         * decode Y/Cb/Cr back to BT.601 full-range RGB, then solve each channel against the
         * patch's own luma. */
        private fun greyWorldSolve(meanY: Float, meanU: Float, meanV: Float): FloatArray {
            val cb = meanU - 128f
            val cr = meanV - 128f
            val r = (meanY + 1.402f * cr).coerceAtLeast(1f)
            val g = (meanY - 0.344136f * cb - 0.714136f * cr).coerceAtLeast(1f)
            val b = (meanY + 1.772f * cb).coerceAtLeast(1f)
            val target = meanY.coerceAtLeast(1f)
            return floatArrayOf(target / r, target / g, target / b)
        }

        /** The operator's `wbSlider`/`wbTintSlider` trim (-1..1 each) on top of the converged auto
         * gains; 0/0 leaves the auto read untouched. Order matches `RggbChannelVector`
         * (red, greenEven, greenOdd, blue). */
        fun whiteBalanceTrim(
            autoGains: FloatArray,
            warmCoolBias: Float = 0f,
            tintBias: Float = 0f,
        ): FloatArray {
            val red = (autoGains[0] + WB_MANUAL_RANGE * warmCoolBias).coerceIn(0.5f, 4f)
            val green = (autoGains[1] - WB_MANUAL_RANGE * tintBias).coerceIn(0.5f, 4f)
            val blue = (autoGains[2] - WB_MANUAL_RANGE * warmCoolBias).coerceIn(0.5f, 4f)
            return floatArrayOf(red, green, green, blue)
        }

        /** One-shot grey-world solve plus trim: where `whiteBalanceStep`'s loop converges to. */
        fun whiteBalanceGains(
            meanY: Float,
            meanU: Float,
            meanV: Float,
            warmCoolBias: Float = 0f,
            tintBias: Float = 0f,
        ): FloatArray = whiteBalanceTrim(greyWorldSolve(meanY, meanU, meanV), warmCoolBias, tintBias)

        private const val WB_MANUAL_RANGE = 0.5f

        /** Joint shutter+ISO metering (README §4): indoor rooms have far less light than outdoor
         * scenes, and there's no free knob to fix that -- slower shutter costs motion-blur risk,
         * higher ISO costs grain, and the right answer is whichever point on that curve gets this
         * *specific* scene's luma to target with the least of both. Shutter moves first (fast
         * end 1/500s toward slow end 1/50s): it doesn't cost anything unconditionally the way ISO
         * grain does, and the kinematic guard (README §6) already screens for the walking speed
         * that would turn a slow shutter into visible blur, so this app's fixed-walkthrough-pace
         * use case can afford to spend that range before ISO has to move past its own baseline.
         * Proportional, not integrating -- gain scales roughly linearly with iso*shutterTime, so
         * one ratio step gets most of the way to `TARGET_LUMA` and a few pre-flight frames
         * converge the rest of the way, same style as the ISO-only metering this replaced. */
        fun meteredExposure(measuredLuma: Float, currentIso: Int, currentShutterNs: Long): Exposure {
            val luma = measuredLuma.coerceAtLeast(1f)
            if (kotlin.math.abs(luma - TARGET_LUMA) < LUMA_DEADZONE) return Exposure(currentIso, currentShutterNs)

            val ratio = (TARGET_LUMA / luma).coerceIn(0.1f, 10.0f)
            val dampedRatio = 1f + (ratio - 1f) * CONVERGENCE_GAIN

            // Joint tradeoff: Shutter expands first (free of grain), then ISO kicks in
            val idealShutterNs = (currentShutterNs * dampedRatio).roundToLong()
                .coerceIn(CameraPipeline.FASTEST_SHUTTER_NS, CameraPipeline.SLOWEST_SHUTTER_NS)
            val shutterContribution = idealShutterNs.toFloat() / currentShutterNs
            val isoRatio = dampedRatio / shutterContribution
            val computedIso = (currentIso * isoRatio).roundToInt()
                .coerceIn(CameraPipeline.MIN_ISO, CameraPipeline.MAX_ISO)

            return Exposure(computedIso, idealShutterNs)
        }

        /** Fraction of the computed correction applied per frame; see the delayed-feedback note
         * in `meteredExposure`. */
        private const val CONVERGENCE_GAIN = 0.60f

        /** Luma units of tolerance around `TARGET_LUMA` before metering moves at all. */
        private const val LUMA_DEADZONE = 8f

        data class Exposure(val iso: Int, val shutterNs: Long)

        /** HUD-only: nearest whole-number denominator for a shutter time in ns, e.g. 1/125s. */
        fun shutterFractionDenominator(shutterNs: Long): Int =
            (1_000_000_000f / shutterNs).roundToInt()

        private const val TARGET_LUMA = 175f

        /** CLAUDE.md's on-device pre-flight validation gate before upload: post-scan
         * completeness/quality issues that block Finish outright, as opposed to
         * COMPLETION_FRACTION which only nudges. Both conditions here mean data that cannot be
         * reconstructed from, not merely "could be better". */
        fun validationIssues(gridFull: Boolean, keyframeCount: Int): List<String> {
            val issues = mutableListOf<String>()
            // Operator-facing copy: these are shown under the finish button as the reason it
            // isn't available, so they say what to do, not which data structure is unhappy.
            if (gridFull) issues += "The mapped area hit its limit — some of the home wasn't captured"
            if (keyframeCount < MIN_KEYFRAMES_FOR_EXPORT) {
                issues += "Only $keyframeCount frame(s) captured so far — keep walking"
            }
            return issues
        }
    }
}
