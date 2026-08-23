package com.glomehometour.capture

import android.Manifest
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Step-1 proof of concept: ARCore-tracked camera -> ZipDepth depth inference -> HUD overlay.
 * ARCore supplies pose + sparse point cloud, used by DepthCalibration to anchor the depth
 * scale to real metric distance so it stays consistent across frames and camera motion.
 */
class MainActivity : AppCompatActivity() {

    // Wall-clock gate targets ~4Hz — see ArCoreCameraSource, which now owns the capture loop.
    // ponytail: actual rate = max(this, inference latency + overhead) — DepthEngine's inference
    // latency is the ceiling if it lands under 4Hz on-device.
    private val targetIntervalMs = 250L

    private lateinit var depthEngine: DepthEngine
    private lateinit var overlayView: OverlayView
    private lateinit var previewView: ImageView
    private lateinit var hudText: TextView
    private var cameraSource: ArCoreCameraSource? = null

    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val inferenceBusy = AtomicBoolean(false)
    private var lastInferenceAtMs = 0L
    private var lastLatencyMs = 0L

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else hudText.text = "camera permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        hudText = findViewById(R.id.hudText)
        depthEngine = DepthEngine(this)

        requestCameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val source = ArCoreCameraSource(
            context = this,
            targetIntervalMs = targetIntervalMs,
            onFrame = { bitmap, rotationDegrees, pose -> onFrame(bitmap, rotationDegrees, pose) },
            onError = { e -> runOnUiThread { hudText.text = "ARCore error: ${e.message}" } },
        )
        cameraSource = source
        source.start()
    }

    private fun onFrame(rawBitmap: Bitmap, rotationDegrees: Int, pose: ArCoreCameraSource.PoseSample?) {
        if (!inferenceBusy.compareAndSet(false, true)) return

        val fullW = rawBitmap.width
        val fullH = rawBitmap.height
        val modelInput = cropToModelInput(rawBitmap, rotationDegrees)
        runOnUiThread { previewView.setImageBitmap(rawBitmap) }

        inferenceExecutor.execute {
            try {
                val result = depthEngine.run(modelInput)
                lastLatencyMs = result.latencyMs

                val edges = CannyEdge.detect(modelInput, DepthEngine.SIZE)

                val calib = pose?.let {
                    DepthCalibration.fit(
                        disparity = result.depth,
                        modelSize = DepthEngine.SIZE,
                        cameraPose = it.pose,
                        points = it.points,
                        numPoints = it.numPoints,
                        fx = it.fx,
                        fy = it.fy,
                        cx = it.cx,
                        cy = it.cy,
                        sensorWidth = fullW,
                        sensorHeight = fullH,
                        rotationDegrees = rotationDegrees,
                    )
                }

                val doneAtMs = System.currentTimeMillis()
                val hz = if (lastInferenceAtMs == 0L) 0.0 else 1000.0 / (doneAtMs - lastInferenceAtMs)
                lastInferenceAtMs = doneAtMs

                runOnUiThread {
                    val bounds = calib?.let { Pair(it.nearRawBound, it.farRawBound) }
                    overlayView.updateDepth(result.depth, DepthEngine.SIZE, edges, bounds, pose?.pose)
                    val calibText = if (calib != null) {
                        "calib: ${calib.pointsUsed} pts"
                    } else {
                        "calib: none [${cameraSource?.lastTrackingState}]"
                    }
                    hudText.text = "latency: %d ms   rate: %.1f Hz   %s".format(result.latencyMs, hz, calibText)
                }
            } finally {
                inferenceBusy.set(false)
            }
        }
    }

    /** Rotate to upright, center-crop to square, scale to the model's SIZE x SIZE input. */
    private fun cropToModelInput(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        val side = minOf(rotated.width, rotated.height)
        val x = (rotated.width - side) / 2
        val y = (rotated.height - side) / 2
        val cropped = Bitmap.createBitmap(rotated, x, y, side, side)

        return Bitmap.createScaledBitmap(cropped, DepthEngine.SIZE, DepthEngine.SIZE, true)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraSource?.stop()
        inferenceExecutor.shutdown()
        depthEngine.close()
    }
}
