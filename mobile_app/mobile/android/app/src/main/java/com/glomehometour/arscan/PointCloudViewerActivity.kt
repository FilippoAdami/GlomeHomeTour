package com.glomehometour.arscan

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.widget.TextView
import android.widget.Toast
import java.io.InputStream
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Standalone 3D Point Cloud Inspection Viewer.
 *
 * Allows inspecting the 3D geometry of any captured or processed scan anytime, anywhere:
 * - 3D Orbit Mode: Intuitive touch rotation, pinch-zoom, and pan on an opaque dark background.
 * - Top-Down Mode: Bird's-eye orthographic floorplan view for boundary inspection.
 * - Immersive Gyroscope Mode: 1:1 real-world physical look-around via rotation vector sensor
 *   paired with touch walking navigation.
 */
class PointCloudViewerActivity : Activity(), SensorEventListener {

    companion object {
        private const val TAG = "PointCloudViewer"
        const val EXTRA_SESSION_NAME = "session_name"
        const val EXTRA_SCAN_FOLDER = "scan_folder"
        const val EXTRA_ROOM_NAME = "room_name"

        fun start(context: Context, sessionName: String, scanFolder: String = "scan_0", roomName: String? = null) {
            val intent = Intent(context, PointCloudViewerActivity::class.java).apply {
                putExtra(EXTRA_SESSION_NAME, sessionName)
                putExtra(EXTRA_SCAN_FOLDER, scanFolder)
                putExtra(EXTRA_ROOM_NAME, roomName)
            }
            context.startActivity(intent)
        }
    }

    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var renderer: PointCloudRenderer
    private lateinit var titleText: TextView
    private lateinit var statsText: TextView
    private lateinit var modeHintText: TextView
    private lateinit var modeOrbitButton: TextView
    private lateinit var modeTopDownButton: TextView
    private lateinit var modeGyroButton: TextView

    private var sensorManager: SensorManager? = null
    private var rotationSensor: Sensor? = null
    private val rawRotationMatrix = FloatArray(16)

    private lateinit var scaleDetector: ScaleGestureDetector
    private var prevTouchX = 0f
    private var prevTouchY = 0f
    private var isMultiTouch = false

    private var sessionName: String = ""
    private var scanFolder: String = "scan_0"
    private var roomName: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_point_cloud_viewer)

        sessionName = intent.getStringExtra(EXTRA_SESSION_NAME) ?: ""
        scanFolder = intent.getStringExtra(EXTRA_SCAN_FOLDER) ?: "scan_0"
        roomName = intent.getStringExtra(EXTRA_ROOM_NAME)

        titleText = findViewById(R.id.titleText)
        statsText = findViewById(R.id.statsText)
        modeHintText = findViewById(R.id.modeHintText)
        modeOrbitButton = findViewById(R.id.modeOrbitButton)
        modeTopDownButton = findViewById(R.id.modeTopDownButton)
        modeGyroButton = findViewById(R.id.modeGyroButton)
        val backButton: TextView = findViewById(R.id.backButton)
        val resetButton: TextView = findViewById(R.id.resetButton)
        val btnPointSize: TextView = findViewById(R.id.btnPointSize)
        val btnToggleTraj: TextView = findViewById(R.id.btnToggleTraj)

        titleText.text = if (!roomName.isNullOrEmpty()) "$roomName · 3D View" else "Point Cloud ($sessionName)"

        backButton.setOnClickListener { finish() }
        resetButton.setOnClickListener {
            renderer.resetOrbit()
            glSurfaceView.requestRender()
        }

        btnPointSize.setOnClickListener {
            when (renderer.pointScale) {
                72.0f -> {
                    renderer.pointScale = 105.0f
                    btnPointSize.text = "⚪ Max"
                }
                105.0f -> {
                    renderer.pointScale = 42.0f
                    btnPointSize.text = "⚪ Fine"
                }
                else -> {
                    renderer.pointScale = 72.0f
                    btnPointSize.text = "⚪ Thick"
                }
            }
            glSurfaceView.requestRender()
        }

        btnToggleTraj.setOnClickListener {
            renderer.showTrajectory = !renderer.showTrajectory
            if (renderer.showTrajectory) {
                btnToggleTraj.text = "🛤 Track"
                btnToggleTraj.setTextColor(getColor(R.color.text_primary))
                btnToggleTraj.alpha = 1.0f
            } else {
                btnToggleTraj.text = "🛤 Track"
                btnToggleTraj.setTextColor(getColor(R.color.text_faint))
                btnToggleTraj.alpha = 0.5f
            }
            glSurfaceView.requestRender()
        }

        modeOrbitButton.setOnClickListener { selectMode(PointCloudRenderer.ViewMode.ORBIT) }
        modeTopDownButton.setOnClickListener { selectMode(PointCloudRenderer.ViewMode.TOP_DOWN) }
        modeGyroButton.setOnClickListener { selectMode(PointCloudRenderer.ViewMode.IMMERSIVE_GYRO) }

        setupGlView()
        setupSensors()
        loadPointCloudAsync()
    }

    private fun setupGlView() {
        glSurfaceView = findViewById(R.id.glSurfaceView)
        glSurfaceView.setEGLContextClientVersion(2)
        renderer = PointCloudRenderer()
        glSurfaceView.setRenderer(renderer)
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY

        scaleDetector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                if (renderer.viewMode == PointCloudRenderer.ViewMode.TOP_DOWN) {
                    renderer.topDownZoom = (renderer.topDownZoom * detector.scaleFactor).coerceIn(0.25f, 8f)
                    glSurfaceView.requestRender()
                    return true
                } else if (renderer.viewMode == PointCloudRenderer.ViewMode.ORBIT) {
                    renderer.orbitDistance = (renderer.orbitDistance / detector.scaleFactor).coerceIn(0.5f, 40f)
                    glSurfaceView.requestRender()
                    return true
                }
                return false
            }
        })

        glSurfaceView.setOnTouchListener { _, event ->
            scaleDetector.onTouchEvent(event)
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    prevTouchX = event.x
                    prevTouchY = event.y
                    isMultiTouch = false
                }
                MotionEvent.ACTION_POINTER_DOWN -> {
                    isMultiTouch = true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.x - prevTouchX
                    val dy = event.y - prevTouchY

                    if (renderer.viewMode == PointCloudRenderer.ViewMode.IMMERSIVE_GYRO) {
                        var lx = renderer.currentLookDir[0]
                        var lz = renderer.currentLookDir[2]
                        var len = sqrt(lx * lx + lz * lz)
                        if (len < 0.1f) {
                            // When tilted steep up/down, walk in direction of top of phone
                            lx = renderer.currentUpDir[0]
                            lz = renderer.currentUpDir[2]
                            len = sqrt(lx * lx + lz * lz)
                        }

                        if (isMultiTouch || event.pointerCount > 1) {
                            // 2-finger gesture in Gyro Walk: vertical drag adjusts eye height, horizontal drag strafes
                            renderer.walkY = (renderer.walkY - dy * 0.004f).coerceIn(
                                renderer.minY + 0.2f,
                                renderer.maxY + 0.5f
                            )
                            if (len > 0.001f) {
                                val strafeX = -lz / len
                                val strafeZ = lx / len
                                val speed = 0.005f
                                renderer.walkX = (renderer.walkX + strafeX * dx * speed).coerceIn(
                                    renderer.minX - 0.5f,
                                    renderer.maxX + 0.5f
                                )
                                renderer.walkZ = (renderer.walkZ + strafeZ * dx * speed).coerceIn(
                                    renderer.minZ - 0.5f,
                                    renderer.maxZ + 0.5f
                                )
                            }
                        } else {
                            // 1-finger gesture in Gyro Walk:
                            // Vertical drag (-dy): walk forward/backward along current horizontal look direction
                            if (len > 0.001f) {
                                val fwdX = lx / len
                                val fwdZ = lz / len
                                val speed = 0.005f
                                renderer.walkX = (renderer.walkX + fwdX * (-dy) * speed).coerceIn(
                                    renderer.minX - 0.5f,
                                    renderer.maxX + 0.5f
                                )
                                renderer.walkZ = (renderer.walkZ + fwdZ * (-dy) * speed).coerceIn(
                                    renderer.minZ - 0.5f,
                                    renderer.maxZ + 0.5f
                                )
                            }

                            // Horizontal drag (dx): adjust yaw heading offset (steer view horizontally)
                            renderer.gyroYawOffset -= dx * 0.25f
                        }
                    } else if (isMultiTouch || event.pointerCount > 1) {
                        // 2-finger camera pan
                        if (renderer.viewMode == PointCloudRenderer.ViewMode.TOP_DOWN) {
                            val aspect = glSurfaceView.width.toFloat() / glSurfaceView.height.coerceAtLeast(1)
                            val spanX = renderer.maxX - renderer.minX
                            val spanZ = renderer.maxZ - renderer.minZ
                            val diagonal = sqrt(spanX * spanX + spanZ * spanZ).coerceAtLeast(1f)
                            val panFactor = diagonal * 1.1f / minOf(1f, aspect) /
                                glSurfaceView.height.coerceAtLeast(1) / renderer.topDownZoom
                            val yaw = Math.toRadians(renderer.orbitYaw.toDouble()).toFloat()
                            renderer.panX += (-dx * cos(yaw) + dy * sin(yaw)) * panFactor
                            renderer.panZ += (-dx * sin(yaw) - dy * cos(yaw)) * panFactor
                        } else {
                            val panFactor = 0.0025f * renderer.orbitDistance
                            renderer.panX -= dx * panFactor
                            renderer.panY += dy * panFactor
                        }
                        glSurfaceView.requestRender()
                    } else {
                        // 1-finger orbit rotation
                        if (renderer.viewMode == PointCloudRenderer.ViewMode.ORBIT) {
                            renderer.orbitYaw = (renderer.orbitYaw + dx * 0.4f) % 360f
                            renderer.orbitPitch = (renderer.orbitPitch - dy * 0.4f).coerceIn(-89.9f, 89.9f)
                        } else if (renderer.viewMode == PointCloudRenderer.ViewMode.TOP_DOWN) {
                            renderer.orbitYaw = (renderer.orbitYaw + dx * 0.4f) % 360f
                        }
                        glSurfaceView.requestRender()
                    }

                    prevTouchX = event.x
                    prevTouchY = event.y
                }
            }
            true
        }
    }

    private fun setupSensors() {
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        rotationSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensorManager?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    }

    private fun selectMode(mode: PointCloudRenderer.ViewMode) {
        renderer.viewMode = mode
        val activeBg = getDrawable(R.drawable.pill)
        val clearBg = null

        modeOrbitButton.background = if (mode == PointCloudRenderer.ViewMode.ORBIT) activeBg else clearBg
        modeOrbitButton.setTextColor(getColor(if (mode == PointCloudRenderer.ViewMode.ORBIT) R.color.text_primary else R.color.text_secondary))

        modeTopDownButton.background = if (mode == PointCloudRenderer.ViewMode.TOP_DOWN) activeBg else clearBg
        modeTopDownButton.setTextColor(getColor(if (mode == PointCloudRenderer.ViewMode.TOP_DOWN) R.color.text_primary else R.color.text_secondary))

        modeGyroButton.background = if (mode == PointCloudRenderer.ViewMode.IMMERSIVE_GYRO) activeBg else clearBg
        modeGyroButton.setTextColor(getColor(if (mode == PointCloudRenderer.ViewMode.IMMERSIVE_GYRO) R.color.text_primary else R.color.text_secondary))

        when (mode) {
            PointCloudRenderer.ViewMode.ORBIT -> {
                modeHintText.text = "1-finger rotate • 2-finger pinch zoom • 2-finger pan"
                glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
                glSurfaceView.requestRender()
            }
            PointCloudRenderer.ViewMode.TOP_DOWN -> {
                modeHintText.text = "Bird's-eye floorplan view • 2-finger pinch to zoom"
                glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
                glSurfaceView.requestRender()
            }
            PointCloudRenderer.ViewMode.IMMERSIVE_GYRO -> {
                modeHintText.text = "📱 Tilt and rotate phone to look around • Drag to walk"
                glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
            }
        }
    }

    private fun loadPointCloudAsync() {
        Thread {
            try {
                val plyStream: InputStream? = ScanProcessor.openPointCloudStream(this, sessionName, scanFolder)
                val acc = ScanPointCloudAccumulator(maxPoints = 65000)

                var loadedCount = 0
                if (plyStream != null) {
                    plyStream.use { loadedCount = acc.readBinaryPly(it) }
                }
                android.util.Log.i(TAG, "loadPointCloudAsync: sessionName=$sessionName, scanFolder=$scanFolder, plyStream=$plyStream, loadedCount=$loadedCount")

                // Try to load camera trajectory from trajectory.csv if present
                val trajPoints = loadTrajectoryPoints()
                if (trajPoints.isNotEmpty()) {
                    for (i in 0 until trajPoints.size / 3) {
                        acc.addTrajectoryPoint(trajPoints[i * 3], trajPoints[i * 3 + 1], trajPoints[i * 3 + 2])
                    }
                }
                acc.clipToTrajectory()
                loadedCount = acc.pointCount

                val bounds = FloatArray(6)
                acc.getBounds(bounds)
                val center = FloatArray(3)
                acc.getCenter(center)

                runOnUiThread {
                    if (loadedCount > 0) {
                        renderer.setBounds(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
                        renderer.setPointCloud(
                            acc.pointCoords,
                            acc.pointCount,
                            acc.minY,
                            acc.maxY,
                            center[0],
                            center[1],
                            center[2],
                        )
                        if (acc.trajectoryCount > 0) {
                            renderer.setTrajectory(acc.trajectoryCoords, acc.trajectoryCount)
                        }
                        renderer.resetOrbit()
                        statsText.text = "${"%,d".format(acc.pointCount)} 3D points • ${acc.trajectoryCount} camera poses"
                        glSurfaceView.requestRender()
                    } else {
                        statsText.text = "No 3D point cloud found for this scan."
                        Toast.makeText(this, "Point cloud preview not available for $sessionName", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load point cloud for $sessionName", e)
                runOnUiThread {
                    statsText.text = "Error reading point cloud."
                }
            }
        }.start()
    }

    private fun loadTrajectoryPoints(): FloatArray {
        try {
            val csv = ScanProcessor.readTrajectoryCsv(this, sessionName, scanFolder) ?: return FloatArray(0)
            return trajectoryPointsFromCsv(csv)
        } catch (e: Exception) {
            Log.w(TAG, "Could not load trajectory for $sessionName", e)
        }
        return FloatArray(0)
    }

    override fun onResume() {
        super.onResume()
        glSurfaceView.onResume()
        rotationSensor?.let {
            sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager?.unregisterListener(this)
        glSurfaceView.onPause()
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return
        if (event.sensor.type == Sensor.TYPE_GAME_ROTATION_VECTOR || event.sensor.type == Sensor.TYPE_ROTATION_VECTOR) {
            if (renderer.viewMode == PointCloudRenderer.ViewMode.IMMERSIVE_GYRO) {
                SensorManager.getRotationMatrixFromVector(rawRotationMatrix, event.values)
                renderer.updateSensorRotation(rawRotationMatrix)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
