package com.glomehometour.arscan

import android.hardware.SensorManager
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.*

/**
 * Ultra-optimized OpenGL ES 2.0 renderer for standalone 3D point cloud inspection.
 *
 * Designed for maximum mobile fluidity and zero battery waste:
 * 1. Single-pass VBO caching in GPU VRAM (GL_STATIC_DRAW).
 * 2. On-demand rendering (RENDERMODE_WHEN_DIRTY) in Orbit and Top-Down modes.
 * 3. Zero-allocation loop: all matrices and vectors are pre-allocated in-place.
 * 4. Height-normalized color spectrum (Floor -> Ceiling) and trajectory line.
 * 5. Supports Orbit, Top-Down, and Immersive Gyroscope modes.
 */
class PointCloudRenderer : GLSurfaceView.Renderer {

    enum class ViewMode {
        ORBIT,
        TOP_DOWN,
        IMMERSIVE_GYRO,
    }

    var viewMode: ViewMode = ViewMode.ORBIT

    // Geometry data
    private var pointVbo = 0
    private var trajVbo = 0
    private var pointCount = 0
    private var trajCount = 0

    // Geometry bounds
    var minX = -10f; private set
    var maxX = 10f; private set
    var minY = 0f; private set
    var maxY = 1f; private set
    var minZ = -10f; private set
    var maxZ = 10f; private set
    var centerX = 0f; private set
    var centerY = 0f; private set
    var centerZ = 0f; private set

    // Orbit parameters
    var orbitYaw = 0f
    var orbitPitch = 30f
    var orbitDistance = 5f
    var panX = 0f
    var panY = 0f
    var panZ = 0f
    var topDownZoom = 1f

    // Immersive camera position (walk translation inside point cloud)
    var walkX = 0f
    var walkY = 1.6f // Eye height
    var walkZ = 0f

    // Device rotation matrix from SensorManager
    private val sensorRotationMatrix = FloatArray(16).apply { Matrix.setIdentityM(this, 0) }
    private val remapMatrix = FloatArray(16)

    // Transformation matrices
    private val projMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val mvpMatrix = FloatArray(16)
    private val eyePos = FloatArray(3)

    // Display controls
    var showTrajectory: Boolean = true
    var pointScale: Float = 72.0f // Thick point size default
    var gyroYawOffset: Float = 0f // Degrees horizontal steering offset
    val currentLookDir = floatArrayOf(0f, 0f, -1f)
    val currentUpDir = floatArrayOf(0f, 1f, 0f)

    // Shaders
    private var pointProgram = 0
    private var trajProgram = 0

    private var surfaceWidth = 1
    private var surfaceHeight = 1

    private val pendingPoints = java.util.concurrent.atomic.AtomicReference<FloatBuffer?>(null)
    private val pendingTraj = java.util.concurrent.atomic.AtomicReference<FloatBuffer?>(null)

    companion object {
        private const val POINT_VERTEX_SHADER = """
            uniform mat4 u_MVP;
            uniform float u_MinY;
            uniform float u_MaxY;
            uniform float u_PointScale;
            attribute vec3 a_Position;
            varying float v_NormalizedHeight;

            void main() {
                gl_Position = u_MVP * vec4(a_Position, 1.0);
                float span = max(u_MaxY - u_MinY, 0.05);
                v_NormalizedHeight = clamp((a_Position.y - u_MinY) / span, 0.0, 1.0);
                float dist = max(gl_Position.w, 0.3);
                gl_PointSize = clamp(u_PointScale / dist, 6.0, 96.0);
            }
        """

        private const val POINT_FRAGMENT_SHADER = """
            precision mediump float;
            varying float v_NormalizedHeight;

            vec3 heightColor(float t) {
                vec3 c0 = vec3(0.08, 0.20, 0.65); // Deep navy (floor)
                vec3 c1 = vec3(0.00, 0.75, 0.90); // Cyan
                vec3 c2 = vec3(0.12, 0.85, 0.40); // Emerald green (eye level)
                vec3 c3 = vec3(0.95, 0.85, 0.20); // Warm yellow
                vec3 c4 = vec3(0.95, 0.35, 0.25); // Coral red (ceiling)

                if (t < 0.25) return mix(c0, c1, t / 0.25);
                if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
                if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
                return mix(c3, c4, (t - 0.75) / 0.25);
            }

            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                float distSq = dot(coord, coord);
                if (distSq > 0.25) discard;
                // Thicker solid core with smooth anti-aliased edge
                float alpha = 1.0 - smoothstep(0.04, 0.25, distSq);
                gl_FragColor = vec4(heightColor(v_NormalizedHeight), alpha);
            }
        """

        private const val TRAJ_VERTEX_SHADER = """
            uniform mat4 u_MVP;
            attribute vec3 a_Position;
            void main() {
                gl_Position = u_MVP * vec4(a_Position, 1.0);
            }
        """

        private const val TRAJ_FRAGMENT_SHADER = """
            precision mediump float;
            void main() {
                // Subtle, elegant semi-transparent gold line
                gl_FragColor = vec4(0.95, 0.78, 0.28, 0.38);
            }
        """
    }

    fun setBounds(minX: Float, maxX: Float, minY: Float, maxY: Float, minZ: Float, maxZ: Float) {
        this.minX = minX
        this.maxX = maxX
        this.minY = minY
        this.maxY = maxY
        this.minZ = minZ
        this.maxZ = maxZ
    }

    /**
     * Uploads point cloud data in background and prepares for GPU upload.
     */
    fun setPointCloud(points: FloatArray, count: Int, minY: Float, maxY: Float, cx: Float, cy: Float, cz: Float) {
        this.minY = minY
        this.maxY = maxY
        this.centerX = cx
        this.centerY = cy
        this.centerZ = cz

        this.walkX = cx
        this.walkY = if (minY.isFinite() && maxY.isFinite() && maxY > minY) {
            (minY + 1.55f).coerceIn(minY + 0.3f, maxY - 0.2f)
        } else {
            cy
        }
        this.walkZ = cz
        this.gyroYawOffset = 0f

        val buf = ByteBuffer.allocateDirect(count * 3 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buf.put(points, 0, count * 3)
        buf.position(0)
        pointCount = count
        pendingPoints.set(buf)
    }

    /**
     * Uploads trajectory breadcrumbs.
     */
    fun setTrajectory(traj: FloatArray, count: Int) {
        val buf = ByteBuffer.allocateDirect(count * 3 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        buf.put(traj, 0, count * 3)
        buf.position(0)
        trajCount = count
        pendingTraj.set(buf)
    }

    fun updateSensorRotation(matrix16: FloatArray) {
        System.arraycopy(matrix16, 0, sensorRotationMatrix, 0, 16)
    }

    fun resetOrbit() {
        orbitYaw = 0f
        orbitPitch = 25f
        orbitDistance = max(3f, max(maxY - minY, 5f) * 1.5f)
        panX = 0f
        panY = 0f
        panZ = 0f
        topDownZoom = 1f
        walkX = centerX
        walkY = if (minY.isFinite() && maxY.isFinite() && maxY > minY) {
            (minY + 1.55f).coerceIn(minY + 0.3f, maxY - 0.2f)
        } else {
            centerY
        }
        walkZ = centerZ
        gyroYawOffset = 0f
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.043f, 0.055f, 0.078f, 1.0f) // #0B0E14
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthFunc(GLES20.GL_LEQUAL)

        // Compile point program
        pointProgram = createProgram(POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER)
        trajProgram = createProgram(TRAJ_VERTEX_SHADER, TRAJ_FRAGMENT_SHADER)

        val buffers = IntArray(2)
        GLES20.glGenBuffers(2, buffers, 0)
        pointVbo = buffers[0]
        trajVbo = buffers[1]
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        surfaceWidth = width
        surfaceHeight = height
        GLES20.glViewport(0, 0, width, height)
        val aspect = width.toFloat() / max(1, height).toFloat()
        Matrix.perspectiveM(projMatrix, 0, 60f, aspect, 0.1f, 100f)
    }

    override fun onDrawFrame(gl: GL10?) {
        // Upload any pending geometry to VBOs
        val pBuf = pendingPoints.getAndSet(null)
        if (pBuf != null) {
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, pointVbo)
            GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, pointCount * 3 * 4, pBuf, GLES20.GL_STATIC_DRAW)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        }
        val tBuf = pendingTraj.getAndSet(null)
        if (tBuf != null) {
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, trajVbo)
            GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, trajCount * 3 * 4, tBuf, GLES20.GL_STATIC_DRAW)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        }

        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        if (viewMode == ViewMode.TOP_DOWN) {
            val aspect = surfaceWidth.toFloat() / max(1, surfaceHeight).toFloat()
            val spanX = maxX - minX
            val spanZ = maxZ - minZ
            val diagonal = sqrt(spanX * spanX + spanZ * spanZ).coerceAtLeast(1f)
            val halfHeight = diagonal * 0.55f / min(1f, aspect) / topDownZoom
            Matrix.orthoM(projMatrix, 0, -halfHeight * aspect, halfHeight * aspect, -halfHeight, halfHeight, 0.1f, 100f)
        } else {
            val aspect = surfaceWidth.toFloat() / max(1, surfaceHeight).toFloat()
            Matrix.perspectiveM(projMatrix, 0, 60f, aspect, 0.1f, 100f)
        }

        computeViewMatrix()
        Matrix.multiplyMM(mvpMatrix, 0, projMatrix, 0, viewMatrix, 0)

        // Draw points
        if (pointCount > 0 && pointVbo != 0) {
            GLES20.glEnable(GLES20.GL_BLEND)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glUseProgram(pointProgram)

            GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(pointProgram, "u_MVP"), 1, false, mvpMatrix, 0)
            GLES20.glUniform1f(GLES20.glGetUniformLocation(pointProgram, "u_MinY"), minY)
            GLES20.glUniform1f(GLES20.glGetUniformLocation(pointProgram, "u_MaxY"), maxY)
            GLES20.glUniform1f(GLES20.glGetUniformLocation(pointProgram, "u_PointScale"), pointScale)

            val posLoc = GLES20.glGetAttribLocation(pointProgram, "a_Position")
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, pointVbo)
            GLES20.glEnableVertexAttribArray(posLoc)
            GLES20.glVertexAttribPointer(posLoc, 3, GLES20.GL_FLOAT, false, 12, 0)

            GLES20.glDrawArrays(GLES20.GL_POINTS, 0, pointCount)

            GLES20.glDisableVertexAttribArray(posLoc)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        }

        // Draw trajectory line
        if (showTrajectory && trajCount > 1 && trajVbo != 0) {
            GLES20.glEnable(GLES20.GL_BLEND)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glLineWidth(2.0f)
            GLES20.glUseProgram(trajProgram)
            GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(trajProgram, "u_MVP"), 1, false, mvpMatrix, 0)

            val posLoc = GLES20.glGetAttribLocation(trajProgram, "a_Position")
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, trajVbo)
            GLES20.glEnableVertexAttribArray(posLoc)
            GLES20.glVertexAttribPointer(posLoc, 3, GLES20.GL_FLOAT, false, 12, 0)

            GLES20.glDrawArrays(GLES20.GL_LINE_STRIP, 0, trajCount)

            GLES20.glDisableVertexAttribArray(posLoc)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        }
    }

    private fun computeViewMatrix() {
        when (viewMode) {
            ViewMode.ORBIT -> {
                val yawRad = Math.toRadians(orbitYaw.toDouble()).toFloat()
                val pitchRad = Math.toRadians(orbitPitch.coerceIn(-89.9f, 89.9f).toDouble()).toFloat()

                val cosP = cos(pitchRad)
                val sinP = sin(pitchRad)
                val cosY = cos(yawRad)
                val sinY = sin(yawRad)

                eyePos[0] = centerX + orbitDistance * cosP * sinY + panX
                eyePos[1] = centerY + orbitDistance * sinP + panY
                eyePos[2] = centerZ + orbitDistance * cosP * cosY

                val upY = if (abs(orbitPitch) > 89.5f) 0f else 1f
                val upZ = when {
                    orbitPitch > 89.5f -> -1f
                    orbitPitch < -89.5f -> 1f
                    else -> 0f
                }
                Matrix.setLookAtM(
                    viewMatrix, 0,
                    eyePos[0], eyePos[1], eyePos[2],
                    centerX + panX, centerY + panY, centerZ,
                    0f, upY, upZ,
                )
            }
            ViewMode.TOP_DOWN -> {
                val yawRad = Math.toRadians(orbitYaw.toDouble()).toFloat()
                eyePos[0] = centerX + panX
                eyePos[1] = centerY + max(orbitDistance, 5f)
                eyePos[2] = centerZ + panZ
                Matrix.setLookAtM(
                    viewMatrix, 0,
                    eyePos[0], eyePos[1], eyePos[2],
                    centerX + panX, centerY, centerZ + panZ,
                    sin(yawRad), 0f, -cos(yawRad),
                )
            }
            ViewMode.IMMERSIVE_GYRO -> {
                // Android SensorManager.getRotationMatrixFromVector returns a 4x4 matrix R in row-major layout:
                // [ R[0]   R[1]   R[2]   0 ] -> Row 0 (World X)
                // [ R[4]   R[5]   R[6]   0 ] -> Row 1 (World Y)
                // [ R[8]   R[9]   R[10]  0 ] -> Row 2 (World Z)
                // where P_world = R * P_device.
                //
                // Device coordinate system:
                // +Xdev: Right of phone screen = [1, 0, 0]^T -> Column 0: [R[0], R[4], R[8]]^T in World
                // +Ydev: Top of phone screen = [0, 1, 0]^T -> Column 1: [R[1], R[5], R[9]]^T in World
                // +Zdev: Out of phone screen = [0, 0, 1]^T -> Column 2: [R[2], R[6], R[10]]^T in World
                // -Zdev: Camera look direction (back of phone) = [0, 0, -1]^T -> -Column 2: [-R[2], -R[6], -R[10]]^T in World
                //
                // Mapping Sensor World (Z is up) -> Point Cloud World (Y is up):
                // PC_X = World_X
                // PC_Y = World_Z
                // PC_Z = -World_Y
                val lookX0 = -sensorRotationMatrix[2]
                val lookY0 = -sensorRotationMatrix[10]
                val lookZ0 = sensorRotationMatrix[6]

                val upX0 = sensorRotationMatrix[1]
                val upY0 = sensorRotationMatrix[9]
                val upZ0 = -sensorRotationMatrix[5]

                // Apply horizontal gyro yaw offset around vertical PC_Y axis
                val yawRad = Math.toRadians(gyroYawOffset.toDouble()).toFloat()
                val cosY = cos(yawRad)
                val sinY = sin(yawRad)

                val lookX = cosY * lookX0 + sinY * lookZ0
                val lookY = lookY0
                val lookZ = -sinY * lookX0 + cosY * lookZ0

                val upX = cosY * upX0 + sinY * upZ0
                val upY = upY0
                val upZ = -sinY * upX0 + cosY * upZ0

                currentLookDir[0] = lookX
                currentLookDir[1] = lookY
                currentLookDir[2] = lookZ

                currentUpDir[0] = upX
                currentUpDir[1] = upY
                currentUpDir[2] = upZ

                Matrix.setLookAtM(
                    viewMatrix, 0,
                    walkX, walkY, walkZ,
                    walkX + lookX, walkY + lookY, walkZ + lookZ,
                    upX, upY, upZ
                )
            }
        }
    }

    private fun createProgram(vertexSrc: String, fragmentSrc: String): Int {
        val vShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSrc)
        val fShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSrc)
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vShader)
        GLES20.glAttachShader(program, fShader)
        GLES20.glLinkProgram(program)
        return program
    }

    private fun loadShader(type: Int, code: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, code)
        GLES20.glCompileShader(shader)
        return shader
    }
}
