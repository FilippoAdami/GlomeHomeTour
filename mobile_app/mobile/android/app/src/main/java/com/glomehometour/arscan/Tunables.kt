package com.glomehometour.arscan

import android.content.Context

/**
 * The handful of capture constants that project_history.md shows actually needing per-device
 * retuning in the field, made settable without a rebuild (SettingsActivity).
 *
 * Everything else stays a `const val` where it lives. A knob nobody turns is support surface for
 * nothing -- these four are here because a `rosemary`-class device has already needed them moved.
 *
 * Defaults are the existing constants themselves, so a fresh install (and every unit test, which
 * never calls [load]) behaves exactly as before.
 */
object Tunables {

    /** Angle between two views before VoxelGrid counts a surface as adequately captured. */
    @Volatile var parallaxMinDeg: Float = VoxelGrid.PARALLAX_MIN_DEG
        set(value) {
            field = value
            parallaxCos = kotlin.math.cos(Math.toRadians(value.toDouble())).toFloat()
        }

    /** Derived from [parallaxMinDeg]; compared against a dot product in VoxelGrid's hot loop. */
    @Volatile var parallaxCos: Float = VoxelGrid.PARALLAX_COS
        private set

    /** Coverage at which the finish button goes green (advisory, never a hard gate). */
    @Volatile var coverageCompleteFraction: Float = CaptureActivity.COMPLETION_FRACTION

    /** Photometric gate: mean luma outside [min, max] drops the frame as dark/blown. */
    @Volatile var photometricMeanMin: Float = PhotometricGate.MEAN_MIN
    @Volatile var photometricMeanMax: Float = PhotometricGate.MEAN_MAX

    /** Keep 1 of every N gate-passed frames. README §4/§5 wants this inside [6, 10]. */
    @Volatile var decimationStride: Long = CaptureActivity.DECIMATION_STRIDE

    fun load(context: Context) {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        parallaxMinDeg = prefs.getFloat(KEY_PARALLAX_DEG, VoxelGrid.PARALLAX_MIN_DEG)
        coverageCompleteFraction = prefs.getFloat(KEY_COVERAGE, CaptureActivity.COMPLETION_FRACTION)
        photometricMeanMin = prefs.getFloat(KEY_LUMA_MIN, PhotometricGate.MEAN_MIN)
        photometricMeanMax = prefs.getFloat(KEY_LUMA_MAX, PhotometricGate.MEAN_MAX)
        decimationStride = prefs.getLong(KEY_STRIDE, CaptureActivity.DECIMATION_STRIDE)
    }

    /**
     * One settable set of values, already clamped to ranges the capture loop survives.
     *
     * Pure and Context-free on purpose: an operator fat-fingering "2500" into the luma gate would
     * otherwise drop every frame of the next scan with no visible cause, and that guard is worth
     * being able to test without a device.
     */
    data class Values(
        val parallaxMinDeg: Float,
        val coverageCompleteFraction: Float,
        val photometricMeanMin: Float,
        val photometricMeanMax: Float,
        val decimationStride: Long,
    ) {
        companion object {
            /** [coveragePercent] is operator-facing 0-100, not the 0-1 fraction used internally. */
            fun clamped(
                parallaxDeg: Float,
                coveragePercent: Float,
                lumaMin: Float,
                lumaMax: Float,
                stride: Long,
            ): Values {
                val min = lumaMin.coerceIn(1f, 120f)
                return Values(
                    parallaxMinDeg = parallaxDeg.coerceIn(5f, 60f),
                    coverageCompleteFraction = (coveragePercent / 100f).coerceIn(0.1f, 1f),
                    photometricMeanMin = min,
                    photometricMeanMax = lumaMax.coerceIn(min + 1f, 255f),
                    decimationStride = stride.coerceIn(1L, 60L),
                )
            }
        }
    }

    fun save(context: Context, values: Values) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putFloat(KEY_PARALLAX_DEG, values.parallaxMinDeg)
            .putFloat(KEY_COVERAGE, values.coverageCompleteFraction)
            .putFloat(KEY_LUMA_MIN, values.photometricMeanMin)
            .putFloat(KEY_LUMA_MAX, values.photometricMeanMax)
            .putLong(KEY_STRIDE, values.decimationStride)
            .apply()
        load(context)
    }

    fun reset(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().clear().apply()
        load(context)
    }

    private const val PREFS = "capture_tunables"
    private const val KEY_PARALLAX_DEG = "parallax_min_deg"
    private const val KEY_COVERAGE = "coverage_complete_fraction"
    private const val KEY_LUMA_MIN = "photometric_mean_min"
    private const val KEY_LUMA_MAX = "photometric_mean_max"
    private const val KEY_STRIDE = "decimation_stride"
}
