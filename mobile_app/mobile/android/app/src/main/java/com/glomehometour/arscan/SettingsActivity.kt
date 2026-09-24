package com.glomehometour.arscan

import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

/**
 * The four capture thresholds project_history.md shows needing per-device retuning, editable in
 * the field instead of by rebuilding. Everything else stays a compile-time constant on purpose.
 *
 * Built in code rather than with PreferenceFragmentCompat: five numbers with clamped ranges
 * don't earn the androidx.preference dependency and its theme requirements.
 */
class SettingsActivity : AppCompatActivity() {

    private lateinit var parallaxField: EditText
    private lateinit var coverageField: EditText
    private lateinit var lumaMinField: EditText
    private lateinit var lumaMaxField: EditText
    private lateinit var strideField: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Tunables.load(this)

        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(24), dp(16), dp(24))
        }
        column.addView(TextView(this).apply {
            text = "Capture settings"
            setTextColor(color(R.color.text_secondary))
            textSize = 18f
            setPadding(0, 0, 0, dp(4))
        })
        column.addView(TextView(this).apply {
            text = "Takes effect on the next scan. Leave these alone unless a device needs it."
            setTextColor(color(R.color.text_faint))
            textSize = 12f
            setPadding(0, 0, 0, dp(16))
        })

        parallaxField = addField(
            column, "Parallax verification angle (°)",
            "Two views this far apart mark a surface captured. Default ${VoxelGrid.PARALLAX_MIN_DEG.toInt()}.",
            Tunables.parallaxMinDeg.toString(),
        )
        coverageField = addField(
            column, "Coverage complete (%)",
            "Where the finish button goes green. Advisory, never blocks finishing. " +
                "Default ${(CaptureActivity.COMPLETION_FRACTION * 100).toInt()}.",
            (Tunables.coverageCompleteFraction * 100f).toString(),
        )
        lumaMinField = addField(
            column, "Photometric gate: min mean luma",
            "Below this a frame is dropped as too dark. Default ${PhotometricGate.MEAN_MIN.toInt()}.",
            Tunables.photometricMeanMin.toString(),
        )
        lumaMaxField = addField(
            column, "Photometric gate: max mean luma",
            "Above this a frame is dropped as blown out. Default ${PhotometricGate.MEAN_MAX.toInt()}.",
            Tunables.photometricMeanMax.toString(),
        )
        strideField = addField(
            column, "Keyframe decimation stride",
            "Keep 1 of every N gate-passed frames. Default ${CaptureActivity.DECIMATION_STRIDE}, " +
                "keep within 6-10.",
            Tunables.decimationStride.toString(),
        )

        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(16), 0, 0)
        }
        buttons.addView(Button(this).apply {
            text = "Reset defaults"
            isAllCaps = false
            background = getDrawable(R.drawable.btn_pill)
            backgroundTintList = android.content.res.ColorStateList.valueOf(color(R.color.action_disabled))
            setTextColor(color(R.color.text_secondary))
            layoutParams = LinearLayout.LayoutParams(0, dp(48), 1f).apply { rightMargin = dp(8) }
            setOnClickListener {
                Tunables.reset(this@SettingsActivity)
                recreate()
            }
        })
        buttons.addView(Button(this).apply {
            text = "Save"
            isAllCaps = false
            background = getDrawable(R.drawable.btn_pill)
            backgroundTintList = android.content.res.ColorStateList.valueOf(color(R.color.action))
            setTextColor(color(R.color.text_primary))
            layoutParams = LinearLayout.LayoutParams(0, dp(48), 1f)
            setOnClickListener { save() }
        })
        column.addView(buttons)

        setContentView(ScrollView(this).apply {
            setBackgroundColor(color(R.color.ink))
            addView(column)
        })
    }

    /** Unparseable input keeps the current value; out-of-range input is clamped (Values.clamped). */
    private fun save() {
        val values = Tunables.Values.clamped(
            parallaxDeg = parallaxField.floatOr(Tunables.parallaxMinDeg),
            coveragePercent = coverageField.floatOr(Tunables.coverageCompleteFraction * 100f),
            lumaMin = lumaMinField.floatOr(Tunables.photometricMeanMin),
            lumaMax = lumaMaxField.floatOr(Tunables.photometricMeanMax),
            stride = strideField.text.toString().trim().toLongOrNull() ?: Tunables.decimationStride,
        )
        Tunables.save(this, values)
        Toast.makeText(this, "Saved -- applies to the next scan", Toast.LENGTH_SHORT).show()
        finish()
    }

    private fun EditText.floatOr(fallback: Float): Float =
        text.toString().trim().toFloatOrNull() ?: fallback

    private fun addField(parent: LinearLayout, title: String, help: String, value: String): EditText {
        parent.addView(TextView(this).apply {
            text = title
            setTextColor(color(R.color.text_primary))
            textSize = 14f
        })
        parent.addView(TextView(this).apply {
            text = help
            setTextColor(color(R.color.text_faint))
            textSize = 11f
        })
        val field = EditText(this).apply {
            setText(value)
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setTextColor(color(R.color.text_primary))
            textSize = 15f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = dp(12) }
        }
        parent.addView(field)
        return field
    }

    private fun color(id: Int) = getColor(id)
    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
}
