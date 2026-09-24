package com.glomehometour.arscan

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Launcher screen: the list of properties being captured, and the way in to everything else.
 *
 * Capture used to be the launcher (the app was one Activity), which left no place to say *what*
 * is being scanned. A property groups the room scans of one listing; a capture is started from
 * inside one (PropertyDetailActivity), not standalone.
 */
class HomeActivity : AppCompatActivity() {

    private lateinit var listContainer: LinearLayout
    private lateinit var emptyText: TextView
    private val store by lazy { PropertyStore(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)
        listContainer = findViewById(R.id.propertyList)
        emptyText = findViewById(R.id.emptyText)
        findViewById<TextView>(R.id.galleryButton).setOnClickListener {
            startActivity(Intent(this, ScanGalleryActivity::class.java))
        }
        findViewById<TextView>(R.id.settingsButton).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        findViewById<Button>(R.id.newPropertyButton).setOnClickListener { promptNewProperty() }
    }

    override fun onResume() {
        super.onResume()
        reload()
    }

    private fun reload() {
        val properties = store.properties()
        listContainer.removeAllViews()
        emptyText.setVisible(properties.isEmpty())
        for (property in properties) listContainer.addView(buildRow(property))
    }

    private fun buildRow(property: PropertyStore.Property): LinearLayout {
        val roomCount = store.rooms(property.id).size
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(14), dp(12), dp(14))
            background = getDrawable(R.drawable.card)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = dp(8) }
            setOnClickListener {
                startActivity(
                    Intent(this@HomeActivity, PropertyDetailActivity::class.java)
                        .putExtra(PropertyStore.EXTRA_PROPERTY_ID, property.id)
                )
            }
        }

        val info = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        info.addView(TextView(this).apply {
            text = property.address
            setTextColor(color(R.color.text_primary))
            textSize = 15f
        })
        info.addView(TextView(this).apply {
            val rooms = if (roomCount == 1) "1 room" else "$roomCount rooms"
            text = "$rooms · ${SimpleDateFormat("yyyy-MM-dd", Locale.US).format(property.createdAtMs)}"
            setTextColor(color(R.color.text_faint))
            textSize = 12f
        })
        row.addView(info)

        row.addView(TextView(this).apply {
            text = "Delete"
            setTextColor(color(R.color.danger))
            textSize = 13f
            setOnClickListener { confirmDelete(property, roomCount) }
        })
        return row
    }

    private fun confirmDelete(property: PropertyStore.Property, roomCount: Int) {
        AlertDialog.Builder(this)
            .setTitle("Delete property?")
            .setMessage(
                "${property.address} and its $roomCount room entries will be removed from the " +
                    "list. The scan files themselves stay on the phone -- delete those from " +
                    "All scans."
            )
            .setPositiveButton("Delete") { _, _ ->
                store.deleteProperty(property.id)
                reload()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun promptNewProperty() {
        val address = EditText(this).apply { hint = "Address" }
        val note = EditText(this).apply { hint = "Note (optional)" }
        val fields = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(8), dp(24), 0)
            addView(address)
            addView(note)
        }
        AlertDialog.Builder(this)
            .setTitle("New property")
            .setView(fields)
            .setPositiveButton("Create") { _, _ ->
                val text = address.text.toString().trim()
                if (text.isEmpty()) return@setPositiveButton
                val property = store.addProperty(text, note.text.toString().trim())
                startActivity(
                    Intent(this, PropertyDetailActivity::class.java)
                        .putExtra(PropertyStore.EXTRA_PROPERTY_ID, property.id)
                )
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun color(id: Int) = getColor(id)
    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
}
