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

/**
 * One property: its labelled room scans, and the way into a new capture.
 *
 * Per-room capture semantics are unchanged (SPEC §2.1: one room scan is still one uninterrupted
 * walk with its own loop closure); "multi-room" is composition at this level only.
 */
class PropertyDetailActivity : AppCompatActivity() {

    private lateinit var listContainer: LinearLayout
    private lateinit var emptyText: TextView
    private val store by lazy { PropertyStore(this) }
    private val propertyId by lazy { intent.getStringExtra(PropertyStore.EXTRA_PROPERTY_ID) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_property_detail)
        listContainer = findViewById(R.id.roomList)
        emptyText = findViewById(R.id.emptyText)
        findViewById<TextView>(R.id.closeButton).setOnClickListener { finish() }
        findViewById<Button>(R.id.addRoomButton).setOnClickListener { promptRoomLabel() }
    }

    override fun onResume() {
        super.onResume()
        val property = propertyId?.let { store.property(it) }
        if (property == null) {
            // The property was deleted from Home while this screen sat in the back stack.
            finish()
            return
        }
        findViewById<TextView>(R.id.addressText).text = property.address
        findViewById<TextView>(R.id.noteText).apply {
            text = property.agentNote.orEmpty()
            setVisible(!property.agentNote.isNullOrBlank())
        }
        reload()
    }

    private fun reload() {
        val rooms = propertyId?.let { store.rooms(it) } ?: emptyList()
        listContainer.removeAllViews()
        emptyText.setVisible(rooms.isEmpty())
        for (room in rooms) listContainer.addView(buildRow(room))
    }

    private fun buildRow(room: PropertyStore.RoomScan): LinearLayout {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(14), dp(12), dp(14))
            background = getDrawable(R.drawable.card)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = dp(8) }
        }

        val info = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        info.addView(TextView(this).apply {
            text = room.label
            setTextColor(color(R.color.text_primary))
            textSize = 15f
        })
        info.addView(TextView(this).apply {
            val coverage = room.coveragePercent?.let { "${it.toInt()}% coverage" } ?: "coverage unknown"
            text = "$coverage · ${room.status.name.lowercase()}"
            setTextColor(color(R.color.text_faint))
            textSize = 12f
        })
        info.addView(TextView(this).apply {
            text = room.sessionName
            setTextColor(color(R.color.text_faint))
            textSize = 10f
        })
        row.addView(info)

        row.addView(TextView(this).apply {
            text = "Remove"
            setTextColor(color(R.color.danger))
            textSize = 13f
            setOnClickListener { confirmRemove(room) }
        })
        return row
    }

    private fun confirmRemove(room: PropertyStore.RoomScan) {
        AlertDialog.Builder(this)
            .setTitle("Remove room?")
            .setMessage(
                "\"${room.label}\" will be unlinked from this property. Its scan file stays on " +
                    "the phone and moves to Unassigned in All scans."
            )
            .setPositiveButton("Remove") { _, _ ->
                store.deleteRoomScan(room.sessionName)
                reload()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun promptRoomLabel() {
        val id = propertyId ?: return
        val existing = store.rooms(id).size
        val label = EditText(this).apply {
            hint = "Room name"
            setText("Room ${existing + 1}")
            setSelection(text.length)
        }
        AlertDialog.Builder(this)
            .setTitle("Scan a room")
            .setView(LinearLayout(this).apply {
                setPadding(dp(24), dp(8), dp(24), 0)
                addView(label)
            })
            .setPositiveButton("Start scan") { _, _ ->
                val text = label.text.toString().trim().ifEmpty { "Room ${existing + 1}" }
                startActivity(
                    Intent(this, CaptureActivity::class.java)
                        .putExtra(PropertyStore.EXTRA_PROPERTY_ID, id)
                        .putExtra(PropertyStore.EXTRA_ROOM_LABEL, text)
                )
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun color(id: Int) = getColor(id)
    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
}
