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
        val existingProperties = store.properties()
        existingProperties.forEach { store.expandLegacyRoomFolders(this, it.id) }
        val properties = store.properties()
        listContainer.removeAllViews()
        emptyText.setVisible(properties.isEmpty())
        for (property in properties) listContainer.addView(buildRow(property))
    }

    private fun buildRow(property: PropertyStore.Property): android.view.View {
        val rooms = store.rooms(property.id)
        val roomCount = rooms.size
        val unprocessedRooms = rooms.filter { !ScanProcessor.isProcessed(this, it.sessionName) }
        val hasCheckpointAny = unprocessedRooms.any { ScanProcessor.hasCheckpoint(this, it.sessionName) }

        val card = layoutInflater.inflate(R.layout.item_property_card, listContainer, false)
        val addressView = card.findViewById<TextView>(R.id.propertyAddress)
        val statusBadge = card.findViewById<TextView>(R.id.propertyStatusBadge)
        val subtitleView = card.findViewById<TextView>(R.id.propertySubtitle)
        val btnListing = card.findViewById<TextView>(R.id.btnPropertyListing)
        val btnAddRoom = card.findViewById<TextView>(R.id.btnPropertyAddRoom)
        val btnDelete = card.findViewById<TextView>(R.id.btnPropertyDelete)

        addressView.text = property.address

        val roomsStr = if (roomCount == 1) "1 room" else "$roomCount rooms"
        val dateStr = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(property.createdAtMs)
        subtitleView.text = "$roomsStr · Created $dateStr"

        when {
            roomCount == 0 -> {
                statusBadge.text = "EMPTY"
                statusBadge.setTextColor(color(R.color.text_faint))
            }
            unprocessedRooms.isEmpty() -> {
                statusBadge.text = "PROCESSED"
                statusBadge.setTextColor(color(R.color.action_ready))
            }
            hasCheckpointAny -> {
                statusBadge.text = "RESUME"
                statusBadge.setTextColor(color(R.color.warn))
            }
            else -> {
                statusBadge.text = "READY"
                statusBadge.setTextColor(color(R.color.action))
            }
        }

        val openListing = {
            startActivity(
                Intent(this, PropertyDetailActivity::class.java)
                    .putExtra(PropertyStore.EXTRA_PROPERTY_ID, property.id)
            )
        }

        card.setOnClickListener { openListing() }
        btnListing.setOnClickListener { openListing() }
        btnAddRoom.setOnClickListener { promptAddRoom(property) }
        btnDelete.setOnClickListener { confirmDelete(property, roomCount) }

        return card
    }

    private fun promptAddRoom(property: PropertyStore.Property) {
        val existingRooms = store.rooms(property.id)
        val existing = existingRooms.size
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
                val intent = Intent(this, CaptureActivity::class.java)
                    .putExtra(PropertyStore.EXTRA_PROPERTY_ID, property.id)
                    .putExtra(PropertyStore.EXTRA_ROOM_LABEL, text)
                if (existingRooms.isNotEmpty()) {
                    val lastRoom = existingRooms.last()
                    AlertDialog.Builder(this)
                        .setTitle("Align with previous room?")
                        .setMessage("Align this room with the doorway of \"${lastRoom.label}\"?")
                        .setPositiveButton("Align (Recommended)") { _, _ ->
                            intent.putExtra(CaptureActivity.EXTRA_ALIGN_TO_SCAN_FOLDER, lastRoom.scanFolder)
                            intent.putExtra(CaptureActivity.EXTRA_ALIGN_TO_SESSION, lastRoom.sessionName)
                            startActivity(intent)
                        }
                        .setNegativeButton("Start fresh") { _, _ ->
                            startActivity(intent)
                        }
                        .show()
                } else {
                    startActivity(intent)
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
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
