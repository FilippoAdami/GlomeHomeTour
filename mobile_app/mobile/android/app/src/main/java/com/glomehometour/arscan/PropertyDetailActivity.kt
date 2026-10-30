package com.glomehometour.arscan

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

import android.content.ClipData
import android.content.res.ColorStateList

/**
 * One property: its labelled room scans, and the way into a new capture.
 *
 * Per-room capture semantics are unchanged (SPEC §2.1: one room scan is still one uninterrupted
 * walk with its own loop closure); "multi-room" is composition at this level only.
 */
class PropertyDetailActivity : AppCompatActivity() {

    private lateinit var listContainer: LinearLayout
    private lateinit var emptyText: TextView
    private lateinit var processPropertyButton: Button
    private lateinit var sharePropertyButton: Button
    private lateinit var propertyBatchProgressLayout: LinearLayout
    private lateinit var propertyBatchProgress: ProgressBar
    private lateinit var propertyBatchProgressText: TextView
    private val store by lazy { PropertyStore(this) }
    private val propertyId by lazy { intent.getStringExtra(PropertyStore.EXTRA_PROPERTY_ID) }
    private var isBatchProcessing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_property_detail)
        listContainer = findViewById(R.id.roomList)
        emptyText = findViewById(R.id.emptyText)
        processPropertyButton = findViewById(R.id.processPropertyButton)
        sharePropertyButton = findViewById(R.id.sharePropertyButton)
        propertyBatchProgressLayout = findViewById(R.id.propertyBatchProgressLayout)
        propertyBatchProgress = findViewById(R.id.propertyBatchProgress)
        propertyBatchProgressText = findViewById(R.id.propertyBatchProgressText)

        findViewById<TextView>(R.id.closeButton).setOnClickListener { finish() }
        findViewById<Button>(R.id.addRoomButton).setOnClickListener { promptRoomLabel() }
        findViewById<Button>(R.id.uploadPropertyButton).setOnClickListener { startUploadFlow() }
        findViewById<Button>(R.id.viewTourButton).setOnClickListener {
            val id = propertyId?.let { store.property(it)?.serverPropertyId } ?: return@setOnClickListener
            startActivity(Intent(this, TourViewerActivity::class.java).putExtra(TourViewerActivity.EXTRA_PROPERTY_ID, id))
        }

        processPropertyButton.setOnClickListener { startBatchProcessing() }
        sharePropertyButton.setOnClickListener { startShareProperty() }
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
        findViewById<Button>(R.id.viewTourButton).visibility =
            if (property.serverPropertyId == null) android.view.View.GONE else android.view.View.VISIBLE
        findViewById<TextView>(R.id.noteText).apply {
            text = property.agentNote.orEmpty()
            setVisible(!property.agentNote.isNullOrBlank())
        }
        reload()
    }

    private fun reload() {
        val rooms = propertyId?.let { store.rooms(it) } ?: emptyList()
        val property = propertyId?.let { store.property(it) }
        listContainer.removeAllViews()
        emptyText.setVisible(rooms.isEmpty())
        for (room in rooms) listContainer.addView(buildRow(room))

        val propertyActionLayout = findViewById<LinearLayout>(R.id.propertyActionLayout)
        if (rooms.isEmpty()) {
            propertyActionLayout.visibility = android.view.View.GONE
        } else {
            propertyActionLayout.visibility = android.view.View.VISIBLE
            val unprocessed = rooms.filter { !ScanProcessor.isProcessed(this, it.sessionName) }
            val hasCheckpointAny = unprocessed.any { ScanProcessor.hasCheckpoint(this, it.sessionName) }

            if (unprocessed.isEmpty()) {
                if (property != null && !property.isProcessed) {
                    property.isProcessed = true
                    store.updateProperty(property)
                }
                processPropertyButton.apply {
                    text = "Processed"
                    backgroundTintList = ColorStateList.valueOf(color(R.color.card))
                    setTextColor(color(R.color.action_ready))
                    isClickable = false
                    isEnabled = false
                }
                sharePropertyButton.apply {
                    backgroundTintList = ColorStateList.valueOf(color(R.color.action))
                    setTextColor(color(R.color.text_primary))
                    isEnabled = !isBatchProcessing
                }
            } else {
                processPropertyButton.apply {
                    text = if (hasCheckpointAny) "Resume Processing" else "Process Property"
                    backgroundTintList = ColorStateList.valueOf(color(R.color.action))
                    setTextColor(color(R.color.text_primary))
                    isClickable = true
                    isEnabled = !isBatchProcessing
                }
                sharePropertyButton.apply {
                    backgroundTintList = ColorStateList.valueOf(color(R.color.card))
                    setTextColor(color(R.color.text_secondary))
                    isEnabled = !isBatchProcessing
                }
            }
        }
    }

    private fun startBatchProcessing() {
        if (isBatchProcessing) return
        val id = propertyId ?: return
        val property = store.property(id) ?: return
        val rooms = store.rooms(id)
        if (rooms.isEmpty()) {
            Toast.makeText(this, "No rooms to process", Toast.LENGTH_SHORT).show()
            return
        }
        val unprocessed = rooms.filter { !ScanProcessor.isProcessed(this, it.sessionName) }
        if (unprocessed.isEmpty()) {
            property.isProcessed = true
            store.updateProperty(property)
            reload()
            Toast.makeText(this, "All rooms already processed!", Toast.LENGTH_SHORT).show()
            return
        }

        isBatchProcessing = true
        processPropertyButton.isEnabled = false
        sharePropertyButton.isEnabled = false
        findViewById<Button>(R.id.addRoomButton).isEnabled = false
        propertyBatchProgressLayout.visibility = android.view.View.VISIBLE
        propertyBatchProgress.progress = 0

        Thread {
            val total = unprocessed.size
            var allSucceeded = true

            for ((idx, room) in unprocessed.withIndex()) {
                val currentNum = idx + 1
                runOnUiThread {
                    propertyBatchProgressText.text = "Processing ${room.label} ($currentNum/$total)…"
                    propertyBatchProgress.progress = 0
                }
                val accessor = ScanProcessor.getAccessor(this, room.sessionName)
                val processor = ScanProcessor(this)
                val success = processor.process(room.sessionName, accessor) { percent ->
                    runOnUiThread {
                        propertyBatchProgress.progress = percent
                        propertyBatchProgressText.text = "Processing ${room.label} ($currentNum/$total): $percent%"
                    }
                }
                if (success) {
                    ScanProcessor.markProcessed(room.sessionName)
                    runOnUiThread { reload() }
                } else {
                    allSucceeded = false
                    runOnUiThread {
                        Toast.makeText(this, "Processing interrupted on ${room.label}", Toast.LENGTH_LONG).show()
                    }
                    break
                }
            }

            runOnUiThread {
                isBatchProcessing = false
                propertyBatchProgressLayout.visibility = android.view.View.GONE
                findViewById<Button>(R.id.addRoomButton).isEnabled = true
                processPropertyButton.isEnabled = true
                sharePropertyButton.isEnabled = true

                if (allSucceeded) {
                    property.isProcessed = true
                    store.updateProperty(property)
                    Toast.makeText(this, "All rooms processed successfully!", Toast.LENGTH_SHORT).show()
                }
                reload()
            }
        }.start()
    }

    private fun startShareProperty() {
        val id = propertyId ?: return
        val property = store.property(id) ?: return
        val rooms = store.rooms(id)
        if (rooms.isEmpty()) {
            Toast.makeText(this, "No rooms to share", Toast.LENGTH_SHORT).show()
            return
        }
        val unprocessed = rooms.filter { !ScanProcessor.isProcessed(this, it.sessionName) }
        if (unprocessed.isNotEmpty()) {
            AlertDialog.Builder(this)
                .setTitle("Processing required")
                .setMessage("${unprocessed.size} room(s) must be processed before sharing the property scan. Process them now?")
                .setPositiveButton("Process all") { _, _ -> startBatchProcessing() }
                .setNegativeButton("Cancel", null)
                .show()
            return
        }

        val dialog = AlertDialog.Builder(this)
            .setTitle("Preparing Property Scan")
            .setMessage("Packaging rooms and generating certificate…")
            .setCancelable(false)
            .show()

        Thread {
            val resultUri = ScanProcessor.packagePropertyScan(this, property, rooms) { percent ->
                runOnUiThread {
                    dialog.setMessage("Packaging property scan: $percent%")
                }
            }
            runOnUiThread {
                dialog.dismiss()
                if (resultUri != null) {
                    val cleanAddress = property.address.replace(Regex("[^a-zA-Z0-9_-]"), "_").trim('_').ifEmpty { "property" }
                    val zipName = "${cleanAddress}_scan.zip"
                    val shareIntent = Intent(Intent.ACTION_SEND).apply {
                        type = "application/zip"
                        putExtra(Intent.EXTRA_STREAM, resultUri)
                        putExtra(Intent.EXTRA_SUBJECT, zipName)
                        clipData = ClipData.newRawUri(zipName, resultUri)
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    startActivity(Intent.createChooser(shareIntent, "Share $zipName"))
                } else {
                    Toast.makeText(this, "Failed to create property scan package", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
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

        val isProcessed = ScanProcessor.isProcessed(this, room.sessionName)
        var hasCheckpoint = if (!isProcessed) ScanProcessor.hasCheckpoint(this, room.sessionName) else false

        val actionArea = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            )
        }

        val processLabel = TextView(this).apply {
            text = when {
                isProcessed -> "Processed"
                hasCheckpoint -> "Resume"
                else -> "Process"
            }
            setTextColor(color(if (isProcessed) R.color.action_ready else R.color.action))
            textSize = 13f
            isClickable = !isProcessed
            isEnabled = !isProcessed
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { rightMargin = dp(14) }
        }

        val shareLabel = TextView(this).apply {
            text = "Share"
            setTextColor(color(R.color.action))
            textSize = 13f
            visibility = if (isProcessed) android.view.View.VISIBLE else android.view.View.GONE
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { rightMargin = dp(14) }
            setOnClickListener { ScanProcessor.shareScan(this@PropertyDetailActivity, room.sessionName) }
        }

        val removeLabel = TextView(this).apply {
            text = "Remove"
            setTextColor(color(R.color.danger))
            textSize = 13f
            setOnClickListener { confirmRemove(room) }
        }

        val progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            visibility = android.view.View.GONE
            layoutParams = LinearLayout.LayoutParams(dp(100), LinearLayout.LayoutParams.WRAP_CONTENT)
        }

        val progressText = TextView(this).apply {
            setTextColor(color(R.color.text_faint))
            textSize = 11f
            visibility = android.view.View.GONE
            setPadding(dp(6), 0, 0, 0)
        }

        actionArea.addView(processLabel)
        actionArea.addView(shareLabel)
        actionArea.addView(removeLabel)
        actionArea.addView(progressBar)
        actionArea.addView(progressText)

        processLabel.setOnClickListener {
            if (ScanProcessor.isProcessed(this, room.sessionName)) return@setOnClickListener
            processLabel.visibility = android.view.View.GONE
            shareLabel.visibility = android.view.View.GONE
            removeLabel.visibility = android.view.View.GONE
            progressBar.visibility = android.view.View.VISIBLE
            progressText.visibility = android.view.View.VISIBLE

            Thread {
                val accessor = ScanProcessor.getAccessor(this, room.sessionName)
                val processor = ScanProcessor(this)
                val success = processor.process(room.sessionName, accessor) { percent ->
                    runOnUiThread {
                        progressBar.progress = percent
                        progressText.text = "$percent%"
                    }
                }
                runOnUiThread {
                    progressBar.visibility = android.view.View.GONE
                    progressText.visibility = android.view.View.GONE
                    if (success) {
                        ScanProcessor.markProcessed(room.sessionName)
                        processLabel.text = "Processed"
                        processLabel.setTextColor(color(R.color.action_ready))
                        processLabel.isClickable = false
                        processLabel.isEnabled = false
                        processLabel.visibility = android.view.View.VISIBLE
                        shareLabel.visibility = android.view.View.VISIBLE
                        removeLabel.visibility = android.view.View.VISIBLE
                        Toast.makeText(this, "Processing complete!", Toast.LENGTH_SHORT).show()
                    } else {
                        hasCheckpoint = true
                        processLabel.text = "Resume"
                        processLabel.setTextColor(color(R.color.action))
                        processLabel.isClickable = true
                        processLabel.isEnabled = true
                        processLabel.visibility = android.view.View.VISIBLE
                        removeLabel.visibility = android.view.View.VISIBLE
                        Toast.makeText(this, "Processing interrupted. Tap Resume to continue.", Toast.LENGTH_LONG).show()
                    }
                }
            }.start()
        }

        row.addView(actionArea)
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
        val existingRooms = store.rooms(id)
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
                if (existingRooms.isNotEmpty()) {
                    val lastRoom = existingRooms.last()
                    AlertDialog.Builder(this)
                        .setTitle("Align with previous room?")
                        .setMessage("Align this room with the doorway of \"${lastRoom.label}\"?")
                        .setPositiveButton("Align (Recommended)") { _, _ ->
                            startActivity(
                                Intent(this, CaptureActivity::class.java)
                                    .putExtra(PropertyStore.EXTRA_PROPERTY_ID, id)
                                    .putExtra(PropertyStore.EXTRA_ROOM_LABEL, text)
                                    .putExtra(CaptureActivity.EXTRA_ALIGN_TO_SESSION, lastRoom.sessionName)
                            )
                        }
                        .setNegativeButton("Start fresh") { _, _ ->
                            startActivity(
                                Intent(this, CaptureActivity::class.java)
                                    .putExtra(PropertyStore.EXTRA_PROPERTY_ID, id)
                                    .putExtra(PropertyStore.EXTRA_ROOM_LABEL, text)
                            )
                        }
                        .show()
                } else {
                    startActivity(
                        Intent(this, CaptureActivity::class.java)
                            .putExtra(PropertyStore.EXTRA_PROPERTY_ID, id)
                            .putExtra(PropertyStore.EXTRA_ROOM_LABEL, text)
                    )
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun startUploadFlow() {
        val id = propertyId ?: return
        if (store.property(id) == null) return
        val rooms = store.rooms(id)
        if (rooms.isEmpty()) {
            AlertDialog.Builder(this)
                .setTitle("No scans available")
                .setMessage("Please scan at least one room before uploading to Glome Web.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        // Locate scan zip file for first room or composite
        val room = rooms.first()
        val zipFile = ScanProcessor.findZipFile(this, room.sessionName)
        if (zipFile == null || !zipFile.isFile) {
            AlertDialog.Builder(this)
                .setTitle("Processing required")
                .setMessage("Room \"${room.label}\" has not been processed yet. Please tap Process before uploading.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        val token = GlomeUploader.savedAuthToken(this)
        if (token != null) {
            uploadRoom(room, zipFile, token)
        } else {
            val email = EditText(this).apply { hint = "Email"; inputType = android.text.InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS }
            val password = EditText(this).apply {
                hint = "Password"
                inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
            }
            val fields = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(24), dp(8), dp(24), 0)
                addView(email)
                addView(password)
            }
            AlertDialog.Builder(this)
                .setTitle("Sign in to Glome")
                .setView(fields)
                .setPositiveButton("Sign in") { _, _ ->
                    val signInDialog = AlertDialog.Builder(this)
                        .setMessage("Signing in…")
                        .setCancelable(false)
                        .show()
                    GlomeUploader.signIn(this, email.text.toString(), password.text.toString(),
                        object : GlomeUploader.SignInCallback {
                            override fun onSuccess(token: String) {
                                signInDialog.dismiss()
                                uploadRoom(room, zipFile, token)
                            }
                            override fun onError(error: String) {
                                signInDialog.dismiss()
                                AlertDialog.Builder(this@PropertyDetailActivity).setTitle("Sign in failed")
                                    .setMessage(error).setPositiveButton("OK", null).show()
                            }
                        })
                }
                .setNegativeButton("Cancel", null)
                .show()
        }
    }

    private fun uploadRoom(room: PropertyStore.RoomScan, zipFile: java.io.File, token: String) {
        val property = propertyId?.let { store.property(it) } ?: return
        val dialog = AlertDialog.Builder(this)
            .setTitle("Uploading to Glome Web")
            .setMessage("Connecting to server...")
            .setCancelable(false)
            .show()
        GlomeUploader.uploadScan(
            context = this,
            zipFile = zipFile,
            propertyAddress = property.address,
            authToken = token,
            callback = object : GlomeUploader.UploadCallback {
                override fun onProgress(percentage: Int) {
                    dialog.setMessage("Uploading ${room.label}: $percentage%")
                }

                override fun onSuccess(propertyId: String) {
                    dialog.dismiss()
                    property.serverPropertyId = propertyId
                    property.uploadStatus = PropertyStore.UploadStatus.UPLOADED
                    store.updateProperty(property)
                    findViewById<Button>(R.id.viewTourButton).visibility = android.view.View.VISIBLE
                    AlertDialog.Builder(this@PropertyDetailActivity)
                        .setTitle("Upload Complete!")
                        .setMessage("Your scan is processing. Use ‘View mesh and Gaussian splat’ here when the reconstruction is ready.")
                        .setPositiveButton("View") { _, _ ->
                            startActivity(Intent(this@PropertyDetailActivity, TourViewerActivity::class.java)
                                .putExtra(TourViewerActivity.EXTRA_PROPERTY_ID, propertyId))
                            reload()
                        }
                        .setNegativeButton("OK") { _, _ -> reload() }
                        .show()
                }

                override fun onError(error: String) {
                    dialog.dismiss()
                    AlertDialog.Builder(this@PropertyDetailActivity)
                        .setTitle("Upload Failed")
                        .setMessage("Error: $error\n\nEnsure the Glome server is running on the local network.")
                        .setPositiveButton("OK", null)
                        .show()
                }
            }
        )
    }

    private fun color(id: Int) = getColor(id)
    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
}
