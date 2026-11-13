package com.glomehometour.arscan

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import android.widget.ViewFlipper
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

        processPropertyButton.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Process Property")
                .setMessage("Low quality scans produce low quality results. Processing is irreversible. Make sure you have inspected all room 3D point clouds before proceeding.\n\nContinue with processing all rooms?")
                .setPositiveButton("Process all") { _, _ -> startBatchProcessing() }
                .setNegativeButton("Cancel", null)
                .show()
        }
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
        store.expandLegacyRoomFolders(this, property.id)
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
            val unprocessed = rooms.distinctBy { it.sessionName }.filter { !ScanProcessor.isProcessed(this, it.sessionName) }
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
        val unprocessed = rooms.distinctBy { it.sessionName }.filter { !ScanProcessor.isProcessed(this, it.sessionName) }
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
        val unprocessed = rooms.distinctBy { it.sessionName }.filter { !ScanProcessor.isProcessed(this, it.sessionName) }
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

    private fun showProcessConfirmationDialog(room: PropertyStore.RoomScan, onProceed: () -> Unit) {
        val dialogView = layoutInflater.inflate(R.layout.dialog_process_confirmation, null)
        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        dialogView.findViewById<TextView>(R.id.dialogTitle).text = "Quality Check & Processing"
        dialogView.findViewById<TextView>(R.id.dialogScanSubtitle).text = "${room.label} (${room.sessionName})"

        dialogView.findViewById<TextView>(R.id.btnDialogInspect3d).setOnClickListener {
            dialog.dismiss()
            PointCloudViewerActivity.start(this, room.sessionName, room.scanFolder, room.label)
        }

        dialogView.findViewById<TextView>(R.id.btnDialogProceed).setOnClickListener {
            dialog.dismiss()
            onProceed()
        }

        dialogView.findViewById<TextView>(R.id.btnDialogCancel).setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun buildRow(room: PropertyStore.RoomScan): android.view.View {
        val card = layoutInflater.inflate(R.layout.item_scan_card, listContainer, false)
        val thumbnailContainer = card.findViewById<View>(R.id.thumbnailContainer)
        val thumbnailFlipper = card.findViewById<ViewFlipper>(R.id.thumbnailFlipper)
        val thumbnailView = card.findViewById<ImageView>(R.id.scanThumbnail)
        val pcPreviewView = card.findViewById<ImageView>(R.id.scanPointcloudPreview)
        val thumbnailBadge = card.findViewById<TextView>(R.id.thumbnailBadge)
        val labelView = card.findViewById<TextView>(R.id.scanLabel)
        val statusBadge = card.findViewById<TextView>(R.id.scanStatusBadge)
        val subtitleView = card.findViewById<TextView>(R.id.scanSubtitle)
        val metricCoverage = card.findViewById<TextView>(R.id.metricCoverage)
        val metricFrames = card.findViewById<TextView>(R.id.metricFrames)
        val metricPoints = card.findViewById<TextView>(R.id.metricPoints)

        val progressLayout = card.findViewById<LinearLayout>(R.id.scanProgressLayout)
        val progressBar = card.findViewById<ProgressBar>(R.id.scanProgressBar)
        val progressText = card.findViewById<TextView>(R.id.scanProgressText)
        val actionsLayout = card.findViewById<LinearLayout>(R.id.scanActionsLayout)

        val btnInspect3d = card.findViewById<TextView>(R.id.btnInspect3d)
        val btnAddCoverage = card.findViewById<TextView>(R.id.btnAddCoverage)
        val btnProcess = card.findViewById<TextView>(R.id.btnProcess)
        val btnDelete = card.findViewById<TextView>(R.id.btnDelete)

        labelView.text = room.label
        val isProcessed = ScanProcessor.isProcessed(this, room.sessionName)
        var hasCheckpoint = if (!isProcessed) ScanProcessor.hasCheckpoint(this, room.sessionName) else false

        subtitleView.text = room.sessionName

        val covInt = room.coveragePercent?.toInt()
        metricCoverage.text = if (covInt != null) "$covInt% Coverage" else "Coverage --"
        metricFrames.text = "Keyframes"
        metricPoints.text = "3D Cloud"

        ScanPreviewLoader.bindScanCardPreview(
            this,
            room.sessionName,
            thumbnailContainer,
            thumbnailFlipper,
            thumbnailView,
            pcPreviewView,
            thumbnailBadge,
            scanFolder = room.scanFolder,
        )

        fun updateStatusUi() {
            when {
                isProcessed -> {
                    statusBadge.text = "PROCESSED"
                    statusBadge.setTextColor(color(R.color.action_ready))
                    btnProcess.text = "Share"
                    btnProcess.setBackgroundResource(R.drawable.btn_card_secondary)
                    btnAddCoverage.isEnabled = false
                    btnAddCoverage.alpha = 0.4f
                    btnAddCoverage.text = "Locked"
                }
                hasCheckpoint -> {
                    statusBadge.text = "RESUME"
                    statusBadge.setTextColor(color(R.color.warn))
                    btnProcess.text = "Resume"
                    btnProcess.setBackgroundResource(R.drawable.btn_card_action)
                    btnAddCoverage.isEnabled = true
                    btnAddCoverage.alpha = 1.0f
                    btnAddCoverage.text = "＋ Add Coverage"
                }
                else -> {
                    statusBadge.text = "READY"
                    statusBadge.setTextColor(color(R.color.action))
                    btnProcess.text = "⚙ Process"
                    btnProcess.setBackgroundResource(R.drawable.btn_card_action)
                    btnAddCoverage.isEnabled = true
                    btnAddCoverage.alpha = 1.0f
                    btnAddCoverage.text = "＋ Add Coverage"
                }
            }
        }
        updateStatusUi()

        btnInspect3d.setOnClickListener {
            PointCloudViewerActivity.start(this, room.sessionName, room.scanFolder, room.label)
        }

        btnAddCoverage.setOnClickListener {
            if (ScanProcessor.isProcessed(this, room.sessionName)) {
                Toast.makeText(this, "Processed scans cannot be extended.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val id = propertyId ?: return@setOnClickListener
            startActivity(
                Intent(this, CaptureActivity::class.java)
                    .putExtra(PropertyStore.EXTRA_PROPERTY_ID, id)
                    .putExtra(PropertyStore.EXTRA_ROOM_LABEL, room.label)
                    .putExtra(CaptureActivity.EXTRA_ALIGN_TO_SCAN_FOLDER, room.scanFolder)
                    .putExtra(CaptureActivity.EXTRA_ALIGN_TO_SESSION, room.sessionName)
            )
        }

        fun runProcessing() {
            actionsLayout.visibility = android.view.View.GONE
            progressLayout.visibility = android.view.View.VISIBLE

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
                    progressLayout.visibility = android.view.View.GONE
                    actionsLayout.visibility = android.view.View.VISIBLE
                    if (success) {
                        ScanProcessor.markProcessed(room.sessionName)
                        reload()
                        Toast.makeText(this, "Processing complete!", Toast.LENGTH_SHORT).show()
                    } else {
                        hasCheckpoint = true
                        updateStatusUi()
                        Toast.makeText(this, "Processing interrupted. Tap Resume to continue.", Toast.LENGTH_LONG).show()
                    }
                }
            }.start()
        }

        btnProcess.setOnClickListener {
            if (ScanProcessor.isProcessed(this, room.sessionName)) {
                ScanProcessor.shareScan(this, room.sessionName)
                return@setOnClickListener
            }
            showProcessConfirmationDialog(room) {
                runProcessing()
            }
        }

        btnDelete.setOnClickListener {
            confirmRemove(room)
        }

        return card
    }

    private fun confirmRemove(room: PropertyStore.RoomScan) {
        AlertDialog.Builder(this)
            .setTitle("Remove room?")
            .setMessage(
                "\"${room.label}\" will be unlinked from this property. Its scan file stays on " +
                    "the phone and moves to Unassigned in All scans."
            )
            .setPositiveButton("Remove") { _, _ ->
                store.deleteRoomScan(room.sessionName, room.scanFolder)
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
                                    .putExtra(CaptureActivity.EXTRA_ALIGN_TO_SCAN_FOLDER, lastRoom.scanFolder)
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
