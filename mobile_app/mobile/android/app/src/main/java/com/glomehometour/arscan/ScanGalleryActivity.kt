package com.glomehometour.arscan

import android.app.AlertDialog
import android.content.ContentUris
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.Gravity
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Lists previous scans -- normally a single "$session.zip" under Documents/GlomeHomeTour/ in
 * MediaStore (or the app's external files dir when the MediaStore probe fell back), see
 * DatasetWriter. Sessions captured before zipping was added may still be loose per-session
 * folders; those are listed too so an operator can clear them out.
 */
class ScanGalleryActivity : AppCompatActivity() {

    private data class ScanEntry(
        val sessionName: String,
        var sizeBytes: Long,
        var lastModifiedMs: Long,
        var mediaZipId: Long? = null,
        var fallbackZipFile: File? = null,
        var legacyMediaSession: Boolean = false,
        var legacyFlatMediaSession: Boolean = false,
        var legacyFallbackDir: File? = null,
    )

    companion object {
        /** Matches a manifest or image file written flat under the album folder by a capture
         * that crashed before finish() could zip and clean it up, e.g.
         * "scan_20260913_133535_transforms.json" or "scan_20260913_133535_images_frame_00005.jpg". */
        private val FLAT_SESSION_FILE = Regex(
            "^(.+?)_(?:images_frame_\\d+\\.jpg|transforms\\.json|trajectory\\.csv|" +
                "coverage_summary\\.json|focus_metadata\\.json|probe\\.jpg)$",
        )
    }

    private lateinit var listContainer: LinearLayout
    private lateinit var emptyText: TextView

    private val store by lazy { PropertyStore(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_scan_gallery)
        listContainer = findViewById(R.id.scanList)
        emptyText = findViewById(R.id.emptyText)
        findViewById<TextView>(R.id.closeButton).setOnClickListener { finish() }
    }

    override fun onResume() {
        super.onResume()
        reload()
    }

    /**
     * Files first, property index second: the zips on disk are the source of truth, and a scan is
     * never hidden because the index disagrees with them. Zips with no index row (captured before
     * properties existed, or unlinked since) land under "Unassigned"; index rows whose zip is gone
     * (deleted by a file manager) show as missing rather than vanishing silently.
     */
    private fun reload() {
        val entries = loadScans().sortedByDescending { it.lastModifiedMs }
        val roomsBySession = store.roomsBySession()
        listContainer.removeAllViews()
        emptyText.setVisible(entries.isEmpty() && roomsBySession.isEmpty())

        val byProperty = entries.groupBy { roomsBySession[it.sessionName]?.propertyId }
        for (property in store.properties()) {
            val owned = byProperty[property.id].orEmpty()
            val missing = store.rooms(property.id).filter { room ->
                entries.none { it.sessionName == room.sessionName }
            }
            if (owned.isEmpty() && missing.isEmpty()) continue
            listContainer.addView(buildHeader(property.address, owned.size + missing.size))
            for (entry in owned) listContainer.addView(buildRow(entry, roomsBySession[entry.sessionName]))
            for (room in missing) listContainer.addView(buildMissingRow(room))
        }

        val unassigned = byProperty[null].orEmpty()
        if (unassigned.isNotEmpty()) {
            listContainer.addView(buildHeader("Unassigned", unassigned.size))
            for (entry in unassigned) listContainer.addView(buildRow(entry, null))
        }
    }

    private fun buildHeader(title: String, count: Int): TextView = TextView(this).apply {
        text = "$title · $count"
        setTextColor(color(R.color.text_secondary))
        textSize = 13f
        setPadding(dp(4), dp(12), dp(4), dp(6))
    }

    /** An index row whose zip is no longer on disk. Offers the one useful action: forget it. */
    private fun buildMissingRow(room: PropertyStore.RoomScan): LinearLayout {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(12), dp(12), dp(12))
            background = getDrawable(R.drawable.card)
            alpha = 0.6f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = dp(8) }
        }
        row.addView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            addView(TextView(this@ScanGalleryActivity).apply {
                text = room.label
                setTextColor(color(R.color.text_primary))
                textSize = 14f
            })
            addView(TextView(this@ScanGalleryActivity).apply {
                text = "File missing (${room.sessionName})"
                setTextColor(color(R.color.danger))
                textSize = 12f
            })
        })
        row.addView(TextView(this).apply {
            text = "Forget"
            setTextColor(color(R.color.danger))
            textSize = 13f
            setOnClickListener {
                store.deleteRoomScan(room.sessionName)
                reload()
            }
        })
        return row
    }

    /** Opens the entry's zip for reading, wherever it lives. Null for legacy loose sessions. */
    private fun openZip(entry: ScanEntry): InputStream? {
        entry.mediaZipId?.let { id ->
            val uri = ContentUris.withAppendedId(
                MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), id,
            )
            return contentResolver.openInputStream(uri)
        }
        return entry.fallbackZipFile?.inputStream()
    }

    private fun loadScans(): List<ScanEntry> {
        val bySession = LinkedHashMap<String, ScanEntry>()
        collectFromMediaStore(bySession)
        collectFromFallbackDir(bySession)
        return bySession.values.toList()
    }

    /** Documents/GlomeHomeTour/<session>.zip (current layout), or, for scans captured before
     * zipping was added, the loose Documents/GlomeHomeTour/<session>/... files (legacy). */
    private fun collectFromMediaStore(out: MutableMap<String, ScanEntry>) {
        val albumPrefix = "Documents/${DatasetWriter.ALBUM}/"
        val projection = arrayOf(
            MediaStore.Files.FileColumns._ID,
            MediaStore.Files.FileColumns.DISPLAY_NAME,
            MediaStore.Files.FileColumns.RELATIVE_PATH,
            MediaStore.Files.FileColumns.SIZE,
            MediaStore.Files.FileColumns.DATE_MODIFIED,
        )
        contentResolver.query(
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY),
            projection,
            "${MediaStore.Files.FileColumns.RELATIVE_PATH} LIKE ?",
            arrayOf("$albumPrefix%"),
            null,
        )?.use { cursor ->
            val idCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
            val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
            val pathCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.RELATIVE_PATH)
            val sizeCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.SIZE)
            val dateCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DATE_MODIFIED)
            while (cursor.moveToNext()) {
                val relPath = cursor.getString(pathCol) ?: continue
                val size = cursor.getLong(sizeCol)
                val modifiedMs = cursor.getLong(dateCol) * 1000L
                if (relPath == albumPrefix) {
                    val displayName = cursor.getString(nameCol) ?: continue
                    val flatMatch = FLAT_SESSION_FILE.matchEntire(displayName)
                    when {
                        displayName.endsWith(".zip") -> {
                            val session = displayName.removeSuffix(".zip")
                            val entry = out.getOrPut(session) { ScanEntry(session, 0L, 0L) }
                            entry.mediaZipId = cursor.getLong(idCol)
                            entry.sizeBytes += size
                            entry.lastModifiedMs = maxOf(entry.lastModifiedMs, modifiedMs)
                        }
                        flatMatch != null -> {
                            val session = flatMatch.groupValues[1]
                            val entry = out.getOrPut(session) { ScanEntry(session, 0L, 0L) }
                            entry.legacyFlatMediaSession = true
                            entry.sizeBytes += size
                            entry.lastModifiedMs = maxOf(entry.lastModifiedMs, modifiedMs)
                        }
                    }
                } else {
                    val session = relPath.removePrefix(albumPrefix).substringBefore('/')
                    if (session.isEmpty()) continue
                    val entry = out.getOrPut(session) { ScanEntry(session, 0L, 0L) }
                    entry.legacyMediaSession = true
                    entry.sizeBytes += size
                    entry.lastModifiedMs = maxOf(entry.lastModifiedMs, modifiedMs)
                }
            }
        }
    }

    /** Fallback location used when the MediaStore probe failed for a given session. */
    private fun collectFromFallbackDir(out: MutableMap<String, ScanEntry>) {
        val root = getExternalFilesDir(null) ?: return
        val files = root.listFiles() ?: return
        for (f in files) {
            if (f.isFile && f.name.endsWith(".zip")) {
                val session = f.name.removeSuffix(".zip")
                val entry = out.getOrPut(session) { ScanEntry(session, 0L, 0L) }
                entry.fallbackZipFile = f
                entry.sizeBytes += f.length()
                entry.lastModifiedMs = maxOf(entry.lastModifiedMs, f.lastModified())
            } else if (f.isDirectory && f.name.startsWith("scan_")) {
                val size = f.walkTopDown().filter { it.isFile }.sumOf { it.length() }
                val entry = out.getOrPut(f.name) { ScanEntry(f.name, 0L, 0L) }
                entry.legacyFallbackDir = f
                entry.sizeBytes += size
                entry.lastModifiedMs = maxOf(entry.lastModifiedMs, f.lastModified())
            }
        }
    }

    private fun buildRow(entry: ScanEntry, room: PropertyStore.RoomScan?): LinearLayout {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(12), dp(12), dp(12))
            background = getDrawable(R.drawable.card)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = dp(8) }
        }

        // Legacy loose-file sessions (captured before finish() zipped, or crashed mid-capture)
        // have no zip to read a frame out of; show no thumbnail rather than an empty grey box
        // that reads as a failed load.
        if (entry.mediaZipId != null || entry.fallbackZipFile != null) {
            val thumbnail = ImageView(this).apply {
                scaleType = ImageView.ScaleType.CENTER_CROP
                setBackgroundColor(color(R.color.action_disabled))
                layoutParams = LinearLayout.LayoutParams(dp(56), dp(56)).apply { rightMargin = dp(12) }
            }
            row.addView(thumbnail)
            ThumbnailFetcher.load(entry.sessionName, thumbnail) { openZip(entry) }
        }

        val info = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        info.addView(TextView(this).apply {
            text = room?.label ?: entry.sessionName
            setTextColor(color(R.color.text_primary))
            textSize = 14f
        })
        info.addView(TextView(this).apply {
            val coverage = room?.coveragePercent?.let { "${it.toInt()}% · " } ?: ""
            text = "$coverage${formatDate(entry.lastModifiedMs)} · ${formatSize(entry.sizeBytes)}"
            setTextColor(color(R.color.text_faint))
            textSize = 12f
        })
        row.addView(info)

        // Right side toggles between the "Delete" label and a progress bar + percentage while a
        // deletion is in flight, so a bulk delete of a large session doesn't look like a freeze.
        val actionArea = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(dp(100), LinearLayout.LayoutParams.WRAP_CONTENT)
        }
        val deleteLabel = TextView(this).apply {
            text = "Delete"
            setTextColor(color(R.color.danger))
            textSize = 13f
            gravity = Gravity.END
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
            )
        }
        val progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            visibility = android.view.View.GONE
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val progressText = TextView(this).apply {
            setTextColor(color(R.color.text_faint))
            textSize = 11f
            visibility = android.view.View.GONE
            setPadding(dp(6), 0, 0, 0)
        }
        actionArea.addView(deleteLabel)
        actionArea.addView(progressBar)
        actionArea.addView(progressText)
        deleteLabel.setOnClickListener {
            confirmDelete(entry) {
                deleteLabel.visibility = android.view.View.GONE
                progressBar.visibility = android.view.View.VISIBLE
                progressText.visibility = android.view.View.VISIBLE
                deleteScan(entry) { percent ->
                    progressBar.progress = percent
                    progressText.text = "$percent%"
                }
            }
        }
        row.addView(actionArea)

        return row
    }

    private fun confirmDelete(entry: ScanEntry, onConfirmed: () -> Unit) {
        AlertDialog.Builder(this)
            .setTitle("Delete this scan?")
            .setMessage("${entry.sessionName} (${formatSize(entry.sizeBytes)}) will be permanently deleted.")
            .setPositiveButton("Delete") { _, _ -> onConfirmed() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /**
     * Runs off the main thread and reports 0-100 progress via [onProgress] (posted back to the UI
     * thread). The common case (a single zip) is one delete; legacy loose sessions still delete
     * one row/file at a time so real progress is available to report on a large one, and their
     * now-empty directory tree is removed afterwards instead of being left behind.
     */
    private fun deleteScan(entry: ScanEntry, onProgress: (Int) -> Unit) {
        Thread {
            val uri = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val legacyIds = mutableListOf<Long>()
            if (entry.legacyMediaSession) {
                val prefix = "Documents/${DatasetWriter.ALBUM}/${entry.sessionName}/"
                contentResolver.query(
                    uri,
                    arrayOf(MediaStore.Files.FileColumns._ID),
                    "${MediaStore.Files.FileColumns.RELATIVE_PATH} LIKE ?",
                    arrayOf("$prefix%"),
                    null,
                )?.use { cursor ->
                    val idCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
                    while (cursor.moveToNext()) legacyIds.add(cursor.getLong(idCol))
                }
            }
            if (entry.legacyFlatMediaSession) {
                contentResolver.query(
                    uri,
                    arrayOf(MediaStore.Files.FileColumns._ID),
                    "${MediaStore.Files.FileColumns.RELATIVE_PATH} = ? AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ?",
                    arrayOf("Documents/${DatasetWriter.ALBUM}/", "${entry.sessionName}_%"),
                    null,
                )?.use { cursor ->
                    val idCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
                    while (cursor.moveToNext()) legacyIds.add(cursor.getLong(idCol))
                }
            }
            val legacyFiles = entry.legacyFallbackDir?.walkTopDown()?.filter { it.isFile }?.toList().orEmpty()

            val total = (if (entry.mediaZipId != null) 1 else 0) +
                (if (entry.fallbackZipFile != null) 1 else 0) + legacyIds.size + legacyFiles.size
            var done = 0
            var lastReported = -1
            fun reportProgress() {
                val percent = if (total == 0) 100 else (done * 100 / total)
                if (percent != lastReported) {
                    lastReported = percent
                    runOnUiThread { onProgress(percent) }
                }
            }
            reportProgress()

            entry.mediaZipId?.let { id ->
                contentResolver.delete(ContentUris.withAppendedId(uri, id), null, null)
                done++; reportProgress()
            }
            entry.fallbackZipFile?.let { file ->
                file.delete()
                done++; reportProgress()
            }
            for (id in legacyIds) {
                contentResolver.delete(ContentUris.withAppendedId(uri, id), null, null)
                done++; reportProgress()
            }
            for (file in legacyFiles) {
                file.delete()
                done++; reportProgress()
            }
            if (entry.legacyMediaSession) {
                val albumDir = File(Environment.getExternalStorageDirectory(), "Documents/${DatasetWriter.ALBUM}")
                val sessionDir = File(albumDir, entry.sessionName)
                try {
                    deleteEmptyDirUpwards(sessionDir, stopAt = albumDir)
                } catch (e: Exception) {
                    // Best-effort: scoped storage may not grant direct File access to this path.
                }
            }
            entry.legacyFallbackDir?.deleteRecursively()

            // The zip was the real data; its index row is meaningless without it.
            store.deleteRoomScan(entry.sessionName)
            runOnUiThread { reload() }
        }.start()
    }

    private fun color(id: Int) = getColor(id)
    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private fun formatDate(ms: Long): String =
        if (ms <= 0L) "Unknown date" else SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(ms)

    private fun formatSize(bytes: Long): String = when {
        bytes >= 1L shl 30 -> "%.1f GB".format(bytes / (1L shl 30).toDouble())
        bytes >= 1L shl 20 -> "%.1f MB".format(bytes / (1L shl 20).toDouble())
        bytes >= 1L shl 10 -> "%.1f KB".format(bytes / (1L shl 10).toDouble())
        else -> "$bytes B"
    }
}

internal fun android.view.View.setVisible(visible: Boolean) {
    visibility = if (visible) android.view.View.VISIBLE else android.view.View.GONE
}
