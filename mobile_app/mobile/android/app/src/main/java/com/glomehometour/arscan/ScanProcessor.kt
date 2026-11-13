package com.glomehometour.arscan

import android.content.ClipData
import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.core.content.FileProvider
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.security.MessageDigest
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.math.atan

/**
 * Executes the on-device preprocessing pipeline on an uncompressed scan session:
 * 1. Evaluates blur, exposure, and texture using QualityGate.
 * 2. Prunes redundant views using KeyframeSelector (ARCore trajectory & floor area budgeting).
 * 3. Rotates kept landscape frames 90° clockwise into standard portrait orientation, updating
 *    intrinsics and camera poses.
 * 4. Generates certificate.json containing provenance, completed stages, orientation state,
 *    keyframe density settings, retained-frame counts, and SHA-256 hashes of manifests and images.
 * 5. Compresses the survivors, manifests, and certificate into "$sessionName.zip" and deletes uncompressed files.
 */
class ScanProcessor(private val context: Context? = null) {

    companion object {
        private const val TAG = "ScanProcessor"
        const val CHECKPOINT_FILE = "processing_checkpoint.json"

        fun getAccessor(context: Context, sessionName: String): FileAccessor {
            val fallbackDir = File(context.getExternalFilesDir(null), sessionName)
            return if (fallbackDir.isDirectory) {
                DirectoryFileAccessor(fallbackDir)
            } else {
                MediaStoreFlatAccessor(context, sessionName)
            }
        }

        fun listScanFolders(context: Context, sessionName: String): List<String> {
            val fallbackDir = File(context.getExternalFilesDir(null), sessionName)
            if (fallbackDir.isDirectory) return DirectoryFileAccessor(fallbackDir).listScanFolders()

            fun folders(zip: java.util.zip.ZipFile): List<String> {
                val scanFolders = zip.entries().asSequence().mapNotNull { entry ->
                    Regex("^(scan_\\d+)/transforms\\.json$").matchEntire(entry.name)?.groupValues?.get(1)
                }.distinct().sortedBy { it.removePrefix("scan_").toIntOrNull() ?: 0 }.toList()
                return scanFolders.ifEmpty { if (zip.getEntry("transforms.json") != null) listOf("") else emptyList() }
            }

            val file = findZipFile(context, sessionName)
            if (file != null) {
                try {
                    return java.util.zip.ZipFile(file).use(::folders)
                } catch (e: Exception) {
                    Log.w(TAG, "Could not inspect room folders in ${file.name}", e)
                }
            }

            val id = queryMediaStoreZipId(context, sessionName)
            if (id != null) {
                val uri = ContentUris.withAppendedId(MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), id)
                try {
                    context.contentResolver.openFileDescriptor(uri, "r")?.use { pfd ->
                        return java.util.zip.ZipFile(File("/proc/self/fd/${pfd.fd}")).use(::folders)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Could not inspect MediaStore room archive for $sessionName", e)
                }
            }

            return getAccessor(context, sessionName).listScanFolders()
        }

        fun findZipFile(context: Context, sessionName: String): File? {
            val fallbackZip = File(context.getExternalFilesDir(null), "$sessionName.zip")
            if (fallbackZip.isFile) return fallbackZip

            val docFile = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
                "${DatasetWriter.ALBUM}/$sessionName.zip",
            )
            if (docFile.isFile) return docFile
            return null
        }

        fun queryMediaStoreZipId(context: Context, sessionName: String): Long? {
            val uri = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val projection = arrayOf(MediaStore.Files.FileColumns._ID)
            val selection = "${MediaStore.Files.FileColumns.DISPLAY_NAME} = ? AND (${MediaStore.Files.FileColumns.RELATIVE_PATH} LIKE ? OR ${MediaStore.Files.FileColumns.RELATIVE_PATH} = ?)"
            val selectionArgs = arrayOf(
                "$sessionName.zip",
                "Documents/${DatasetWriter.ALBUM}%",
                "Documents/${DatasetWriter.ALBUM}",
            )
            return try {
                context.contentResolver.query(uri, projection, selection, selectionArgs, null)?.use { cursor ->
                    if (cursor.moveToFirst()) cursor.getLong(0) else null
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to query MediaStore for zip $sessionName", e)
                null
            }
        }

        fun readTrajectoryCsv(context: Context, sessionName: String, scanFolder: String): String? {
            val roomPath = "$scanFolder/trajectory.csv"
            val dirs = listOfNotNull(
                context.getExternalFilesDir(null)?.let { File(it, sessionName) },
                File(context.filesDir, sessionName),
            )
            for (dir in dirs) {
                try {
                    val roomFile = File(dir, roomPath)
                    if (roomFile.isFile) return roomFile.readText()
                    if (scanFolder == "scan_0") {
                        val legacyFile = File(dir, "trajectory.csv")
                        if (legacyFile.isFile) return legacyFile.readText()
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Could not read trajectory from ${dir.name}", e)
                }
            }

            try {
                val accessor = getAccessor(context, sessionName)
                accessor.readText(roomPath)?.let { return it }
                if (scanFolder == "scan_0") accessor.readText("trajectory.csv")?.let { return it }
            } catch (e: Exception) {
                Log.w(TAG, "Could not read trajectory for $sessionName/$scanFolder", e)
            }

            val zipFile = findZipFile(context, sessionName)
            if (zipFile != null) {
                try {
                    java.util.zip.ZipFile(zipFile).use { zip ->
                        val entry = zip.getEntry(roomPath)
                            ?: if (scanFolder == "scan_0") zip.getEntry("trajectory.csv") else null
                        if (entry != null) return zip.getInputStream(entry).bufferedReader().use { it.readText() }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Could not read trajectory from ${zipFile.name}", e)
                }
            }

            val zipId = queryMediaStoreZipId(context, sessionName) ?: return null
            val zipUri = ContentUris.withAppendedId(MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), zipId)
            return try {
                context.contentResolver.openInputStream(zipUri)?.use rawStream@ { raw ->
                    java.util.zip.ZipInputStream(raw).use zipStream@ { zip ->
                        var entry = zip.nextEntry
                        while (entry != null) {
                            if (entry.name == roomPath || (scanFolder == "scan_0" && entry.name == "trajectory.csv")) {
                                return@zipStream zip.readBytes().toString(Charsets.UTF_8)
                            }
                            entry = zip.nextEntry
                        }
                        null
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Could not read trajectory from MediaStore archive for $sessionName", e)
                null
            }
        }

        private val processedCache = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()

        private data class PropertyRoomSource(val file: File, val scanFolder: String)

        fun markProcessed(sessionName: String) {
            processedCache.add(sessionName)
        }

        fun isProcessed(context: Context, sessionName: String): Boolean {
            if (processedCache.contains(sessionName)) return true
            val found = findZipFile(context, sessionName) != null || queryMediaStoreZipId(context, sessionName) != null
            if (found) {
                processedCache.add(sessionName)
            }
            return found
        }

        fun hasCheckpoint(context: Context, sessionName: String): Boolean {
            return try {
                val accessor = getAccessor(context, sessionName)
                hasCheckpoint(accessor)
            } catch (e: Exception) {
                false
            }
        }

        fun findZipUri(context: Context, sessionName: String): Uri? {
            val file = findZipFile(context, sessionName)
            if (file != null) {
                try {
                    return FileProvider.getUriForFile(
                        context,
                        "${context.applicationContext.packageName}.fileprovider",
                        file,
                    )
                } catch (e: Exception) {
                    Log.w(TAG, "FileProvider failed for file: ${file.absolutePath}", e)
                }
            }
            val id = queryMediaStoreZipId(context, sessionName)
            if (id != null) {
                return ContentUris.withAppendedId(
                    MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY),
                    id,
                )
            }
            return null
        }

        fun shareScan(context: Context, sessionName: String) {
            val uri = findZipUri(context, sessionName)
            if (uri == null) {
                Toast.makeText(context, "Zip file not available to share", Toast.LENGTH_SHORT).show()
                return
            }
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "application/zip"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "$sessionName.zip")
                clipData = ClipData.newRawUri("$sessionName.zip", uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            context.startActivity(Intent.createChooser(shareIntent, "Share $sessionName.zip"))
        }

        fun getSessionZipFile(context: Context, sessionName: String): File? {
            val f = findZipFile(context, sessionName)
            if (f != null && f.isFile) return f
            val uri = findZipUri(context, sessionName)
            if (uri != null) {
                try {
                    val temp = File(context.cacheDir, "temp_${sessionName}.zip")
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(temp).use { output -> input.copyTo(output) }
                    }
                    if (temp.isFile && temp.length() > 0) return temp
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to copy MediaStore zip to cache", e)
                }
            }
            return null
        }

        fun packagePropertyScan(
            context: Context,
            property: PropertyStore.Property,
            rooms: List<PropertyStore.RoomScan>,
            onProgress: ((Int) -> Unit)? = null,
        ): Uri? {
            if (rooms.isEmpty()) return null
            val cleanAddress = property.address.replace(Regex("[^a-zA-Z0-9_-]"), "_").trim('_').ifEmpty { "property" }
            val zipName = "${cleanAddress}_scan.zip"
            val targetZipFile = File(context.getExternalFilesDir(null), zipName)

            val roomSources = mutableListOf<PropertyRoomSource>()
            val tempFilesToDelete = mutableListOf<File>()
            val sourcesBySession = mutableMapOf<String, File?>()

            for (room in rooms) {
                val zipFile = sourcesBySession.getOrPut(room.sessionName) {
                    getSessionZipFile(context, room.sessionName)
                }
                if (zipFile != null && zipFile.isFile) {
                    if (zipFile.parentFile == context.cacheDir) {
                        tempFilesToDelete.add(zipFile)
                    }
                    roomSources.add(PropertyRoomSource(zipFile, room.scanFolder.ifBlank { "scan_0" }))
                } else {
                    val fallbackDir = File(context.getExternalFilesDir(null), room.sessionName)
                    if (fallbackDir.isDirectory) {
                        roomSources.add(PropertyRoomSource(fallbackDir, room.scanFolder.ifBlank { "scan_0" }))
                    }
                }
            }

            try {
                val ok = packagePropertyRoomsArchive(targetZipFile, roomSources, onProgress)
                if (!ok) return null

                // Copy to Documents/GlomeHomeTour for user visibility if possible
                try {
                    val docDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), DatasetWriter.ALBUM)
                    if (docDir.isDirectory || docDir.mkdirs()) {
                        val dest = File(docDir, zipName)
                        targetZipFile.copyTo(dest, overwrite = true)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Could not copy property zip to Documents", e)
                }

                return FileProvider.getUriForFile(
                    context,
                    "${context.applicationContext.packageName}.fileprovider",
                    targetZipFile,
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to package property scan", e)
                return null
            } finally {
                tempFilesToDelete.forEach { it.delete() }
            }
        }

        fun packagePropertyScanArchive(
            targetZipFile: File,
            roomSources: List<File>,
            onProgress: ((Int) -> Unit)? = null,
        ): Boolean {
            val expandedSources = roomSources.flatMap { source ->
                listRoomFolders(source).map { folder -> PropertyRoomSource(source, folder) }
            }
            return packagePropertyRoomsArchive(targetZipFile, expandedSources, onProgress)
        }

        private fun packagePropertyRoomsArchive(
            targetZipFile: File,
            roomSources: List<PropertyRoomSource>,
            onProgress: ((Int) -> Unit)? = null,
        ): Boolean {
            if (roomSources.isEmpty()) return false
            val manifestHashes = mutableMapOf<String, String>()
            val imageHashes = mutableMapOf<String, String>()
            val countedCertificates = mutableSetOf<String>()

            var overallRawW = 1920
            var overallRawH = 1080
            var overallProcessedW = 1080
            var overallProcessedH = 1920
            var overallAlreadyPortrait = false
            var overallTotalInput = 0
            var overallRetainedCount = 0

            try {
                onProgress?.invoke(5)
                targetZipFile.parentFile?.mkdirs()
                FileOutputStream(targetZipFile).buffered(65536).use { fos ->
                    ZipOutputStream(fos).use { zos ->
                        zos.setLevel(java.util.zip.Deflater.BEST_SPEED)

                        for ((roomIdx, roomSource) in roomSources.withIndex()) {
                            val source = roomSource.file
                            val sourceFolder = roomSource.scanFolder
                            val scanFolder = "scan_$roomIdx"
                            val startPct = 5 + (roomIdx * 85 / roomSources.size)
                            val endPct = 5 + ((roomIdx + 1) * 85 / roomSources.size)
                            onProgress?.invoke(startPct)

                            if (source.isFile) {
                                java.util.zip.ZipFile(source).use { zf ->
                                    val entries = zf.entries().asSequence().toList()
                                    for (entry in entries) {
                                        if (entry.isDirectory) continue
                                        val entryName = entry.name
                                        if (!belongsToRoom(entryName, sourceFolder)) continue
                                        if (entryName.endsWith("processing_checkpoint.json")) continue
                                        if (entryName == "certificate.json" || entryName.endsWith("/certificate.json")) {
                                            if (!countedCertificates.add(source.absolutePath)) continue
                                            try {
                                                val certText = zf.getInputStream(entry).bufferedReader().readText()
                                                val certObj = JSONObject(certText)
                                                val fObj = certObj.optJSONObject("frames")
                                                if (fObj != null) {
                                                    overallTotalInput += fObj.optInt("total_input_frames", 0)
                                                    overallRetainedCount += fObj.optInt("retained_frames_count", 0)
                                                }
                                                val oObj = certObj.optJSONObject("orientation_state")
                                                if (oObj != null) {
                                                    overallRawW = oObj.optInt("raw_width", overallRawW)
                                                    overallRawH = oObj.optInt("raw_height", overallRawH)
                                                    overallProcessedW = oObj.optInt("processed_width", overallProcessedW)
                                                    overallProcessedH = oObj.optInt("processed_height", overallProcessedH)
                                                    overallAlreadyPortrait = oObj.optBoolean("is_already_portrait", overallAlreadyPortrait)
                                                }
                                            } catch (e: Exception) {
                                                Log.w(TAG, "Error reading room certificate for ${source.name}", e)
                                            }
                                            continue
                                        }

                                        val cleanRel = roomRelativePath(entryName, sourceFolder)
                                        val targetEntryName = "$scanFolder/$cleanRel"

                                        if (cleanRel == "transforms.json") {
                                            val trText = zf.getInputStream(entry).bufferedReader().readText()
                                            val trObj = JSONObject(trText)
                                            val frames = trObj.optJSONArray("frames")
                                            if (frames != null) {
                                                for (fi in 0 until frames.length()) {
                                                    val frame = frames.getJSONObject(fi)
                                                    val origFp = frame.optString("file_path", "")
                                                    val cleanFp = origFp.replace(Regex("""^(\./)?(scan_\d+/)?images/"""), "images/")
                                                    frame.put("file_path", cleanFp)
                                                }
                                            }
                                            val cleanBytes = trObj.toString(2).toByteArray(Charsets.UTF_8)
                                            val sha = computeSha256(cleanBytes.inputStream())
                                            manifestHashes[targetEntryName] = sha
                                            zos.putNextEntry(ZipEntry(targetEntryName))
                                            zos.write(cleanBytes)
                                            zos.closeEntry()
                                        } else if (cleanRel in listOf("trajectory.csv", "coverage_summary.json", "focus_metadata.json", "pointcloud.ply")) {
                                            val bytes = zf.getInputStream(entry).readBytes()
                                            val sha = computeSha256(bytes.inputStream())
                                            manifestHashes[targetEntryName] = sha
                                            zos.putNextEntry(ZipEntry(targetEntryName))
                                            zos.write(bytes)
                                            zos.closeEntry()
                                        } else if (cleanRel.startsWith("images/")) {
                                            zos.putNextEntry(ZipEntry(targetEntryName))
                                            val digest = MessageDigest.getInstance("SHA-256")
                                            zf.getInputStream(entry).use { inStream ->
                                                val buf = ByteArray(65536)
                                                var len: Int
                                                while (inStream.read(buf).also { len = it } != -1) {
                                                    digest.update(buf, 0, len)
                                                    zos.write(buf, 0, len)
                                                }
                                            }
                                            zos.closeEntry()
                                            val sha = digest.digest().joinToString("") { "%02x".format(it) }
                                            imageHashes[targetEntryName] = sha
                                        }
                                    }
                                }
                            } else if (source.isDirectory) {
                                val files = source.walkTopDown().filter { it.isFile }.toList()
                                for (file in files) {
                                    val entryName = file.relativeTo(source).path
                                    if (!belongsToRoom(entryName, sourceFolder)) continue
                                    if (entryName.endsWith("processing_checkpoint.json")) continue
                                    if (entryName == "certificate.json" || entryName.endsWith("/certificate.json")) {
                                        if (!countedCertificates.add(source.absolutePath)) continue
                                        try {
                                            val certObj = JSONObject(file.readText())
                                            val fObj = certObj.optJSONObject("frames")
                                            if (fObj != null) {
                                                overallTotalInput += fObj.optInt("total_input_frames", 0)
                                                overallRetainedCount += fObj.optInt("retained_frames_count", 0)
                                            }
                                            val oObj = certObj.optJSONObject("orientation_state")
                                            if (oObj != null) {
                                                overallRawW = oObj.optInt("raw_width", overallRawW)
                                                overallRawH = oObj.optInt("raw_height", overallRawH)
                                                overallProcessedW = oObj.optInt("processed_width", overallProcessedW)
                                                overallProcessedH = oObj.optInt("processed_height", overallProcessedH)
                                                overallAlreadyPortrait = oObj.optBoolean("is_already_portrait", overallAlreadyPortrait)
                                            }
                                        } catch (e: Exception) {
                                            Log.w(TAG, "Error reading room certificate for ${source.name}", e)
                                        }
                                        continue
                                    }

                                    val cleanRel = roomRelativePath(entryName, sourceFolder)
                                    val targetEntryName = "$scanFolder/$cleanRel"

                                    if (cleanRel == "transforms.json") {
                                        val trText = file.readText()
                                        val trObj = JSONObject(trText)
                                        val frames = trObj.optJSONArray("frames")
                                        if (frames != null) {
                                            for (fi in 0 until frames.length()) {
                                                val frame = frames.getJSONObject(fi)
                                                val origFp = frame.optString("file_path", "")
                                                val cleanFp = origFp.replace(Regex("""^(\./)?(scan_\d+/)?images/"""), "images/")
                                                frame.put("file_path", cleanFp)
                                            }
                                        }
                                        val cleanBytes = trObj.toString(2).toByteArray(Charsets.UTF_8)
                                        val sha = computeSha256(cleanBytes.inputStream())
                                        manifestHashes[targetEntryName] = sha
                                        zos.putNextEntry(ZipEntry(targetEntryName))
                                        zos.write(cleanBytes)
                                        zos.closeEntry()
                                    } else if (cleanRel in listOf("trajectory.csv", "coverage_summary.json", "focus_metadata.json", "pointcloud.ply")) {
                                        val bytes = file.readBytes()
                                        val sha = computeSha256(bytes.inputStream())
                                        manifestHashes[targetEntryName] = sha
                                        zos.putNextEntry(ZipEntry(targetEntryName))
                                        zos.write(bytes)
                                        zos.closeEntry()
                                    } else if (cleanRel.startsWith("images/")) {
                                        zos.putNextEntry(ZipEntry(targetEntryName))
                                        val digest = MessageDigest.getInstance("SHA-256")
                                        file.inputStream().use { inStream ->
                                            val buf = ByteArray(65536)
                                            var len: Int
                                            while (inStream.read(buf).also { len = it } != -1) {
                                                digest.update(buf, 0, len)
                                                zos.write(buf, 0, len)
                                            }
                                        }
                                        zos.closeEntry()
                                        val sha = digest.digest().joinToString("") { "%02x".format(it) }
                                        imageHashes[targetEntryName] = sha
                                    }
                                }
                            }
                            onProgress?.invoke(endPct)
                        }

                        // Generate and write single root certificate.json
                        onProgress?.invoke(92)
                        val finalRetainedCount = if (overallRetainedCount > 0) overallRetainedCount else imageHashes.size
                        val finalTotalInput = maxOf(overallTotalInput, finalRetainedCount)

                        val certJson = buildCertificateJson(
                            rawW = overallRawW,
                            rawH = overallRawH,
                            isAlreadyPortrait = overallAlreadyPortrait,
                            processedW = overallProcessedW,
                            processedH = overallProcessedH,
                            totalInputFrames = finalTotalInput,
                            retainedFramesCount = finalRetainedCount,
                            manifestHashes = manifestHashes,
                            imageHashes = imageHashes,
                        )
                        val certBytes = certJson.toString(2).toByteArray(Charsets.UTF_8)
                        zos.putNextEntry(ZipEntry("certificate.json"))
                        zos.write(certBytes)
                        zos.closeEntry()
                    }
                }
                onProgress?.invoke(100)
                return true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to package property scan archive", e)
                return false
            }
        }

        private fun ensurePortraitBitmap(bmp: Bitmap?): Bitmap? {
            if (bmp == null) return null
            return if (bmp.width > bmp.height) {
                val matrix = Matrix().apply { postRotate(90f) }
                val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
                bmp.recycle()
                rotated
            } else {
                bmp
            }
        }

        private fun listRoomFolders(source: File): List<String> {
            if (source.isDirectory) {
                val folders = source.listFiles()?.filter {
                    it.isDirectory && it.name.matches(Regex("^scan_\\d+$")) && File(it, "transforms.json").isFile
                }?.map { it.name }?.sortedBy { it.removePrefix("scan_").toIntOrNull() ?: 0 }.orEmpty()
                return folders.ifEmpty { if (File(source, "transforms.json").isFile) listOf("") else emptyList() }
            }
            return try {
                java.util.zip.ZipFile(source).use { zf ->
                    val folders = zf.entries().asSequence().mapNotNull { entry ->
                        Regex("^(scan_\\d+)/transforms\\.json$").matchEntire(entry.name)?.groupValues?.get(1)
                    }.distinct().sortedBy { it.removePrefix("scan_").toIntOrNull() ?: 0 }.toList()
                    folders.ifEmpty { if (zf.getEntry("transforms.json") != null) listOf("") else emptyList() }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Could not list room folders in ${source.name}", e)
                emptyList()
            }
        }

        private fun belongsToRoom(entryName: String, scanFolder: String): Boolean {
            if (entryName == "certificate.json") return true
            if (scanFolder.isEmpty()) return !entryName.matches(Regex("^scan_\\d+/.*"))
            return entryName.startsWith("$scanFolder/") ||
                (scanFolder == "scan_0" && !entryName.matches(Regex("^scan_\\d+/.*")))
        }

        private fun roomRelativePath(entryName: String, scanFolder: String): String =
            if (scanFolder.isNotEmpty() && entryName.startsWith("$scanFolder/")) {
                entryName.removePrefix("$scanFolder/")
            } else {
                entryName
            }

        fun getFirstFrameBitmap(
            context: Context,
            sessionName: String,
            sampleSize: Int = 4,
            scanFolder: String = "scan_0",
        ): Bitmap? {
            val options = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            fun isFrame(path: String) = path.startsWith("$scanFolder/images/") ||
                (scanFolder == "scan_0" && path.startsWith("images/"))

            // Check fallback dir first
            val fallbackDir = File(context.getExternalFilesDir(null), sessionName)
            if (fallbackDir.isDirectory) {
                val frameFiles = fallbackDir.walkTopDown()
                    .filter { it.isFile && it.name.startsWith("frame_") && it.name.endsWith(".jpg") && isFrame(it.relativeTo(fallbackDir).invariantSeparatorsPath) }
                    .toList()
                    .sortedBy { it.name }
                if (frameFiles.isNotEmpty()) {
                    return ensurePortraitBitmap(BitmapFactory.decodeFile(frameFiles.first().absolutePath, options))
                }
            }

            // Check MediaStore flat files
            val uri = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val projection = arrayOf(MediaStore.Files.FileColumns._ID, MediaStore.Files.FileColumns.DISPLAY_NAME)
            val selection = "(${MediaStore.Files.FileColumns.RELATIVE_PATH} = ? OR ${MediaStore.Files.FileColumns.RELATIVE_PATH} = ?) AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ? AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ?"
            val selectionArgs = arrayOf(
                "Documents/${DatasetWriter.ALBUM}/",
                "Documents/${DatasetWriter.ALBUM}",
                "${sessionName}_%",
                "%.jpg",
            )
            val frames = mutableListOf<Pair<Long, String>>()
            try {
                context.contentResolver.query(uri, projection, selection, selectionArgs, "${MediaStore.Files.FileColumns.DISPLAY_NAME} ASC")?.use { cursor ->
                    val idCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
                    val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
                    while (cursor.moveToNext()) {
                        val dName = cursor.getString(nameCol) ?: continue
                        if (dName.startsWith("${sessionName}_${scanFolder}_images_frame_") ||
                            (scanFolder == "scan_0" && dName.startsWith("${sessionName}_images_frame_"))) {
                            frames.add(cursor.getLong(idCol) to dName)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Querying MediaStore for first frame failed", e)
            }
            if (frames.isNotEmpty()) {
                val firstId = frames.first().first
                val firstUri = ContentUris.withAppendedId(uri, firstId)
                try {
                    context.contentResolver.openInputStream(firstUri)?.use { input ->
                        return ensurePortraitBitmap(BitmapFactory.decodeStream(input, null, options))
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to decode first frame from MediaStore", e)
                }
            }

            // Check fallback zip archive in app-private external files dir
            val fallbackZip = File(context.getExternalFilesDir(null), "$sessionName.zip")
            if (fallbackZip.isFile) {
                try {
                    java.util.zip.ZipFile(fallbackZip).use { zf ->
                        val entries = zf.entries().asSequence()
                            .filter { !it.isDirectory && isFrame(it.name) && it.name.endsWith(".jpg") }
                            .sortedBy { it.name }
                            .toList()
                        if (entries.isNotEmpty()) {
                            zf.getInputStream(entries.first()).use { stream ->
                                return ensurePortraitBitmap(BitmapFactory.decodeStream(stream, null, options))
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to read first frame from fallback zip", e)
                }
            }

            // Check MediaStore zip archive via ContentResolver
            val zipId = queryMediaStoreZipId(context, sessionName)
            if (zipId != null) {
                val zipUri = ContentUris.withAppendedId(
                    MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), zipId
                )
                // 1. Try random-access ZipFile via /proc/self/fd (instant central directory read)
                try {
                    context.contentResolver.openFileDescriptor(zipUri, "r")?.use { pfd ->
                        val fdFile = File("/proc/self/fd/${pfd.fd}")
                        java.util.zip.ZipFile(fdFile).use { zf ->
                            val entries = zf.entries().asSequence()
                                .filter { !it.isDirectory && isFrame(it.name) && it.name.endsWith(".jpg") }
                                .sortedBy { it.name }
                                .toList()
                            if (entries.isNotEmpty()) {
                                zf.getInputStream(entries.first()).use { stream ->
                                    return ensurePortraitBitmap(BitmapFactory.decodeStream(stream, null, options))
                                }
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.d(TAG, "ZipFile via /proc/self/fd failed for $sessionName, falling back to streaming", e)
                }

                // 2. Streaming fallback via ZipInputStream (returns first found frame immediately)
                try {
                    context.contentResolver.openInputStream(zipUri)?.use { rawStream ->
                        java.util.zip.ZipInputStream(rawStream).use { zis ->
                            var entry = zis.nextEntry
                            var count = 0
                            while (entry != null) {
                                if (!entry.isDirectory && isFrame(entry.name) && entry.name.endsWith(".jpg")) {
                                    val bytes = zis.readBytes()
                                    val bmp = ensurePortraitBitmap(BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options))
                                    if (bmp != null) return bmp
                                }
                                count++
                                if (count > 50) break
                                entry = zis.nextEntry
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to read first frame from MediaStore zip", e)
                }
            }

            // Check property scan archives (if session belongs to a packaged property)
            try {
                val store = PropertyStore(context)
                val room = store.roomScan(sessionName, scanFolder)
                if (room != null) {
                    val property = store.property(room.propertyId)
                    if (property != null) {
                        if (room.scanFolder.isNotBlank()) {
                            val scanPrefix = "${room.scanFolder}/images/"
                            val cleanAddress = property.address.replace(Regex("[^a-zA-Z0-9_-]"), "_").trim('_').ifEmpty { "property" }
                            val propZip = File(context.getExternalFilesDir(null), "${cleanAddress}_scan.zip")
                            if (propZip.isFile) {
                                try {
                                    java.util.zip.ZipFile(propZip).use { zf ->
                                        val entry = zf.entries().asSequence()
                                            .filter { !it.isDirectory && it.name.startsWith(scanPrefix) && it.name.endsWith(".jpg") }
                                            .minByOrNull { it.name }
                                        if (entry != null) {
                                            zf.getInputStream(entry).use { stream ->
                                                return ensurePortraitBitmap(BitmapFactory.decodeStream(stream, null, options))
                                            }
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.w(TAG, "Failed reading first frame from property zip", e)
                                }
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed checking property store for first frame", e)
            }

            return null
        }

        fun getLastFrameBitmap(context: Context, sessionName: String, scanFolder: String = "scan_0"): Bitmap? {
            fun isFrame(path: String) = path.startsWith("$scanFolder/images/") ||
                (scanFolder == "scan_0" && path.startsWith("images/"))
            // Check fallback dir first
            val fallbackDir = File(context.getExternalFilesDir(null), sessionName)
            if (fallbackDir.isDirectory) {
                val frameFiles = fallbackDir.walkTopDown()
                    .filter { it.isFile && it.name.startsWith("frame_") && it.name.endsWith(".jpg") && isFrame(it.relativeTo(fallbackDir).invariantSeparatorsPath) }
                    .toList()
                    .sortedBy { it.name }
                if (frameFiles.isNotEmpty()) {
                    return ensurePortraitBitmap(BitmapFactory.decodeFile(frameFiles.last().absolutePath))
                }
            }

            // Check MediaStore flat files
            val uri = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val projection = arrayOf(MediaStore.Files.FileColumns._ID, MediaStore.Files.FileColumns.DISPLAY_NAME)
            val selection = "(${MediaStore.Files.FileColumns.RELATIVE_PATH} = ? OR ${MediaStore.Files.FileColumns.RELATIVE_PATH} = ?) AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ? AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ?"
            val selectionArgs = arrayOf(
                "Documents/${DatasetWriter.ALBUM}/",
                "Documents/${DatasetWriter.ALBUM}",
                "${sessionName}_%",
                "%.jpg",
            )
            val frames = mutableListOf<Pair<Long, String>>()
            try {
                context.contentResolver.query(uri, projection, selection, selectionArgs, "${MediaStore.Files.FileColumns.DISPLAY_NAME} ASC")?.use { cursor ->
                    val idCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
                    val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
                    while (cursor.moveToNext()) {
                        val dName = cursor.getString(nameCol) ?: continue
                        if (dName.startsWith("${sessionName}_${scanFolder}_images_frame_") ||
                            (scanFolder == "scan_0" && dName.startsWith("${sessionName}_images_frame_"))) {
                            frames.add(cursor.getLong(idCol) to dName)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Querying MediaStore for last frame failed", e)
            }
            if (frames.isNotEmpty()) {
                val lastId = frames.last().first
                val lastUri = ContentUris.withAppendedId(uri, lastId)
                try {
                    context.contentResolver.openInputStream(lastUri)?.use { input ->
                        return ensurePortraitBitmap(BitmapFactory.decodeStream(input))
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to decode last frame from MediaStore", e)
                }
            }

            // Check fallback zip archive in app-private external files dir
            val fallbackZip = File(context.getExternalFilesDir(null), "$sessionName.zip")
            if (fallbackZip.isFile) {
                try {
                    java.util.zip.ZipFile(fallbackZip).use { zf ->
                        val entries = zf.entries().asSequence()
                            .filter { !it.isDirectory && isFrame(it.name) && it.name.endsWith(".jpg") }
                            .sortedBy { it.name }
                            .toList()
                        if (entries.isNotEmpty()) {
                            zf.getInputStream(entries.last()).use { stream ->
                                return ensurePortraitBitmap(BitmapFactory.decodeStream(stream))
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to read last frame from fallback zip", e)
                }
            }

            // Check MediaStore zip archive via ContentResolver
            val zipId = queryMediaStoreZipId(context, sessionName)
            if (zipId != null) {
                val zipUri = ContentUris.withAppendedId(
                    MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), zipId
                )
                try {
                    context.contentResolver.openInputStream(zipUri)?.use { rawStream ->
                        java.util.zip.ZipInputStream(rawStream).use { zis ->
                            var entry = zis.nextEntry
                            var lastBitmap: Bitmap? = null
                            while (entry != null) {
                                if (!entry.isDirectory && isFrame(entry.name) && entry.name.endsWith(".jpg")) {
                                    val bytes = zis.readBytes()
                                    lastBitmap = ensurePortraitBitmap(BitmapFactory.decodeByteArray(bytes, 0, bytes.size))
                                }
                                entry = zis.nextEntry
                            }
                            if (lastBitmap != null) {
                                return lastBitmap
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to read last frame from MediaStore zip", e)
                }
            }

            // Check property scan archives (if session belongs to a packaged property)
            try {
                val store = PropertyStore(context)
                val room = store.roomScan(sessionName, scanFolder)
                if (room != null) {
                    val property = store.property(room.propertyId)
                    if (property != null) {
                        if (room.scanFolder.isNotBlank()) {
                            val scanPrefix = "${room.scanFolder}/images/"
                            val cleanAddress = property.address.replace(Regex("[^a-zA-Z0-9_-]"), "_").trim('_').ifEmpty { "property" }
                            val propZip = File(context.getExternalFilesDir(null), "${cleanAddress}_scan.zip")
                            if (propZip.isFile) {
                                try {
                                    java.util.zip.ZipFile(propZip).use { zf ->
                                        val entry = zf.entries().asSequence()
                                            .filter { !it.isDirectory && it.name.startsWith(scanPrefix) && it.name.endsWith(".jpg") }
                                            .maxByOrNull { it.name }
                                        if (entry != null) {
                                            zf.getInputStream(entry).use { stream ->
                                                return ensurePortraitBitmap(BitmapFactory.decodeStream(stream))
                                            }
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.w(TAG, "Failed reading last frame from property zip", e)
                                }
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed checking property store for last frame", e)
            }

            return null
        }

        fun openPointCloudStream(context: Context, sessionName: String, scanFolder: String = "scan_0"): InputStream? {
            // Check fallback dir first (external app files)
            val fallbackDir = File(context.getExternalFilesDir(null), sessionName)
            if (fallbackDir.isDirectory) {
                val f1 = File(fallbackDir, "$scanFolder/pointcloud.ply")
                if (f1.isFile) return f1.inputStream()
                if (scanFolder == "scan_0") {
                    val f2 = File(fallbackDir, "pointcloud.ply")
                    if (f2.isFile) return f2.inputStream()
                }
            }

            // Check internal files
            val internalDir = File(context.filesDir, sessionName)
            if (internalDir.isDirectory) {
                val f1 = File(internalDir, "$scanFolder/pointcloud.ply")
                if (f1.isFile) return f1.inputStream()
                if (scanFolder == "scan_0") {
                    val f2 = File(internalDir, "pointcloud.ply")
                    if (f2.isFile) return f2.inputStream()
                }
            }

            // Check MediaStore flat files
            val uri = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val projection = arrayOf(MediaStore.Files.FileColumns._ID, MediaStore.Files.FileColumns.DISPLAY_NAME)
            val selection = "(${MediaStore.Files.FileColumns.RELATIVE_PATH} = ? OR ${MediaStore.Files.FileColumns.RELATIVE_PATH} = ?) AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ?"
            val selectionArgs = arrayOf(
                "Documents/${DatasetWriter.ALBUM}/",
                "Documents/${DatasetWriter.ALBUM}",
                "${sessionName}_%",
            )
            val roomPointCloudName = flatMediaName(sessionName, scanFolder, "pointcloud.ply")
            val legacyPointCloudName = if (scanFolder == "scan_0") {
                flatMediaName(sessionName, null, "pointcloud.ply")
            } else {
                null
            }
            try {
                context.contentResolver.query(uri, projection, selection, selectionArgs, null)?.use { cursor ->
                    val idCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
                    val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
                    while (cursor.moveToNext()) {
                        val dName = cursor.getString(nameCol) ?: continue
                        if (dName == roomPointCloudName || dName == legacyPointCloudName) {
                            val id = cursor.getLong(idCol)
                            val itemUri = ContentUris.withAppendedId(uri, id)
                            return context.contentResolver.openInputStream(itemUri)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to query MediaStore for pointcloud.ply", e)
            }

            // Check zip archive on disk
            val zipFile = findZipFile(context, sessionName)
            if (zipFile != null && zipFile.isFile) {
                try {
                    val zf = java.util.zip.ZipFile(zipFile)
                    val entry = zf.getEntry("$scanFolder/pointcloud.ply")
                        ?: if (scanFolder == "scan_0") zf.getEntry("pointcloud.ply") else null
                    if (entry != null) {
                        return zf.getInputStream(entry)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to read pointcloud.ply from zip file", e)
                }
            }

            // Check MediaStore zip stream
            val zipId = queryMediaStoreZipId(context, sessionName)
            if (zipId != null) {
                try {
                    val zipUri = ContentUris.withAppendedId(MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), zipId)
                    context.contentResolver.openInputStream(zipUri)?.use { rawStream ->
                        val zis = java.util.zip.ZipInputStream(rawStream)
                        var entry = zis.nextEntry
                        while (entry != null) {
                            if (entry.name == "$scanFolder/pointcloud.ply" ||
                                (scanFolder == "scan_0" && entry.name == "pointcloud.ply")
                            ) {
                                val bytes = zis.readBytes()
                                return java.io.ByteArrayInputStream(bytes)
                            }
                            entry = zis.nextEntry
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to stream pointcloud.ply from MediaStore zip", e)
                }
            }

            return null
        }

        fun createCenterCutout(src: Bitmap, borderFraction: Float = 0.20f): Bitmap {
            val w = src.width
            val h = src.height
            val cropX = (w * borderFraction).toInt()
            val cropY = (h * borderFraction).toInt()
            val cropW = maxOf(1, w - 2 * cropX)
            val cropH = maxOf(1, h - 2 * cropY)
            return Bitmap.createBitmap(src, cropX, cropY, cropW, cropH)
        }

        fun hasCheckpoint(accessor: FileAccessor): Boolean {
            return accessor.readText(CHECKPOINT_FILE) != null
        }

        /**
         * Rolls pose by 90 deg clockwise to match portrait image rotation.
         * c2w' = c2w * R_ROLL
         */
        fun rollPose(c2w: FloatArray): FloatArray {
            val out = FloatArray(16)
            for (r in 0 until 4) {
                val off = r * 4
                out[off + 0] = c2w[off + 1]
                out[off + 1] = -c2w[off + 0]
                out[off + 2] = c2w[off + 2]
                out[off + 3] = c2w[off + 3]
            }
            return out
        }

        fun toPortraitIntrinsics(flX: Float, flY: Float, cx: Float, cy: Float, w: Int, h: Int): FloatArray {
            // Returns [flX', flY', cx', cy', w', h', camera_angle_x']
            val nFlX = flY
            val nFlY = flX
            val nCx = h - cy
            val nCy = cx
            val nW = h.toFloat()
            val nH = w.toFloat()
            val angleX = (2.0 * atan((nW / (2.0f * nFlX)).toDouble())).toFloat()
            return floatArrayOf(nFlX, nFlY, nCx, nCy, nW, nH, angleX)
        }

        fun computeSha256(stream: InputStream): String {
            val digest = MessageDigest.getInstance("SHA-256")
            val buffer = ByteArray(8192)
            var bytesRead: Int
            while (stream.read(buffer).also { bytesRead = it } != -1) {
                digest.update(buffer, 0, bytesRead)
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        fun buildCertificateJson(
            producerName: String = "com.glomehometour.arscan",
            producerVersion: String = "0.1",
            policyVersion: String = "1.0.0",
            completedStages: List<String> = listOf(
                "quality_gate",
                "keyframe_selection",
                "orientation_normalization",
                "manifest_consolidation",
            ),
            rawW: Int,
            rawH: Int,
            isAlreadyPortrait: Boolean,
            processedW: Int,
            processedH: Int,
            totalInputFrames: Int,
            retainedFramesCount: Int,
            densityKeyframeFilterEnabled: Boolean = Tunables.keyframeFilterEnabled,
            densityKeyframesPerM2Lo: Float = Tunables.keyframeMinPerM2,
            densityKeyframesPerM2Hi: Float = Tunables.keyframeMaxPerM2,
            densityBaseKeyframes: Float = 50.0f,
            densityStandoffM: Float = 0.8f,
            densityMinRoomAreaM2: Float = 9.0f,
            densityMinTranslationM: Float = 0.08f,
            densityMinRotationDeg: Float = 4.0f,
            densityQualityGateBlurThreshold: Float = Tunables.qualityGateBlurThreshold,
            densityDecimationStride: Long = Tunables.decimationStride,
            manifestHashes: Map<String, String>,
            imageHashes: Map<String, String>,
        ): JSONObject {
            val producer = JSONObject().apply {
                put("name", producerName)
                put("version", producerVersion)
            }
            val stagesArr = JSONArray().apply {
                completedStages.forEach { put(it) }
            }
            val orientation = JSONObject().apply {
                put("raw_width", rawW)
                put("raw_height", rawH)
                put("raw_orientation", if (rawW >= rawH) "landscape" else "portrait")
                put("is_already_portrait", isAlreadyPortrait)
                put("processed_width", processedW)
                put("processed_height", processedH)
                put("processed_orientation", "portrait")
                put("rotation_applied_deg", if (!isAlreadyPortrait) 90 else 0)
                put("pose_rolled", !isAlreadyPortrait)
            }
            val density = JSONObject().apply {
                put("keyframe_filter_enabled", densityKeyframeFilterEnabled)
                put("keyframes_per_m2_lo", densityKeyframesPerM2Lo.toDouble())
                put("keyframes_per_m2_hi", densityKeyframesPerM2Hi.toDouble())
                put("base_keyframes", densityBaseKeyframes.toDouble())
                put("standoff_m", densityStandoffM.toDouble())
                put("min_room_area_m2", densityMinRoomAreaM2.toDouble())
                put("min_translation_m", densityMinTranslationM.toDouble())
                put("min_rotation_deg", densityMinRotationDeg.toDouble())
                put("quality_gate_blur_threshold", densityQualityGateBlurThreshold.toDouble())
                put("decimation_stride", densityDecimationStride)
            }
            val frames = JSONObject().apply {
                put("total_input_frames", totalInputFrames)
                put("retained_frames_count", retainedFramesCount)
                put("discarded_frames_count", totalInputFrames - retainedFramesCount)
            }
            val manifestsObj = JSONObject().apply {
                manifestHashes.forEach { (name, hash) -> put(name, hash) }
            }
            val imagesObj = JSONObject().apply {
                imageHashes.forEach { (name, hash) -> put(name, hash) }
            }

            return JSONObject().apply {
                put("schema_version", "1.0.0")
                put("producer", producer)
                put("policy_version", policyVersion)
                put("completed_stages", stagesArr)
                put("orientation_state", orientation)
                put("keyframe_density_settings", density)
                put("frames", frames)
                put("manifest_hashes", manifestsObj)
                put("image_hashes", imagesObj)
            }
        }
    }

    interface FileAccessor {
        fun readText(path: String): String?
        fun openInput(path: String): InputStream?
        fun openOutput(path: String): OutputStream?
        fun delete(path: String): Boolean
        fun listImages(scanFolder: String = ""): List<String>
        fun listScanFolders(): List<String>
        fun compressToZip(zipOutputStream: OutputStream): String
        fun cleanupUncompressed()
    }

    /**
     * Runs processing on the given session. Must be called from a background thread.
     * Reports progress 0-100 via [onProgress].
     *
     * Processes each scan folder sequentially, produces certificate.json alongside
     * the scan folders, and compresses into "$sessionName.zip".
     */
    fun process(
        sessionName: String,
        accessor: FileAccessor,
        onProgress: (Int) -> Unit,
    ): Boolean {
        try {
            onProgress(2)
            val scanFolders = accessor.listScanFolders()
            if (scanFolders.isEmpty()) {
                throw IllegalStateException("No scan folders or transforms.json found for $sessionName")
            }

            val numFolders = scanFolders.size
            var overallTotalInput = 0
            var overallRetainedCount = 0
            val manifestHashes = mutableMapOf<String, String>()
            val imageHashes = mutableMapOf<String, String>()

            var overallRawW = 1920
            var overallRawH = 1080
            var overallProcessedW = 1080
            var overallProcessedH = 1920
            var overallAlreadyPortrait = false

            // Check for previous checkpoint to resume from
            val checkpointText = accessor.readText(CHECKPOINT_FILE)
            val completedFolders = mutableMapOf<String, JSONObject>()
            if (checkpointText != null) {
                try {
                    val cpJson = JSONObject(checkpointText)
                    overallRawW = cpJson.optInt("overall_raw_w", overallRawW)
                    overallRawH = cpJson.optInt("overall_raw_h", overallRawH)
                    overallProcessedW = cpJson.optInt("overall_processed_w", overallProcessedW)
                    overallProcessedH = cpJson.optInt("overall_processed_h", overallProcessedH)
                    overallAlreadyPortrait = cpJson.optBoolean("overall_already_portrait", overallAlreadyPortrait)
                    val cfArr = cpJson.optJSONArray("completed_folders")
                    if (cfArr != null) {
                        for (i in 0 until cfArr.length()) {
                            val fObj = cfArr.getJSONObject(i)
                            val fName = fObj.optString("folder")
                            completedFolders[fName] = fObj
                        }
                    }
                    Log.i(TAG, "Resuming $sessionName from checkpoint with ${completedFolders.size} completed folders")
                } catch (e: Exception) {
                    Log.w(TAG, "Corrupt checkpoint file for $sessionName, starting fresh", e)
                }
            }

            for ((folderIdx, scanFolder) in scanFolders.withIndex()) {
                val folderStartProg = 2 + (folderIdx * 84 / numFolders)
                val folderEndProg = 2 + ((folderIdx + 1) * 84 / numFolders)
                fun folderProgress(fraction: Float) {
                    val p = (folderStartProg + fraction * (folderEndProg - folderStartProg)).toInt()
                    onProgress(p.coerceIn(2, 86))
                }

                // If this folder was already completed in a prior run, recover its stats & hashes and skip
                val completedObj = completedFolders[scanFolder]
                if (completedObj != null) {
                    val totalIn = completedObj.getInt("total_in")
                    val keptCount = completedObj.getInt("retained_count")
                    overallTotalInput += totalIn
                    overallRetainedCount += keptCount
                    val mHashes = completedObj.optJSONObject("manifest_hashes")
                    if (mHashes != null) {
                        for (k in mHashes.keys()) manifestHashes[k] = mHashes.getString(k)
                    }
                    val iHashes = completedObj.optJSONObject("image_hashes")
                    if (iHashes != null) {
                        for (k in iHashes.keys()) imageHashes[k] = iHashes.getString(k)
                    }
                    folderProgress(1.0f)
                    continue
                }

                val transformsRelPath = if (scanFolder.isEmpty()) "transforms.json" else "$scanFolder/transforms.json"
                val transformsText = accessor.readText(transformsRelPath)
                    ?: throw IllegalStateException("No transforms.json found in $scanFolder for $sessionName")

                val json = JSONObject(transformsText)
                val framesArr = json.getJSONArray("frames")
                val totalIn = framesArr.length()
                if (totalIn == 0) throw IllegalStateException("Empty frames in $transformsRelPath")

                val rawW = json.optInt("w", 1920)
                val rawH = json.optInt("h", 1080)
                val isAlreadyPortrait = rawW < rawH
                overallRawW = rawW
                overallRawH = rawH
                overallAlreadyPortrait = isAlreadyPortrait

                // 1. Collect quality gate stats across frames in this scan folder (0.0 to 0.40 of folder budget)
                val qGate = QualityGate(
                    relativeBlurThreshold = Tunables.qualityGateBlurThreshold,
                )
                val numCores = minOf(Runtime.getRuntime().availableProcessors(), 6)
                val executor = java.util.concurrent.Executors.newFixedThreadPool(maxOf(1, numCores))
                val statsArray = arrayOfNulls<QualityGate.FrameStats>(totalIn)
                val progressCounter = java.util.concurrent.atomic.AtomicInteger(0)

                val futures = (0 until totalIn).map { i ->
                    executor.submit {
                        val fObj = framesArr.getJSONObject(i)
                        val imgRelPath = fObj.getString("file_path").removePrefix("./")
                        val fullImgPath = if (scanFolder.isEmpty()) imgRelPath else "$scanFolder/$imgRelPath"
                        val stream = accessor.openInput(fullImgPath)
                        if (stream != null) {
                            try {
                                // Subsampled decode (inSampleSize = 2, 960x540) ONLY for quality gating metrics.
                                // Preserves full high-resolution original images on disk.
                                val opts = BitmapFactory.Options().apply { inSampleSize = 2 }
                                val bitmap = BitmapFactory.decodeStream(stream, null, opts)
                                if (bitmap != null) {
                                    val bw = bitmap.width
                                    val bh = bitmap.height
                                    val pixels = IntArray(bw * bh)
                                    bitmap.getPixels(pixels, 0, bw, 0, 0, bw, bh)
                                    bitmap.recycle()

                                    val gray = ByteArray(bw * bh)
                                    for (pIdx in pixels.indices) {
                                        val p = pixels[pIdx]
                                        val r = (p shr 16) and 0xFF
                                        val g = (p shr 8) and 0xFF
                                        val b = p and 0xFF
                                        gray[pIdx] = ((r * 77 + g * 150 + b * 29) shr 8).toByte()
                                    }
                                    statsArray[i] = qGate.computeStats(gray, bw, bh, i, fullImgPath)
                                }
                            } finally {
                                stream.close()
                            }
                        }
                        val done = progressCounter.incrementAndGet()
                        if (done % 25 == 0 || done == totalIn) {
                            folderProgress(done.toFloat() / totalIn * 0.40f)
                        }
                    }
                }
                futures.forEach { it.get() }
                val statsList = statsArray.filterNotNull()

                val qResult = if (statsList.isEmpty()) {
                    QualityGate.QualityGateResult(
                        acceptedIndices = (0 until totalIn).toList(),
                        discardedIndices = emptyList(),
                        verdicts = (0 until totalIn).map { idx ->
                            QualityGate.FrameVerdict(
                                index = idx,
                                filePath = "",
                                blurScore = 30.0f,
                                relativeSharpness = 1.0f,
                                accepted = true,
                                rejectionReason = null,
                            )
                        },
                        rejectedBlurCount = 0,
                        rejectedExposureCount = 0,
                        rejectedTextureCount = 0,
                    )
                } else {
                    qGate.evaluate(statsList)
                }
                val acceptedIdxSet = qResult.acceptedIndices.toHashSet()
                val blurScoresByIndex = qResult.verdicts.associate { it.index to it.blurScore }

                folderProgress(0.45f)

                // 2. Pre-COLMAP keyframe selection for this scan folder (0.45 to 0.55 of folder budget)
                val keptIndices: List<Int> = if (Tunables.keyframeFilterEnabled) {
                    val candidatePoseFrames = mutableListOf<KeyframeSelector.PoseFrame>()
                    for (i in 0 until totalIn) {
                        if (!acceptedIdxSet.contains(i)) continue
                        val fObj = framesArr.getJSONObject(i)
                        val mArr = fObj.getJSONArray("transform_matrix")
                        val mat = FloatArray(16) { mIdx ->
                            val row = mIdx / 4
                            val col = mIdx % 4
                            mArr.getJSONArray(row).getDouble(col).toFloat()
                        }
                        val blur = blurScoresByIndex[i] ?: 25.0f
                        candidatePoseFrames.add(
                            KeyframeSelector.PoseFrame(
                                index = i,
                                filePath = fObj.getString("file_path"),
                                c2w = mat,
                                sharpnessScore = blur,
                                numPoints = fObj.optInt("num_points", 100),
                            )
                        )
                    }

                    val selector = KeyframeSelector(
                        keyframesPerM2Lo = Tunables.keyframeMinPerM2,
                        keyframesPerM2Hi = Tunables.keyframeMaxPerM2,
                    )
                    val selResult = selector.selectKeyframes(candidatePoseFrames)
                    selResult.selectedFrames.map { it.index }
                } else {
                    qResult.acceptedIndices
                }

                folderProgress(0.55f)
                val keptSet = keptIndices.toHashSet()
                val keptCount = keptIndices.size

                // 3. Rotate kept frames to portrait & discard rejected images (0.55 to 0.85 of folder budget)
                // Kept frames are processed and saved at FULL 1080p HIGH RESOLUTION (no downsampling!).
                val newFrames = arrayOfNulls<JSONObject>(keptCount)
                val rotateCounter = java.util.concurrent.atomic.AtomicInteger(0)

                val rotateFutures = keptIndices.mapIndexed { stepIdx, origIdx ->
                    executor.submit {
                        val fObj = framesArr.getJSONObject(origIdx)
                        val imgRelPath = fObj.getString("file_path").removePrefix("./")
                        val fullImgPath = if (scanFolder.isEmpty()) imgRelPath else "$scanFolder/$imgRelPath"

                        val newFObj = JSONObject(fObj.toString())
                        if (!isAlreadyPortrait) {
                            val stream = accessor.openInput(fullImgPath)
                            if (stream != null) {
                                // Full resolution 1080p decode and rotation
                                val bmp = BitmapFactory.decodeStream(stream)
                                stream.close()
                                if (bmp != null) {
                                    if (bmp.width > bmp.height) { // Idempotent: only rotate if still in landscape
                                        val matrix = Matrix().apply { postRotate(90f) }
                                        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
                                        bmp.recycle()
                                        val digest = MessageDigest.getInstance("SHA-256")
                                        accessor.openOutput(fullImgPath)?.use { out ->
                                            java.security.DigestOutputStream(out, digest).use { dOut ->
                                                rotated.compress(Bitmap.CompressFormat.JPEG, 95, dOut)
                                            }
                                        }
                                        rotated.recycle()
                                        val hash = digest.digest().joinToString("") { "%02x".format(it) }
                                        synchronized(imageHashes) {
                                            imageHashes[fullImgPath] = hash
                                        }
                                    } else {
                                        bmp.recycle()
                                    }
                                }
                            }

                            val mArr = fObj.getJSONArray("transform_matrix")
                            val mat = FloatArray(16) { mIdx ->
                                mArr.getJSONArray(mIdx / 4).getDouble(mIdx % 4).toFloat()
                            }
                            val rolled = rollPose(mat)
                            val newMArr = JSONArray()
                            for (r in 0 until 4) {
                                val rowArr = JSONArray()
                                for (c in 0 until 4) rowArr.put(rolled[r * 4 + c].toDouble())
                                newMArr.put(rowArr)
                            }
                            newFObj.put("transform_matrix", newMArr)

                            if (fObj.has("fl_x") && fObj.has("fl_y") && fObj.has("cx") && fObj.has("cy")) {
                                val pIntr = toPortraitIntrinsics(
                                    fObj.getDouble("fl_x").toFloat(),
                                    fObj.getDouble("fl_y").toFloat(),
                                    fObj.getDouble("cx").toFloat(),
                                    fObj.getDouble("cy").toFloat(),
                                    rawW, rawH,
                                )
                                newFObj.put("fl_x", pIntr[0].toDouble())
                                newFObj.put("fl_y", pIntr[1].toDouble())
                                newFObj.put("cx", pIntr[2].toDouble())
                                newFObj.put("cy", pIntr[3].toDouble())
                            }
                        }
                        newFrames[stepIdx] = newFObj
                        val rDone = rotateCounter.incrementAndGet()
                        folderProgress(0.55f + (rDone.toFloat() / maxOf(1, keptCount)) * 0.30f)
                    }
                }
                rotateFutures.forEach { it.get() }
                executor.shutdown()

                // Delete discarded frame files
                for (i in 0 until totalIn) {
                    if (!keptSet.contains(i)) {
                        val fObj = framesArr.getJSONObject(i)
                        val imgRelPath = fObj.getString("file_path").removePrefix("./")
                        val fullImgPath = if (scanFolder.isEmpty()) imgRelPath else "$scanFolder/$imgRelPath"
                        accessor.delete(fullImgPath)
                    }
                }

                // Update header intrinsics
                val finalFramesArr = JSONArray()
                newFrames.filterNotNull().forEach { finalFramesArr.put(it) }
                json.put("frames", finalFramesArr)
                if (!isAlreadyPortrait) {
                    val headerIntr = toPortraitIntrinsics(
                        json.getDouble("fl_x").toFloat(),
                        json.getDouble("fl_y").toFloat(),
                        json.getDouble("cx").toFloat(),
                        json.getDouble("cy").toFloat(),
                        rawW, rawH,
                    )
                    json.put("fl_x", headerIntr[0].toDouble())
                    json.put("fl_y", headerIntr[1].toDouble())
                    json.put("cx", headerIntr[2].toDouble())
                    json.put("cy", headerIntr[3].toDouble())
                    json.put("w", headerIntr[4].toInt())
                    json.put("h", headerIntr[5].toInt())
                    json.put("camera_angle_x", headerIntr[6].toDouble())
                }

                overallProcessedW = json.getInt("w")
                overallProcessedH = json.getInt("h")

                // Write updated transforms.json for this scan folder
                accessor.openOutput(transformsRelPath)?.use { out ->
                    out.write(json.toString(2).toByteArray())
                }

                // Collect manifest hashes for this scan folder
                listOf("transforms.json", "trajectory.csv", "coverage_summary.json", "focus_metadata.json", "pointcloud.ply").forEach { mName ->
                    val fullMPath = if (scanFolder.isEmpty()) mName else "$scanFolder/$mName"
                    accessor.openInput(fullMPath)?.use { stream ->
                        manifestHashes[fullMPath] = computeSha256(stream)
                    }
                }

                // Collect image hashes for kept frames (if not already hashed during rotation write)
                for (origIdx in keptIndices) {
                    val fObj = framesArr.getJSONObject(origIdx)
                    val imgRelPath = fObj.getString("file_path").removePrefix("./")
                    val fullImgPath = if (scanFolder.isEmpty()) imgRelPath else "$scanFolder/$imgRelPath"
                    if (!imageHashes.containsKey(fullImgPath)) {
                        accessor.openInput(fullImgPath)?.use { stream ->
                            imageHashes[fullImgPath] = computeSha256(stream)
                        }
                    }
                }

                overallTotalInput += totalIn
                overallRetainedCount += keptCount

                // Checkpoint this completed folder dynamically to disk
                val folderRecord = JSONObject().apply {
                    put("folder", scanFolder)
                    put("total_in", totalIn)
                    put("retained_count", keptCount)
                    put("raw_w", rawW)
                    put("raw_h", rawH)
                    put("processed_w", overallProcessedW)
                    put("processed_h", overallProcessedH)
                    put("is_already_portrait", isAlreadyPortrait)
                    val mObj = JSONObject()
                    manifestHashes.filterKeys { it.startsWith(if (scanFolder.isEmpty()) "" else "$scanFolder/") }.forEach { (k, v) -> mObj.put(k, v) }
                    put("manifest_hashes", mObj)
                    val iObj = JSONObject()
                    imageHashes.filterKeys { it.startsWith(if (scanFolder.isEmpty()) "" else "$scanFolder/") }.forEach { (k, v) -> iObj.put(k, v) }
                    put("image_hashes", iObj)
                }
                completedFolders[scanFolder] = folderRecord

                val cpToSave = JSONObject().apply {
                    put("session_name", sessionName)
                    put("overall_raw_w", overallRawW)
                    put("overall_raw_h", overallRawH)
                    put("overall_processed_w", overallProcessedW)
                    put("overall_processed_h", overallProcessedH)
                    put("overall_already_portrait", overallAlreadyPortrait)
                    val arr = JSONArray()
                    completedFolders.values.forEach { arr.put(it) }
                    put("completed_folders", arr)
                }
                accessor.openOutput(CHECKPOINT_FILE)?.use { out ->
                    out.write(cpToSave.toString(2).toByteArray())
                }
            }

            onProgress(86)

            // 4. Generate certificate.json at root alongside the scan folders (86% to 92%)
            val certJson = buildCertificateJson(
                rawW = overallRawW,
                rawH = overallRawH,
                isAlreadyPortrait = overallAlreadyPortrait,
                processedW = overallProcessedW,
                processedH = overallProcessedH,
                totalInputFrames = overallTotalInput,
                retainedFramesCount = overallRetainedCount,
                manifestHashes = manifestHashes,
                imageHashes = imageHashes,
            )

            accessor.openOutput("certificate.json")?.use { out ->
                out.write(certJson.toString(2).toByteArray())
            }

            onProgress(92)

            // Remove checkpoint file before final zip packaging
            accessor.delete(CHECKPOINT_FILE)

            // 5. Zip surviving files and delete uncompressed originals (92% to 100%)
            accessor.cleanupUncompressed()
            markProcessed(sessionName)
            onProgress(100)
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Scan processing failed for $sessionName", e)
            return false
        }
    }

    /**
     * FileAccessor implementation for local file directory.
     */
    class DirectoryFileAccessor(val dir: File) : FileAccessor {
        override fun readText(path: String): String? {
            val f = File(dir, path)
            return if (f.isFile) f.readText() else null
        }

        override fun openInput(path: String): InputStream? {
            val f = File(dir, path)
            return if (f.isFile) f.inputStream() else null
        }

        override fun openOutput(path: String): OutputStream? {
            val f = File(dir, path)
            f.parentFile?.mkdirs()
            return FileOutputStream(f)
        }

        override fun delete(path: String): Boolean {
            val f = File(dir, path)
            return f.delete()
        }

        override fun listScanFolders(): List<String> {
            val scanDirs = dir.listFiles()?.filter {
                it.isDirectory && it.name.matches(Regex("""^scan_\d+$""")) && File(it, "transforms.json").isFile
            }?.map { it.name }?.sortedBy { it.removePrefix("scan_").toIntOrNull() ?: 0 } ?: emptyList()
            if (scanDirs.isNotEmpty()) return scanDirs
            if (File(dir, "transforms.json").isFile) return listOf("")
            return emptyList()
        }

        override fun listImages(scanFolder: String): List<String> {
            val base = if (scanFolder.isEmpty()) dir else File(dir, scanFolder)
            val imgDir = File(base, "images")
            if (!imgDir.isDirectory) return emptyList()
            val prefix = if (scanFolder.isEmpty()) "images/" else "$scanFolder/images/"
            return imgDir.listFiles()?.filter { it.isFile && it.name.endsWith(".jpg") }
                ?.map { "$prefix${it.name}" } ?: emptyList()
        }

        override fun compressToZip(zipOutputStream: OutputStream): String {
            val zipName = "${dir.name}.zip"
            val zipFile = File(dir.parentFile, zipName)
            java.io.BufferedOutputStream(FileOutputStream(zipFile), 65536).use { bufOut ->
                ZipOutputStream(bufOut).use { zos ->
                    zos.setLevel(java.util.zip.Deflater.BEST_SPEED)
                    val buf = ByteArray(65536)
                    dir.walkTopDown().filter { it.isFile }.forEach { file ->
                        zos.putNextEntry(ZipEntry(file.relativeTo(dir).path))
                        java.io.BufferedInputStream(file.inputStream(), 65536).use { bufIn ->
                            var read: Int
                            while (bufIn.read(buf).also { read = it } != -1) {
                                zos.write(buf, 0, read)
                            }
                        }
                        zos.closeEntry()
                    }
                }
            }
            return zipFile.absolutePath
        }

        override fun cleanupUncompressed() {
            compressToZip(java.io.ByteArrayOutputStream())
            dir.deleteRecursively()
        }
    }

    /**
     * FileAccessor implementation for flat MediaStore entries.
     */
    class MediaStoreFlatAccessor(
        private val context: Context,
        private val sessionName: String,
        mediaAlbumPathRaw: String = "Documents/${DatasetWriter.ALBUM}",
    ) : FileAccessor {
        private val filesUri = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        private val mediaAlbumPathSlash = if (mediaAlbumPathRaw.endsWith("/")) mediaAlbumPathRaw else "$mediaAlbumPathRaw/"
        private val mediaAlbumPathNoSlash = mediaAlbumPathRaw.trimEnd('/')

        override fun listScanFolders(): List<String> {
            val folders = mutableListOf<String>()
            val regex = Regex("""^${Regex.escape(sessionName)}_(scan_\d+)_transforms\.json$""")
            var hasRootTransforms = false
            context.contentResolver.query(
                filesUri,
                arrayOf(MediaStore.Files.FileColumns.DISPLAY_NAME),
                "(${MediaStore.Files.FileColumns.RELATIVE_PATH} = ? OR ${MediaStore.Files.FileColumns.RELATIVE_PATH} = ?) AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ?",
                arrayOf(mediaAlbumPathSlash, mediaAlbumPathNoSlash, "${sessionName}_%transforms.json"),
                null,
            )?.use { cursor ->
                val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
                while (cursor.moveToNext()) {
                    val dName = cursor.getString(nameCol)
                    val match = regex.matchEntire(dName)
                    if (match != null) {
                        folders.add(match.groupValues[1])
                    } else if (dName == "${sessionName}_transforms.json") {
                        hasRootTransforms = true
                    }
                }
            }
            if (folders.isNotEmpty()) {
                return folders.distinct().sortedBy { it.removePrefix("scan_").toIntOrNull() ?: 0 }
            }
            if (hasRootTransforms) return listOf("")
            return emptyList()
        }

        private fun queryId(displayName: String): Long? {
            context.contentResolver.query(
                filesUri,
                arrayOf(MediaStore.Files.FileColumns._ID),
                "(${MediaStore.Files.FileColumns.RELATIVE_PATH} = ? OR ${MediaStore.Files.FileColumns.RELATIVE_PATH} = ?) AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} = ?",
                arrayOf(mediaAlbumPathSlash, mediaAlbumPathNoSlash, displayName),
                null,
            )?.use { cursor ->
                if (cursor.moveToFirst()) return cursor.getLong(0)
            }
            return null
        }

        override fun readText(path: String): String? {
            val flatName = flatMediaName(sessionName, if (path.contains('/')) path.substringBeforeLast('/') else null, path.substringAfterLast('/'))
            val id = queryId(flatName) ?: return null
            return context.contentResolver.openInputStream(ContentUris.withAppendedId(filesUri, id))?.use {
                it.bufferedReader().readText()
            }
        }

        override fun openInput(path: String): InputStream? {
            val flatName = flatMediaName(sessionName, if (path.contains('/')) path.substringBeforeLast('/') else null, path.substringAfterLast('/'))
            val id = queryId(flatName) ?: return null
            return context.contentResolver.openInputStream(ContentUris.withAppendedId(filesUri, id))
        }

        override fun openOutput(path: String): OutputStream? {
            val flatName = flatMediaName(sessionName, if (path.contains('/')) path.substringBeforeLast('/') else null, path.substringAfterLast('/'))
            val id = queryId(flatName)
            val uri = if (id != null) {
                ContentUris.withAppendedId(filesUri, id)
            } else {
                val mime = if (path.endsWith(".jpg")) "image/jpeg" else if (path.endsWith(".json")) "application/json" else "text/plain"
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, flatName)
                    put(MediaStore.MediaColumns.MIME_TYPE, mime)
                    put(MediaStore.MediaColumns.RELATIVE_PATH, mediaAlbumPathSlash)
                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }
                context.contentResolver.insert(filesUri, values) ?: return null
            }
            val stream = context.contentResolver.openOutputStream(uri, "wt") ?: return null
            return object : OutputStream() {
                override fun write(b: Int) = stream.write(b)
                override fun write(b: ByteArray, off: Int, len: Int) = stream.write(b, off, len)
                override fun close() {
                    stream.close()
                    context.contentResolver.update(
                        uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null,
                    )
                }
            }
        }

        override fun delete(path: String): Boolean {
            val flatName = flatMediaName(sessionName, if (path.contains('/')) path.substringBeforeLast('/') else null, path.substringAfterLast('/'))
            val id = queryId(flatName) ?: return false
            context.contentResolver.delete(ContentUris.withAppendedId(filesUri, id), null, null)
            return true
        }

        override fun listImages(scanFolder: String): List<String> {
            val res = mutableListOf<String>()
            val prefix = if (scanFolder.isEmpty()) "${sessionName}_images_" else "${sessionName}_${scanFolder}_images_"
            val pathPrefix = if (scanFolder.isEmpty()) "images/" else "$scanFolder/images/"
            context.contentResolver.query(
                filesUri,
                arrayOf(MediaStore.Files.FileColumns.DISPLAY_NAME),
                "(${MediaStore.Files.FileColumns.RELATIVE_PATH} = ? OR ${MediaStore.Files.FileColumns.RELATIVE_PATH} = ?) AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ?",
                arrayOf(mediaAlbumPathSlash, mediaAlbumPathNoSlash, "$prefix%.jpg"),
                null,
            )?.use { cursor ->
                val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
                while (cursor.moveToNext()) {
                    val name = cursor.getString(nameCol)
                    res.add("$pathPrefix${name.removePrefix(prefix)}")
                }
            }
            return res
        }

        override fun compressToZip(zipOutputStream: OutputStream): String {
            return ""
        }

        override fun cleanupUncompressed() {
            // Zip surviving files to "$sessionName.zip"
            val entries = mutableListOf<Pair<Long, String>>()
            context.contentResolver.query(
                filesUri,
                arrayOf(MediaStore.Files.FileColumns._ID, MediaStore.Files.FileColumns.DISPLAY_NAME),
                "(${MediaStore.Files.FileColumns.RELATIVE_PATH} = ? OR ${MediaStore.Files.FileColumns.RELATIVE_PATH} = ?) AND ${MediaStore.Files.FileColumns.DISPLAY_NAME} LIKE ?",
                arrayOf(mediaAlbumPathSlash, mediaAlbumPathNoSlash, "${sessionName}_%"),
                null,
            )?.use { cursor ->
                val idCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID)
                val nameCol = cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns.DISPLAY_NAME)
                while (cursor.moveToNext()) {
                    val dName = cursor.getString(nameCol)
                    if (dName.endsWith(CHECKPOINT_FILE)) {
                        context.contentResolver.delete(ContentUris.withAppendedId(filesUri, cursor.getLong(idCol)), null, null)
                    } else if (!dName.endsWith(".zip")) {
                        entries.add(cursor.getLong(idCol) to unflattenMediaName(sessionName, dName))
                    }
                }
            }

            val zipName = "$sessionName.zip"
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, zipName)
                put(MediaStore.MediaColumns.MIME_TYPE, "application/zip")
                put(MediaStore.MediaColumns.RELATIVE_PATH, mediaAlbumPathSlash)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val zipUri = context.contentResolver.insert(filesUri, values)
                ?: throw IllegalStateException("Could not create MediaStore zip")

            context.contentResolver.openOutputStream(zipUri, "wt")?.use { rawOut ->
                java.io.BufferedOutputStream(rawOut, 65536).use { bufOut ->
                    ZipOutputStream(bufOut).use { zos ->
                        zos.setLevel(java.util.zip.Deflater.BEST_SPEED)
                        val buf = ByteArray(65536)
                        for ((id, entryName) in entries) {
                            context.contentResolver.openInputStream(ContentUris.withAppendedId(filesUri, id))?.use { input ->
                                java.io.BufferedInputStream(input, 65536).use { bufIn ->
                                    zos.putNextEntry(ZipEntry(entryName))
                                    var read: Int
                                    while (bufIn.read(buf).also { read = it } != -1) {
                                        zos.write(buf, 0, read)
                                    }
                                    zos.closeEntry()
                                }
                            }
                        }
                    }
                }
            }
            context.contentResolver.update(
                zipUri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null,
            )

            // Delete loose files
            for ((id, _) in entries) {
                context.contentResolver.delete(ContentUris.withAppendedId(filesUri, id), null, null)
            }
        }
    }
}

internal fun trajectoryPointsFromCsv(csv: String): FloatArray {
    val points = ArrayList<Float>()
    csv.lineSequence().drop(1).forEach { line ->
        val fields = line.split(',')
        if (fields.size < 4 || (fields.size >= 9 && fields[8].trim() != "TRACKING")) return@forEach
        val x = fields[1].trim().toFloatOrNull() ?: return@forEach
        val y = fields[2].trim().toFloatOrNull() ?: return@forEach
        val z = fields[3].trim().toFloatOrNull() ?: return@forEach
        if (x.isFinite() && y.isFinite() && z.isFinite()) {
            points.add(x); points.add(y); points.add(z)
        }
    }
    return points.toFloatArray()
}
