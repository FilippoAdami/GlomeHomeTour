package com.glomehometour.arscan

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.math.atan

/**
 * Executes the on-device preprocessing pipeline on an uncompressed scan session:
 * 1. Evaluates blur, exposure, and texture using QualityGate.
 * 2. Prunes redundant views using KeyframeSelector (ARCore trajectory & floor area budgeting).
 * 3. Rotates kept landscape frames 90° clockwise into standard portrait orientation, updating
 *    intrinsics and camera poses.
 * 4. Compresses the survivors and manifests into "$sessionName.zip" and deletes uncompressed files.
 */
class ScanProcessor(private val context: Context) {

    companion object {
        private const val TAG = "ScanProcessor"

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
    }

    interface FileAccessor {
        fun readText(path: String): String?
        fun openInput(path: String): InputStream?
        fun openOutput(path: String): OutputStream?
        fun delete(path: String): Boolean
        fun listImages(): List<String>
        fun compressToZip(zipOutputStream: OutputStream): String
        fun cleanupUncompressed()
    }

    /**
     * Runs processing on the given session. Must be called from a background thread.
     * Reports progress 0-100 via [onProgress].
     */
    fun process(
        sessionName: String,
        accessor: FileAccessor,
        onProgress: (Int) -> Unit,
    ): Boolean {
        try {
            onProgress(2)
            val transformsText = accessor.readText("transforms.json")
                ?: throw IllegalStateException("No transforms.json found for $sessionName")

            val json = JSONObject(transformsText)
            val framesArr = json.getJSONArray("frames")
            val totalIn = framesArr.length()
            if (totalIn == 0) throw IllegalStateException("Empty frames in transforms.json")

            val rawW = json.optInt("w", 1920)
            val rawH = json.optInt("h", 1080)
            val isAlreadyPortrait = rawW < rawH

            // 1. Collect quality gate stats across all frames (2% to 40%)
            val qGate = QualityGate(
                relativeBlurThreshold = Tunables.qualityGateBlurThreshold,
            )
            val statsList = mutableListOf<QualityGate.FrameStats>()

            for (i in 0 until totalIn) {
                val fObj = framesArr.getJSONObject(i)
                val relPath = fObj.getString("file_path").removePrefix("./")
                val stream = accessor.openInput(relPath)
                if (stream != null) {
                    try {
                        val bitmap = BitmapFactory.decodeStream(stream)
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
                            statsList.add(qGate.computeStats(gray, bw, bh, i, relPath))
                        }
                    } finally {
                        stream.close()
                    }
                }
                val pct = 2 + ((i + 1) * 38 / totalIn)
                onProgress(pct)
            }

            val qResult = qGate.evaluate(statsList)
            val acceptedIdxSet = qResult.acceptedIndices.toHashSet()
            val blurScoresByIndex = qResult.verdicts.associate { it.index to it.blurScore }

            onProgress(42)

            // 2. Pre-COLMAP keyframe selection (42% to 52%)
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

            onProgress(55)
            val keptSet = keptIndices.toHashSet()

            // 3. Rotate kept frames to portrait & discard rejected images (55% to 80%)
            val newFrames = JSONArray()
            val keptCount = keptIndices.size

            for ((stepIdx, origIdx) in keptIndices.withIndex()) {
                val fObj = framesArr.getJSONObject(origIdx)
                val relPath = fObj.getString("file_path").removePrefix("./")

                // Update pose and intrinsics if not already portrait
                val newFObj = JSONObject(fObj.toString())
                if (!isAlreadyPortrait) {
                    // Rotate image in place
                    val stream = accessor.openInput(relPath)
                    if (stream != null) {
                        val bmp = BitmapFactory.decodeStream(stream)
                        stream.close()
                        if (bmp != null) {
                            val matrix = Matrix().apply { postRotate(90f) }
                            val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
                            bmp.recycle()
                            accessor.openOutput(relPath)?.use { out ->
                                rotated.compress(Bitmap.CompressFormat.JPEG, 95, out)
                            }
                            rotated.recycle()
                        }
                    }

                    // Update transform matrix
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

                    // Update frame intrinsics if present
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
                newFrames.put(newFObj)
                val p = 55 + ((stepIdx + 1) * 25 / maxOf(1, keptCount))
                onProgress(p)
            }

            // Delete discarded frame files
            for (i in 0 until totalIn) {
                if (!keptSet.contains(i)) {
                    val fObj = framesArr.getJSONObject(i)
                    val relPath = fObj.getString("file_path").removePrefix("./")
                    accessor.delete(relPath)
                }
            }

            // Update header intrinsics
            json.put("frames", newFrames)
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

            // Write updated transforms.json
            accessor.openOutput("transforms.json")?.use { out ->
                out.write(json.toString(2).toByteArray())
            }

            onProgress(82)

            // 4. Zip surviving files and delete uncompressed originals (82% to 100%)
            accessor.cleanupUncompressed()
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

        override fun listImages(): List<String> {
            val imgDir = File(dir, "images")
            if (!imgDir.isDirectory) return emptyList()
            return imgDir.listFiles()?.filter { it.isFile && it.name.endsWith(".jpg") }
                ?.map { "images/${it.name}" } ?: emptyList()
        }

        override fun compressToZip(zipOutputStream: OutputStream): String {
            val zipName = "${dir.name}.zip"
            val zipFile = File(dir.parentFile, zipName)
            ZipOutputStream(FileOutputStream(zipFile)).use { zos ->
                dir.walkTopDown().filter { it.isFile }.forEach { file ->
                    zos.putNextEntry(ZipEntry(file.relativeTo(dir).path))
                    file.inputStream().use { it.copyTo(zos) }
                    zos.closeEntry()
                }
            }
            return zipFile.absolutePath
        }

        override fun cleanupUncompressed() {
            val zipName = "${dir.name}.zip"
            val zipFile = File(dir.parentFile, zipName)
            ZipOutputStream(FileOutputStream(zipFile)).use { zos ->
                dir.walkTopDown().filter { it.isFile }.forEach { file ->
                    zos.putNextEntry(ZipEntry(file.relativeTo(dir).path))
                    file.inputStream().use { it.copyTo(zos) }
                    zos.closeEntry()
                }
            }
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
            val stream = context.contentResolver.openOutputStream(uri) ?: return null
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

        override fun listImages(): List<String> {
            val res = mutableListOf<String>()
            val prefix = "${sessionName}_images_"
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
                    res.add("images/${name.removePrefix(prefix)}")
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
                    if (!dName.endsWith(".zip")) {
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

            context.contentResolver.openOutputStream(zipUri)?.use { out ->
                ZipOutputStream(out).use { zos ->
                    for ((id, entryName) in entries) {
                        context.contentResolver.openInputStream(ContentUris.withAppendedId(filesUri, id))?.use { input ->
                            zos.putNextEntry(ZipEntry(entryName))
                            input.copyTo(zos)
                            zos.closeEntry()
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
