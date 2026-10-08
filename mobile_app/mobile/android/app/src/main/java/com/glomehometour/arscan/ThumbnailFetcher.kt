package com.glomehometour.arscan

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import android.widget.ImageView
import java.io.InputStream
import java.util.concurrent.Executors
import java.util.zip.ZipInputStream

/**
 * First captured frame out of a session zip, for the gallery list.
 *
 * Streams the zip rather than opening it as a file: the current sessions live in MediaStore under
 * Documents/, where the app has a content URI and no File path.
 *
 * ponytail: in-memory LRU only, and a fixed 1/8 downsample instead of measuring the JPEG first --
 * a few tens of scans per phone at ~135px each costs nothing. Add a disk cache if a gallery ever
 * holds thousands.
 */
object ThumbnailFetcher {

    private val cache = LruCache<String, Bitmap>(64)
    private val executor = Executors.newSingleThreadExecutor()

    /** Loads [sessionName]'s thumbnail into [target], skipping the work if the view was rebound. */
    fun load(sessionName: String, target: ImageView, openZip: () -> InputStream?) {
        target.tag = sessionName
        val cached = cache.get(sessionName)
        if (cached != null) {
            target.setImageBitmap(cached)
            return
        }
        target.setImageBitmap(null)
        executor.execute {
            val bitmap = try {
                openZip()?.use { firstFrame(it) }
            } catch (e: Exception) {
                null
            } ?: return@execute
            cache.put(sessionName, bitmap)
            target.post { if (target.tag == sessionName) target.setImageBitmap(bitmap) }
        }
    }

    private fun firstFrame(stream: InputStream): Bitmap? {
        val bytes = stream.readBytes()
        try {
            ZipInputStream(bytes.inputStream()).use { zis ->
                var entry = zis.nextEntry
                while (entry != null) {
                    if (!entry.isDirectory && entry.name.startsWith("images/") && entry.name.endsWith(".jpg")) {
                        val options = BitmapFactory.Options().apply { inSampleSize = 8 }
                        return BitmapFactory.decodeStream(zis, null, options)
                    }
                    entry = zis.nextEntry
                }
            }
        } catch (e: Exception) {
            // Not a zip
        }
        return try {
            val options = BitmapFactory.Options().apply { inSampleSize = 8 }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
        } catch (e: Exception) {
            null
        }
    }
}
