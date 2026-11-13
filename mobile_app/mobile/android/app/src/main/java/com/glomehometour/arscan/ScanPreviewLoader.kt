package com.glomehometour.arscan

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.LruCache
import android.util.TypedValue
import android.view.MotionEvent
import android.view.View
import android.view.animation.AnimationUtils
import android.widget.ImageView
import android.widget.TextView
import android.widget.ViewFlipper
import java.util.concurrent.Executors
import kotlin.math.abs

/**
 * Handles dual-preview visualization (First Frame Photo & 3D Point Cloud Preview) and
 * horizontal swipe gestures on scan cards.
 */
object ScanPreviewLoader {

    private const val TAG = "ScanPreviewLoader"
    private val photoCache = LruCache<String, Bitmap>(64)
    private val executor = Executors.newFixedThreadPool(3)

    /**
     * Binds the card's thumbnail container to load both the first frame photo and 3D point cloud
     * preview, with default view set to the first photo and swipe gesture enabled.
     */
    fun bindScanCardPreview(
        context: Context,
        sessionName: String,
        container: View,
        flipper: ViewFlipper,
        photoView: ImageView,
        pointCloudView: ImageView,
        badgeView: TextView,
        customPhotoStreamProvider: (() -> java.io.InputStream?)? = null,
        scanFolder: String = "scan_0",
    ) {
        // 1. Default to Photo (Page 0)
        flipper.displayedChild = 0
        badgeView.text = "📷 Photo"

        // 2. Setup smooth swipe gesture & tap handling
        setupSwipeGesture(context, container, flipper, badgeView)

        // 3. Load First Frame Photo asynchronously
        loadFirstFrame(context, sessionName, scanFolder, photoView, customPhotoStreamProvider)

        // 4. Load 3D Point Cloud Preview asynchronously
        loadPointCloudPreview(context, sessionName, scanFolder, pointCloudView)
    }

    private fun setupSwipeGesture(
        context: Context,
        container: View,
        flipper: ViewFlipper,
        badge: TextView,
    ) {
        val animSlideInRight = AnimationUtils.loadAnimation(context, R.anim.slide_in_right)
        val animSlideOutLeft = AnimationUtils.loadAnimation(context, R.anim.slide_out_left)
        val animSlideInLeft = AnimationUtils.loadAnimation(context, R.anim.slide_in_left)
        val animSlideOutRight = AnimationUtils.loadAnimation(context, R.anim.slide_out_right)

        fun switchToPointCloud() {
            if (flipper.displayedChild != 1) {
                flipper.inAnimation = animSlideInRight
                flipper.outAnimation = animSlideOutLeft
                flipper.displayedChild = 1
                badge.text = "☁️ 3D"
            }
        }

        fun switchToPhoto() {
            if (flipper.displayedChild != 0) {
                flipper.inAnimation = animSlideInLeft
                flipper.outAnimation = animSlideOutRight
                flipper.displayedChild = 0
                badge.text = "📷 Photo"
            }
        }

        fun toggle() {
            if (flipper.displayedChild == 0) {
                switchToPointCloud()
            } else {
                switchToPhoto()
            }
        }

        val touchSlop = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, 8f, context.resources.displayMetrics
        )
        val swipeThreshold = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, 22f, context.resources.displayMetrics
        )

        container.setOnTouchListener(object : View.OnTouchListener {
            private var startX = 0f
            private var startY = 0f
            private var isHorizontalSwipe = false

            override fun onTouch(v: View, event: MotionEvent): Boolean {
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        startX = event.x
                        startY = event.y
                        isHorizontalSwipe = false
                        v.parent?.requestDisallowInterceptTouchEvent(true)
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val dx = event.x - startX
                        val dy = event.y - startY
                        if (!isHorizontalSwipe) {
                            if (abs(dx) > touchSlop && abs(dx) > abs(dy)) {
                                isHorizontalSwipe = true
                                v.parent?.requestDisallowInterceptTouchEvent(true)
                            } else if (abs(dy) > touchSlop) {
                                // Allow parent vertical scroll view to handle vertical scrolling
                                v.parent?.requestDisallowInterceptTouchEvent(false)
                            }
                        }
                    }
                    MotionEvent.ACTION_UP -> {
                        val dx = event.x - startX
                        val dy = event.y - startY
                        if (abs(dx) > swipeThreshold && abs(dx) > abs(dy)) {
                            if (dx < 0) {
                                // Swiped left -> Switch to 3D Point Cloud Preview
                                switchToPointCloud()
                            } else {
                                // Swiped right -> Switch to First Frame Photo
                                switchToPhoto()
                            }
                        } else if (abs(dx) < touchSlop && abs(dy) < touchSlop) {
                            // Tap to toggle
                            toggle()
                        }
                        v.parent?.requestDisallowInterceptTouchEvent(false)
                    }
                    MotionEvent.ACTION_CANCEL -> {
                        v.parent?.requestDisallowInterceptTouchEvent(false)
                    }
                }
                return true
            }
        })
    }

    private fun loadFirstFrame(
        context: Context,
        sessionName: String,
        scanFolder: String,
        target: ImageView,
        customStreamProvider: (() -> java.io.InputStream?)? = null,
    ) {
        val tagKey = "photo_${sessionName}_$scanFolder"
        target.tag = tagKey

        val cacheKey = "${sessionName}_$scanFolder"
        val cached = photoCache.get(cacheKey)
        if (cached != null && !cached.isRecycled) {
            target.setImageBitmap(cached)
            return
        }

        // Check disk cache
        val diskFile = java.io.File(context.cacheDir, "photo_thumb_${sessionName}_$scanFolder.jpg")
        if (diskFile.exists() && diskFile.length() > 0) {
            try {
                val diskBmp = android.graphics.BitmapFactory.decodeFile(diskFile.absolutePath)
                if (diskBmp != null) {
                    photoCache.put(cacheKey, diskBmp)
                    target.setImageBitmap(diskBmp)
                    return
                }
            } catch (_: Exception) {}
        }

        target.setImageBitmap(null)
        executor.execute {
            try {
                var bmp: Bitmap? = null
                if (customStreamProvider != null) {
                    try {
                        customStreamProvider()?.use { stream ->
                            val zis = java.util.zip.ZipInputStream(stream)
                            var entry = zis.nextEntry
                            val options = android.graphics.BitmapFactory.Options().apply { inSampleSize = 4 }
                            while (entry != null) {
                                if (!entry.isDirectory && entry.name.startsWith("$scanFolder/images/") && entry.name.endsWith(".jpg")) {
                                    val bytes = zis.readBytes()
                                    val rawBmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
                                    if (rawBmp != null) {
                                        bmp = if (rawBmp.width > rawBmp.height) {
                                            val matrix = android.graphics.Matrix().apply { postRotate(90f) }
                                            val rot = Bitmap.createBitmap(rawBmp, 0, 0, rawBmp.width, rawBmp.height, matrix, true)
                                            rawBmp.recycle()
                                            rot
                                        } else {
                                            rawBmp
                                        }
                                        break
                                    }
                                }
                                entry = zis.nextEntry
                            }
                        }
                    } catch (_: Exception) {}
                }

                if (bmp == null) {
                    bmp = ScanProcessor.getFirstFrameBitmap(context, sessionName, scanFolder = scanFolder)
                }

                val finalBmp = bmp ?: return@execute

                photoCache.put(cacheKey, finalBmp)
                try {
                    diskFile.outputStream().use { out ->
                        finalBmp.compress(Bitmap.CompressFormat.JPEG, 85, out)
                    }
                } catch (_: Exception) {}

                target.post {
                    if (target.tag == tagKey) {
                        target.setImageBitmap(finalBmp)
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load first frame for $sessionName", e)
            }
        }
    }

    private fun loadPointCloudPreview(context: Context, sessionName: String, scanFolder: String, target: ImageView) {
        val tagKey = "pc_${sessionName}_$scanFolder"
        target.tag = tagKey

        val cached = PointCloudPreviewGenerator.getInMemory(sessionName, scanFolder)
        if (cached != null && !cached.isRecycled) {
            target.setImageBitmap(cached)
            return
        }

        target.setImageBitmap(null)
        executor.execute {
            try {
                val bmp = PointCloudPreviewGenerator.getOrGeneratePreview(context, sessionName, scanFolder)
                target.post {
                    if (target.tag == tagKey) {
                        target.setImageBitmap(bmp)
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to generate/load point cloud preview for $sessionName", e)
            }
        }
    }
}
