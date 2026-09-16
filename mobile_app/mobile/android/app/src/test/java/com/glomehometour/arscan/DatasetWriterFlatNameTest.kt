package com.glomehometour.arscan

import org.junit.Assert.assertEquals
import org.junit.Test

/** flatMediaName/unflattenMediaName must round-trip so the post-capture zip lands each MediaStore
 * file at the same path the fallback (nested-folder) writer would have used. */
class DatasetWriterFlatNameTest {

    @Test
    fun `root file flattens and unflattens back to its bare name`() {
        val flat = flatMediaName("scan_20260913_133535", null, "transforms.json")
        assertEquals("scan_20260913_133535_transforms.json", flat)
        assertEquals("transforms.json", unflattenMediaName("scan_20260913_133535", flat))
    }

    @Test
    fun `image file flattens and unflattens back to its images subpath`() {
        val flat = flatMediaName("scan_20260913_133535", "images", "frame_00005.jpg")
        assertEquals("scan_20260913_133535_images_frame_00005.jpg", flat)
        assertEquals("images/frame_00005.jpg", unflattenMediaName("scan_20260913_133535", flat))
    }
}
