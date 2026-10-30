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
    fun `scan subfolder image flattens and unflattens back to scan_N images path`() {
        val flat = flatMediaName("scan_20260913_133535", "scan_0/images", "frame_00005.jpg")
        assertEquals("scan_20260913_133535_scan_0_images_frame_00005.jpg", flat)
        assertEquals("scan_0/images/frame_00005.jpg", unflattenMediaName("scan_20260913_133535", flat))
    }

    @Test
    fun `scan subfolder manifest flattens and unflattens back to scan_N manifest path`() {
        val flat = flatMediaName("scan_20260913_133535", "scan_0", "transforms.json")
        assertEquals("scan_20260913_133535_scan_0_transforms.json", flat)
        assertEquals("scan_0/transforms.json", unflattenMediaName("scan_20260913_133535", flat))
    }

    @Test
    fun `certificate json flattens and unflattens back to root certificate json`() {
        val flat = flatMediaName("scan_20260913_133535", null, "certificate.json")
        assertEquals("scan_20260913_133535_certificate.json", flat)
        assertEquals("certificate.json", unflattenMediaName("scan_20260913_133535", flat))
    }
}
