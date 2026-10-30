package com.glomehometour.arscan

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.InputStream
import java.io.OutputStream

class ScanProcessorCheckpointTest {

    @Test
    fun `ScanProcessor resumes from checkpoint without reprocessing completed scan folders`() {
        val files = mutableMapOf<String, ByteArray>()
        var scan0ProcessedCount = 0

        fun addRoom(folder: String, imgName: String) {
            val transforms = """
                {
                  "w": 1080, "h": 1920, "fl_x": 1000.0, "fl_y": 1000.0, "cx": 540.0, "cy": 960.0,
                  "frames": [
                    {
                      "file_path": "images/$imgName",
                      "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
                    }
                  ]
                }
            """.trimIndent()
            files["$folder/transforms.json"] = transforms.toByteArray()
            files["$folder/trajectory.csv"] = "timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking,exported\n100,0,0,0,0,0,0,1,TRACKING,true\n".toByteArray()
            files["$folder/coverage_summary.json"] = "{}".toByteArray()
            files["$folder/focus_metadata.json"] = "{}".toByteArray()
            files["$folder/images/$imgName"] = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xD9.toByte())
        }

        addRoom("scan_0", "frame_00000.jpg")
        addRoom("scan_1", "frame_00000.jpg")

        var shouldFailOnScan1 = true

        val accessor = object : ScanProcessor.FileAccessor {
            override fun readText(path: String): String? = files[path]?.toString(Charsets.UTF_8)
            override fun openInput(path: String): InputStream? {
                if (path == "scan_0/images/frame_00000.jpg") {
                    scan0ProcessedCount++
                }
                return files[path]?.let { ByteArrayInputStream(it) }
            }
            override fun openOutput(path: String): OutputStream? {
                if (shouldFailOnScan1 && path == "scan_1/transforms.json") {
                    throw RuntimeException("Simulated crash/interruption while processing scan_1")
                }
                return object : java.io.ByteArrayOutputStream() {
                    override fun close() {
                        super.close()
                        files[path] = toByteArray()
                    }
                }
            }
            override fun delete(path: String): Boolean = files.remove(path) != null
            override fun listImages(scanFolder: String): List<String> =
                files.keys.filter { it.startsWith(if (scanFolder.isEmpty()) "images/" else "$scanFolder/images/") }
            override fun listScanFolders(): List<String> = listOf("scan_0", "scan_1")
            override fun compressToZip(zipOutputStream: OutputStream): String = ""
            override fun cleanupUncompressed() {}
        }

        val processor = ScanProcessor()
        assertFalse(ScanProcessor.hasCheckpoint(accessor))

        // Run 1: Fails mid-way on scan_1
        val ok1 = processor.process("session_chk", accessor) {}
        assertFalse("First run must report failure on interruption", ok1)
        assertTrue("Checkpoint must be created after scan_0 completes", ScanProcessor.hasCheckpoint(accessor))

        val checkpointText = accessor.readText(ScanProcessor.CHECKPOINT_FILE)
        val cpJson = JSONObject(checkpointText!!)
        val completed = cpJson.getJSONArray("completed_folders")
        assertEquals(1, completed.length())
        assertEquals("scan_0", completed.getJSONObject(0).getString("folder"))

        val initialScan0PassCount = scan0ProcessedCount
        assertTrue("scan_0 images must have been read", initialScan0PassCount > 0)

        // Run 2: Resume with crash condition cleared
        shouldFailOnScan1 = false
        val ok2 = processor.process("session_chk", accessor) {}
        assertTrue("Second resumed run must succeed", ok2)

        // scan_0 must NOT have been re-processed
        assertEquals("scan_0 images must not be read again on resume", initialScan0PassCount, scan0ProcessedCount)

        // Checkpoint must be cleaned up after successful completion
        assertFalse("Checkpoint file must be removed after successful processing", ScanProcessor.hasCheckpoint(accessor))
        assertTrue("certificate.json must be generated", files.containsKey("certificate.json"))

        val certJson = JSONObject(files["certificate.json"]!!.toString(Charsets.UTF_8))
        assertEquals(2, certJson.getJSONObject("frames").getInt("total_input_frames"))
        assertEquals(2, certJson.getJSONObject("frames").getInt("retained_frames_count"))
    }
}
