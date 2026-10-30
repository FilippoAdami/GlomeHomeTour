package com.glomehometour.arscan

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.InputStream
import java.io.OutputStream

class ScanProcessorCertificateTest {

    @Test
    fun `computeSha256 matches known sha256 checksum`() {
        // echo -n "GlomeHomeTour" | sha256sum
        // 73781d5359fd832f68f145d1b084aa026e9c8a563418c1486bf9bc49571a3715
        val input = ByteArrayInputStream("GlomeHomeTour".toByteArray(Charsets.UTF_8))
        val hash = ScanProcessor.computeSha256(input)
        assertEquals("73781d5359fd832f68f145d1b084aa026e9c8a563418c1486bf9bc49571a3715", hash)
    }

    @Test
    fun `buildCertificateJson contains all required provenance, orientation, density, and hash fields`() {
        val manifestHashes = mapOf(
            "transforms.json" to "53bfca1e7feea0aeb95311b51e5e29780521665a6c3f0b2f1559eebbc831a31c",
            "trajectory.csv" to "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        val imageHashes = mapOf(
            "images/frame_00000.jpg" to "a1b2c3d4e5f60000000000000000000000000000000000000000000000000000",
            "images/frame_00001.jpg" to "f1e2d3c4b5a60000000000000000000000000000000000000000000000000000",
        )

        val cert = ScanProcessor.buildCertificateJson(
            producerName = "com.glomehometour.arscan",
            producerVersion = "0.1",
            policyVersion = "1.0.0",
            rawW = 1920,
            rawH = 1080,
            isAlreadyPortrait = false,
            processedW = 1080,
            processedH = 1920,
            totalInputFrames = 100,
            retainedFramesCount = 20,
            manifestHashes = manifestHashes,
            imageHashes = imageHashes,
        )

        assertEquals("1.0.0", cert.getString("schema_version"))
        assertEquals("com.glomehometour.arscan", cert.getJSONObject("producer").getString("name"))
        assertEquals("0.1", cert.getJSONObject("producer").getString("version"))
        assertEquals("1.0.0", cert.getString("policy_version"))

        val stages = cert.getJSONArray("completed_stages")
        assertEquals(4, stages.length())
        assertEquals("quality_gate", stages.getString(0))
        assertEquals("manifest_consolidation", stages.getString(3))

        val orientation = cert.getJSONObject("orientation_state")
        assertEquals(1920, orientation.getInt("raw_width"))
        assertEquals(1080, orientation.getInt("raw_height"))
        assertEquals("landscape", orientation.getString("raw_orientation"))
        assertFalse(orientation.getBoolean("is_already_portrait"))
        assertEquals(1080, orientation.getInt("processed_width"))
        assertEquals(1920, orientation.getInt("processed_height"))
        assertEquals("portrait", orientation.getString("processed_orientation"))
        assertEquals(90, orientation.getInt("rotation_applied_deg"))
        assertTrue(orientation.getBoolean("pose_rolled"))

        val density = cert.getJSONObject("keyframe_density_settings")
        assertTrue(density.getBoolean("keyframe_filter_enabled"))
        assertEquals(8.0, density.getDouble("keyframes_per_m2_lo"), 1e-4)
        assertEquals(11.0, density.getDouble("keyframes_per_m2_hi"), 1e-4)
        assertEquals(50.0, density.getDouble("base_keyframes"), 1e-4)
        assertEquals(0.8, density.getDouble("standoff_m"), 1e-4)
        assertEquals(9.0, density.getDouble("min_room_area_m2"), 1e-4)
        assertEquals(0.08, density.getDouble("min_translation_m"), 1e-4)
        assertEquals(4.0, density.getDouble("min_rotation_deg"), 1e-4)
        assertEquals(0.35, density.getDouble("quality_gate_blur_threshold"), 1e-4)

        val frames = cert.getJSONObject("frames")
        assertEquals(100, frames.getInt("total_input_frames"))
        assertEquals(20, frames.getInt("retained_frames_count"))
        assertEquals(80, frames.getInt("discarded_frames_count"))

        val mHashes = cert.getJSONObject("manifest_hashes")
        assertEquals("53bfca1e7feea0aeb95311b51e5e29780521665a6c3f0b2f1559eebbc831a31c", mHashes.getString("transforms.json"))

        val iHashes = cert.getJSONObject("image_hashes")
        assertEquals(2, iHashes.length())
        assertEquals("a1b2c3d4e5f60000000000000000000000000000000000000000000000000000", iHashes.getString("images/frame_00000.jpg"))
    }

    @Test
    fun `ScanProcessor processes multi-room scan folders sequentially and writes certificate alongside them`() {
        val files = mutableMapOf<String, ByteArray>()

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

        val accessor = object : ScanProcessor.FileAccessor {
            override fun readText(path: String): String? = files[path]?.toString(Charsets.UTF_8)
            override fun openInput(path: String): InputStream? = files[path]?.let { ByteArrayInputStream(it) }
            override fun openOutput(path: String): OutputStream? = object : java.io.ByteArrayOutputStream() {
                override fun close() {
                    super.close()
                    files[path] = toByteArray()
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
        var progressReported = 0
        val ok = processor.process("session_multi", accessor) { progressReported = it }

        assertTrue(ok)
        assertEquals(100, progressReported)
        assertTrue("certificate.json must be written at root", files.containsKey("certificate.json"))

        val certJson = JSONObject(files["certificate.json"]!!.toString(Charsets.UTF_8))
        assertEquals("1.0.0", certJson.getString("schema_version"))
        assertEquals(2, certJson.getJSONObject("frames").getInt("total_input_frames"))
        assertEquals(2, certJson.getJSONObject("frames").getInt("retained_frames_count"))

        val mHashes = certJson.getJSONObject("manifest_hashes")
        assertTrue(mHashes.has("scan_0/transforms.json"))
        assertTrue(mHashes.has("scan_0/trajectory.csv"))
        assertTrue(mHashes.has("scan_0/coverage_summary.json"))
        assertTrue(mHashes.has("scan_0/focus_metadata.json"))
        assertTrue(mHashes.has("scan_1/transforms.json"))
        assertTrue(mHashes.has("scan_1/trajectory.csv"))
        assertTrue(mHashes.has("scan_1/coverage_summary.json"))
        assertTrue(mHashes.has("scan_1/focus_metadata.json"))

        val iHashes = certJson.getJSONObject("image_hashes")
        assertEquals(2, iHashes.length())
        assertTrue(iHashes.has("scan_0/images/frame_00000.jpg"))
        assertTrue(iHashes.has("scan_1/images/frame_00000.jpg"))
    }

    @Test
    fun `ScanProcessor truncates transforms_json so no trailing extra data exists and certificate matches clean manifest`() {
        val files = mutableMapOf<String, ByteArray>()

        // Simulate an initial unpruned transforms.json with 200 frames (longer file, > budgetLo 122)
        val initialFrames = (0 until 200).joinToString(",") { i ->
            val img = "frame_%05d.jpg".format(i)
            val tx = i * 0.05f
            """
            {
              "file_path": "images/$img",
              "transform_matrix": [[1,0,0,$tx],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            }
            """.trimIndent()
        }
        val unprunedTransforms = """
            {
              "w": 1080, "h": 1920, "fl_x": 1000.0, "fl_y": 1000.0, "cx": 540.0, "cy": 960.0,
              "frames": [$initialFrames]
            }
        """.trimIndent()

        val unprunedBytes = JSONObject(unprunedTransforms).toString(2).toByteArray(Charsets.UTF_8)
        files["scan_0/transforms.json"] = unprunedBytes
        files["scan_0/trajectory.csv"] = "timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking,exported\n100,0,0,0,0,0,0,1,TRACKING,true\n".toByteArray()
        files["scan_0/coverage_summary.json"] = "{}".toByteArray()
        files["scan_0/focus_metadata.json"] = "{}".toByteArray()
        for (i in 0 until 200) {
            val img = "frame_%05d.jpg".format(i)
            files["scan_0/images/$img"] = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xD9.toByte())
        }

        val initialLen = unprunedBytes.size

        val accessor = object : ScanProcessor.FileAccessor {
            override fun readText(path: String): String? = files[path]?.toString(Charsets.UTF_8)
            override fun openInput(path: String): InputStream? = files[path]?.let { ByteArrayInputStream(it) }
            override fun openOutput(path: String): OutputStream? = object : java.io.ByteArrayOutputStream() {
                override fun close() {
                    super.close()
                    // Emulate truncate: overwrite completely
                    files[path] = toByteArray()
                }
            }
            override fun delete(path: String): Boolean = files.remove(path) != null
            override fun listImages(scanFolder: String): List<String> =
                files.keys.filter { it.startsWith("scan_0/images/") }
            override fun listScanFolders(): List<String> = listOf("scan_0")
            override fun compressToZip(zipOutputStream: OutputStream): String = ""
            override fun cleanupUncompressed() {}
        }

        val processor = ScanProcessor()
        val ok = processor.process("session_trunc", accessor) {}
        assertTrue(ok)

        val writtenTransformsBytes = files["scan_0/transforms.json"]!!
        // The pruned file has fewer frames, so it must be shorter than the initial length
        assertTrue("Rewritten transforms.json must be truncated", writtenTransformsBytes.size < initialLen)

        // Parse with JSONObject - must parse cleanly without any trailing garbage
        val parsedJson = JSONObject(writtenTransformsBytes.toString(Charsets.UTF_8))
        assertTrue(parsedJson.has("frames"))

        // Verify certificate hash matches the exact clean file
        val certJson = JSONObject(files["certificate.json"]!!.toString(Charsets.UTF_8))
        val certHash = certJson.getJSONObject("manifest_hashes").getString("scan_0/transforms.json")
        val actualHash = ScanProcessor.computeSha256(ByteArrayInputStream(writtenTransformsBytes))
        assertEquals("Certificate hash must match clean truncated transforms.json", actualHash, certHash)
    }

    @Test
    fun `packagePropertyScanArchive produces unified multi-room zip with single root certificate and valid SHA256 hashes`() {
        val tempDir = java.nio.file.Files.createTempDirectory("package_prop_test").toFile()
        try {
            fun createRoomZip(roomZip: java.io.File, frameCount: Int, totalInput: Int) {
                java.util.zip.ZipOutputStream(java.io.FileOutputStream(roomZip)).use { zos ->
                    val frameList = (0 until frameCount).joinToString(",") { i ->
                        """{"file_path":"images/frame_%05d.jpg","transform_matrix":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}""".format(i)
                    }
                    val transforms = """{"w":1080,"h":1920,"fl_x":1000.0,"fl_y":1000.0,"cx":540.0,"cy":960.0,"frames":[$frameList]}"""
                    zos.putNextEntry(java.util.zip.ZipEntry("transforms.json"))
                    zos.write(transforms.toByteArray(Charsets.UTF_8))
                    zos.closeEntry()

                    zos.putNextEntry(java.util.zip.ZipEntry("trajectory.csv"))
                    zos.write("timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking,exported\n100,0,0,0,0,0,0,1,TRACKING,true\n".toByteArray())
                    zos.closeEntry()

                    zos.putNextEntry(java.util.zip.ZipEntry("coverage_summary.json"))
                    zos.write("{}".toByteArray())
                    zos.closeEntry()

                    zos.putNextEntry(java.util.zip.ZipEntry("focus_metadata.json"))
                    zos.write("{}".toByteArray())
                    zos.closeEntry()

                    for (i in 0 until frameCount) {
                        zos.putNextEntry(java.util.zip.ZipEntry("images/frame_%05d.jpg".format(i)))
                        zos.write(byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xD9.toByte(), i.toByte()))
                        zos.closeEntry()
                    }

                    val cert = """{"schema_version":"1.0.0","frames":{"total_input_frames":$totalInput,"retained_frames_count":$frameCount},"orientation_state":{"raw_width":1920,"raw_height":1080,"is_already_portrait":false,"processed_width":1080,"processed_height":1920}}"""
                    zos.putNextEntry(java.util.zip.ZipEntry("certificate.json"))
                    zos.write(cert.toByteArray(Charsets.UTF_8))
                    zos.closeEntry()
                }
            }

            val room0Zip = java.io.File(tempDir, "room_0.zip")
            val room1Zip = java.io.File(tempDir, "room_1.zip")
            createRoomZip(room0Zip, frameCount = 2, totalInput = 10)
            createRoomZip(room1Zip, frameCount = 1, totalInput = 5)

            val outputZip = java.io.File(tempDir, "unified_property.zip")
            var lastProgress = 0
            val ok = ScanProcessor.packagePropertyScanArchive(
                targetZipFile = outputZip,
                roomSources = listOf(room0Zip, room1Zip),
                onProgress = { lastProgress = it }
            )

            assertTrue(ok)
            assertEquals(100, lastProgress)
            assertTrue(outputZip.isFile)

            val zipEntries = mutableMapOf<String, ByteArray>()
            java.util.zip.ZipFile(outputZip).use { zf ->
                for (entry in zf.entries().asSequence()) {
                    zipEntries[entry.name] = zf.getInputStream(entry).readBytes()
                }
            }

            // Verify structure: scan_0, scan_1, and root certificate.json
            assertTrue("Must contain scan_0/transforms.json", zipEntries.containsKey("scan_0/transforms.json"))
            assertTrue("Must contain scan_0/trajectory.csv", zipEntries.containsKey("scan_0/trajectory.csv"))
            assertTrue("Must contain scan_0/coverage_summary.json", zipEntries.containsKey("scan_0/coverage_summary.json"))
            assertTrue("Must contain scan_0/focus_metadata.json", zipEntries.containsKey("scan_0/focus_metadata.json"))
            assertTrue("Must contain scan_0/images/frame_00000.jpg", zipEntries.containsKey("scan_0/images/frame_00000.jpg"))
            assertTrue("Must contain scan_0/images/frame_00001.jpg", zipEntries.containsKey("scan_0/images/frame_00001.jpg"))

            assertTrue("Must contain scan_1/transforms.json", zipEntries.containsKey("scan_1/transforms.json"))
            assertTrue("Must contain scan_1/trajectory.csv", zipEntries.containsKey("scan_1/trajectory.csv"))
            assertTrue("Must contain scan_1/coverage_summary.json", zipEntries.containsKey("scan_1/coverage_summary.json"))
            assertTrue("Must contain scan_1/focus_metadata.json", zipEntries.containsKey("scan_1/focus_metadata.json"))
            assertTrue("Must contain scan_1/images/frame_00000.jpg", zipEntries.containsKey("scan_1/images/frame_00000.jpg"))

            assertTrue("Must contain root certificate.json", zipEntries.containsKey("certificate.json"))

            // Verify NO per-scan certificates exist
            assertFalse("Must NOT contain scan_0/certificate.json", zipEntries.containsKey("scan_0/certificate.json"))
            assertFalse("Must NOT contain scan_1/certificate.json", zipEntries.containsKey("scan_1/certificate.json"))

            // Verify certificate contents
            val certJson = JSONObject(zipEntries["certificate.json"]!!.toString(Charsets.UTF_8))
            assertEquals("1.0.0", certJson.getString("schema_version"))
            val framesObj = certJson.getJSONObject("frames")
            assertEquals(15, framesObj.getInt("total_input_frames"))
            assertEquals(3, framesObj.getInt("retained_frames_count"))
            assertEquals(12, framesObj.getInt("discarded_frames_count"))

            val mHashes = certJson.getJSONObject("manifest_hashes")
            val iHashes = certJson.getJSONObject("image_hashes")

            assertEquals(8, mHashes.length())
            assertEquals(3, iHashes.length())

            // Verify all recorded hashes match the actual byte contents in the ZIP
            for (manifestKey in listOf(
                "scan_0/transforms.json", "scan_0/trajectory.csv", "scan_0/coverage_summary.json", "scan_0/focus_metadata.json",
                "scan_1/transforms.json", "scan_1/trajectory.csv", "scan_1/coverage_summary.json", "scan_1/focus_metadata.json"
            )) {
                assertTrue("Missing manifest hash for $manifestKey", mHashes.has(manifestKey))
                val expectedHash = mHashes.getString(manifestKey)
                val actualHash = ScanProcessor.computeSha256(ByteArrayInputStream(zipEntries[manifestKey]!!))
                assertEquals("Hash mismatch for $manifestKey", actualHash, expectedHash)
            }

            for (imgKey in listOf("scan_0/images/frame_00000.jpg", "scan_0/images/frame_00001.jpg", "scan_1/images/frame_00000.jpg")) {
                assertTrue("Missing image hash for $imgKey", iHashes.has(imgKey))
                val expectedHash = iHashes.getString(imgKey)
                val actualHash = ScanProcessor.computeSha256(ByteArrayInputStream(zipEntries[imgKey]!!))
                assertEquals("Hash mismatch for $imgKey", actualHash, expectedHash)
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }
}
