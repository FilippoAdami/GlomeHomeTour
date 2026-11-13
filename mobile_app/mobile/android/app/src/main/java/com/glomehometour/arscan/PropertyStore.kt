package com.glomehometour.arscan

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

/**
 * Local index of properties and the room scans captured for them.
 *
 * This is an index *on top of* the session zips DatasetWriter/ScanGalleryActivity already
 * produce, not a replacement for them: a row's [RoomScan.sessionName] is the zip's name, and the
 * zips stay the source of truth for what actually exists on disk (a zip deleted by a file
 * manager leaves an orphan row here, which the gallery reconciles by listing files first and
 * looking rows up second).
 *
 * ponytail: one JSON file rewritten whole under a lock, not a Room database -- the plan called
 * for Room, but the access pattern is "load every row, show them all" over tens of rows, which
 * needs neither SQL nor the KSP annotation processor this build doesn't otherwise have. Move to
 * Room when something needs an actual query (server-side sync, or scan counts in the thousands).
 */
class PropertyStore(context: Context) {

    enum class UploadStatus { LOCAL_ONLY, UPLOADING, UPLOADED, UPLOAD_FAILED }
    enum class ScanStatus { CAPTURING, DONE, UPLOADED, UPLOAD_FAILED }

    data class Property(
        val id: String = UUID.randomUUID().toString(),
        var address: String,
        var agentNote: String? = null,
        var serverPropertyId: String? = null,
        val createdAtMs: Long = System.currentTimeMillis(),
        var uploadStatus: UploadStatus = UploadStatus.LOCAL_ONLY,
        var isProcessed: Boolean = false,
    )

    data class RoomScan(
        val id: String = UUID.randomUUID().toString(),
        val propertyId: String,
        var label: String,
        var sessionName: String,
        var scanFolder: String = "scan_0",
        var coveragePercent: Float? = null,
        val capturedAtMs: Long = System.currentTimeMillis(),
        var status: ScanStatus = ScanStatus.CAPTURING,
    )

    private val file = File(context.filesDir, FILE_NAME)
    private val lock = Any()

    fun properties(): List<Property> = read().first.sortedByDescending { it.createdAtMs }

    fun property(id: String): Property? = read().first.firstOrNull { it.id == id }

    fun rooms(propertyId: String): List<RoomScan> =
        read().second.filter { it.propertyId == propertyId }.sortedBy { it.capturedAtMs }

    /** Session name -> all room rows, since legacy archives can contain several rooms. */
    fun roomsBySession(): Map<String, List<RoomScan>> = read().second.groupBy { it.sessionName }

    fun roomScan(sessionName: String, scanFolder: String): RoomScan? =
        read().second.firstOrNull { it.sessionName == sessionName && it.scanFolder == scanFolder }

    fun isPropertyProcessed(context: Context, propertyId: String): Boolean {
        val property = property(propertyId) ?: return false
        val rList = rooms(propertyId)
        if (rList.isEmpty()) return false
        val allProcessed = rList.all { ScanProcessor.isProcessed(context, it.sessionName) }
        if (allProcessed && !property.isProcessed) {
            property.isProcessed = true
            updateProperty(property)
        }
        return allProcessed
    }

    fun addProperty(address: String, note: String?): Property {
        val property = Property(address = address, agentNote = note?.takeIf { it.isNotBlank() })
        mutate { properties, _ -> properties.add(property) }
        return property
    }

    fun deleteProperty(id: String) = mutate { properties, rooms ->
        properties.removeAll { it.id == id }
        rooms.removeAll { it.propertyId == id }
    }

    /** Called when a capture finishes; upserts on session name so a retried write can't duplicate. */
    fun recordRoomScan(
        propertyId: String,
        label: String,
        sessionName: String,
        coveragePercent: Float?,
        scanFolder: String = "scan_0",
    ) =
        mutate { _, rooms ->
            val existing = rooms.firstOrNull {
                it.propertyId == propertyId && it.sessionName == sessionName && it.scanFolder == scanFolder
            }
            if (existing != null) {
                existing.label = label
                existing.coveragePercent = coveragePercent
                existing.status = ScanStatus.DONE
            } else {
                rooms.add(
                    RoomScan(
                        propertyId = propertyId,
                        label = label,
                        sessionName = sessionName,
                        scanFolder = scanFolder,
                        coveragePercent = coveragePercent,
                        status = ScanStatus.DONE,
                    )
                )
            }
        }

    fun deleteRoomScan(sessionName: String, scanFolder: String? = null) = mutate { _, rooms ->
        rooms.removeAll { it.sessionName == sessionName && (scanFolder == null || it.scanFolder == scanFolder) }
    }

    /** Expands the pre-folder-aware single index row when its archive contains several rooms. */
    fun expandLegacyRooms(propertyId: String, sessionName: String, scanFolders: List<String>) = mutate { _, rooms ->
        val matching = rooms.filter { it.propertyId == propertyId && it.sessionName == sessionName }
        val legacy = matching.singleOrNull()?.takeIf { it.scanFolder.isBlank() } ?: return@mutate
        val folders = scanFolders.distinct().sortedBy { folder ->
            folder.removePrefix("scan_").toIntOrNull() ?: Int.MAX_VALUE
        }
        if (folders.isEmpty()) return@mutate

        rooms.remove(legacy)
        val lastIndex = folders.lastIndex
        for ((index, folder) in folders.withIndex()) {
            val numericIndex = folder.removePrefix("scan_").toIntOrNull() ?: index
            rooms.add(
                if (index == lastIndex) legacy.copy(scanFolder = folder)
                else RoomScan(
                    propertyId = propertyId,
                    label = "Room ${numericIndex + 1}",
                    sessionName = sessionName,
                    scanFolder = folder,
                    capturedAtMs = legacy.capturedAtMs - (lastIndex - index),
                    status = ScanStatus.DONE,
                )
            )
        }
    }

    fun expandLegacyRoomFolders(context: Context, propertyId: String) {
        for (room in rooms(propertyId).filter { it.scanFolder.isBlank() }.distinctBy { it.sessionName }) {
            val folders = try {
                ScanProcessor.listScanFolders(context, room.sessionName)
            } catch (e: Exception) {
                Log.w(TAG, "Could not expand room folders for ${room.sessionName}", e)
                emptyList()
            }
            expandLegacyRooms(propertyId, room.sessionName, folders)
        }
    }

    fun updateProperty(property: Property) = mutate { props, _ ->
        val idx = props.indexOfFirst { it.id == property.id }
        if (idx >= 0) {
            props[idx] = property
        }
    }

    private fun mutate(block: (MutableList<Property>, MutableList<RoomScan>) -> Unit) {
        synchronized(lock) {
            val (properties, rooms) = read()
            val p = properties.toMutableList()
            val r = rooms.toMutableList()
            block(p, r)
            write(p, r)
        }
    }

    private fun read(): Pair<List<Property>, List<RoomScan>> {
        if (!file.exists()) return emptyList<Property>() to emptyList()
        return try {
            val root = JSONObject(file.readText())
            val properties = root.optJSONArray("properties").mapObjects { o ->
                Property(
                    id = o.getString("id"),
                    address = o.getString("address"),
                    agentNote = o.optString("agent_note").takeIf { it.isNotEmpty() },
                    serverPropertyId = o.optString("server_property_id").takeIf { it.isNotEmpty() },
                    createdAtMs = o.optLong("created_at_ms"),
                    uploadStatus = enumOr(o.optString("upload_status"), UploadStatus.LOCAL_ONLY),
                    isProcessed = o.optBoolean("is_processed", false),
                )
            }
            val rooms = root.optJSONArray("rooms").mapObjects { o ->
                RoomScan(
                    id = o.getString("id"),
                    propertyId = o.getString("property_id"),
                    label = o.getString("label"),
                    sessionName = o.getString("session_name"),
                    scanFolder = if (o.has("scan_folder")) o.optString("scan_folder", "scan_0") else "",
                    coveragePercent = if (o.isNull("coverage_percent")) null
                        else o.getDouble("coverage_percent").toFloat(),
                    capturedAtMs = o.optLong("captured_at_ms"),
                    status = enumOr(o.optString("status"), ScanStatus.DONE),
                )
            }
            properties to rooms
        } catch (e: Exception) {
            // A corrupt index must not take the capture app down with it: the zips on disk are
            // the real data, and the gallery still lists them all under "Unassigned".
            Log.w(TAG, "property index unreadable, starting empty", e)
            emptyList<Property>() to emptyList()
        }
    }

    private fun write(properties: List<Property>, rooms: List<RoomScan>) {
        val root = JSONObject()
        root.put("properties", JSONArray().apply {
            for (p in properties) put(JSONObject().apply {
                put("id", p.id)
                put("address", p.address)
                put("agent_note", p.agentNote ?: "")
                put("server_property_id", p.serverPropertyId ?: "")
                put("created_at_ms", p.createdAtMs)
                put("upload_status", p.uploadStatus.name)
                put("is_processed", p.isProcessed)
            })
        })
        root.put("rooms", JSONArray().apply {
            for (r in rooms) put(JSONObject().apply {
                put("id", r.id)
                put("property_id", r.propertyId)
                put("label", r.label)
                put("session_name", r.sessionName)
                put("scan_folder", r.scanFolder)
                put("coverage_percent", r.coveragePercent ?: JSONObject.NULL)
                put("captured_at_ms", r.capturedAtMs)
                put("status", r.status.name)
            })
        })
        val tmp = File(file.parentFile, "$FILE_NAME.tmp")
        tmp.writeText(root.toString())
        if (!tmp.renameTo(file)) {
            tmp.delete()
            throw IllegalStateException("Could not commit the property index")
        }
    }

    private inline fun <T> JSONArray?.mapObjects(f: (JSONObject) -> T): List<T> {
        val a = this ?: return emptyList()
        return (0 until a.length()).map { f(a.getJSONObject(it)) }
    }

    private inline fun <reified T : Enum<T>> enumOr(name: String, fallback: T): T =
        enumValues<T>().firstOrNull { it.name == name } ?: fallback

    companion object {
        private const val TAG = "ArScan"
        private const val FILE_NAME = "properties.json"

        const val EXTRA_PROPERTY_ID = "property_id"
        const val EXTRA_ROOM_LABEL = "room_label"
    }
}
