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
        val createdAtMs: Long = System.currentTimeMillis(),
        var uploadStatus: UploadStatus = UploadStatus.LOCAL_ONLY,
    )

    data class RoomScan(
        val id: String = UUID.randomUUID().toString(),
        val propertyId: String,
        var label: String,
        var sessionName: String,
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

    /** Session name -> room, for the gallery's file-first listing to look rows up against. */
    fun roomsBySession(): Map<String, RoomScan> = read().second.associateBy { it.sessionName }

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
    fun recordRoomScan(propertyId: String, label: String, sessionName: String, coveragePercent: Float?) =
        mutate { _, rooms ->
            val existing = rooms.firstOrNull { it.sessionName == sessionName }
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
                        coveragePercent = coveragePercent,
                        status = ScanStatus.DONE,
                    )
                )
            }
        }

    fun deleteRoomScan(sessionName: String) = mutate { _, rooms ->
        rooms.removeAll { it.sessionName == sessionName }
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
                    createdAtMs = o.optLong("created_at_ms"),
                    uploadStatus = enumOr(o.optString("upload_status"), UploadStatus.LOCAL_ONLY),
                )
            }
            val rooms = root.optJSONArray("rooms").mapObjects { o ->
                RoomScan(
                    id = o.getString("id"),
                    propertyId = o.getString("property_id"),
                    label = o.getString("label"),
                    sessionName = o.getString("session_name"),
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
                put("created_at_ms", p.createdAtMs)
                put("upload_status", p.uploadStatus.name)
            })
        })
        root.put("rooms", JSONArray().apply {
            for (r in rooms) put(JSONObject().apply {
                put("id", r.id)
                put("property_id", r.propertyId)
                put("label", r.label)
                put("session_name", r.sessionName)
                put("coverage_percent", r.coveragePercent ?: JSONObject.NULL)
                put("captured_at_ms", r.capturedAtMs)
                put("status", r.status.name)
            })
        })
        val tmp = File(file.parentFile, "$FILE_NAME.tmp")
        tmp.writeText(root.toString())
        tmp.renameTo(file)
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
