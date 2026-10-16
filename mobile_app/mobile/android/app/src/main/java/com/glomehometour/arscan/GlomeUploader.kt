package com.glomehometour.arscan

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.Executors

/**
 * Uploads a local scan archive (.zip) directly to the Glome Local API server.
 * Uses standard java.net.HttpURLConnection so no additional HTTP client dependencies are required.
 */
object GlomeUploader {

    private const val TAG = "GlomeUploader"
    private const val DEFAULT_SERVER_URL = "http://10.0.2.2:8000" // Default for Android emulator to host machine
    private const val PREFS = "glome_server"
    private const val SERVER_URL_KEY = "base_url"
    private const val TOKEN_KEY = "auth_token"
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    interface UploadCallback {
        fun onProgress(percentage: Int)
        fun onSuccess(propertyId: String)
        fun onError(error: String)
    }

    interface SignInCallback {
        fun onSuccess(token: String)
        fun onError(error: String)
    }

    fun serverBaseUrl(context: Context): String =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(SERVER_URL_KEY, DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL

    fun saveServerBaseUrl(context: Context, value: String): Boolean {
        val normalized = value.trim().trimEnd('/')
        return try {
            val parsed = java.net.URI(normalized)
            if (parsed.scheme !in setOf("http", "https") || parsed.host.isNullOrBlank() ||
                parsed.path.isNotEmpty() || parsed.query != null || parsed.fragment != null || parsed.userInfo != null
            ) {
                false
            } else {
                val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                val edit = prefs.edit().putString(SERVER_URL_KEY, normalized)
                if (prefs.getString(SERVER_URL_KEY, DEFAULT_SERVER_URL) != normalized) edit.remove(TOKEN_KEY)
                edit.apply()
                true
            }
        } catch (_: Exception) {
            false
        }
    }

    fun savedAuthToken(context: Context): String? =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(TOKEN_KEY, null)

    fun saveAuthToken(context: Context, token: String) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(TOKEN_KEY, token).apply()
    }

    fun viewerUrl(context: Context, propertyId: String): String {
        val server = URL(serverBaseUrl(context))
        return "${server.protocol}://${server.host}:5173/mobile-viewer/$propertyId"
    }

    fun signIn(context: Context, email: String, password: String, callback: SignInCallback) {
        executor.execute {
            try {
                val connection = URL("${serverBaseUrl(context)}/api/auth/signin")
                    .openConnection() as HttpURLConnection
                val code: Int
                val body: String
                try {
                    connection.requestMethod = "POST"
                    connection.doOutput = true
                    connection.connectTimeout = 15000
                    connection.readTimeout = 15000
                    connection.setRequestProperty("Content-Type", "application/json")
                    connection.outputStream.use {
                        it.write(JSONObject().put("email", email.trim()).put("password", password).toString().toByteArray())
                    }
                    code = connection.responseCode
                    body = (if (code in 200..299) connection.inputStream else connection.errorStream)
                        ?.bufferedReader()?.use { it.readText() }.orEmpty()
                } finally {
                    connection.disconnect()
                }
                if (code !in 200..299) throw IllegalStateException(JSONObject(body).optString("detail", "Sign in failed"))
                val token = JSONObject(body).getString("token")
                saveAuthToken(context, token)
                mainHandler.post { callback.onSuccess(token) }
            } catch (e: Exception) {
                mainHandler.post { callback.onError(e.message ?: "Sign in failed") }
            }
        }
    }

    /**
     * Uploads the room zip file to the Glome backend /api/properties/upload endpoint.
     */
    fun uploadScan(
        context: Context,
        zipFile: File,
        propertyAddress: String,
        authToken: String? = null,
        baseUrl: String? = null,
        callback: UploadCallback
    ) {
        executor.execute {
            var connection: HttpURLConnection? = null
            try {
                if (!zipFile.exists() || zipFile.length() == 0L) {
                    mainHandler.post { callback.onError("Scan zip file does not exist or is empty") }
                    return@execute
                }

                val boundary = "GlomeBoundary" + UUID.randomUUID().toString()
                val lineEnd = "\r\n"
                val twoHyphens = "--"

                val url = URL("${baseUrl ?: serverBaseUrl(context)}/api/properties/upload")
                connection = (url.openConnection() as HttpURLConnection).apply {
                    doInput = true
                    doOutput = true
                    useCaches = false
                    requestMethod = "POST"
                    setRequestProperty("Connection", "Keep-Alive")
                    setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
                    if (!authToken.isNullOrBlank()) {
                        setRequestProperty("Authorization", "Bearer $authToken")
                    }
                    connectTimeout = 15000
                    readTimeout = 120000 // 2 minutes for large archives
                }

                val totalBytes = zipFile.length()
                var bytesUploaded = 0L

                val outputStream: OutputStream = connection.outputStream

                // 1. Form field: address
                outputStream.write((twoHyphens + boundary + lineEnd).toByteArray())
                outputStream.write(("Content-Disposition: form-data; name=\"address\"" + lineEnd + lineEnd).toByteArray())
                outputStream.write((propertyAddress + lineEnd).toByteArray())

                // 2. File field: file
                outputStream.write((twoHyphens + boundary + lineEnd).toByteArray())
                outputStream.write(
                    ("Content-Disposition: form-data; name=\"file\"; filename=\"" + zipFile.name + "\"" + lineEnd).toByteArray()
                )
                outputStream.write(("Content-Type: application/zip" + lineEnd + lineEnd).toByteArray())

                // Stream file contents with progress calculation
                FileInputStream(zipFile).use { fileInput ->
                    val buffer = ByteArray(32 * 1024)
                    var bytesRead: Int
                    while (fileInput.read(buffer).also { bytesRead = it } != -1) {
                        outputStream.write(buffer, 0, bytesRead)
                        bytesUploaded += bytesRead
                        val pct = ((bytesUploaded * 100) / totalBytes).toInt()
                        mainHandler.post { callback.onProgress(pct) }
                    }
                }

                outputStream.write(lineEnd.toByteArray())
                outputStream.write((twoHyphens + boundary + twoHyphens + lineEnd).toByteArray())
                outputStream.flush()
                outputStream.close()

                val responseCode = connection.responseCode
                if (responseCode in 200..299) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    Log.i(TAG, "Upload succeeded: $response")
                    mainHandler.post { callback.onSuccess(JSONObject(response).getString("propertyId")) }
                } else {
                    val err = connection.errorStream?.bufferedReader()?.use { it.readText() } ?: "HTTP $responseCode"
                    Log.e(TAG, "Upload failed: $err")
                    mainHandler.post { callback.onError(err) }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Upload exception", e)
                mainHandler.post { callback.onError(e.message ?: "Upload network exception") }
            } finally {
                connection?.disconnect()
            }
        }
    }
}
