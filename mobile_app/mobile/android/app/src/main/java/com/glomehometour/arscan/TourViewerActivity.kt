package com.glomehometour.arscan

import android.os.Bundle
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity

class TourViewerActivity : AppCompatActivity() {
    private var webView: WebView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val propertyId = intent.getStringExtra(EXTRA_PROPERTY_ID) ?: run { finish(); return }
        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = false
            settings.allowContentAccess = false
            webViewClient = WebViewClient()
            webChromeClient = WebChromeClient()
            setBackgroundColor(0xFF020617.toInt())
            loadUrl(GlomeUploader.viewerUrl(this@TourViewerActivity, propertyId))
        }
        setContentView(webView)
    }

    override fun onDestroy() {
        webView?.apply {
            stopLoading()
            destroy()
        }
        webView = null
        super.onDestroy()
    }

    companion object {
        const val EXTRA_PROPERTY_ID = "server_property_id"
    }
}
