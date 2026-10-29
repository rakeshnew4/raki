package com.so.samsungpro

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {

    private lateinit var serverUrlInput: EditText
    private lateinit var statusText: TextView
    private var pendingStartUrl: String? = null

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val text = intent.getStringExtra(AudioStreamService.EXTRA_STATUS_TEXT) ?: return
            statusText.text = text
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        val micGranted = results[Manifest.permission.RECORD_AUDIO] == true
        if (micGranted) {
            pendingStartUrl?.let { startStreaming(it) }
        } else {
            Toast.makeText(this, "Microphone permission is required to stream audio", Toast.LENGTH_LONG).show()
        }
        pendingStartUrl = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        serverUrlInput = findViewById(R.id.serverUrlInput)
        statusText = findViewById(R.id.statusText)

        findViewById<Button>(R.id.startButton).setOnClickListener {
            val url = serverUrlInput.text.toString().trim()
            if (url.isEmpty()) {
                Toast.makeText(this, "Enter a server WebSocket URL", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            requestPermissionsAndStart(url)
        }

        findViewById<Button>(R.id.stopButton).setOnClickListener {
            val intent = Intent(this, AudioStreamService::class.java).apply {
                action = AudioStreamService.ACTION_STOP
            }
            startService(intent)
        }
    }

    override fun onStart() {
        super.onStart()
        val filter = IntentFilter(AudioStreamService.BROADCAST_STATUS)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(statusReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(statusReceiver, filter)
        }
    }

    override fun onStop() {
        super.onStop()
        unregisterReceiver(statusReceiver)
    }

    private fun requestPermissionsAndStart(url: String) {
        val needed = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            needed.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        val missing = needed.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            startStreaming(url)
        } else {
            pendingStartUrl = url
            permissionLauncher.launch(missing.toTypedArray())
        }
    }

    private fun startStreaming(url: String) {
        statusText.text = "Connecting..."
        val intent = Intent(this, AudioStreamService::class.java).apply {
            action = AudioStreamService.ACTION_START
            putExtra(AudioStreamService.EXTRA_SERVER_URL, url)
        }
        ContextCompat.startForegroundService(this, intent)
    }
}
