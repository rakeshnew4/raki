package com.so.samsungpro

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioRecordingConfiguration
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * Captures microphone audio continuously and streams raw PCM16 mono frames
 * to a server over a WebSocket, as a foreground service so it survives
 * backgrounding / screen-off.
 */
class AudioStreamService : Service() {

    companion object {
        const val ACTION_START = "com.so.samsungpro.action.START"
        const val ACTION_STOP = "com.so.samsungpro.action.STOP"
        const val EXTRA_SERVER_URL = "server_url"

        const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        // v2 suffix: channel importance is immutable after first creation on-device,
        // so a new id is needed to apply the lowered importance to existing installs.
        private const val NOTIFICATION_CHANNEL_ID = "audio_stream_channel_v2"
        private const val NOTIFICATION_ID = 1

        const val BROADCAST_STATUS = "com.so.samsungpro.STATUS"
        const val EXTRA_STATUS_TEXT = "status_text"
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build()

    private var webSocket: WebSocket? = null
    private var audioRecord: AudioRecord? = null

    /** True whenever the user wants streaming to be running (start→stop); drives reconnects. */
    private val active = AtomicBoolean(false)

    /** True only while the capture thread loop should keep reading from the mic. */
    private val isRecording = AtomicBoolean(false)
    private var recordingThread: Thread? = null
    private var reconnectAttempts = 0

    /** True while the OS has silenced our mic capture (e.g. an active phone call). */
    private val isSilenced = AtomicBoolean(false)
    private var audioRecordingCallback: AudioManager.AudioRecordingCallback? = null

    // SpeechRecognizer must only be touched from the main thread.
    private val mainHandler = Handler(Looper.getMainLooper())
    private var speechRecognizer: SpeechRecognizer? = null
    private val recognizerActive = AtomicBoolean(false)

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val url = intent.getStringExtra(EXTRA_SERVER_URL) ?: return START_NOT_STICKY
                startForegroundWithNotification()
                connectAndStream(url)
            }
            ACTION_STOP -> {
                stopStreaming()
                stopSelf()
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        stopStreaming()
        super.onDestroy()
    }

    private fun startForegroundWithNotification() {
        // IMPORTANCE_MIN is the least intrusive Android allows for a foreground
        // service notification: no sound, no status bar icon, collapsed into the
        // shade's "silent" section. A foreground mic service cannot go lower than
        // this — Android requires some persistent notification while recording.
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            "Audio Streaming",
            NotificationManager.IMPORTANCE_MIN
        ).apply {
            setSound(null, null)
            enableVibration(false)
            setShowBadge(false)
        }
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)

        val notification: Notification = NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("SamsungPro")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setPriority(NotificationCompat.PRIORITY_MIN)
            .setSilent(true)
            .setShowWhen(false)
            .setOngoing(true)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun connectAndStream(url: String) {
        active.set(true)

        val request = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                reconnectAttempts = 0
                broadcastStatus("Connected — streaming")
                startRecording(webSocket)
                startHindiRecognition(webSocket)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                broadcastStatus("Connection error: ${t.message}")
                scheduleReconnect(url)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                broadcastStatus("Disconnected: $reason")
                if (active.get()) {
                    scheduleReconnect(url)
                }
            }
        })
    }

    private fun scheduleReconnect(url: String) {
        stopRecordingOnly()
        stopHindiRecognition()
        if (!active.get()) return
        reconnectAttempts++
        val delayMs = minOf(30_000L, 1000L * reconnectAttempts)
        broadcastStatus("Reconnecting in ${delayMs / 1000}s...")
        thread {
            Thread.sleep(delayMs)
            if (active.get()) {
                connectAndStream(url)
            }
        }
    }

    private fun startRecording(socket: WebSocket) {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            broadcastStatus("Microphone permission not granted")
            stopSelf()
            return
        }

        val minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        val bufferSize = maxOf(minBufferSize, SAMPLE_RATE / 10 * 2) // ~100ms of audio

        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize
        )

        if (record.state != AudioRecord.STATE_INITIALIZED) {
            broadcastStatus("Failed to initialize AudioRecord")
            record.release()
            stopSelf()
            return
        }

        audioRecord = record
        isRecording.set(true)
        isSilenced.set(false)
        record.startRecording()
        registerSilenceWatcher(record)

        recordingThread = thread(name = "AudioCaptureThread") {
            val buffer = ByteArray(bufferSize)
            while (isRecording.get()) {
                val read = record.read(buffer, 0, buffer.size)
                // Keep draining the mic buffer even while silenced, so it doesn't
                // overflow/block — just skip forwarding silence to the server.
                if (read > 0 && !isSilenced.get()) {
                    val chunk = if (read == buffer.size) buffer else buffer.copyOf(read)
                    socket.send(chunk.toByteString())
                }
            }
        }
    }

    private fun registerSilenceWatcher(record: AudioRecord) {
        val audioManager = getSystemService(AudioManager::class.java) ?: return
        val sessionId = record.audioSessionId

        val callback = object : AudioManager.AudioRecordingCallback() {
            override fun onRecordingConfigChanged(configs: MutableList<AudioRecordingConfiguration>) {
                val silenced = configs.any { it.clientAudioSessionId == sessionId && it.isClientSilenced }
                if (isSilenced.compareAndSet(!silenced, silenced)) {
                    broadcastStatus(if (silenced) "Paused (mic silenced by system, e.g. a call)" else "Connected — streaming")
                }
            }
        }
        audioRecordingCallback = callback
        audioManager.registerAudioRecordingCallback(callback, null)
    }

    private fun unregisterSilenceWatcher() {
        audioRecordingCallback?.let {
            getSystemService(AudioManager::class.java)?.unregisterAudioRecordingCallback(it)
        }
        audioRecordingCallback = null
    }

    private fun stopRecordingOnly() {
        isRecording.set(false)
        unregisterSilenceWatcher()
        audioRecord?.let {
            try {
                it.stop()
            } catch (_: IllegalStateException) {
                // already stopped
            }
            it.release()
        }
        audioRecord = null
        recordingThread?.interrupt()
        recordingThread = null
    }

    private fun stopStreaming() {
        active.set(false)
        stopRecordingOnly()
        stopHindiRecognition()
        webSocket?.close(1000, "Client stopped streaming")
        webSocket = null
        broadcastStatus("Stopped")
    }

    /**
     * Runs Android's on-device/native SpeechRecognizer in Hindi alongside the raw
     * audio stream, forwarding recognized text to the server as JSON text frames.
     * Note: this competes with [startRecording]'s AudioRecord for the microphone;
     * on some devices one of the two captures may be silenced by the OS.
     */
    private fun startHindiRecognition(socket: WebSocket) {
        mainHandler.post {
            if (!SpeechRecognizer.isRecognitionAvailable(this)) {
                broadcastStatus("Hindi speech recognition unavailable on this device")
                return@post
            }
            val recognizer = SpeechRecognizer.createSpeechRecognizer(this)
            speechRecognizer = recognizer
            recognizer.setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle) {
                    results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull()
                        ?.takeIf { it.isNotBlank() }
                        ?.let { sendTranscript(socket, it, final = true) }
                    restartListening(recognizer, delayMs = 0)
                }

                override fun onPartialResults(partialResults: Bundle) {
                    partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull()
                        ?.takeIf { it.isNotBlank() }
                        ?.let { sendTranscript(socket, it, final = false) }
                }

                override fun onError(error: Int) {
                    // ERROR_NO_MATCH / ERROR_SPEECH_TIMEOUT are routine during continuous
                    // dictation (silence between utterances); back off briefly either way
                    // so a persistent failure (e.g. mic held by AudioRecord) doesn't spin.
                    restartListening(recognizer, delayMs = 400)
                }

                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
            recognizerActive.set(true)
            startListening(recognizer)
        }
    }

    private fun startListening(recognizer: SpeechRecognizer) {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "hi-IN")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)
        }
        recognizer.startListening(intent)
    }

    private fun restartListening(recognizer: SpeechRecognizer, delayMs: Long) {
        if (!recognizerActive.get() || !active.get()) return
        mainHandler.postDelayed({
            if (recognizerActive.get() && active.get()) {
                startListening(recognizer)
            }
        }, delayMs)
    }

    private fun sendTranscript(socket: WebSocket, text: String, final: Boolean) {
        val json = JSONObject().apply {
            put("type", "transcript")
            put("lang", "hi-IN")
            put("final", final)
            put("text", text)
        }
        socket.send(json.toString())
    }

    private fun stopHindiRecognition() {
        recognizerActive.set(false)
        mainHandler.post {
            speechRecognizer?.apply {
                stopListening()
                cancel()
                destroy()
            }
            speechRecognizer = null
        }
    }

    private fun broadcastStatus(text: String) {
        val intent = Intent(BROADCAST_STATUS).apply {
            putExtra(EXTRA_STATUS_TEXT, text)
            setPackage(packageName)
        }
        sendBroadcast(intent)
    }
}
