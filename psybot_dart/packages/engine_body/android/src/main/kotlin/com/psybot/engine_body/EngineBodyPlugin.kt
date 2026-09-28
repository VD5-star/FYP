package com.psybot.engine_body

import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.util.Size
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceRequest
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.PluginRegistry
import io.flutter.view.TextureRegistry
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.os.Handler
import android.os.Looper

private class FixedStateLifecycleOwner : LifecycleOwner {
    private val registry = LifecycleRegistry(this)

    override val lifecycle: Lifecycle
        get() = registry

    fun start() {
        registry.currentState = Lifecycle.State.STARTED
        registry.currentState = Lifecycle.State.RESUMED
    }

    fun stop() {
        registry.currentState = Lifecycle.State.DESTROYED
    }
}

class EngineBodyPlugin :
    FlutterPlugin,
    ActivityAware,
    MethodChannel.MethodCallHandler,
    EventChannel.StreamHandler {

    private var applicationContext: Context? = null
    private var activity: Activity? = null
    private var textureRegistry: TextureRegistry? = null
    private var methodChannel: MethodChannel? = null
    private var eventChannel: EventChannel? = null
    private var eventSink: EventChannel.EventSink? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    private var analysisExecutor: ExecutorService? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var poseDetector: PoseDetector? = null
    private var textureEntry: TextureRegistry.SurfaceTextureEntry? = null
    private var lifecycleOwner: FixedStateLifecycleOwner? = null
    private var isMirrored = false
    private var targetFps = 15
    private var frameIntervalNanos = 1_000_000_000L / 15
    private var lastAnalyzedAtNanos = 0L
    private var frameCountInWindow = 0
    private var windowStartNanos = 0L
    private var measuredFpsValue: Double? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        applicationContext = binding.applicationContext
        textureRegistry = binding.textureRegistry
        methodChannel = MethodChannel(binding.binaryMessenger, METHOD_CHANNEL)
        methodChannel?.setMethodCallHandler(this)
        eventChannel = EventChannel(binding.binaryMessenger, EVENT_CHANNEL)
        eventChannel?.setStreamHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        teardownCamera()
        methodChannel?.setMethodCallHandler(null)
        methodChannel = null
        eventChannel?.setStreamHandler(null)
        eventChannel = null
        eventSink = null
        textureRegistry = null
        applicationContext = null
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        activity = binding.activity
    }

    override fun onDetachedFromActivityForConfigChanges() {
        activity = null
    }

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
        activity = binding.activity
    }

    override fun onDetachedFromActivity() {
        activity = null
    }

    override fun onListen(arguments: Any?, sink: EventChannel.EventSink?) {
        eventSink = sink
    }

    override fun onCancel(arguments: Any?) {
        eventSink = null
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "hasCameraPermission" -> result.success(hasCameraPermission())
            "requestCameraPermission" -> result.success(hasCameraPermission())
            "start" -> handleStart(call, result)
            "stop" -> {
                teardownCamera()
                result.success(null)
            }
            "measuredFps" -> result.success(measuredFpsValue)
            else -> result.notImplemented()
        }
    }

    private fun hasCameraPermission(): Boolean {
        val context = applicationContext ?: return false
        return ContextCompat.checkSelfPermission(
            context,
            android.Manifest.permission.CAMERA,
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun handleStart(call: MethodCall, result: MethodChannel.Result) {
        val context = applicationContext
        if (context == null) {
            result.error("camera_unavailable", "No application context.", null)
            return
        }
        if (!hasCameraPermission()) {
            result.error("permission_denied", "Camera permission is not granted.", null)
            return
        }

        teardownCamera()

        val args = call.arguments as? Map<*, *> ?: emptyMap<Any?, Any?>()
        val modelName = args["model"] as? String ?: "fast"
        val facingName = args["facing"] as? String ?: "front"
        targetFps = (args["targetFps"] as? Number)?.toInt() ?: 15
        val analysisWidth = (args["analysisWidth"] as? Number)?.toInt() ?: 480
        frameIntervalNanos = 1_000_000_000L / targetFps.coerceAtLeast(1)
        isMirrored = facingName == "front"

        val cameraSelector = if (facingName == "back") {
            CameraSelector.DEFAULT_BACK_CAMERA
        } else {
            CameraSelector.DEFAULT_FRONT_CAMERA
        }

        val detector = try {
            buildDetector(modelName)
        } catch (error: Exception) {
            result.error("model_unavailable", error.message, null)
            return
        }
        poseDetector = detector

        val entry = textureRegistry?.createSurfaceTexture()
        if (entry == null) {
            result.error("camera_unavailable", "Could not register a texture.", null)
            return
        }
        textureEntry = entry

        val executor = Executors.newSingleThreadExecutor()
        analysisExecutor = executor

        val owner = FixedStateLifecycleOwner()
        lifecycleOwner = owner

        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                cameraProvider = provider

                val preview = Preview.Builder().build()
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(Size(analysisWidth, analysisWidth * 4 / 3))
                    .build()
                analysis.setAnalyzer(executor) { image -> analyzeFrame(image) }

                var responded = false
                preview.setSurfaceProvider { request ->
                    val texture = entry.surfaceTexture()
                    texture.setDefaultBufferSize(
                        request.resolution.width,
                        request.resolution.height,
                    )
                    val surface = Surface(texture)
                    request.provideSurface(surface, executor) { surface.release() }

                    if (!responded) {
                        responded = true
                        mainHandler.post {
                            owner.start()
                            result.success(
                                mapOf(
                                    "textureId" to entry.id(),
                                    "previewWidth" to request.resolution.width,
                                    "previewHeight" to request.resolution.height,
                                    "rotation" to currentRotationDegrees(),
                                    "isMirrored" to isMirrored,
                                ),
                            )
                        }
                    }
                }

                mainHandler.post {
                    try {
                        provider.unbindAll()
                        provider.bindToLifecycle(owner, cameraSelector, preview, analysis)
                        owner.start()
                    } catch (error: Exception) {
                        teardownCamera()
                        result.error("camera_unavailable", error.message, null)
                    }
                }
            } catch (error: Exception) {
                mainHandler.post {
                    teardownCamera()
                    result.error("camera_unavailable", error.message, null)
                }
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun buildDetector(modelName: String): PoseDetector {
        return if (modelName == "accurate") {
            val options = AccuratePoseDetectorOptions.Builder()
                .setDetectorMode(AccuratePoseDetectorOptions.STREAM_MODE)
                .build()
            PoseDetection.getClient(options)
        } else {
            val options = PoseDetectorOptions.Builder()
                .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
                .build()
            PoseDetection.getClient(options)
        }
    }

    private fun currentRotationDegrees(): Int {
        val rotation = activity?.windowManager?.defaultDisplay?.rotation ?: Surface.ROTATION_0
        return when (rotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
    }

    private fun analyzeFrame(image: ImageProxy) {
        val now = System.nanoTime()
        if (now - lastAnalyzedAtNanos < frameIntervalNanos) {
            image.close()
            return
        }
        lastAnalyzedAtNanos = now

        val mediaImage = image.image
        val detector = poseDetector
        if (mediaImage == null || detector == null) {
            image.close()
            return
        }

        val inputImage = InputImage.fromMediaImage(mediaImage, image.imageInfo.rotationDegrees)
        val width = inputImage.width.toDouble()
        val height = inputImage.height.toDouble()
        val timestampMicros = image.imageInfo.timestamp / 1000L

        detector.process(inputImage)
            .addOnSuccessListener { pose -> onPoseDetected(pose, width, height, timestampMicros) }
            .addOnCompleteListener { image.close() }
    }

    private fun onPoseDetected(pose: Pose, width: Double, height: Double, timestampMicros: Long) {
        recordFrameForFps()
        val sink = eventSink ?: return
        if (width <= 0.0 || height <= 0.0) return

        val landmarks = pose.allPoseLandmarks.map { landmark: PoseLandmark ->
            mapOf(
                "type" to landmark.landmarkType,
                "x" to (landmark.position.x / width),
                "y" to (landmark.position.y / height),
                "z" to (landmark.position3D.z / width),
                "likelihood" to landmark.inFrameLikelihood.toDouble(),
                "inFrameLikelihood" to landmark.inFrameLikelihood.toDouble(),
            )
        }

        val frame = mapOf(
            "timestampMicros" to timestampMicros,
            "isMirrored" to isMirrored,
            "landmarks" to landmarks,
        )

        mainHandler.post { sink.success(frame) }
    }

    private fun recordFrameForFps() {
        val now = System.nanoTime()
        if (windowStartNanos == 0L) {
            windowStartNanos = now
            frameCountInWindow = 0
        }
        frameCountInWindow += 1
        val elapsed = now - windowStartNanos
        if (elapsed >= 1_000_000_000L) {
            measuredFpsValue = frameCountInWindow * 1_000_000_000.0 / elapsed
            windowStartNanos = now
            frameCountInWindow = 0
        }
    }

    private fun teardownCamera() {
        analysisExecutor?.shutdown()
        analysisExecutor = null
        mainHandler.post {
            cameraProvider?.unbindAll()
            cameraProvider = null
        }
        lifecycleOwner?.stop()
        lifecycleOwner = null
        poseDetector?.close()
        poseDetector = null
        textureEntry?.release()
        textureEntry = null
        lastAnalyzedAtNanos = 0L
        windowStartNanos = 0L
        frameCountInWindow = 0
        measuredFpsValue = null
    }

    companion object {
        private const val METHOD_CHANNEL = "psybot/engine_body/methods"
        private const val EVENT_CHANNEL = "psybot/engine_body/frames"
    }
}
