import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../core/body_frame.dart';
import '../core/framing.dart';
import 'tracker_config.dart';

class BodyTracker {
  BodyTracker({
    @visibleForTesting MethodChannel? methodChannel,
    @visibleForTesting EventChannel? eventChannel,
    this.gate = const FramingGate(),
  }) : _methods = methodChannel ?? const MethodChannel(_methodChannelName),
       _events = eventChannel ?? const EventChannel(_eventChannelName);

  static const _methodChannelName = 'psybot/engine_body/methods';
  static const _eventChannelName = 'psybot/engine_body/frames';

  final MethodChannel _methods;
  final EventChannel _events;

  final FramingGate gate;

  Stream<BodyFrame>? _frames;

  static bool get isSupported => Platform.isAndroid;

  Stream<BodyFrame> get frames =>
      _frames ??= _events
          .receiveBroadcastStream()
          .map(_decode)
          .where((frame) => frame != null)
          .cast<BodyFrame>();

  BodyFrame? _decode(Object? event) {
    if (event is! Map) return null;
    try {
      return BodyFrame.fromMap(event.cast<Object?, Object?>(), gate: gate);
    } on Object {
      return null;
    }
  }

  Future<bool> hasPermission() async {
    if (!isSupported) return false;
    try {
      return await _methods.invokeMethod<bool>('hasCameraPermission') ?? false;
    } on PlatformException {
      return false;
    }
  }

  Future<bool> requestPermission() async {
    if (!isSupported) return false;
    try {
      return await _methods.invokeMethod<bool>('requestCameraPermission') ??
          false;
    } on PlatformException {
      return false;
    }
  }

  Future<PreviewInfo?> start([
    TrackerConfig config = const TrackerConfig(),
  ]) async {
    if (!isSupported) {
      throw const BodyTrackerException(
        TrackerError.platformUnsupported,
        'Body tracking requires Android. On desktop, use SessionReplay to '
        'analyse a session recorded on a device.',
      );
    }

    try {
      final result = await _methods.invokeMethod<Map<Object?, Object?>>(
        'start',
        config.toMap(),
      );
      return PreviewInfo.fromMap(result);
    } on PlatformException catch (e) {
      throw BodyTrackerException(_errorFrom(e.code), e.message ?? e.code);
    }
  }

  Future<void> stop() async {
    if (!isSupported) return;
    try {
      await _methods.invokeMethod<void>('stop');
    } on PlatformException {
      return;
    }
  }

  Future<double?> get measuredFps async {
    if (!isSupported) return null;
    try {
      return await _methods.invokeMethod<double>('measuredFps');
    } on PlatformException {
      return null;
    }
  }

  static TrackerError _errorFrom(String code) => switch (code) {
    'permission_denied' => TrackerError.cameraPermissionDenied,
    'camera_unavailable' => TrackerError.cameraUnavailable,
    'model_unavailable' => TrackerError.modelUnavailable,
    _ => TrackerError.unknown,
  };
}