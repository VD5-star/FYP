import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'package:camera/camera.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import 'angles.dart';
import 'cameras.dart';
import 'points.dart';
import 'pose_source.dart';

bool get poseSupported => Platform.isAndroid || Platform.isIOS;

PoseSource createPoseSource() {
  if (!poseSupported) {
    return UnsupportedPoseSource(
        'tracking needs a phone, this build has no camera pose');
  }
  return MlKitPoseSource();
}

const Map<PoseLandmarkType, String> landmarkNameOf =
    <PoseLandmarkType, String>{
  PoseLandmarkType.nose: 'nose',
  PoseLandmarkType.leftEyeInner: 'leftEyeInner',
  PoseLandmarkType.leftEye: 'leftEye',
  PoseLandmarkType.leftEyeOuter: 'leftEyeOuter',
  PoseLandmarkType.rightEyeInner: 'rightEyeInner',
  PoseLandmarkType.rightEye: 'rightEye',
  PoseLandmarkType.rightEyeOuter: 'rightEyeOuter',
  PoseLandmarkType.leftEar: 'leftEar',
  PoseLandmarkType.rightEar: 'rightEar',
  PoseLandmarkType.leftMouth: 'leftMouth',
  PoseLandmarkType.rightMouth: 'rightMouth',
  PoseLandmarkType.leftShoulder: 'leftShoulder',
  PoseLandmarkType.rightShoulder: 'rightShoulder',
  PoseLandmarkType.leftElbow: 'leftElbow',
  PoseLandmarkType.rightElbow: 'rightElbow',
  PoseLandmarkType.leftWrist: 'leftWrist',
  PoseLandmarkType.rightWrist: 'rightWrist',
  PoseLandmarkType.leftPinky: 'leftPinky',
  PoseLandmarkType.rightPinky: 'rightPinky',
  PoseLandmarkType.leftIndex: 'leftIndex',
  PoseLandmarkType.rightIndex: 'rightIndex',
  PoseLandmarkType.leftThumb: 'leftThumb',
  PoseLandmarkType.rightThumb: 'rightThumb',
  PoseLandmarkType.leftHip: 'leftHip',
  PoseLandmarkType.rightHip: 'rightHip',
  PoseLandmarkType.leftKnee: 'leftKnee',
  PoseLandmarkType.rightKnee: 'rightKnee',
  PoseLandmarkType.leftAnkle: 'leftAnkle',
  PoseLandmarkType.rightAnkle: 'rightAnkle',
  PoseLandmarkType.leftHeel: 'leftHeel',
  PoseLandmarkType.rightHeel: 'rightHeel',
  PoseLandmarkType.leftFootIndex: 'leftFootIndex',
  PoseLandmarkType.rightFootIndex: 'rightFootIndex',
};

int slotFor(PoseLandmarkType type) {
  final String? name = landmarkNameOf[type];
  if (name == null) return -1;
  return idx[name] ?? -1;
}

PoseFrame frameFromPose(Pose pose, double width, double height,
    double timestamp, {bool mirror = true}) {
  final Points points = Points.nan(landmarkCount);
  final List<double> visibility = List<double>.filled(landmarkCount, 0.0);
  pose.landmarks.forEach((PoseLandmarkType type, PoseLandmark mark) {
    final int slot = slotFor(type);
    if (slot < 0) return;
    final double x = mirror ? width - mark.x : mark.x;
    points.set(slot, x, mark.y);
    visibility[slot] = mark.likelihood.clamp(0.0, 1.0);
  });
  return PoseFrame(
    points: points,
    visibility: visibility,
    width: width,
    height: height,
    timestamp: timestamp,
  );
}

InputImageRotation rotationOf(int degrees) {
  switch (degrees % 360) {
    case 90:
      return InputImageRotation.rotation90deg;
    case 180:
      return InputImageRotation.rotation180deg;
    case 270:
      return InputImageRotation.rotation270deg;
    default:
      return InputImageRotation.rotation0deg;
  }
}

InputImageFormat formatOf(CameraImage image) {
  if (Platform.isIOS) return InputImageFormat.bgra8888;
  if (image.format.group == ImageFormatGroup.nv21) {
    return InputImageFormat.nv21;
  }
  return InputImageFormat.yuv_420_888;
}

CameraFacing facingOf(CameraLensDirection lens) {
  switch (lens) {
    case CameraLensDirection.front:
      return CameraFacing.front;
    case CameraLensDirection.back:
      return CameraFacing.back;
    case CameraLensDirection.external:
      return CameraFacing.external;
  }
}

List<CameraChoice> choicesFrom(List<CameraDescription> found) {
  final List<CameraChoice> out = <CameraChoice>[];
  for (int i = 0; i < found.length; i++) {
    out.add(CameraChoice(
      id: found[i].name,
      facing: facingOf(found[i].lensDirection),
      index: i,
    ));
  }
  return out;
}

class MlKitPoseSource implements PoseSource {
  MlKitPoseSource({this.mirror = true, this.savedCameraId});

  final bool mirror;
  final String? savedCameraId;

  final PoseDetector _detector = PoseDetector(
    options: PoseDetectorOptions(mode: PoseDetectionMode.stream),
  );

  CameraController? _controller;
  CameraDescription? _description;
  PoseFrame? _latest;
  bool _running = false;
  bool _busy = false;
  String _status = 'starting the camera';
  DateTime? _origin;

  List<CameraDescription> _found = <CameraDescription>[];
  List<CameraChoice> _cameras = <CameraChoice>[];
  int _cameraIndex = -1;

  @override
  bool get supported => true;

  @override
  bool get running => _running;

  @override
  String get status => _status;

  CameraController? get controller => _controller;

  @override
  PoseFrame? get latest => _latest;

  @override
  List<CameraChoice> get cameras => _cameras;

  @override
  int get cameraIndex => _cameraIndex;

  bool get mirroring {
    if (_cameraIndex < 0 || _cameraIndex >= _cameras.length) return mirror;
    return _cameras[_cameraIndex].mirrored;
  }

  @override
  Future<bool> useCamera(int index) async {
    if (index < 0 || index >= _cameras.length) return false;
    if (index == _cameraIndex && _running) return true;
    final bool wasRunning = _running;
    await stop();
    _cameraIndex = index;
    if (!wasRunning) return true;
    return _open();
  }

  @override
  Future<bool> start() async {
    if (_running) return true;
    if (!await _list()) return false;
    if (_cameraIndex < 0) {
      _status = 'no camera found';
      return false;
    }
    return _open();
  }

  Future<bool> _list() async {
    try {
      _found = await availableCameras();
    } catch (_) {
      _found = <CameraDescription>[];
      _status = 'the camera list could not be read';
      return false;
    }
    _cameras = choicesFrom(_found);
    if (_cameras.isEmpty) {
      _cameraIndex = -1;
      _status = 'no camera found';
      return false;
    }
    if (_cameraIndex < 0 || _cameraIndex >= _cameras.length) {
      _cameraIndex = resolveIndex(_cameras, savedCameraId);
    }
    return true;
  }

  Future<bool> _open() async {
    if (_cameraIndex < 0 || _cameraIndex >= _found.length) return false;
    try {
      final CameraDescription desc = _found[_cameraIndex];
      _description = desc;
      final CameraController controller = CameraController(
        desc,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isIOS
            ? ImageFormatGroup.bgra8888
            : ImageFormatGroup.nv21,
      );
      await controller.initialize();
      await controller.startImageStream(_onImage);
      _controller = controller;
      _origin = DateTime.now();
      _running = true;
      _status = 'tracking';
      return true;
    } catch (_) {
      _status = 'the camera would not start';
      return false;
    }
  }

  void _onImage(CameraImage image) {
    if (_busy || !_running) return;
    _busy = true;
    _process(image).whenComplete(() => _busy = false);
  }

  Future<void> _process(CameraImage image) async {
    final CameraDescription? desc = _description;
    if (desc == null) return;
    try {
      final InputImage? input = _toInput(image, desc);
      if (input == null) return;
      final List<Pose> poses = await _detector.processImage(input);
      if (poses.isEmpty) {
        _latest = null;
        return;
      }
      final DateTime? origin = _origin;
      final double t = origin == null
          ? 0.0
          : DateTime.now().difference(origin).inMicroseconds / 1e6;
      _latest = frameFromPose(poses.first, image.width.toDouble(),
          image.height.toDouble(), t, mirror: mirroring);
    } catch (_) {
      _latest = null;
    }
  }

  InputImage? _toInput(CameraImage image, CameraDescription desc) {
    final Uint8List bytes = _bytesOf(image);
    if (bytes.isEmpty) return null;
    return InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotationOf(desc.sensorOrientation),
        format: formatOf(image),
        bytesPerRow: image.planes.isEmpty
            ? image.width
            : image.planes.first.bytesPerRow,
      ),
    );
  }

  Uint8List _bytesOf(CameraImage image) {
    if (image.planes.length == 1) return image.planes.first.bytes;
    final BytesBuilder builder = BytesBuilder(copy: false);
    for (final Plane plane in image.planes) {
      builder.add(plane.bytes);
    }
    return builder.toBytes();
  }

  @override
  Future<void> stop() async {
    _running = false;
    final CameraController? c = _controller;
    _controller = null;
    if (c != null) {
      try {
        if (c.value.isStreamingImages) await c.stopImageStream();
      } catch (_) {
        _status = 'the camera stream already stopped';
      }
      try {
        await c.dispose();
      } catch (_) {
        _status = 'the camera did not close cleanly';
      }
    }
  }

  @override
  void dispose() {
    stop();
    try {
      _detector.close();
    } catch (_) {
      _status = 'the detector did not close cleanly';
    }
  }
}
