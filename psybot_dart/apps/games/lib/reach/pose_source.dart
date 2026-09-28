import 'angles.dart';
import 'cameras.dart';
import 'points.dart';

class PoseFrame {
  const PoseFrame({
    required this.points,
    required this.visibility,
    required this.width,
    required this.height,
    required this.timestamp,
  });

  final Points points;
  final List<double> visibility;
  final double width;
  final double height;
  final double timestamp;
}

abstract class PoseSource {
  bool get supported;

  bool get running;

  String get status;

  Future<bool> start();

  Future<void> stop();

  PoseFrame? get latest;

  List<CameraChoice> get cameras;

  int get cameraIndex;

  Future<bool> useCamera(int index);

  void dispose();
}

class FakePoseSource implements PoseSource {
  FakePoseSource({
    this.frames = const <PoseFrame>[],
    this.loop = false,
    this.cameras = const <CameraChoice>[],
  }) {
    _cameraIndex = preferredIndex(cameras);
  }

  final List<PoseFrame> frames;
  final bool loop;

  @override
  final List<CameraChoice> cameras;
  int _cursor = 0;
  int _cameraIndex = -1;
  bool _running = false;
  PoseFrame? _latest;

  @override
  bool get supported => true;

  @override
  bool get running => _running;

  @override
  int get cameraIndex => _cameraIndex;

  @override
  Future<bool> useCamera(int index) async {
    if (index < 0 || index >= cameras.length) return false;
    _cameraIndex = index;
    return true;
  }

  @override
  String get status => _running ? 'fake pose source' : 'stopped';

  @override
  Future<bool> start() async {
    _running = true;
    return true;
  }

  @override
  Future<void> stop() async {
    _running = false;
  }

  @override
  PoseFrame? get latest => _latest;

  void push(PoseFrame frame) {
    _latest = frame;
  }

  PoseFrame? advance() {
    if (frames.isEmpty) return _latest;
    if (_cursor >= frames.length) {
      if (!loop) return _latest;
      _cursor = 0;
    }
    _latest = frames[_cursor++];
    return _latest;
  }

  @override
  void dispose() {
    _running = false;
  }
}

class UnsupportedPoseSource implements PoseSource {
  UnsupportedPoseSource(this.reason);

  final String reason;

  @override
  bool get supported => false;

  @override
  bool get running => false;

  @override
  String get status => reason;

  @override
  Future<bool> start() async => false;

  @override
  Future<void> stop() async {}

  @override
  PoseFrame? get latest => null;

  @override
  List<CameraChoice> get cameras => const <CameraChoice>[];

  @override
  int get cameraIndex => -1;

  @override
  Future<bool> useCamera(int index) async => false;

  @override
  void dispose() {}
}

PoseFrame poseFrameFromRows(List<List<double>> rows, List<double> visibility,
    double width, double height, double timestamp) {
  final Points p = Points.nan(landmarkCount);
  for (int i = 0; i < rows.length && i < landmarkCount; i++) {
    p.set(i, rows[i][0], rows[i][1]);
  }
  return PoseFrame(
    points: p,
    visibility: visibility,
    width: width,
    height: height,
    timestamp: timestamp,
  );
}
