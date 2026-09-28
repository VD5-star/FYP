import 'package:flutter_test/flutter_test.dart';
import 'package:psybot_games/core/zones.dart';
import 'package:psybot_games/reach/cameras.dart';
import 'package:psybot_games/reach/logic.dart';
import 'package:psybot_games/reach/pose_source.dart';

const double w = 1280;
const double h = 720;

CameraChoice cam(String id, CameraFacing facing, int index) =>
    CameraChoice(id: id, facing: facing, index: index);

List<CameraChoice> get pair => <CameraChoice>[
      cam('back0', CameraFacing.back, 0),
      cam('front0', CameraFacing.front, 1),
    ];

void main() {
  group('choosing which camera watches you', () {
    test('the front camera is picked first when there is one', () {
      expect(preferredIndex(pair), 1);
      expect(pair[preferredIndex(pair)].facing, CameraFacing.front);
    });

    test('with no front camera the first one is used', () {
      final List<CameraChoice> backs = <CameraChoice>[
        cam('back0', CameraFacing.back, 0),
        cam('ext0', CameraFacing.external, 1),
      ];
      expect(preferredIndex(backs), 0);
    });

    test('an empty list asks for nothing', () {
      expect(preferredIndex(const <CameraChoice>[]), -1);
      expect(nextIndex(const <CameraChoice>[], 0), -1);
    });

    test('a front camera is mirrored and a back one is not', () {
      expect(cam('a', CameraFacing.front, 0).mirrored, isTrue);
      expect(cam('b', CameraFacing.back, 1).mirrored, isFalse);
      expect(cam('c', CameraFacing.external, 2).mirrored, isFalse);
    });

    test('cycling wraps back to the start', () {
      expect(nextIndex(pair, 0), 1);
      expect(nextIndex(pair, 1), 0);
    });

    test('cycling from nothing lands on the preferred one', () {
      expect(nextIndex(pair, -1), 1);
    });

    test('a remembered camera is found again by name', () {
      expect(resolveIndex(pair, 'back0'), 0);
      expect(resolveIndex(pair, 'front0'), 1);
    });

    test('a camera that is gone falls back to the preferred one', () {
      expect(resolveIndex(pair, 'unplugged'), 1);
      expect(resolveIndex(pair, null), 1);
    });

    test('two cameras facing the same way are numbered', () {
      final List<CameraChoice> twoBacks = <CameraChoice>[
        cam('back0', CameraFacing.back, 0),
        cam('back1', CameraFacing.back, 1),
        cam('front0', CameraFacing.front, 2),
      ];
      expect(labelFor(twoBacks, twoBacks[0]), 'back 1');
      expect(labelFor(twoBacks, twoBacks[1]), 'back 2');
      expect(labelFor(twoBacks, twoBacks[2]), 'front');
    });
  });

  group('the camera button only shows when there is a choice', () {
    test('one camera means no button', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.setCameras(<CameraChoice>[cam('only', CameraFacing.front, 0)], 0);
      expect(logic.hasCameraChoice, isFalse);
      expect(logic.zones['camera'], isNull);
    });

    test('no camera at all means no button', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.setCameras(const <CameraChoice>[], -1);
      expect(logic.hasCameraChoice, isFalse);
      expect(logic.zones['camera'], isNull);
    });

    test('two cameras give a button you can actually press', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.setCameras(pair, 1);
      expect(logic.hasCameraChoice, isTrue);
      final Zone? z = logic.zones['camera'];
      expect(z, isNotNull);
      expect(z!.w, greaterThanOrEqualTo(touchMin));
      expect(z.h, greaterThanOrEqualTo(touchMin));
      expect(logic.zones.at(z.cx, z.cy), 'camera');
    });

    test('the button sits on screen at every size', () {
      for (final List<double> size in <List<double>>[
        <double>[640, 480],
        <double>[1280, 720],
        <double>[1920, 1080],
      ]) {
        final ReachLogic logic =
            ReachLogic(width: size[0], height: size[1]);
        logic.setCameras(pair, 1);
        final Zone? z = logic.zones['camera'];
        expect(z, isNotNull, reason: 'missing at ${size[0]}x${size[1]}');
        expect(z!.x, greaterThanOrEqualTo(0.0));
        expect(z.y, greaterThanOrEqualTo(0.0));
        expect(z.x1, lessThanOrEqualTo(size[0]));
        expect(z.y1, lessThanOrEqualTo(size[1]));
      }
    });

    test('the button never covers begin at any size', () {
      for (final List<double> size in <List<double>>[
        <double>[640, 480],
        <double>[1280, 720],
        <double>[1920, 1080],
      ]) {
        final ReachLogic logic =
            ReachLogic(width: size[0], height: size[1]);
        logic.setCameras(pair, 1);
        final Zone begin = logic.zones['begin']!;
        final Zone camera = logic.zones['camera']!;
        expect(camera.y, greaterThanOrEqualTo(begin.y1),
            reason: 'overlap at x');
      }
    });

    test('the label says which camera is live', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.setCameras(pair, 1);
      expect(logic.cameraLabel, 'front');
      logic.setCameras(pair, 0);
      expect(logic.cameraLabel, 'back');
    });
  });

  group('pressing the button switches camera', () {
    test('it moves to the next camera and tells the source', () async {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.setCameras(pair, 1);
      final List<int> asked = <int>[];
      logic.onPickCamera = (int i) async {
        asked.add(i);
        return true;
      };
      await logic.cycleCamera();
      expect(asked, <int>[0]);
      expect(logic.cameraIndex, 0);
      expect(logic.cameraLabel, 'back');
    });

    test('a camera that refuses to open is not selected', () async {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.setCameras(pair, 1);
      logic.onPickCamera = (int i) async => false;
      await logic.cycleCamera();
      expect(logic.cameraIndex, 1);
      expect(logic.cameraLabel, 'front');
    });

    test('switching clears the tracker so no pose carries over', () async {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.setCameras(pair, 1);
      logic.onPickCamera = (int i) async => true;
      await logic.cycleCamera();
      expect(logic.tracker.points, isNull);
      expect(logic.tracker.torso, isNull);
    });

    test('with one camera pressing does nothing', () async {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.setCameras(<CameraChoice>[cam('only', CameraFacing.front, 0)], 0);
      var called = false;
      logic.onPickCamera = (int i) async {
        called = true;
        return true;
      };
      await logic.cycleCamera();
      expect(called, isFalse);
      expect(logic.cameraIndex, 0);
    });

    test('the tap routes through the menu handler', () async {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.setCameras(pair, 1);
      logic.onPickCamera = (int i) async => true;
      logic.tapMenu('camera');
      await Future<void>.delayed(Duration.zero);
      expect(logic.cameraIndex, 0);
    });
  });

  group('a fake source answers the same questions', () {
    test('it reports the cameras it was given', () {
      final FakePoseSource source = FakePoseSource(cameras: pair);
      expect(source.cameras.length, 2);
      expect(source.cameraIndex, 1);
    });

    test('it accepts a real index and refuses a silly one', () async {
      final FakePoseSource source = FakePoseSource(cameras: pair);
      expect(await source.useCamera(0), isTrue);
      expect(source.cameraIndex, 0);
      expect(await source.useCamera(9), isFalse);
      expect(await source.useCamera(-1), isFalse);
      expect(source.cameraIndex, 0);
    });

    test('a source with no cameras has no selection', () {
      final FakePoseSource source = FakePoseSource();
      expect(source.cameras, isEmpty);
      expect(source.cameraIndex, -1);
    });

    test('an unsupported source refuses every camera', () async {
      final UnsupportedPoseSource source = UnsupportedPoseSource('no camera');
      expect(source.cameras, isEmpty);
      expect(source.cameraIndex, -1);
      expect(await source.useCamera(0), isFalse);
    });
  });
}
