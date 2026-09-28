import 'package:mood_engine/mood_engine.dart';
import 'package:test/test.dart';

void main() {
  group('angles never read upside down', () {
    test('wrapping lands in the same half turn', () {
      final cases = <List<double>>[
        <double>[0.0, 0.0],
        <double>[45.0, 45.0],
        <double>[181.0, -179.0],
        <double>[-171.0, -171.0],
        <double>[360.0, 0.0],
        <double>[-190.0, 170.0],
        <double>[179.9, 179.9],
        <double>[180.0, -180.0],
        <double>[540.0, -180.0],
      ];
      for (final c in cases) {
        expect(PoseEstimator.wrap180(c[0]), closeTo(c[1], 1e-9),
            reason: '${c[0]} should wrap to ${c[1]}');
      }
    });

    test('wrapping twice changes nothing', () {
      for (var a = -720.0; a <= 720.0; a += 7.5) {
        final once = PoseEstimator.wrap180(a);
        expect(PoseEstimator.wrap180(once), closeTo(once, 1e-9));
      }
    });

    test('every wrapped angle sits inside one turn', () {
      for (var a = -1000.0; a <= 1000.0; a += 13.0) {
        final w = PoseEstimator.wrap180(a);
        expect(w, greaterThanOrEqualTo(-180.0));
        expect(w, lessThanOrEqualTo(180.0));
      }
    });
  });

  group('what the pose reports stays within bounds', () {
    test('attention is never outside zero to one', () {
      for (var yaw = -90.0; yaw <= 90.0; yaw += 5.0) {
        for (var pitch = -90.0; pitch <= 90.0; pitch += 15.0) {
          final p = PoseResult(yaw: yaw, pitch: pitch);
          final a = PoseEstimator.attentionOf(p);
          expect(a, greaterThanOrEqualTo(0.0));
          expect(a, lessThanOrEqualTo(1.0));
        }
      }
    });

    test('looking straight ahead beats looking away', () {
      final ahead = PoseResult(yaw: 0.0, pitch: 0.0);
      final away = PoseResult(yaw: 60.0, pitch: 0.0);
      expect(PoseEstimator.attentionOf(ahead),
          greaterThan(PoseEstimator.attentionOf(away)));
    });
  });
}
