import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/pose_fixtures.dart';

void main() {
  final t0 = DateTime(2026, 1, 1);

  group('SessionRecorder', () {
    test('round-trips landmarks without loss', () {
      final recorder = SessionRecorder(sessionId: 'test', startedAt: t0);
      final original = PoseFixtures.stillSequence(frames: 5);
      for (final frame in original) {
        recorder.add(frame);
      }

      final replay = SessionReplay.decode(recorder.encode());
      expect(replay.frames, hasLength(5));

      for (var i = 0; i < original.length; i++) {
        final before = original[i];
        final after = replay.frames[i];
        expect(after.timestamp, before.timestamp);
        expect(after.landmarks, hasLength(before.landmarks.length));

        for (var j = 0; j < before.landmarks.length; j++) {
          expect(after.landmarks[j].type, before.landmarks[j].type);
          expect(after.landmarks[j].x, closeTo(before.landmarks[j].x, 1e-12));
          expect(after.landmarks[j].y, closeTo(before.landmarks[j].y, 1e-12));
          expect(
            after.landmarks[j].inFrameLikelihood,
            closeTo(before.landmarks[j].inFrameLikelihood, 1e-12),
          );
        }
      }
    });

    test('replayed analysis matches live analysis exactly', () {
      final frames = [
        for (var i = 0; i < 20; i++)
          PoseFixtures.frame(
            i.isEven ? PoseFixtures.standing() : PoseFixtures.leftArmRaised(),
            at: t0.add(Duration(milliseconds: 66 * i)),
          ),
      ];

      final live = MovementAnalyser();
      final recorder = SessionRecorder(sessionId: 'test', startedAt: t0);
      for (final frame in frames) {
        live.add(frame);
        recorder.add(frame);
      }

      final replayed = MovementAnalyser();
      for (final frame in SessionReplay.decode(recorder.encode()).frames) {
        replayed.add(frame);
      }

      expect(
        replayed.signals.movementEnergy,
        closeTo(live.signals.movementEnergy, 1e-12),
      );
      expect(replayed.signals.sampleCount, live.signals.sampleCount);
      expect(replayed.signals.framing, live.signals.framing);
    });

    test('framing is recomputed on replay, not stored', () {
      final recorder = SessionRecorder(sessionId: 'test', startedAt: t0);
      recorder.add(
        PoseFixtures.frame(PoseFixtures.standing(legsVisible: false), at: t0),
      );

      final encoded = recorder.encode();
      expect(encoded, isNot(contains('upperBodyOnly')));

      final strict = SessionReplay.decode(
        encoded,
        gate: const FramingGate(groupThreshold: 0.95),
      );
      expect(strict.frames.single.framing.framing, Framing.partial);

      final lenient = SessionReplay.decode(encoded);
      expect(lenient.frames.single.framing.framing, Framing.upperBodyOnly);
    });

    test('a truncated recording still yields the frames before the cut', () {
      final recorder = SessionRecorder(sessionId: 'test', startedAt: t0);
      for (final frame in PoseFixtures.stillSequence(frames: 4)) {
        recorder.add(frame);
      }

      final truncated = '${recorder.encode()}\n{"landmarks": [{"typ';
      final replay = SessionReplay.decode(truncated);
      expect(replay.frames, hasLength(4));
    });

    test('rejects an unknown schema version rather than misreading it', () {
      expect(
        () => SessionReplay.decode('{"schema": 99, "sessionId": "x"}'),
        throwsA(isA<FormatException>()),
      );
    });

    test('rejects an empty recording', () {
      expect(() => SessionReplay.decode(''), throwsA(isA<FormatException>()));
    });

    test('measures the frame rate the device actually delivered', () {
      final recorder = SessionRecorder(sessionId: 'test', startedAt: t0);
      for (final frame in PoseFixtures.stillSequence(frames: 31, fps: 15)) {
        recorder.add(frame);
      }
      final replay = SessionReplay.decode(recorder.encode());
      expect(replay.measuredFrameRate, closeTo(15, 0.1));
    });

    test('frame rate is null when there is too little data to measure', () {
      final recorder = SessionRecorder(sessionId: 'test', startedAt: t0);
      recorder.add(PoseFixtures.frame(PoseFixtures.standing(), at: t0));
      expect(SessionReplay.decode(recorder.encode()).measuredFrameRate, isNull);
    });

    test('stores coordinates only, never image data', () {
      final recorder = SessionRecorder(sessionId: 'test', startedAt: t0);
      recorder.add(PoseFixtures.frame(PoseFixtures.standing(), at: t0));
      final encoded = recorder.encode();

      for (final forbidden in ['image', 'jpeg', 'png', 'bytes', 'base64']) {
        expect(
          encoded.toLowerCase(),
          isNot(contains(forbidden)),
          reason: 'session recordings must never carry image data',
        );
      }
    });
  });

  group('SessionReplay.stream', () {
    test('replays frames in order', () async {
      final recorder = SessionRecorder(sessionId: 'test', startedAt: t0);
      for (final frame in PoseFixtures.stillSequence(frames: 6)) {
        recorder.add(frame);
      }

      final replay = SessionReplay.decode(recorder.encode());
      final seen = await replay.stream().toList();
      expect(seen, hasLength(6));
      for (var i = 1; i < seen.length; i++) {
        expect(seen[i].timestamp.isAfter(seen[i - 1].timestamp), isTrue);
      }
    });
  });
}
