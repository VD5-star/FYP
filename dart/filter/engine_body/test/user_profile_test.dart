import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('UserProfile', () {
    test('round-trips a calibration', () {
      final profile = UserProfile();
      final config = HysteresisConfig(
        name: 'squat',
        downBelow: 118,
        upAbove: 162,
      );
      expect(
        profile.store('squat', config, observedMin: 107, observedMax: 175),
        isTrue,
      );

      final restored = UserProfile.fromJson(profile.toJson());
      final back = restored.configFor('squat', squatConfig);
      expect(back.downBelow, closeTo(118, 1e-9));
      expect(back.upAbove, closeTo(162, 1e-9));
    });

    test('falls back for an uncalibrated exercise', () {
      expect(
        UserProfile().configFor('pushup', pushupConfig),
        same(pushupConfig),
      );
    });

    test('a null config is rejected rather than stored', () {
      expect(UserProfile().store('squat', null), isFalse);
    });

    test('corrupt JSON yields an uncalibrated profile, not a crash', () {
      for (final bad in ['{ not json', '[]', 'null', '', '{"exercises": 5}']) {
        final profile = UserProfile.fromJson(bad);
        expect(profile.configFor('squat', squatConfig), same(squatConfig));
      }
    });

    test('a partially written entry is discarded, not guessed at', () {
      final profile =
          UserProfile.fromJson('{"exercises": {"squat": {"downBelow": 120}}}');
      expect(profile.configFor('squat', squatConfig), same(squatConfig));
    });

    test('non-finite stored values are refused', () {
      final profile = UserProfile.fromJson(
          '{"exercises": {"squat": {"downBelow": 120, "upAbove": 1e999}}}');
      expect(profile.configFor('squat', squatConfig), same(squatConfig));
    });

    test('a stored profile that no longer validates is rejected', () {
      final profile = UserProfile(exercises: {
        'squat': const CalibratedThresholds(downBelow: 100, upAbove: 105),
      });
      expect(profile.configFor('squat', squatConfig), same(squatConfig));
    });

    test('stores angles only', () {
      final profile = UserProfile();
      profile.store(
        'squat',
        HysteresisConfig(name: 'squat', downBelow: 118, upAbove: 162),
        observedMin: 107,
        observedMax: 175,
      );

      const permitted = {
        'exercises',
        'squat',
        'downBelow',
        'upAbove',
        'observedMin',
        'observedMax',
      };
      final keys = RegExp(r'"([^"]+)":')
          .allMatches(profile.toJson())
          .map((m) => m.group(1)!)
          .toSet();
      expect(keys.difference(permitted), isEmpty);
    });

    test('a calibrated profile makes a limited user countable', () {
      final profile = UserProfile();
      profile.store(
        'squat',
        HysteresisConfig(name: 'squat', downBelow: 125, upAbove: 160),
      );

      final calibrated = profile.configFor('squat', squatConfig);
      final counter = RepCounter(calibrated);

      Duration at(int f) => Duration(microseconds: (f * 1000000 / 30).round());
      var frame = 0;
      for (var rep = 0; rep < 3; rep++) {
        for (var i = 0; i < 15; i++) {
          counter.update(175, at(frame++));
        }
        for (var i = 0; i < 15; i++) {
          counter.update(115, at(frame++));
        }
      }
      for (var i = 0; i < 15; i++) {
        counter.update(175, at(frame++));
      }

      expect(counter.count, greaterThanOrEqualTo(2));
    });
  });
}
