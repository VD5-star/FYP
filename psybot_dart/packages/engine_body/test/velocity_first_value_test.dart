import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  Duration at(int frame) =>
      Duration(microseconds: (frame * 1000000 / 30).round());

  group('VelocityRepCounter travel reference', () {
    test('a flat start still counts the repetition that follows', () {
      final angles = <double>[
        ...List<double>.filled(5, 180),
        for (var i = 1; i <= 25; i++) 180 - 4.0 * i,
        for (var i = 1; i <= 25; i++) 80 + 4.0 * i,
      ];

      final counter = VelocityRepCounter();
      for (var i = 0; i < angles.length; i++) {
        counter.update(angles[i], at(i));
      }

      expect(counter.count, 1);
    });

    test('a flat run in the middle does not lose the reference', () {
      final angles = <double>[
        for (var i = 0; i < 25; i++) 180 - 4.0 * i,
        ...List<double>.filled(10, 80),
        for (var i = 1; i <= 25; i++) 80 + 4.0 * i,
      ];

      final counter = VelocityRepCounter();
      for (var i = 0; i < angles.length; i++) {
        counter.update(angles[i], at(i));
      }

      expect(counter.count, 1);
    });

    test('a flat signal alone counts nothing', () {
      final counter = VelocityRepCounter();
      for (var i = 0; i < 120; i++) {
        counter.update(180, at(i));
      }
      expect(counter.count, 0);
    });
  });
}
