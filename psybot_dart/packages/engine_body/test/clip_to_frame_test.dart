import 'package:engine_body/engine_body.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const size = Size(1280, 720);

  group('clipToFrame', () {
    test('trims a bone at the edge without moving the inside end', () {
      final clipped = clipToFrame(
        const Offset(100, 100),
        const Offset(1680, 300),
        size,
      );

      expect(clipped, isNotNull);
      expect(clipped!.from.dx, closeTo(100, 1e-9));
      expect(clipped.from.dy, closeTo(100, 1e-9));
      expect(clipped.to.dx, lessThanOrEqualTo(size.width + 1e-9));
      expect(clipped.to.dy, lessThanOrEqualTo(size.height + 1e-9));
    });

    test('drops a bone entirely outside the frame', () {
      expect(
        clipToFrame(const Offset(-80, -80), const Offset(-10, -10), size),
        isNull,
      );
    });

    test('leaves a bone fully inside the frame alone', () {
      const a = Offset(200, 200);
      const b = Offset(400, 500);
      final clipped = clipToFrame(a, b, size);

      expect(clipped, isNotNull);
      expect(clipped!.from, a);
      expect(clipped.to, b);
    });

    test('trims both ends when the bone crosses the whole frame', () {
      final clipped = clipToFrame(
        const Offset(-200, 360),
        const Offset(1480, 360),
        size,
      );

      expect(clipped, isNotNull);
      expect(clipped!.from.dx, closeTo(0, 1e-9));
      expect(clipped.to.dx, closeTo(size.width, 1e-9));
      expect(clipped.from.dy, closeTo(360, 1e-9));
    });

    test('handles a vertical bone leaving the bottom', () {
      final clipped = clipToFrame(
        const Offset(640, 300),
        const Offset(640, 1200),
        size,
      );

      expect(clipped, isNotNull);
      expect(clipped!.from.dy, closeTo(300, 1e-9));
      expect(clipped.to.dy, closeTo(size.height, 1e-9));
      expect(clipped.to.dx, closeTo(640, 1e-9));
    });

    test('a zero-length bone inside the frame survives', () {
      const p = Offset(640, 360);
      final clipped = clipToFrame(p, p, size);
      expect(clipped, isNotNull);
      expect(clipped!.from, p);
    });

    test('a zero-length bone outside the frame is dropped', () {
      const p = Offset(-40, 360);
      expect(clipToFrame(p, p, size), isNull);
    });
  });
}
