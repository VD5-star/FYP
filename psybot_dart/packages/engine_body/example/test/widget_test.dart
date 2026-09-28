import 'package:engine_body/engine_body.dart';
import 'package:engine_body_example/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows an explanation instead of failing on desktop', (
    tester,
  ) async {
    await tester.pumpWidget(const BodyEngineDemo());
    await tester.pump();

    if (BodyTracker.isSupported) return;
    expect(find.textContaining('requires Android'), findsOneWidget);
  });

  testWidgets('the skeleton painter draws a frame without throwing', (
    tester,
  ) async {
    final frame = BodyFrame(
      timestamp: DateTime(2026),
      landmarks: [
        for (final type in LandmarkType.values)
          BodyLandmark(
            type: type,
            x: 0.5,
            y: 0.5,
            z: 0,
            likelihood: 0.9,
            inFrameLikelihood: 0.9,
          ),
      ],
      framing: const FramingGate().assess([]),
      isMirrored: false,
    );

    await tester.pumpWidget(
      MaterialApp(
        home: CustomPaint(
          size: const Size(320, 480),
          painter: SkeletonPainter(frame: frame),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
  });

  testWidgets('an empty frame paints nothing rather than crashing', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: CustomPaint(
          size: const Size(320, 480),
          painter: SkeletonPainter(frame: BodyFrame.empty),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
  });
}
