import 'package:engine_body/engine_body.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const overlayKey = Key('overlay');
  const overlay = SizedBox.expand(key: overlayKey);

  Future<void> pump(WidgetTester tester, PreviewInfo preview) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Center(
          child: SizedBox(
            width: 400,
            height: 800,
            child: BodyCameraPreview(preview: preview, overlay: overlay),
          ),
        ),
      ),
    );
  }

  group('BodyCameraPreview geometry', () {
    testWidgets('a quarter-turned camera lays out upright, not sideways', (
      tester,
    ) async {
      await pump(
        tester,
        const PreviewInfo(
          textureId: 1,
          width: 640,
          height: 480,
          rotation: 90,
          isMirrored: false,
        ),
      );

      final size = tester.getSize(find.byKey(overlayKey));
      expect(
        size.width / size.height,
        closeTo(480 / 640, 0.01),
        reason: 'a 90-degree sensor must display as portrait',
      );
    });

    testWidgets('an unrotated camera keeps its native aspect ratio', (
      tester,
    ) async {
      await pump(
        tester,
        const PreviewInfo(
          textureId: 1,
          width: 640,
          height: 480,
          rotation: 0,
          isMirrored: false,
        ),
      );

      final size = tester.getSize(find.byKey(overlayKey));
      expect(size.width / size.height, closeTo(640 / 480, 0.01));
    });

    testWidgets('the overlay is not rotated with the texture', (tester) async {
      await pump(
        tester,
        const PreviewInfo(
          textureId: 1,
          width: 640,
          height: 480,
          rotation: 90,
          isMirrored: false,
        ),
      );

      final rotated = find.byType(RotatedBox);
      expect(rotated, findsOneWidget);
      expect(
        find.descendant(of: rotated, matching: find.byKey(overlayKey)),
        findsNothing,
        reason: 'the overlay is already upright and must not be rotated',
      );
      expect(
        find.descendant(of: rotated, matching: find.byType(Texture)),
        findsOneWidget,
      );
    });

    testWidgets('mirroring is applied to the image and overlay together', (
      tester,
    ) async {
      await pump(
        tester,
        const PreviewInfo(
          textureId: 1,
          width: 640,
          height: 480,
          rotation: 90,
          isMirrored: true,
        ),
      );

      final transform = find.ancestor(
        of: find.byKey(overlayKey),
        matching: find.byType(Transform),
      );
      expect(
        transform,
        findsWidgets,
        reason: 'the overlay must be inside the mirroring transform',
      );
    });

    testWidgets('defaults to showing the whole frame rather than cropping', (
      tester,
    ) async {
      await pump(
        tester,
        const PreviewInfo(
          textureId: 1,
          width: 640,
          height: 480,
          rotation: 90,
          isMirrored: false,
        ),
      );

      final fitted = tester.widget<FittedBox>(find.byType(FittedBox));
      expect(fitted.fit, BoxFit.contain);

      final size = tester.getSize(find.byKey(overlayKey));
      expect(size.width / size.height, closeTo(480 / 640, 0.01));
    });

    testWidgets('a zero-sized preview renders black instead of throwing', (
      tester,
    ) async {
      await pump(
        tester,
        const PreviewInfo(
          textureId: 0,
          width: 0,
          height: 0,
          rotation: 0,
          isMirrored: false,
        ),
      );
      expect(tester.takeException(), isNull);
      expect(find.byType(Texture), findsNothing);
    });
  });
}
