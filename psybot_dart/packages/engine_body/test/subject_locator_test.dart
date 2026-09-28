import 'package:engine_body/engine_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const frameWidth = 1280;
  const frameHeight = 720;

  Duration seconds(double s) => Duration(microseconds: (s * 1000000).round());

  BodyFrame subject({
    double centreX = 0.5,
    double centreY = 0.5,
    double height = 400.0 / frameHeight,
    double confidence = 1.0,
  }) {
    final halfH = height / 2;
    final halfW = height / 4;
    const types = <LandmarkType>[
      LandmarkType.leftShoulder,
      LandmarkType.rightShoulder,
      LandmarkType.leftHip,
      LandmarkType.rightHip,
      LandmarkType.nose,
    ];
    final corners = <(double, double)>[
      (centreX - halfW, centreY - halfH),
      (centreX + halfW, centreY - halfH),
      (centreX - halfW, centreY + halfH),
      (centreX + halfW, centreY + halfH),
      (centreX, centreY),
    ];

    return BodyFrame(
      timestamp: DateTime.fromMillisecondsSinceEpoch(0),
      landmarks: [
        for (var i = 0; i < types.length; i++)
          BodyLandmark(
            type: types[i],
            x: corners[i].$1,
            y: corners[i].$2,
            z: 0,
            likelihood: confidence,
            inFrameLikelihood: confidence,
          ),
      ],
      framing: FramingReport.noSubject,
      isMirrored: false,
    );
  }

  group('SubjectLocator', () {
    test('starts with no subject', () {
      final locator = SubjectLocator();
      expect(locator.hasSubject, isFalse);
      expect(
        locator.region(0, frameWidth, frameHeight, Duration.zero),
        isNull,
      );
      expect(locator.shouldSearch(Duration.zero), isFalse);
    });

    test('remembers where the subject was', () {
      final locator = SubjectLocator()
        ..seen(
          subject(centreX: 300 / frameWidth, centreY: 200 / frameHeight,
              height: 240 / frameHeight),
          Duration.zero,
        );
      expect(locator.hasSubject, isTrue);

      final region =
          locator.region(0, frameWidth, frameHeight, seconds(0.1));
      expect(region, isNotNull);
      expect(region!.x0, lessThan(300));
      expect(region.x1, greaterThan(300));
      expect(region.y0, lessThan(200));
      expect(region.y1, greaterThan(200));
    });

    test('ignores landmarks it cannot trust', () {
      final locator = SubjectLocator()
        ..seen(subject(confidence: 0.1), Duration.zero);
      expect(locator.hasSubject, isFalse);
    });

    test('the search actually magnifies', () {
      final locator = SubjectLocator()..seen(subject(), Duration.zero);
      for (var attempt = 0; attempt < locator.attempts; attempt++) {
        final region =
            locator.region(attempt, frameWidth, frameHeight, seconds(0.1));
        if (region == null) continue;
        expect(
          region.scale,
          greaterThan(1.0),
          reason: 'attempt $attempt would re-send the same pixels',
        );
      }
    });

    test('later attempts look for a smaller subject', () {
      final locator = SubjectLocator()..seen(subject(), Duration.zero);
      final scales = <double>[
        for (var a = 0; a < locator.attempts; a++)
          if (locator.region(a, frameWidth, frameHeight, seconds(0.1))
              case final r?)
            r.scale,
      ];
      expect(scales, isNotEmpty);
      expect(
        scales.reduce((a, b) => a > b ? a : b),
        greaterThan(scales.reduce((a, b) => a < b ? a : b) * 1.5),
      );
    });

    test('the search ends', () {
      final locator = SubjectLocator()..seen(subject(), Duration.zero);
      expect(
        locator.region(
            locator.attempts, frameWidth, frameHeight, seconds(0.1)),
        isNull,
      );
    });

    test('prediction leads a moving subject', () {
      final moving = SubjectLocator();
      for (var i = 0; i < 4; i++) {
        moving.seen(
          subject(centreX: (300 + i * 60) / frameWidth),
          seconds(i * 0.1),
        );
      }

      final still = SubjectLocator()
        ..seen(subject(centreX: 480 / frameWidth), seconds(0.3));

      final movingRegion =
          moving.region(0, frameWidth, frameHeight, seconds(0.6));
      final stillRegion =
          still.region(0, frameWidth, frameHeight, seconds(0.6));

      expect(movingRegion, isNotNull);
      expect(stillRegion, isNotNull);
      expect(
        movingRegion!.centreX,
        greaterThan(stillRegion!.centreX + 20),
      );
    });

    test('prediction is bounded', () {
      final locator = SubjectLocator();
      for (var i = 0; i < 4; i++) {
        locator.seen(
          subject(centreX: (300 + i * 60) / frameWidth),
          seconds(i * 0.1),
        );
      }

      final near = locator.region(0, frameWidth, frameHeight, seconds(1.0));
      final far = locator.region(0, frameWidth, frameHeight, seconds(30));
      expect(near, isNotNull);
      expect(far, isNotNull);
      expect((far!.centreX - near!.centreX).abs(), lessThan(1.0));
    });

    test('searching is eager at first', () {
      final locator = SubjectLocator()..seen(subject(), Duration.zero);
      for (final t in [0.03, 0.2, 0.5, 0.9]) {
        expect(
          locator.shouldSearch(seconds(t)),
          isTrue,
          reason: 'would not search at $t s',
        );
        locator.searched(seconds(t));
      }
    });

    test('searching backs off but never stops', () {
      final locator = SubjectLocator()..seen(subject(), Duration.zero);

      var searches = 0;
      for (var frame = 0; frame < 60 * 30; frame++) {
        final t = seconds(frame / 30);
        if (locator.shouldSearch(t)) {
          locator.searched(t);
          searches++;
        }
      }

      expect(searches, greaterThan(0));
      expect(
        searches,
        lessThan(200),
        reason: 'the back-off is not working',
      );
      expect(locator.shouldSearch(seconds(3600)), isTrue);
    });

    test('the back-off does not start during the eager window', () {
      final locator = SubjectLocator()..seen(subject(), Duration.zero);
      for (var frame = 0; frame <= 30; frame++) {
        final t = seconds(frame / 30);
        if (locator.shouldSearch(t)) locator.searched(t);
      }
      expect(locator.interval, lessThanOrEqualTo(searchInterval));
    });

    test('finding the subject resets the back-off', () {
      final locator = SubjectLocator()..seen(subject(), Duration.zero);
      for (final t in [2.0, 4.0, 8.0, 16.0]) {
        if (locator.shouldSearch(seconds(t))) locator.searched(seconds(t));
      }
      expect(locator.interval, greaterThan(searchInterval));

      locator.seen(subject(), seconds(20));
      expect(locator.interval, searchInterval);
    });

    test('regions stay inside the frame', () {
      final locator = SubjectLocator();
      const places = <(double, double)>[
        (20, 20),
        (1260, 700),
        (640, 10),
      ];

      for (final (x, y) in places) {
        locator.reset();
        locator.seen(
          subject(
            centreX: x / frameWidth,
            centreY: y / frameHeight,
            height: 300 / frameHeight,
          ),
          Duration.zero,
        );

        for (var a = 0; a < locator.attempts; a++) {
          final region =
              locator.region(a, frameWidth, frameHeight, seconds(0.1));
          if (region == null) continue;
          expect(region.x0, greaterThanOrEqualTo(0));
          expect(region.y0, greaterThanOrEqualTo(0));
          expect(region.x1, lessThanOrEqualTo(frameWidth));
          expect(region.y1, lessThanOrEqualTo(frameHeight));
          expect(region.x0, lessThan(region.x1));
          expect(region.y0, lessThan(region.y1));
        }
      }
    });

    test('maps crop coordinates back to the frame', () {
      const region =
          SearchRegion(x0: 200, y0: 100, x1: 456, y1: 356, scale: 2.0);

      final centre = region.toFrame(256, 256);
      expect(centre.x, closeTo(328, 1e-9));
      expect(centre.y, closeTo(228, 1e-9));

      final origin = region.toFrame(0, 0);
      expect(origin.x, closeTo(200, 1e-9));
      expect(origin.y, closeTo(100, 1e-9));
    });

    test('a degenerate region is refused', () {
      const tiny = SearchRegion(x0: 10, y0: 10, x1: 12, y1: 12, scale: 4);
      expect(tiny.isValid, isFalse);

      const usable =
          SearchRegion(x0: 100, y0: 100, x1: 200, y1: 200, scale: 3);
      expect(usable.isValid, isTrue);
      expect(usable.width, 100);
      expect(usable.height, 100);
    });

    test('reset forgets everything', () {
      final locator = SubjectLocator()..seen(subject(), Duration.zero);
      locator.reset();
      expect(locator.hasSubject, isFalse);
      expect(locator.shouldSearch(seconds(1)), isFalse);
    });
  });
}
