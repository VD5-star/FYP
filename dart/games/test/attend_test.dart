import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:psybot_games/attend/logic.dart';
import 'package:psybot_games/attend/psycho.dart';
import 'package:psybot_games/attend/scenes.dart';
import 'package:psybot_games/attend/session.dart';
import 'package:psybot_games/attend/sounds.dart';
import 'package:psybot_games/core/synth.dart';
import 'package:psybot_games/core/zones.dart';
import 'package:psybot_games/jigsaw/raster.dart';

const double probeSeconds = loopSeconds;

Map<String, Float64List>? _cache;

Map<String, Float64List> sounds() {
  _cache ??= <String, Float64List>{
    for (final String n in soundNames)
      n: renderSound(n, seconds: probeSeconds)
  };
  return _cache!;
}

double correlation(Float64List a, Float64List b) {
  final int n = math.min(a.length, b.length);
  double ma = 0;
  double mb = 0;
  for (int i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  double num = 0;
  double da = 0;
  double db = 0;
  for (int i = 0; i < n; i++) {
    final double x = a[i] - ma;
    final double y = b[i] - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  if (da <= 0 || db <= 0) return 0.0;
  return num / math.sqrt(da * db);
}

void main() {
  group('the session lasts exactly as long as you asked', () {
    test('every duration lands on the minute', () {
      for (int m = minMinutes; m <= maxMinutes; m++) {
        final Session s = Session(minutes: m, seed: 5);
        expect((s.total - m * 60).abs(), lessThan(0.01),
            reason: '$m minutes gave ${s.total}');
      }
    });

    test('running it frame by frame ends on time', () {
      for (final int m in <int>[3, 5, 10]) {
        final Session s = Session(minutes: m, seed: 11);
        double t = 0;
        int guard = 0;
        while (!s.finished && guard < 200000) {
          s.advance(1 / 60);
          t += 1 / 60;
          guard++;
        }
        expect(s.finished, isTrue);
        expect((t - m * 60).abs(), lessThan(0.5),
            reason: '$m minutes ran $t');
      }
    });

    test('the plan visits every sound and switches enough', () {
      for (int m = minMinutes; m <= maxMinutes; m++) {
        final Session s = Session(minutes: m, seed: m * 7);
        final Set<String> seen = <String>{};
        int switches = 0;
        for (final Step st in s.steps) {
          if (st.target != null) seen.add(st.target!);
          if (st.stage == stageSwitch) switches++;
        }
        expect(seen.length, soundNames.length);
        expect(switches, greaterThanOrEqualTo(minSwitches));
      }
    });

    test('it always ends with divide then settle', () {
      final Session s = Session(minutes: 6, seed: 3);
      expect(s.steps[s.steps.length - 2].stage, stageDivide);
      expect(s.steps.last.stage, stageSettle);
    });

    test('the same sound never follows itself', () {
      for (int seed = 0; seed < 12; seed++) {
        final Session s = Session(minutes: 7, seed: seed);
        for (int i = 1; i < s.steps.length; i++) {
          final String? a = s.steps[i - 1].target;
          final String? b = s.steps[i].target;
          if (a != null && b != null && s.steps[i].stage == stageSwitch) {
            expect(a == b, isFalse);
          }
        }
      }
    });

    test('minutes clamp and name correctly at the edges', () {
      expect(clampMinutes(1), minMinutes);
      expect(clampMinutes(99), maxMinutes);
      expect(shapeFor(3).name, 'light');
      expect(shapeFor(4).name, 'light');
      expect(shapeFor(5).name, 'steady');
      expect(shapeFor(6).name, 'steady');
      expect(shapeFor(7).name, 'full');
      expect(shapeFor(8).name, 'full');
      expect(shapeFor(9).name, 'deep');
      expect(shapeFor(10).name, 'deep');
    });

    test('the clock reads properly', () {
      expect(clockText(0), '0:00');
      expect(clockText(65), '1:05');
      expect(clockText(600), '10:00');
    });

    test('pausing freezes it and resuming carries on', () {
      final Session s = Session(minutes: 4, seed: 2);
      s.advance(10);
      final double at = s.position;
      s.togglePause();
      for (int i = 0; i < 100; i++) {
        s.advance(1 / 60);
      }
      expect(s.position, at);
      s.togglePause();
      s.advance(1.0);
      expect(s.position, greaterThan(at));
    });

    test('a long frame cannot skip a whole step', () {
      final Session s = Session(minutes: 3, seed: 9);
      s.advance(1000.0);
      expect(s.position, lessThanOrEqualTo(maxStep + 1e-9));
    });
  });

  group('the five sounds stay kind to the ear', () {
    test('all of them sit inside the limits', () {
      sounds().forEach((String name, Float64List x) {
        final PsychoReport r = reportOn(name, x);
        expect(r.verdict, isEmpty,
            reason: '$name broke ${r.verdict}');
        expect(r.sharp, lessThanOrEqualTo(psychoLimits['sharp']!));
        expect(r.treble, lessThanOrEqualTo(psychoLimits['treble']!));
        expect(r.harsh, lessThanOrEqualTo(psychoLimits['harsh']!));
      });
    });

    test('all of them are tonal, not noise', () {
      sounds().forEach((String name, Float64List x) {
        expect(tonality(x), greaterThan(0.62), reason: name);
      });
    });

    test('none of them clips', () {
      sounds().forEach((String name, Float64List x) {
        double peak = 0;
        for (final double v in x) {
          if (v.abs() > peak) peak = v.abs();
        }
        expect(peak, lessThanOrEqualTo(1.0), reason: name);
        expect(peak, greaterThan(0.05), reason: '$name is silent');
      });
    });

    test('no two sounds sit on top of each other', () {
      final List<double> centres = <double>[
        for (final String n in soundNames) medianFreq(sounds()[n]!)
      ]..sort();
      for (int i = 1; i < centres.length; i++) {
        final double ratio = centres[i] / math.max(1.0, centres[i - 1]);
        expect(ratio, greaterThan(1.25),
            reason: '${centres[i - 1]} vs ${centres[i]}');
      }
    });

    test('every sound is at the same level as the others', () {
      final List<double> louds = <double>[
        for (final String n in soundNames) loudness(sounds()[n]!, sampleRate)
      ];
      final double lo = louds.reduce(math.min);
      final double hi = louds.reduce(math.max);
      expect(hi / lo, lessThan(1.35), reason: '$louds');
    });
  });

  group('the sounds move so the ear can tell them apart', () {
    test('each one breathes', () {
      sounds().forEach((String name, Float64List x) {
        final Float64List e = rmsEnvelope(x);
        final double lo = math.max(1e-9, percentile(e, 10));
        final double hi = percentile(e, 90);
        expect(hi / lo, greaterThan(1.15),
            reason: '$name swings ${hi / lo}');
      });
    });

    test('no two breathe together', () {
      final Map<String, Float64List> env = <String, Float64List>{
        for (final String n in soundNames) n: rmsEnvelope(sounds()[n]!)
      };
      double worst = 0;
      for (int i = 0; i < soundNames.length; i++) {
        for (int j = i + 1; j < soundNames.length; j++) {
          final double c = correlation(
            env[soundNames[i]]!,
            env[soundNames[j]]!,
          ).abs();
          if (c > worst) worst = c;
        }
      }
      expect(worst, lessThanOrEqualTo(0.75), reason: 'worst $worst');
    });

    test('the bowl strikes without overlapping itself', () {
      for (final double secs in <double>[12.0, 24.0, loopSeconds]) {
        final int strikes = math.max(4, (secs / 2.4).round());
        final int n = frameCount(secs, sampleRate);
        final int gap = n ~/ strikes;
        final int dn = math.min(n, n ~/ strikes);
        expect(dn, lessThanOrEqualTo(gap),
            reason: '$secs s: strike $dn vs gap $gap');
      }
    });
  });

  group('the running screen has one round button', () {
    test('and nothing else', () {
      final AttendLogic a = AttendLogic()..ready = true;
      a.begin();
      expect(a.zones.keys.toList(), <String>['open']);
      final Zone o = a.zones['open']!;
      expect(o.w, o.h);
      expect(o.cx, greaterThan(a.w * 0.8));
      expect((o.cy - a.h / 2).abs(), lessThanOrEqualTo(8));
      expect(o.touchable, isTrue);
    });

    test('it is round at every window size', () {
      for (final List<double> wh in <List<double>>[
        <double>[760, 560],
        <double>[1100, 700],
        <double>[1920, 1080],
      ]) {
        final AttendLogic a =
            AttendLogic(w: wh[0], h: wh[1])..ready = true;
        a.begin();
        final Zone o = a.zones['open']!;
        expect(o.w, o.h);
        expect((o.cy - wh[1] / 2).abs(), lessThanOrEqualTo(8));
        expect(o.x1, lessThanOrEqualTo(wh[0]));
      }
    });
  });

  group('the stop menu holds everything', () {
    AttendLogic running() {
      final AttendLogic a = AttendLogic()..ready = true;
      a.begin();
      a.tap('open');
      return a;
    }

    test('opening it pauses for real', () {
      final AttendLogic a = running();
      expect(a.paused, isTrue);
      final double at = a.sess!.position;
      for (int i = 0; i < 120; i++) {
        a.step(1 / 60, i / 60);
      }
      expect(a.sess!.position, at);
      expect(a.soundShouldPlay, isFalse);
    });

    test('it has the sound slider and four buttons', () {
      final AttendLogic a = running();
      for (final String k in <String>[
        'volume',
        'resume',
        'again',
        'home',
        'quit'
      ]) {
        expect(a.zones.has(k), isTrue, reason: 'missing $k');
        expect(a.zones[k]!.touchable, isTrue, reason: '$k too small');
      }
      expect(a.zones.has('settings'), isFalse);
    });

    test('the panel is a middling size and centred', () {
      final AttendLogic a = running();
      final Zone p = a.zones['panel']!;
      expect(p.w, inInclusiveRange(260, 520));
      expect(p.h, inInclusiveRange(200, 340));
      expect((p.cx - a.w / 2).abs(), lessThanOrEqualTo(2));
      expect((p.cy - a.h / 2).abs(), lessThanOrEqualTo(2));
    });

    test('sound sits on top, across the middle', () {
      final AttendLogic a = running();
      final Zone v = a.zones['volume']!;
      final Zone p = a.zones['panel']!;
      expect(v.y, lessThan(a.zones['resume']!.y));
      expect((v.cx - p.cx).abs(), lessThanOrEqualTo(2));
    });

    test('the four buttons form two rows of two', () {
      final AttendLogic a = running();
      final Map<double, List<String>> rows = <double, List<String>>{};
      for (final String k in <String>['resume', 'again', 'home', 'quit']) {
        rows.putIfAbsent(a.zones[k]!.y, () => <String>[]).add(k);
      }
      expect(rows.length, 2);
      for (final List<String> r in rows.values) {
        expect(r.length, 2);
      }
    });

    test('dragging sound left mutes and right maxes', () {
      final AttendLogic a = running();
      final Zone v = a.zones['volume']!;
      a.pointerDown(v.x + sliderPad, v.cy);
      a.pointerUp(v.x + sliderPad, v.cy);
      expect(a.volume, volMin);
      a.pointerDown(v.x1 - sliderPad, v.cy);
      a.pointerUp(v.x1 - sliderPad, v.cy);
      expect(a.volume, volMax);
    });

    test('changing sound never leaves the menu or starts the sound', () {
      final AttendLogic a = running();
      final Zone v = a.zones['volume']!;
      a.pointerDown(v.cx, v.cy);
      a.pointerMove(v.x1 - sliderPad, v.cy);
      a.pointerUp(v.x1 - sliderPad, v.cy);
      expect(a.paused, isTrue);
      expect(a.inStopMenu, isTrue);
      expect(a.soundShouldPlay, isFalse);
    });

    test('tapping the panel background does nothing', () {
      final AttendLogic a = running();
      final Zone p = a.zones['panel']!;
      a.pointerDown(p.x + 4, p.y + 4);
      a.pointerUp(p.x + 4, p.y + 4);
      expect(a.inStopMenu, isTrue);
      expect(a.paused, isTrue);
    });

    test('keep going restores the clock and the sound', () {
      final AttendLogic a = running();
      a.tap('resume');
      expect(a.paused, isFalse);
      expect(a.zones.keys.toList(), <String>['open']);
      expect(a.soundShouldPlay, isTrue);
      final double at = a.sess!.position;
      a.step(0.5, 1.0);
      expect(a.sess!.position, greaterThan(at));
    });

    test('start over gives a fresh session', () {
      final AttendLogic a = running();
      a.step(0, 0);
      final Session before = a.sess!;
      a.tap('again');
      expect(a.sess, isNot(same(before)));
      expect(a.sess!.position, 0);
      expect(a.inStopMenu, isFalse);
    });

    test('back returns to the menu and leaves it silent', () {
      final AttendLogic a = running();
      a.tap('home');
      expect(a.page, pageMenu);
      expect(a.sess, isNull);
      expect(a.soundShouldPlay, isFalse);
      expect(a.zones.has('begin'), isTrue);
    });

    test('close asks to quit', () {
      final AttendLogic a = running();
      a.tap('quit');
      expect(a.quitting, isTrue);
    });
  });

  group('the menu works', () {
    test('the minutes slider reaches every value', () {
      final AttendLogic a = AttendLogic();
      final Zone r = a.zones['minutes']!;
      final Set<int> seen = <int>{};
      for (double x = r.x; x <= r.x1; x += 1) {
        a.pointerDown(r.cx, r.cy);
        a.pointerMove(x, r.cy);
        a.pointerUp(x, r.cy);
        seen.add(a.minutes);
      }
      expect(seen,
          <int>{for (int m = minMinutes; m <= maxMinutes; m++) m});
    });

    test('begin does nothing until the sounds are ready', () {
      final AttendLogic a = AttendLogic();
      a.tap('begin');
      expect(a.page, pageMenu);
      a.ready = true;
      a.tap('begin');
      expect(a.page, pageRun);
    });

    test('the session matches the slider', () {
      final AttendLogic a = AttendLogic()..ready = true;
      a.minutes = 6;
      a.begin();
      expect(a.sess!.total, closeTo(360, 0.01));
    });

    test('settings opens and closes without losing the menu', () {
      final AttendLogic a = AttendLogic();
      a.tap('settings');
      expect(a.page, pageSettings);
      expect(a.zones.has('volume'), isTrue);
      final Zone v = a.zones['volume']!;
      expect(a.zones.at(v.cx, v.cy), 'volume');
      a.tap('close');
      expect(a.page, pageMenu);
      expect(a.zones.has('begin'), isTrue);
    });
  });

  group('nothing is off screen and nothing overlaps', () {
    test('at every size and on every page', () {
      for (final List<double> wh in <List<double>>[
        <double>[760, 560],
        <double>[1100, 700],
        <double>[1920, 1080],
      ]) {
        final AttendLogic a =
            AttendLogic(w: wh[0], h: wh[1])..ready = true;
        final List<void Function()> pages = <void Function()>[
          a.menuZones,
          a.runZones,
          a.pauseZones,
          a.settingsZones,
          a.overZones,
        ];
        for (final void Function() make in pages) {
          make();
          final List<MapEntry<String, Zone>> items = a.zones.items.entries
              .where((MapEntry<String, Zone> e) => e.key != 'panel')
              .toList();
          for (final MapEntry<String, Zone> e in items) {
            expect(e.value.x, greaterThanOrEqualTo(0), reason: e.key);
            expect(e.value.y, greaterThanOrEqualTo(0), reason: e.key);
            expect(e.value.x1, lessThanOrEqualTo(wh[0]), reason: e.key);
            expect(e.value.y1, lessThanOrEqualTo(wh[1]), reason: e.key);
            expect(e.value.touchable, isTrue, reason: e.key);
          }
          for (int i = 0; i < items.length; i++) {
            for (int j = i + 1; j < items.length; j++) {
              expect(items[i].value.overlaps(items[j].value), isFalse,
                  reason: '${items[i].key} vs ${items[j].key}');
            }
          }
        }
      }
    });

    test('a full screen panel never swallows its own buttons', () {
      final AttendLogic a = AttendLogic()..ready = true;
      a.begin();
      a.tap('open');
      for (final String k in <String>[
        'volume',
        'resume',
        'again',
        'home',
        'quit'
      ]) {
        final Zone z = a.zones[k]!;
        expect(a.zones.at(z.cx, z.cy), k);
      }
    });
  });

  group('scenes exist for every sound', () {
    test('and each one is dim enough to sit behind words', () {
      final Map<String, Raster> all = renderAllScenes(w: 220, h: 140);
      expect(all.length, soundNames.length);
      all.forEach((String name, Raster r) {
        double sum = 0;
        final int n = r.w * r.h;
        for (int i = 0; i < n; i++) {
          final int o = i * 4;
          sum += 0.114 * r.rgba[o] +
              0.587 * r.rgba[o + 1] +
              0.299 * r.rgba[o + 2];
        }
        final double mean = sum / n;
        expect(mean, lessThanOrEqualTo(maxLuma + 1), reason: name);
        expect(mean, greaterThan(4), reason: '$name is black');
      });
    });

    test('blending two scenes gives something between them', () {
      final Map<String, Raster> all = renderAllScenes(w: 60, h: 40);
      final Raster mixed = blendScenes(all, <String, double>{
        soundNames[0]: 0.5,
        soundNames[1]: 0.5,
      }, 60, 40);
      expect(mixed.w, 60);
      expect(mixed.h, 40);
      double sum = 0;
      for (int i = 0; i < 60 * 40; i++) {
        sum += mixed.rgba[i * 4];
      }
      expect(sum, greaterThan(0));
    });
  });

  group('nothing is scored and nothing is kept', () {
    test('the session carries no score', () {
      final Session s = Session(minutes: 3, seed: 1);
      final String dump = s.toString().toLowerCase();
      for (final String bad in <String>[
        'score',
        'points',
        'streak',
        'best',
        'rank'
      ]) {
        expect(dump.contains(bad), isFalse);
      }
    });

    test('the source never writes to disk', () {
      final Directory dir = Directory('lib/attend');
      final RegExp bad = RegExp(
          r'(File\(|\.writeAs|RandomAccessFile|Hive|SharedPreferences)');
      for (final FileSystemEntity f in dir.listSync()) {
        if (f is File && f.path.endsWith('.dart')) {
          expect(bad.hasMatch(f.readAsStringSync()), isFalse,
              reason: f.path);
        }
      }
    });
  });
}
