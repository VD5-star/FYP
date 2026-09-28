import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:psybot_games/core/synth.dart';
import 'package:psybot_games/core/zones.dart';
import 'package:psybot_games/reach/angles.dart';
import 'package:psybot_games/reach/audio.dart';
import 'package:engine_body/engine_body.dart' show BodyModel, LandmarkType;
import 'package:psybot_games/reach/body.dart';
import 'package:psybot_games/reach/bridge.dart';
import 'package:psybot_games/reach/draw.dart';
import 'package:psybot_games/reach/engine.dart';
import 'package:psybot_games/reach/game.dart';
import 'package:psybot_games/reach/logic.dart';
import 'package:psybot_games/reach/points.dart';
import 'package:psybot_games/reach/pose_source.dart';
import 'package:psybot_games/reach/targets.dart';

const double w = 1280;
const double h = 720;

const List<List<double>> bodyOffsets = <List<double>>[
  <double>[-50, -90],
  <double>[50, -90],
  <double>[-80, 0],
  <double>[80, 0],
  <double>[-90, 70],
  <double>[90, 70],
  <double>[-40, 90],
  <double>[40, 90],
  <double>[-45, 220],
  <double>[45, 220],
  <double>[-48, 340],
  <double>[48, 340],
];

const List<String> bodyNames = <String>[
  'leftShoulder',
  'rightShoulder',
  'leftElbow',
  'rightElbow',
  'leftWrist',
  'rightWrist',
  'leftHip',
  'rightHip',
  'leftKnee',
  'rightKnee',
  'leftAnkle',
  'rightAnkle',
];

class Rig {
  Rig(this.points, this.visibility, this.torso);
  final Points points;
  final List<double> visibility;
  final double torso;
}

Rig makeBody({double cx = 640, double cy = 360, double torso = 180,
    bool cutTop = false}) {
  final Points p = Points.nan(landmarkCount);
  final List<double> vis = List<double>.filled(landmarkCount, 0.0);
  final int nose = idx['nose']!;
  p.set(nose, cx, cutTop ? 2.0 : cy - 150);
  vis[nose] = 1.0;
  for (int i = 0; i < bodyNames.length; i++) {
    final int k = idx[bodyNames[i]]!;
    p.set(k, cx + bodyOffsets[i][0], cy + bodyOffsets[i][1]);
    vis[k] = 1.0;
  }
  return Rig(p, vis, torso);
}

List<List<double>> visibleBody(Rig rig) {
  final List<double> xs = <double>[];
  final List<double> ys = <double>[];
  for (int i = 0; i < rig.points.count; i++) {
    if (rig.visibility[i] >= 0.15 && rig.points.finite(i)) {
      xs.add(rig.points.xs[i]);
      ys.add(rig.points.ys[i]);
    }
  }
  return <List<double>>[xs, ys];
}

double nearestGap(List<List<double>> body, double x, double y) {
  double best = double.infinity;
  for (int i = 0; i < body[0].length; i++) {
    final double d = hypot(body[0][i] - x, body[1][i] - y);
    if (d < best) best = d;
  }
  return best;
}

List<String> reachSources() {
  final Directory dir = Directory('lib/reach');
  return dir
      .listSync()
      .whereType<File>()
      .where((File f) => f.path.endsWith('.dart'))
      .map((File f) => f.readAsStringSync())
      .toList();
}

int simulateBonus(String mode, {int steps = 900}) {
  final ReachEngine g = ReachEngine(mode: mode, duration: 600.0);
  final Points pts = Points(landmarkCount);
  final List<double> vis = List<double>.filled(landmarkCount, 1.0);
  int seen = 0;
  double t = 0.0;
  for (int i = 0; i < steps; i++) {
    t += 0.05;
    g.update(pts, vis, idx, 200.0, t, w, h);
    final Target? tg = g.target;
    if (tg != null) {
      if (tg.isBonus) seen += 1;
      for (int k = 0; k < pts.count; k++) {
        pts.set(k, tg.x, tg.y);
      }
    } else {
      for (int k = 0; k < pts.count; k++) {
        pts.set(k, 0.0, 0.0);
      }
    }
  }
  return seen;
}

ReachEngine simulateRun(String mode, {int steps = 900}) {
  final ReachEngine g = ReachEngine(mode: mode, duration: 600.0);
  final Points pts = Points(landmarkCount);
  final List<double> vis = List<double>.filled(landmarkCount, 1.0);
  double t = 0.0;
  for (int i = 0; i < steps; i++) {
    t += 0.05;
    g.update(pts, vis, idx, 200.0, t, w, h);
    final Target? tg = g.target;
    if (tg != null) {
      for (int k = 0; k < pts.count; k++) {
        pts.set(k, tg.x, tg.y);
      }
    } else {
      for (int k = 0; k < pts.count; k++) {
        pts.set(k, 0.0, 0.0);
      }
    }
  }
  return g;
}

List<List<String>> overlapping(Zones z, {Set<String> ignore = const <String>{}}) {
  final List<String> keys =
      z.keys.where((String k) => !ignore.contains(k)).toList();
  final List<List<String>> clash = <List<String>>[];
  for (int i = 0; i < keys.length; i++) {
    for (int j = i + 1; j < keys.length; j++) {
      if (z[keys[i]]!.overlaps(z[keys[j]]!)) {
        clash.add(<String>[keys[i], keys[j]]);
      }
    }
  }
  return clash;
}

List<String> offScreen(Zones z, double width, double height) {
  return z.keys.where((String k) {
    final Zone r = z[k]!;
    return r.x < 0 || r.y < 0 || r.x1 > width || r.y1 > height;
  }).toList();
}

void main() {
  group('landmark map', () {
    test('one source of truth, 33 entries', () {
      expect(landmarkNames.length, 33);
      expect(idx.length, 33);
      expect(idx['nose'], 0);
      expect(idx['leftShoulder'], 11);
      expect(idx['rightShoulder'], 12);
      expect(idx['leftWrist'], 15);
      expect(idx['rightFootIndex'], 32);
      for (int i = 0; i < landmarkNames.length; i++) {
        expect(idx[landmarkNames[i]], i);
      }
    });
  });

  group('minutes slider', () {
    test('covers 1..10 and every value is reachable', () {
      final Zones z = menuZones(w, h, timed);
      final Zone r = z['minutes']!;
      final Set<int> seen = <int>{};
      for (double x = r.x; x <= r.x1; x += 1.0) {
        seen.add(sliderValue(r, x, lo: minMinutes, hi: maxMinutes));
      }
      expect(seen, <int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    });

    test('dragging the slider sets every minute', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      final Zone r = logic.zones['minutes']!;
      final Set<int> seen = <int>{};
      for (double x = r.x; x <= r.x1; x += 2.0) {
        logic.pointerDown(r.cx, r.cy);
        logic.pointerMove(x, r.cy);
        logic.pointerUp(x, r.cy);
        seen.add(logic.minutes);
      }
      expect(seen, <int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    });

    test('the knob sits where python put it', () {
      final Zone r = menuZones(w, h, timed)['minutes']!;
      expect(sliderKnob(r, minMinutes, lo: minMinutes, hi: maxMinutes)
          .round(), 432);
      expect(sliderKnob(r, maxMinutes, lo: minMinutes, hi: maxMinutes)
          .round(), 848);
    });

    test('clamps out of range', () {
      expect(clampMinutes(0), 1);
      expect(clampMinutes(-5), 1);
      expect(clampMinutes(99), 10);
    });

    test('shape names are right at the boundaries', () {
      const List<List<String>> want = <List<String>>[
        <String>['quick', 'a short warm up'],
        <String>['quick', 'a short warm up'],
        <String>['steady', 'long enough to find a rhythm'],
        <String>['steady', 'long enough to find a rhythm'],
        <String>['full', 'a proper session'],
        <String>['full', 'a proper session'],
        <String>['full', 'a proper session'],
        <String>['long', 'settle in and keep moving'],
        <String>['long', 'settle in and keep moving'],
        <String>['long', 'settle in and keep moving'],
      ];
      for (int m = 1; m <= 10; m++) {
        expect(shape(m), want[m - 1], reason: 'minute $m');
      }
    });

    test('the word never goes backwards', () {
      final List<String> order = <String>['quick', 'steady', 'full', 'long'];
      int last = 0;
      for (int m = 1; m <= 10; m++) {
        final int here = order.indexOf(shape(m)[0]);
        expect(here, greaterThanOrEqualTo(last));
        last = here;
      }
    });
  });

  group('calm mode hides the clock', () {
    test('timed has a minutes zone, calm does not', () {
      expect(menuZones(w, h, timed).has('minutes'), isTrue);
      expect(menuZones(w, h, calm).has('minutes'), isFalse);
    });

    test('calm keeps the modes and begin', () {
      final Zones z = menuZones(w, h, calm);
      expect(z.has('timed'), isTrue);
      expect(z.has('calm'), isTrue);
      expect(z.has('begin'), isTrue);
    });

    test('tapping where the slider was does nothing in calm', () {
      final ReachLogic logic =
          ReachLogic(width: w, height: h, mode: calm);
      expect(logic.zones.has('minutes'), isFalse);
      final Zone ghost = menuZones(w, h, timed)['minutes']!;
      final int before = logic.minutes;
      logic.pointerDown(ghost.cx, ghost.cy);
      expect(logic.sliding, isFalse);
      logic.pointerMove(ghost.x1 - 4, ghost.cy);
      logic.pointerUp(ghost.x1 - 4, ghost.cy);
      expect(logic.minutes, before);
      expect(logic.page, pageMenu);
      expect(logic.zones.at(ghost.cx, ghost.cy), isNull);
    });

    test('switching back to timed restores the slider', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      expect(logic.zones.has('minutes'), isTrue);
      final Zone c = logic.zones['calm']!;
      logic.pointerDown(c.cx, c.cy);
      logic.pointerUp(c.cx, c.cy);
      expect(logic.mode, calm);
      expect(logic.zones.has('minutes'), isFalse);
      final Zone t = logic.zones['timed']!;
      logic.pointerDown(t.cx, t.cy);
      logic.pointerUp(t.cx, t.cy);
      expect(logic.mode, timed);
      expect(logic.zones.has('minutes'), isTrue);
    });

    test('picking a mode does not begin the game', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      final Zone c = logic.zones['calm']!;
      logic.pointerDown(c.cx, c.cy);
      logic.pointerUp(c.cx, c.cy);
      expect(logic.page, pageMenu);
    });

    test('begin starts the session with the chosen minutes', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      final Zone r = logic.zones['minutes']!;
      logic.pointerDown(r.cx, r.cy);
      logic.pointerUp(r.x1 - 2, r.cy);
      expect(logic.minutes, maxMinutes);
      final Zone b = logic.zones['begin']!;
      logic.pointerDown(b.cx, b.cy);
      logic.pointerUp(b.cx, b.cy);
      expect(logic.page, pagePlay);
      expect(logic.engine.duration, maxMinutes * 60.0);
    });

    test('a slid press is not a tap', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      final Zone b = logic.zones['begin']!;
      logic.pointerDown(b.cx, b.cy);
      logic.pointerUp(b.cx + 300, b.cy);
      expect(logic.page, pageMenu);
    });
  });

  group('bonus targets', () {
    test('timed spawns bonus targets, calm spawns exactly zero', () {
      expect(simulateBonus(timed), greaterThan(0));
      expect(simulateBonus(calm), 0);
    });

    test('the counts match the measured python run exactly', () {
      expect(simulateBonus(timed), 150);
      expect(simulateBonus(calm), 0);
      expect(simulateRun(timed).state.hits.length, 899);
      expect(simulateRun(calm).state.hits.length, 899);
      final ReachEngine g = simulateRun(timed);
      expect(g.state.bonusTaken, 149);
      expect(g.state.bonusTime, 447.0);
    });

    test('calm still scores and never banks bonus time', () {
      final ReachEngine g = simulateRun(calm);
      expect(g.state.hits.length, greaterThan(0));
      expect(g.state.score, greaterThan(0));
      expect(g.state.bonusTime, 0.0);
      expect(g.state.bonusTaken, 0);
      for (final Hit hit in g.state.hits) {
        expect(hit.bonus, isFalse);
        expect(hit.band, isNot(bonus));
      }
    });

    test('timed banks bonus time when bonus targets are taken', () {
      final ReachEngine g = simulateRun(timed);
      expect(g.state.bonusTaken, greaterThan(0));
      expect(g.state.bonusTime, g.state.bonusTaken * bonusSeconds);
      expect(g.totalTime(), greaterThan(g.duration));
    });

    test('a bonus target is only ever due after a streak of five', () {
      final ReachEngine g = ReachEngine(mode: timed, duration: 600.0);
      final Points pts = Points(landmarkCount);
      final List<double> vis = List<double>.filled(landmarkCount, 1.0);
      double t = 0.0;
      final List<int> streaks = <int>[];
      for (int i = 0; i < 400; i++) {
        t += 0.05;
        g.update(pts, vis, idx, 200.0, t, w, h);
        final Target? tg = g.target;
        if (tg == null) continue;
        if (tg.isBonus) streaks.add(g.state.streak);
        for (int k = 0; k < pts.count; k++) {
          pts.set(k, tg.x, tg.y);
        }
      }
      expect(streaks, isNotEmpty);
      for (final int s in streaks) {
        expect(s % bonusEvery, 0);
      }
    });

    test('bonus targets carry no points and are bigger', () {
      final TargetSpawner sp = TargetSpawner(seed: 3);
      final Target plain =
          sp.spawn(640, 360, 180, 0.0, w, h, bonusTarget: false)!;
      final Target reward =
          sp.spawn(640, 360, 180, 0.0, w, h, bonusTarget: true)!;
      expect(reward.band, bonus);
      expect(reward.isBonus, isTrue);
      expect(reward.points, 0);
      expect(plain.isBonus, isFalse);
      expect(plain.points, greaterThan(0));
      expect(reward.radius, greaterThan(plain.radius));
    });
  });

  group('targets, bands and lifetime', () {
    test('constants match the python source', () {
      expect(armLength, 0.97);
      expect(defaultBodyGap, 0.95);
      expect(bonusEvery, 5);
      expect(bonusSeconds, 3.0);
      expect(defaultLifetime, 1.5);
      expect(bands[near], <double>[0.95, 1.15, 1]);
      expect(bands[far], <double>[1.15, 1.40, 2]);
      expect(bands[veryFar], <double>[1.40, 99.0, 3]);
    });

    test('bands score 1, 2 and 3 by distance', () {
      final TargetSpawner sp = TargetSpawner(seed: 3);
      final Rig rig = makeBody();
      final List<List<double>> body = visibleBody(rig);
      final Map<String, List<double>> byBand = <String, List<double>>{};
      for (int i = 0; i < 600; i++) {
        final Target? t = sp.spawn(640, 360, rig.torso, 0.0, w, h,
            bodyX: body[0], bodyY: body[1]);
        if (t == null) continue;
        final double gap = nearestGap(body, t.x, t.y) / rig.torso;
        byBand.putIfAbsent(t.band, () => <double>[]).add(gap);
        if (t.band == near) expect(t.points, 1);
        if (t.band == far) expect(t.points, 2);
        if (t.band == veryFar) expect(t.points, 3);
      }
      expect(byBand.keys, containsAll(<String>[near, far, veryFar]));
      for (final double g in byBand[near]!) {
        expect(g, inInclusiveRange(0.95, 1.15));
      }
      for (final double g in byBand[far]!) {
        expect(g, inInclusiveRange(1.15, 1.40));
      }
      for (final double g in byBand[veryFar]!) {
        expect(g, greaterThanOrEqualTo(1.40));
      }
    });

    test('spawns keep clear of the body', () {
      final TargetSpawner sp = TargetSpawner(seed: 3);
      final Rig rig = makeBody();
      final List<List<double>> body = visibleBody(rig);
      final List<double> gaps = <double>[];
      for (int i = 0; i < 500; i++) {
        final Target? t = sp.spawn(640, 360, rig.torso, 0.0, w, h,
            bodyX: body[0], bodyY: body[1]);
        if (t == null) continue;
        gaps.add(nearestGap(body, t.x, t.y) / rig.torso);
      }
      expect(gaps, isNotEmpty);
      expect(gaps.reduce(math.min), greaterThanOrEqualTo(defaultBodyGap));
    });

    test('spawns stay fully on screen', () {
      final TargetSpawner sp = TargetSpawner(seed: 5);
      for (int i = 0; i < 400; i++) {
        final Target? t = sp.spawn(640, 360, 180, 0.0, w, h);
        if (t == null) continue;
        expect(t.x - t.radius, greaterThanOrEqualTo(-1.0));
        expect(t.y - t.radius, greaterThanOrEqualTo(-1.0));
        expect(t.x + t.radius, lessThanOrEqualTo(w + 1.0));
        expect(t.y + t.radius, lessThanOrEqualTo(h + 1.0));
      }
    });

    test('a body cut off at the top gets a fence', () {
      final Rig cut = makeBody(cutTop: true);
      final List<List<double>> body = visibleBody(cut);
      final EdgeFence fence =
          edgePoints(body[0], body[1], cut.torso, w, h);
      expect(fence.isEmpty, isFalse);
      final Rig whole = makeBody();
      final List<List<double>> full = visibleBody(whole);
      expect(edgePoints(full[0], full[1], whole.torso, w, h).isEmpty,
          isTrue);
    });

    test('nothing spawns in the cut-off zone above the body', () {
      final Rig cut = makeBody(cutTop: true);
      final List<List<double>> body = visibleBody(cut);
      double minY = body[1].reduce(math.min);
      double minX = body[0].reduce(math.min);
      double maxX = body[0].reduce(math.max);
      minX -= cut.torso * 0.5;
      maxX += cut.torso * 0.5;
      final TargetSpawner sp = TargetSpawner(seed: 4);
      int above = 0;
      for (int i = 0; i < 500; i++) {
        final Target? t = sp.spawn(640, 360, cut.torso, 0.0, w, h,
            bodyX: body[0], bodyY: body[1]);
        if (t == null) continue;
        if (t.y < minY + cut.torso * 0.5 && t.x >= minX && t.x <= maxX) {
          above += 1;
        }
      }
      expect(above, 0);
    });

    test('a target shrinks and then expires at its lifetime', () {
      final TargetSpawner sp = TargetSpawner(seed: 1);
      final Target t = sp.spawn(640, 360, 180, 0.0, w, h)!;
      expect(t.scale(0.0), closeTo(1.0, 1e-9));
      expect(t.expired(0.0), isFalse);
      expect(t.scale(0.75), closeTo(shrinkFloor + 0.65 * 0.5, 1e-9));
      expect(t.expired(1.49), isFalse);
      expect(t.expired(1.5), isTrue);
      expect(t.scale(1.5), closeTo(shrinkFloor, 1e-9));
      expect(t.currentRadius(1.5), closeTo(t.radius * shrinkFloor, 1e-9));
      expect(t.life(9.0), 0.0);
    });

    test('contains uses the shrunken square', () {
      final Target t = Target(
        x: 100,
        y: 100,
        radius: 20,
        band: near,
        points: 1,
        born: 0.0,
        lifetime: 1.5,
      );
      expect(t.contains(100, 100, 0.0), isTrue);
      expect(t.contains(119, 119, 0.0), isTrue);
      expect(t.contains(121, 100, 0.0), isFalse);
      expect(t.contains(double.nan, 100, 0.0), isFalse);
      expect(t.contains(115, 100, 1.5), isFalse);
      expect(t.containsAny(<double>[500, 100], <double>[500, 100], 0.0),
          isTrue);
      expect(t.containsAny(<double>[500], <double>[500], 0.0), isFalse);
      expect(t.containsAny(<double>[], <double>[], 0.0), isFalse);
    });
  });

  group('hit detection and scoring', () {
    test('a wrist on the target scores and respawns', () {
      final ReachEngine g = ReachEngine(mode: timed, duration: 30, seed: 7);
      final Rig rig = makeBody();
      g.update(rig.points, rig.visibility, idx, rig.torso, 0.0, w, h);
      final Target? first = g.target;
      expect(first, isNotNull);
      rig.points.set(idx['rightWrist']!, first!.x, first.y);
      final List<Hit> hits =
          g.update(rig.points, rig.visibility, idx, rig.torso, 0.1, w, h);
      expect(hits.length, 1);
      expect(hits.first.points, first.points);
      expect(g.state.score, first.points);
      expect(g.state.streak, 1);
      expect(identical(g.target, first), isFalse);
    });

    test('any part of a limb can touch, not just the extremities', () {
      for (final List<String> limb in <List<String>>[
        <String>['leftShoulder', 'leftElbow'],
        <String>['leftHip', 'leftKnee'],
      ]) {
        final ReachEngine g = ReachEngine(mode: timed, duration: 30, seed: 2);
        final Rig rig = makeBody();
        g.update(rig.points, rig.visibility, idx, rig.torso, 0.0, w, h);
        final Target t = g.target!;
        final int a = idx[limb[0]]!;
        final int b = idx[limb[1]]!;
        final double mx = (rig.points.xs[a] + rig.points.xs[b]) / 2;
        final double my = (rig.points.ys[a] + rig.points.ys[b]) / 2;
        final double dx = t.x - mx;
        final double dy = t.y - my;
        final Points moved = rig.points.copy();
        for (int i = 0; i < moved.count; i++) {
          if (rig.visibility[i] >= 0.15 && rig.points.finite(i)) {
            moved.set(i, rig.points.xs[i] + dx, rig.points.ys[i] + dy);
          }
        }
        final List<Hit> hits =
            g.update(moved, rig.visibility, idx, rig.torso, 0.05, w, h);
        expect(hits, isNotEmpty, reason: '${limb[0]}..${limb[1]}');
      }
    });

    test('limb interpolation produces the documented point count', () {
      final ReachEngine g = ReachEngine(mode: timed, duration: 30, seed: 2);
      final Rig rig = makeBody();
      final List<List<double>> touch =
          g.touchPoints(rig.points, rig.visibility, idx);
      final int landmarks =
          rig.visibility.where((double v) => v >= 0.15).length;
      int segments = 0;
      for (final List<String> limb in limbs) {
        final int a = idx[limb[0]]!;
        final int b = idx[limb[1]]!;
        if (rig.visibility[a] >= 0.15 && rig.visibility[b] >= 0.15 &&
            rig.points.finite(a) && rig.points.finite(b)) {
          segments += 1;
        }
      }
      expect(touch[0].length, landmarks + segments * (limbSteps - 1));
      expect(touch[0].length, touch[1].length);
      expect(limbs.length, 16);
      expect(limbSteps, 4);
    });

    test('an expired target counts as a miss and breaks the streak', () {
      final ReachEngine g = ReachEngine(mode: timed, duration: 30, seed: 7);
      final Rig rig = makeBody();
      g.update(rig.points, rig.visibility, idx, rig.torso, 0.0, w, h);
      final Target first = g.target!;
      rig.points.set(idx['rightWrist']!, first.x, first.y);
      g.update(rig.points, rig.visibility, idx, rig.torso, 0.05, w, h);
      expect(g.state.streak, 1);
      rig.points.set(idx['rightWrist']!, -999, -999);
      g.update(rig.points, rig.visibility, idx, rig.torso, 5.0, w, h);
      expect(g.state.misses, greaterThan(0));
      expect(g.state.streak, 0);
    });

    test('no body means no play', () {
      final ReachEngine g = ReachEngine(mode: timed, duration: 30, seed: 7);
      expect(g.update(null, null, idx, null, 0.0, w, h), isEmpty);
      expect(g.target, isNull);
      final Rig rig = makeBody();
      expect(g.update(rig.points, rig.visibility, idx, 0.0, 0.0, w, h),
          isEmpty);
      expect(g.target, isNull);
    });

    test('too few visible torso points means no anchor', () {
      final ReachEngine g = ReachEngine(mode: timed, duration: 30, seed: 7);
      final Rig rig = makeBody();
      for (final String name in <String>[
        'leftShoulder',
        'rightShoulder',
        'leftHip'
      ]) {
        rig.visibility[idx[name]!] = 0.0;
      }
      g.update(rig.points, rig.visibility, idx, rig.torso, 0.0, w, h);
      expect(g.target, isNull);
    });
  });

  group('game state machine', () {
    test('timed runs out, calm never does', () {
      final ReachEngine t = ReachEngine(mode: timed, duration: 10.0);
      final Rig rig = makeBody();
      t.update(rig.points, rig.visibility, idx, rig.torso, 0.0, w, h);
      expect(t.remaining(0.0), closeTo(10.0, 1e-6));
      t.update(rig.points, rig.visibility, idx, rig.torso, 11.0, w, h);
      expect(t.state.phase, ending);

      final ReachEngine c = ReachEngine(mode: calm, duration: 10.0);
      c.update(rig.points, rig.visibility, idx, rig.torso, 0.0, w, h);
      c.update(rig.points, rig.visibility, idx, rig.torso, 999.0, w, h);
      expect(c.remaining(999.0), double.infinity);
      expect(c.state.phase, running);
    });

    test('ending becomes finished after the fade', () {
      final ReachEngine g = ReachEngine(mode: timed, duration: 1.0);
      final Rig rig = makeBody();
      g.update(rig.points, rig.visibility, idx, rig.torso, 0.0, w, h);
      g.beginEnding(2.0);
      expect(g.state.phase, ending);
      expect(g.endingProgress(2.0), 0.0);
      g.update(rig.points, rig.visibility, idx, rig.torso, 2.5, w, h);
      expect(g.state.phase, ending);
      g.update(rig.points, rig.visibility, idx, rig.torso,
          2.0 + endingSeconds, w, h);
      expect(g.state.phase, finished);
      expect(g.target, isNull);
    });

    test('pause freezes the clock and shifts the target', () {
      final ReachEngine g = ReachEngine(mode: timed, duration: 60.0);
      final Rig rig = makeBody();
      g.update(rig.points, rig.visibility, idx, rig.torso, 0.0, w, h);
      final double born = g.target!.born;
      g.pause(1.0);
      expect(g.state.phase, paused);
      expect(g.elapsed(50.0), closeTo(1.0, 1e-6));
      expect(g.update(rig.points, rig.visibility, idx, rig.torso, 5.0, w, h),
          isEmpty);
      g.resume(4.0);
      expect(g.state.phase, running);
      expect(g.state.pausedTotal, closeTo(3.0, 1e-6));
      expect(g.target!.born, closeTo(born + 3.0, 1e-6));
      expect(g.elapsed(5.0), closeTo(2.0, 1e-6));
    });

    test('reset clears the score and the target', () {
      final ReachEngine g = ReachEngine(mode: timed, duration: 30, seed: 7);
      final Rig rig = makeBody();
      g.update(rig.points, rig.visibility, idx, rig.torso, 0.0, w, h);
      rig.points.set(idx['rightWrist']!, g.target!.x, g.target!.y);
      g.update(rig.points, rig.visibility, idx, rig.torso, 0.1, w, h);
      expect(g.state.score, greaterThan(0));
      g.reset();
      expect(g.state.score, 0);
      expect(g.state.hits, isEmpty);
      expect(g.state.phase, running);
      expect(g.target, isNull);
    });

    test('summary counts hits, bands and bonus separately', () {
      final ReachEngine g = simulateRun(timed, steps: 300);
      final Summary s = g.summary(20.0);
      final int bonusHits =
          g.state.hits.where((Hit hit) => hit.bonus).length;
      expect(s.hits, g.state.hits.length - bonusHits);
      expect(s.bonus, g.state.bonusTaken);
      expect(s.score, g.state.score);
      int counted = 0;
      s.bands.forEach((String k, int v) => counted += v);
      expect(counted, g.state.hits.length);
      expect(s.bands.keys, isNot(contains('nonsense')));
    });
  });

  group('smoothing through the shared engine', () {
    test('it strips jitter without dragging the hand behind', () {
      final Rig rig = makeBody();
      final BodyTracker tracker = BodyTracker();
      final int wrist = idx['leftWrist']!;

      int seed = 12345;
      double gauss() {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        final double u1 = ((seed >> 8) & 0xffff) / 65535.0;
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        final double u2 = ((seed >> 8) & 0xffff) / 65535.0;
        return math.sqrt(-2 * math.log(math.max(1e-12, u1))) *
            math.cos(2 * math.pi * u2);
      }

      final List<Points> noisy = <Points>[];
      final List<Points> out = <Points>[];
      double t = 0.0;
      for (int k = 0; k < 120; k++) {
        t += 1 / 30;
        final Points jit = Points.nan(landmarkCount);
        for (int i = 0; i < landmarkCount; i++) {
          jit.set(i, rig.points.xs[i] + gauss() * 2.0,
              rig.points.ys[i] + gauss() * 2.0);
        }
        noisy.add(jit);
        final Points? got = tracker.update(PoseFrame(
          points: jit,
          visibility: rig.visibility,
          width: w,
          height: h,
          timestamp: t,
        ));
        if (got != null) out.add(got.copy());
      }

      expect(out.length, greaterThan(60));

      List<double> steps(List<Points> track) {
        final List<double> d = <double>[];
        for (int i = 1; i < track.length; i++) {
          d.add(hypot(track[i].xs[wrist] - track[i - 1].xs[wrist],
              track[i].ys[wrist] - track[i - 1].ys[wrist]));
        }
        return d;
      }

      final double rawJitter = median(steps(noisy));
      final double smoothJitter = median(steps(out));
      expect(rawJitter, inInclusiveRange(2.0, 4.5));
      expect(smoothJitter, lessThan(rawJitter));

      final List<double> offset = <double>[];
      for (int i = 20; i < out.length; i++) {
        offset.add(hypot(out[i].xs[wrist] - rig.points.xs[wrist],
            out[i].ys[wrist] - rig.points.ys[wrist]));
      }
      expect(median(offset), lessThan(6.0));
    });

    test('a still body keeps its place', () {
      final Rig rig = makeBody();
      final BodyTracker tracker = BodyTracker();
      Points? last;
      double t = 0.0;
      for (int k = 0; k < 60; k++) {
        t += 1 / 30;
        last = tracker.update(PoseFrame(
          points: rig.points,
          visibility: rig.visibility,
          width: w,
          height: h,
          timestamp: t,
        ));
      }
      expect(last, isNotNull);
      final int wrist = idx['leftWrist']!;
      expect(last!.xs[wrist], closeTo(rig.points.xs[wrist], 4.0));
      expect(last.ys[wrist], closeTo(rig.points.ys[wrist], 4.0));
    });

    test('the beta used for smoothing is the pixel value times the unit', () {
      expect(smoothBetaPixels, 0.02);
      expect(smoothMinCutoff, 1.2);
      expect(smoothVisibility, 0.15);
    });
  });

  group('skeleton, anchor and occlusion through the shared engine', () {
    test('the torso is measured in isotropic units', () {
      final Rig rig = makeBody();
      final FrameScale scale = FrameScale(w, h);
      final Points iso = pixelsToIso(rig.points, scale);
      expect(isoTorso(iso) * scale.unit, closeTo(180.0, 1e-9));
    });

    test('isotropic scaling keeps a circle a circle', () {
      final FrameScale scale = FrameScale(480.0, 640.0);
      final double r = 120.0;
      double worstLow = double.infinity;
      double worstHigh = 0.0;
      for (int k = 0; k < 360; k++) {
        final double a = k * math.pi / 180.0;
        final double px = 240.0 + r * math.cos(a);
        final double py = 320.0 + r * math.sin(a);
        final double ix = scale.toIsoX(px) - scale.toIsoX(240.0);
        final double iy = scale.toIsoY(py) - scale.toIsoY(320.0);
        final double got = hypot(ix, iy) * scale.unit;
        worstLow = math.min(worstLow, got);
        worstHigh = math.max(worstHigh, got);
      }
      expect(worstLow, closeTo(r, 1e-9));
      expect(worstHigh, closeTo(r, 1e-9));
    });

    test('a round trip through the bridge changes nothing', () {
      final Rig rig = makeBody();
      final FrameScale scale = FrameScale(w, h);
      final Points back = isoToPixels(pixelsToIso(rig.points, scale), scale);
      for (int i = 0; i < landmarkCount; i++) {
        if (!rig.points.finite(i)) continue;
        expect(back.xs[i], closeTo(rig.points.xs[i], 1e-9));
        expect(back.ys[i], closeTo(rig.points.ys[i], 1e-9));
      }
    });

    test('the calibrator learns bone lengths and mirrors them', () {
      final BodyTracker tracker = BodyTracker(smoothing: false);
      final Rig rig = makeBody();
      double t = 0.0;
      for (int i = 0; i < 40; i++) {
        t += 1 / 30;
        tracker.update(PoseFrame(
          points: rig.points,
          visibility: rig.visibility,
          width: w,
          height: h,
          timestamp: t,
        ));
      }
      final BodyModel model = tracker.skeleton.model;
      expect(model.lengths.length, greaterThanOrEqualTo(8));
      expect(model.lengths[LandmarkType.leftElbow], isNotNull);
      expect(model.lengths[LandmarkType.leftElbow],
          closeTo(model.lengths[LandmarkType.rightElbow]!, 1e-12));
    });

    test('a joint that leaves the frame is dropped', () {
      final BodyTracker tracker = BodyTracker(smoothing: false);
      final Rig rig = makeBody();
      final int wrist = idx['leftWrist']!;
      final Points gone = rig.points.copy();
      gone.set(wrist, -500, -500);
      tracker.update(PoseFrame(
        points: gone,
        visibility: rig.visibility,
        width: w,
        height: h,
        timestamp: 0.0,
      ));
      expect(tracker.discarded, isNotNull);
      expect(tracker.discarded![wrist], isTrue);
    });

    test('a quiet joint is held still and a moving one is released', () {
      final Rig rig = makeBody();
      final int wrist = idx['leftWrist']!;

      final BodyTracker quiet = BodyTracker(smoothing: false);
      double t = 0.0;
      for (int i = 0; i < 30; i++) {
        t += 1 / 30;
        quiet.update(PoseFrame(
          points: rig.points,
          visibility: rig.visibility,
          width: w,
          height: h,
          timestamp: t,
        ));
      }
      final double restX = quiet.points!.xs[wrist];

      final Points nudged = rig.points.copy();
      nudged.set(wrist, rig.points.xs[wrist] + 180.0 * 0.002,
          rig.points.ys[wrist]);
      t += 1 / 30;
      quiet.update(PoseFrame(
        points: nudged,
        visibility: rig.visibility,
        width: w,
        height: h,
        timestamp: t,
      ));
      expect((quiet.points!.xs[wrist] - restX).abs(), lessThan(0.30));

      final Points shoved = rig.points.copy();
      shoved.set(wrist, rig.points.xs[wrist] + 180.0 * 0.5,
          rig.points.ys[wrist]);
      t += 1 / 30;
      quiet.update(PoseFrame(
        points: shoved,
        visibility: rig.visibility,
        width: w,
        height: h,
        timestamp: t,
      ));
      expect(quiet.points!.xs[wrist], greaterThan(restX + 40.0));
    });

    test('the tracker runs one torso length through the whole pipeline', () {
      final BodyTracker tracker = BodyTracker();
      final Rig rig = makeBody();
      double t = 0.0;
      for (int i = 0; i < 40; i++) {
        t += 1 / 30;
        tracker.update(PoseFrame(
          points: rig.points,
          visibility: rig.visibility,
          width: w,
          height: h,
          timestamp: t,
        ));
      }
      expect(tracker.points, isNotNull);
      expect(tracker.torso, isNotNull);
      expect(tracker.torso, closeTo(180.0, 3.0));
      expect(tracker.wrists().length, 2);
      tracker.reset();
      expect(tracker.points, isNull);
      expect(tracker.torso, isNull);
    });

    test('a tiny body is refused as too far away', () {
      final BodyTracker tracker = BodyTracker();
      final Rig rig = makeBody(torso: 10, cx: 100, cy: 100);
      final Points small = Points.nan(landmarkCount);
      for (int i = 0; i < landmarkCount; i++) {
        if (rig.visibility[i] >= 0.15) {
          small.set(i, 100 + rig.points.xs[i] * 0.02,
              100 + rig.points.ys[i] * 0.02);
        }
      }
      tracker.update(PoseFrame(
        points: small,
        visibility: rig.visibility,
        width: w,
        height: h,
        timestamp: 0.0,
      ));
      expect(tracker.torso, isNull);
    });

    test('the minimum torso is the old pixel floor in units', () {
      expect(minTorsoUnits * 720.0, closeTo(40.0, 1e-9));
    });
  });

  group('menus, pause and summary', () {
    test('pause has continue and finish, both touchable', () {
      final Zones z = pauseZones(w, h);
      expect(z.has('resume'), isTrue);
      expect(z.has('finish'), isTrue);
      expect(z['resume']!.touchable, isTrue);
      expect(z['finish']!.touchable, isTrue);
    });

    test('the panel never swallows its own buttons', () {
      final Zones z = pauseZones(w, h);
      expect(z.at(z['resume']!.cx, z['resume']!.cy), 'resume');
      expect(z.at(z['finish']!.cx, z['finish']!.cy), 'finish');
      expect(z.at(z['panel']!.x + 4, z['panel']!.y + 4), 'panel');
    });

    test('the stop button opens pause, continue and finish work', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.begin();
      final Rig rig = makeBody();
      logic.step(
          PoseFrame(
            points: rig.points,
            visibility: rig.visibility,
            width: w,
            height: h,
            timestamp: 0.0,
          ),
          0.0);
      final Zone stop = logic.zones['stop']!;
      expect(stop.touchable, isTrue);
      logic.pointerDown(stop.cx, stop.cy);
      logic.pointerUp(stop.cx, stop.cy);
      expect(logic.showPause, isTrue);
      expect(logic.zones.has('resume'), isTrue);

      final Zone resume = logic.zones['resume']!;
      logic.pointerDown(resume.cx, resume.cy);
      logic.pointerUp(resume.cx, resume.cy);
      expect(logic.showPause, isFalse);
      expect(logic.engine.state.phase, running);

      logic.pointerDown(stop.cx, stop.cy);
      logic.pointerUp(stop.cx, stop.cy);
      final Zone finishBtn = logic.zones['finish']!;
      logic.pointerDown(finishBtn.cx, finishBtn.cy);
      logic.pointerUp(finishBtn.cx, finishBtn.cy);
      expect(logic.engine.state.phase, finished);
      expect(logic.showSummary, isTrue);
    });

    test('tapping the pause panel itself does nothing', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.begin();
      logic.engine.pause(0.0);
      logic.rebuildZones();
      final Zone panel = logic.zones['panel']!;
      logic.pointerDown(panel.x + 4, panel.y + 4);
      logic.pointerUp(panel.x + 4, panel.y + 4);
      expect(logic.showPause, isTrue);
    });

    test('summary has again, menu and close', () {
      final Zones z = summaryZones(w, h);
      for (final String k in <String>['again', 'menu', 'close']) {
        expect(z.has(k), isTrue, reason: k);
        expect(z[k]!.touchable, isTrue, reason: k);
      }
      expect(overlapping(z), isEmpty);
    });

    test('the three summary buttons each do their job', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.begin();
      logic.engine.finish(1.0);
      logic.finishedAt = 1.0;
      logic.rebuildZones();
      expect(logic.showSummary, isTrue);

      final Zone menu = logic.zones['menu']!;
      logic.pointerDown(menu.cx, menu.cy);
      logic.pointerUp(menu.cx, menu.cy);
      expect(logic.page, pageMenu);

      logic.begin();
      logic.engine.finish(1.0);
      logic.finishedAt = 1.0;
      logic.rebuildZones();
      final Zone again = logic.zones['again']!;
      logic.pointerDown(again.cx, again.cy);
      logic.pointerUp(again.cx, again.cy);
      expect(logic.page, pagePlay);
      expect(logic.engine.state.phase, running);

      logic.engine.finish(2.0);
      logic.finishedAt = 2.0;
      logic.rebuildZones();
      final Zone close = logic.zones['close']!;
      logic.pointerDown(close.cx, close.cy);
      logic.pointerUp(close.cx, close.cy);
      expect(logic.closed, isTrue);
    });

    test('the summary fades in over 0.9 seconds', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.begin();
      logic.finishedAt = 10.0;
      expect(logic.summaryAlpha(10.0), 0.0);
      expect(logic.summaryAlpha(10.45), closeTo(0.5, 1e-9));
      expect(logic.summaryAlpha(10.9), 1.0);
      expect(logic.summaryAlpha(50.0), 1.0);
      expect(summaryFade, 0.9);
    });

    test('menu zones never overlap and stay on screen', () {
      for (final List<double> size in <List<double>>[
        <double>[640, 480],
        <double>[1280, 720],
        <double>[1920, 1080],
      ]) {
        for (final String mode in <String>[timed, calm]) {
          final Zones z = menuZones(size[0], size[1], mode);
          expect(overlapping(z), isEmpty,
              reason: 'menu $mode ${size[0]}x${size[1]}');
          expect(offScreen(z, size[0], size[1]), isEmpty,
              reason: 'menu $mode ${size[0]}x${size[1]}');
          for (final String k in z.keys) {
            expect(z[k]!.touchable, isTrue, reason: '$mode $k');
          }
        }
      }
    });

    test('pause and summary fit 640x480 and 1920x1080', () {
      for (final List<double> size in <List<double>>[
        <double>[640, 480],
        <double>[1920, 1080],
      ]) {
        final Zones p = pauseZones(size[0], size[1]);
        expect(offScreen(p, size[0], size[1]), isEmpty,
            reason: 'pause ${size[0]}x${size[1]}');
        expect(overlapping(p, ignore: <String>{'panel'}), isEmpty);
        expect(p['resume']!.touchable, isTrue);
        expect(p['finish']!.touchable, isTrue);

        final Zones s = summaryZones(size[0], size[1]);
        expect(offScreen(s, size[0], size[1]), isEmpty,
            reason: 'summary ${size[0]}x${size[1]}');
        expect(overlapping(s), isEmpty);
        for (final String k in <String>['again', 'menu', 'close']) {
          expect(s[k]!.touchable, isTrue, reason: '$k ${size[0]}');
        }
      }
    });

    test('the pause buttons sit inside the panel', () {
      for (final List<double> size in <List<double>>[
        <double>[640, 480],
        <double>[1280, 720],
        <double>[1920, 1080],
      ]) {
        final Zones z = pauseZones(size[0], size[1]);
        final Zone panel = z['panel']!;
        for (final String k in <String>['resume', 'finish']) {
          final Zone b = z[k]!;
          expect(b.x, greaterThanOrEqualTo(panel.x));
          expect(b.x1, lessThanOrEqualTo(panel.x1));
          expect(b.y, greaterThanOrEqualTo(panel.y));
          expect(b.y1, lessThanOrEqualTo(panel.y1));
        }
      }
    });

    test('the play screen only offers stop', () {
      final Zones z = playZones(w, h);
      expect(z.keys.toList(), <String>['stop']);
      expect(z['stop']!.touchable, isTrue);
      expect(offScreen(z, w, h), isEmpty);
    });

    test('resize relays out every screen', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.resize(640, 480);
      expect(offScreen(logic.zones, 640, 480), isEmpty);
      logic.begin();
      logic.resize(1920, 1080);
      expect(offScreen(logic.zones, 1920, 1080), isEmpty);
    });
  });

  group('summary hides score and bonus in calm', () {
    const Summary rich = Summary(
      score: 1240,
      hits: 38,
      misses: 4,
      bonus: 3,
      seconds: 184.0,
      bands: <String, int>{near: 12, far: 9, veryFar: 17},
    );

    test('calm lists neither score nor bonus', () {
      final List<String> labels =
          summaryStats(rich, calm).map((List<String> r) => r[0]).toList();
      expect(labels, <String>['touched', 'time']);
      expect(labels, isNot(contains('score')));
      expect(labels, isNot(contains('bonus')));
    });

    test('timed lists score first and bonus last', () {
      final List<List<String>> stats = summaryStats(rich, timed);
      final List<String> labels =
          stats.map((List<String> r) => r[0]).toList();
      expect(labels, <String>['score', 'touched', 'time', 'bonus']);
      expect(stats.first[1], '1240');
      expect(stats.last[1], '3');
    });

    test('the time reads as minutes and seconds', () {
      expect(clockText(184.0), '3:04');
      expect(clockText(0.0), '0:00');
      expect(clockText(59.9), '0:59');
      expect(clockText(600.0), '10:00');
      expect(summaryStats(rich, calm)[1][1], '3:04');
    });

    test('the band breakdown is sorted and complete', () {
      expect(bandBreakdown(rich), 'far 9   near 12   very_far 17');
      expect(
          bandBreakdown(const Summary(
            score: 0,
            hits: 0,
            misses: 0,
            bonus: 0,
            seconds: 0,
            bands: <String, int>{},
          )),
          '');
    });

    test('drawing a summary at any alpha and mode never throws', () {
      for (final String mode in <String>[timed, calm]) {
        for (final double a in <double>[0.0, 0.2, 0.5, 1.0]) {
          final ui.PictureRecorder rec = ui.PictureRecorder();
          final Canvas canvas = Canvas(rec);
          drawSummary(canvas, const Size(w, h), summaryZones(w, h), rich,
              mode, a);
          rec.endRecording().dispose();
        }
      }
    });
  });

  group('no keyboard and no gestures anywhere in lib/reach', () {
    test('the sources mention no keyboard handling', () {
      const List<String> banned = <String>[
        'RawKeyboard',
        'KeyboardListener',
        'RawKeyboardListener',
        'LogicalKeyboardKey',
        'PhysicalKeyboardKey',
        'HardwareKeyboard',
        'onKeyEvent',
        'onKey:',
        'KeyDownEvent',
        'FocusNode',
        'Shortcuts(',
        'CallbackAction',
        'services.dart',
      ];
      final List<String> sources = reachSources();
      expect(sources, isNotEmpty);
      for (final String src in sources) {
        for (final String word in banned) {
          expect(src.contains(word), isFalse, reason: 'found $word');
        }
      }
    });

    test('the finger gesture system is gone', () {
      const List<String> banned = <String>[
        'fingers',
        'CommandWatcher',
        'HeldCommand',
        'gesture',
        'Gesture',
        'hand_landmarker',
        'handLandmarker',
      ];
      for (final String src in reachSources()) {
        for (final String word in banned) {
          expect(src.contains(word), isFalse, reason: 'found $word');
        }
      }
    });

    test('there is no fingers module on disk', () {
      expect(File('lib/reach/fingers.dart').existsSync(), isFalse);
      expect(File('lib/reach/hand_landmarker.task').existsSync(), isFalse);
    });

    test('the sources carry no comments', () {
      for (final String src in reachSources()) {
        for (final String line in src.split('\n')) {
          final String t = line.trim();
          expect(t.startsWith('//'), isFalse, reason: line);
        }
      }
    });

    test('in-game ui uses no material buttons', () {
      const List<String> banned = <String>[
        'ElevatedButton',
        'TextButton',
        'OutlinedButton',
        'IconButton',
        'FloatingActionButton',
        'MaterialButton',
      ];
      for (final String src in reachSources()) {
        for (final String word in banned) {
          expect(src.contains(word), isFalse, reason: 'found $word');
        }
      }
    });
  });

  group('pose source is the only platform boundary', () {
    test('a fake source feeds frames without a camera', () async {
      final Rig rig = makeBody();
      final FakePoseSource fake = FakePoseSource(
        frames: <PoseFrame>[
          PoseFrame(
            points: rig.points,
            visibility: rig.visibility,
            width: w,
            height: h,
            timestamp: 0.0,
          ),
        ],
      );
      expect(fake.supported, isTrue);
      expect(await fake.start(), isTrue);
      expect(fake.running, isTrue);
      expect(fake.latest, isNull);
      expect(fake.advance(), isNotNull);
      expect(fake.latest!.points.count, landmarkCount);
      await fake.stop();
      expect(fake.running, isFalse);
    });

    test('an unsupported source explains itself and never crashes', () async {
      final UnsupportedPoseSource none =
          UnsupportedPoseSource('tracking needs a phone');
      expect(none.supported, isFalse);
      expect(await none.start(), isFalse);
      expect(none.latest, isNull);
      expect(none.status, contains('phone'));
      none.dispose();
    });

    test('the game still runs with no pose at all', () {
      final ReachLogic logic = ReachLogic(width: w, height: h);
      logic.trackingNote = 'body tracking needs a phone';
      expect(logic.inMenu, isTrue);
      logic.begin();
      for (int i = 0; i < 40; i++) {
        logic.step(null, i * 0.05);
      }
      expect(logic.tracking, isFalse);
      expect(logic.engine.state.phase, running);
      expect(logic.engine.target, isNull);
      expect(logic.zones.has('stop'), isTrue);
    });

    test('a driven fake source plays the game end to end', () {
      final ReachLogic logic =
          ReachLogic(width: w, height: h, mode: timed, seed: 5);
      logic.begin();
      final Rig rig = makeBody();
      double t = 0.0;
      int hits = 0;
      for (int i = 0; i < 200; i++) {
        t += 0.05;
        final Points p = rig.points.copy();
        final Target? tg = logic.engine.target;
        if (tg != null) {
          p.set(idx['rightWrist']!, tg.x, tg.y);
        }
        logic.step(
            PoseFrame(
              points: p,
              visibility: rig.visibility,
              width: w,
              height: h,
              timestamp: t,
            ),
            t);
        hits = logic.engine.state.hits.length;
      }
      expect(logic.tracking, isTrue);
      expect(hits, greaterThan(0));
      expect(logic.engine.state.score, greaterThan(0));
    });

    test('the landmark slots line up with the shared name map', () {
      for (int i = 0; i < landmarkNames.length; i++) {
        expect(indexOfName(landmarkNames[i]), i);
      }
      expect(indexOfName('nonsense'), -1);
    });
  });

  group('audio constants match the python source', () {
    test('tones and gains', () {
      expect(tones, <double>[196, 220, 247, 262, 294, 330]);
      expect(bonusTone, 392.0);
      expect(endTone, 147.0);
      expect(hitGain, 0.34);
      expect(bonusGain, 0.30);
      expect(endGain, 0.28);
      expect(maxVoices, 8);
      expect(sampleRate, 44100);
    });

    test('a tone is audible, bounded and ends in silence', () {
      for (final double f in tones) {
        final List<double> mono = clickTone(freq: f, ms: hitMs, seed: 1);
        expect(mono.length, greaterThan(1000));
        double peak = 0;
        for (final double v in mono) {
          peak = math.max(peak, v.abs());
        }
        expect(peak, closeTo(1.0, 1e-6));
        expect(mono.first, 0.0);
        expect(mono.last, 0.0);
      }
    });

    test('the ending tone is the longest', () {
      final int hit = clickTone(freq: tones.first, ms: hitMs).length;
      final int reward = clickTone(freq: bonusTone, ms: hitMs * 1.6).length;
      final int end = clickTone(freq: endTone, ms: hitMs * 3.4).length;
      expect(reward, greaterThan(hit));
      expect(end, greaterThan(reward));
    });

    test('volume curve is silent at zero and rises to one', () {
      expect(volumeCurve(0), 0.0);
      expect(volumeCurve(100), closeTo(1.0, 1e-9));
      double last = -1;
      for (int v = 1; v <= 100; v++) {
        final double g = volumeCurve(v);
        expect(g, greaterThan(last));
        last = g;
      }
    });
  });

  group('the widget opens on every platform', () {
    testWidgets('desktop shows the menu and explains tracking',
        (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: ReachGame(
            sound: false,
            poseSource:
                UnsupportedPoseSource('tracking needs a phone'),
          ),
        ),
      ));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 32));
      expect(find.byType(ReachGame), findsOneWidget);
      expect(find.byType(CustomPaint), findsWidgets);
      expect(tester.takeException(), isNull);
    });

    testWidgets('a tap on begin starts the session',
        (WidgetTester tester) async {
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: ReachGame(
            sound: false,
            poseSource: FakePoseSource(),
          ),
        ),
      ));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 32));

      final _ReachProbe probe = _ReachProbe(tester);
      final Zone begin = probe.logic.zones['begin']!;
      await tester.tapAt(Offset(begin.cx, begin.cy));
      await tester.pump();
      expect(probe.logic.page, pagePlay);
      expect(tester.takeException(), isNull);
    });

    testWidgets('it survives a resize', (WidgetTester tester) async {
      tester.view.physicalSize = const Size(640, 480);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: ReachGame(sound: false, poseSource: FakePoseSource()),
        ),
      ));
      await tester.pump();
      tester.view.physicalSize = const Size(1920, 1080);
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 32));
      expect(tester.takeException(), isNull);
    });
  });
}

class _ReachProbe {
  _ReachProbe(WidgetTester tester) {
    final CustomPaint paint = tester
        .widgetList<CustomPaint>(find.byType(CustomPaint))
        .firstWhere((CustomPaint p) => p.painter is ReachPainter);
    logic = (paint.painter as ReachPainter).logic;
  }

  late final ReachLogic logic;
}
