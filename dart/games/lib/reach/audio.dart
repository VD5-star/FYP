import 'dart:math' as math;
import 'dart:typed_data';

import '../core/audio.dart';
import '../core/synth.dart';

const List<double> tones = <double>[196, 220, 247, 262, 294, 330];
const double bonusTone = 392.0;
const double endTone = 147.0;

const double hitGain = 0.34;
const double bonusGain = 0.30;
const double endGain = 0.28;
const double hitMs = 150.0;
const int maxVoices = 8;
const double attackMs = 9.0;

class ReachAudio {
  ReachAudio({AudioEngine? engine, int seed = 0})
      : engine = engine ?? AudioEngine(),
        _rng = math.Random(seed);

  final AudioEngine engine;
  final math.Random _rng;
  bool loaded = false;

  Future<void> load() async {
    if (loaded) return;
    await engine.init();
    for (int i = 0; i < tones.length; i++) {
      final Float32List mono = clickTone(
          freq: tones[i], ms: hitMs, seed: i, attackMs: attackMs);
      await engine.load('hit$i', toStereo(mono, gain: hitGain));
    }
    final Float32List reward = clickTone(
        freq: bonusTone, ms: hitMs * 1.6, seed: 71, attackMs: attackMs);
    await engine.load('bonus', toStereo(reward, gain: bonusGain));
    final Float32List soft = clickTone(
        freq: endTone, ms: hitMs * 3.4, seed: 91, attackMs: attackMs);
    await engine.load('end', toStereo(soft, gain: endGain));
    loaded = true;
  }

  String pickTone() => 'hit${_rng.nextInt(tones.length)}';

  Future<void> hit({double pan = 0.0}) =>
      engine.play(pickTone(), pan: pan.clamp(-1.0, 1.0));

  Future<void> reward({double pan = 0.0}) =>
      engine.play('bonus', pan: pan.clamp(-1.0, 1.0));

  Future<void> done() => engine.play('end');

  int setVolume(int v) => engine.setVolume(v);

  int get volume => engine.volume;

  bool get muted => engine.muted;

  String get levelName => engine.levelName;

  void dispose() => engine.dispose();
}
