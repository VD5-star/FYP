import 'dart:math' as math;
import 'dart:typed_data';

import '../core/audio.dart';
import '../core/synth.dart';

const List<double> tones = <double>[196, 220, 247, 262, 294, 330];
const double doneTone = 147.0;
const double clickGain = 0.34;
const double clickMs = 150.0;
const double doneGain = 0.30;
const int maxVoices = 8;

class JigsawAudio {
  JigsawAudio({AudioEngine? engine, int seed = 0})
      : engine = engine ?? AudioEngine(),
        _rng = math.Random(seed);

  final AudioEngine engine;
  final math.Random _rng;
  bool loaded = false;

  Future<void> load() async {
    if (loaded) return;
    await engine.init();
    for (int i = 0; i < tones.length; i++) {
      final Float32List mono =
          clickTone(freq: tones[i], ms: clickMs, seed: i);
      await engine.load('tone$i', toStereo(mono, gain: clickGain));
    }
    final Float32List soft =
        clickTone(freq: doneTone, ms: clickMs * 3.4, seed: 91);
    await engine.load('done', toStereo(soft, gain: doneGain));
    loaded = true;
  }

  String toneFor(double doneFrac) {
    final double f = doneFrac.clamp(0.0, 1.0);
    int i = (f * tones.length).floor();
    if (i >= tones.length) i = tones.length - 1;
    if (i < 0) i = 0;
    if (_rng.nextDouble() < 0.25) {
      i = (i + 1).clamp(0, tones.length - 1);
    }
    return 'tone$i';
  }

  Future<void> place({double doneFrac = 0.0, double pan = 0.0}) =>
      engine.play(toneFor(doneFrac), pan: pan.clamp(-1.0, 1.0));

  Future<void> done() => engine.play('done');

  int setVolume(int v) => engine.setVolume(v);

  int get volume => engine.volume;

  bool get muted => engine.muted;

  String get soundName =>
      engine.failed ? 'no sound' : 'sound ${engine.levelName}';

  void dispose() => engine.dispose();
}
