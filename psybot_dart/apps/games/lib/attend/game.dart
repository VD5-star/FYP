import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import '../core/audio.dart';
import '../core/synth.dart';
import '../jigsaw/raster.dart';
import 'draw.dart';
import 'logic.dart';
import 'scenes.dart';
import 'sounds.dart';

class SceneBundle {
  SceneBundle(this.rasters);

  final Map<String, Raster> rasters;
}

class SoundBundle {
  SoundBundle(this.buffers);

  final Map<String, Float32List> buffers;
}

SceneBundle buildScenesJob(List<int> size) {
  final Map<String, Raster> out =
      renderAllScenes(w: size[0], h: size[1]);
  return SceneBundle(out);
}

SoundBundle buildSoundsJob(double seconds) {
  final Map<String, Float64List> raw =
      renderAllSounds(seconds: seconds);
  final Map<String, Float32List> out = <String, Float32List>{};
  raw.forEach((String name, Float64List mono) {
    final Float32List m = Float32List(mono.length);
    for (int i = 0; i < mono.length; i++) {
      m[i] = mono[i];
    }
    out[name] = toStereo(m, pan: soundPans[name] ?? 0.0, gain: 0.9);
  });
  return SoundBundle(out);
}

class AttendGame extends StatefulWidget {
  const AttendGame({super.key, this.seed, this.onExit});

  final int? seed;
  final VoidCallback? onExit;

  @override
  State<AttendGame> createState() => _AttendGameState();
}

class _AttendGameState extends State<AttendGame>
    with SingleTickerProviderStateMixin {
  late final AttendLogic logic;
  late final Ticker _ticker;
  final ValueNotifier<int> _frame = ValueNotifier<int>(0);
  final AudioEngine _audio = AudioEngine();
  final Map<String, ui.Image> _scenes = <String, ui.Image>{};
  Duration _last = Duration.zero;
  double _now = 0;
  bool _soundsReady = false;

  @override
  void initState() {
    super.initState();
    logic = AttendLogic(seed: widget.seed);
    _ticker = createTicker(_tick)..start();
    _boot();
  }

  Future<void> _boot() async {
    unawaited(_loadScenes());
    unawaited(_loadSounds());
  }

  Future<void> _loadScenes() async {
    final SceneBundle bundle = await Isolate.run(
        () => buildScenesJob(<int>[sceneW, sceneH]));
    for (final MapEntry<String, Raster> e in bundle.rasters.entries) {
      final ui.Image img =
          await imageFromRaster(e.value);
      _scenes[e.key] = img;
    }
    if (mounted) setState(_maybeReady);
  }

  Future<void> _loadSounds() async {
    await _audio.init();
    final SoundBundle bundle =
        await Isolate.run(() => buildSoundsJob(loopSeconds));
    for (final MapEntry<String, Float32List> e in bundle.buffers.entries) {
      await _audio.load(e.key, e.value);
    }
    _audio.setVolume(logic.volume);
    _soundsReady = true;
    if (mounted) setState(_maybeReady);
  }

  void _maybeReady() {
    logic.ready = _scenes.length == soundNames.length && _soundsReady;
  }

  void _tick(Duration elapsed) {
    final double dt = _last == Duration.zero
        ? 0.0
        : (elapsed - _last).inMicroseconds / 1e6;
    _last = elapsed;
    _now = elapsed.inMicroseconds / 1e6;
    logic.step(dt.clamp(0.0, 0.25), _now);
    _syncAudio();
    if (logic.quitting) {
      logic.quitting = false;
      widget.onExit?.call();
    }
    _frame.value++;
  }

  void _syncAudio() {
    if (!_audio.ready || !_soundsReady) return;
    if (_audio.volume != logic.volume) _audio.setVolume(logic.volume);
    if (logic.soundShouldPlay) {
      if (!_audio.looping) {
        for (final String n in soundNames) {
          unawaited(_audio.loop(n, volume: 1.0));
        }
      }
    } else if (_audio.looping) {
      _audio.stopLoops();
    }
  }

  @override
  void dispose() {
    _ticker.dispose();
    _frame.dispose();
    _audio.dispose();
    for (final ui.Image i in _scenes.values) {
      i.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (BuildContext context, BoxConstraints c) {
        if (logic.w != c.maxWidth || logic.h != c.maxHeight) {
          logic.resize(c.maxWidth, c.maxHeight);
        }
        return Listener(
          behavior: HitTestBehavior.opaque,
          onPointerDown: (PointerDownEvent e) =>
              logic.pointerDown(e.localPosition.dx, e.localPosition.dy),
          onPointerMove: (PointerMoveEvent e) =>
              logic.pointerMove(e.localPosition.dx, e.localPosition.dy),
          onPointerUp: (PointerUpEvent e) =>
              logic.pointerUp(e.localPosition.dx, e.localPosition.dy),
          child: CustomPaint(
            size: Size(c.maxWidth, c.maxHeight),
            painter: AttendPainter(
              logic: logic,
              now: _now,
              scenes: _scenes,
              repaint: _frame,
            ),
          ),
        );
      },
    );
  }
}

Future<ui.Image> imageFromRaster(Raster r) {
  final Completer<ui.Image> done = Completer<ui.Image>();
  ui.decodeImageFromPixels(
      r.rgba, r.w, r.h, ui.PixelFormat.rgba8888, done.complete);
  return done.future;
}
