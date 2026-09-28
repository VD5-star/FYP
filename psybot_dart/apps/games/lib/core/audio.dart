import 'dart:typed_data';

import 'package:flutter_soloud/flutter_soloud.dart';

import 'synth.dart';

class AudioEngine {
  AudioEngine({int volume = volDefault}) : _volume = volume.clamp(volMin, volMax);

  final SoLoud _soloud = SoLoud.instance;
  final Map<String, AudioSource> _sources = <String, AudioSource>{};
  final Map<String, SoundHandle> _loops = <String, SoundHandle>{};
  int _volume;
  bool _ready = false;
  bool _failed = false;
  int _seq = 0;

  bool get ready => _ready;
  bool get failed => _failed;
  int get volume => _volume;
  bool get muted => _failed || _volume <= volMin;
  String get levelName => _volume <= volMin ? 'off' : '$_volume';

  Future<bool> init() async {
    if (_ready) return true;
    try {
      await _soloud.init(sampleRate: sampleRate, channels: Channels.stereo);
      _soloud.setGlobalVolume(volumeCurve(_volume));
      _ready = true;
    } catch (_) {
      _failed = true;
      _ready = false;
    }
    return _ready;
  }

  
  Future<void> load(String name, Float32List stereo) async {
    if (!_ready) return;
    try {
      final Uint8List wav = wavFromFloat(stereo);
      _sources[name] = await _soloud.loadMem('$name.${_seq++}.wav', wav);
    } catch (_) {
      
    }
  }

  bool has(String name) => _sources.containsKey(name);

  Future<void> play(String name, {double volume = 1.0, double pan = 0.0}) async {
    if (!_ready || muted) return;
    final AudioSource? s = _sources[name];
    if (s == null) return;
    try {
      _soloud.play(s, volume: volume, pan: pan);
    } catch (_) {
      
    }
  }

  Future<void> loop(String name, {double volume = 1.0}) async {
    if (!_ready || muted) return;
    final AudioSource? s = _sources[name];
    if (s == null || _loops.containsKey(name)) return;
    try {
      _loops[name] = _soloud.play(s, volume: volume, looping: true);
    } catch (_) {
      
    }
  }

  void setLoopVolume(String name, double v) {
    final SoundHandle? h = _loops[name];
    if (h == null || !_ready) return;
    try {
      _soloud.setVolume(h, v);
    } catch (_) {
      
    }
  }

  void stopLoops() {
    if (!_ready) return;
    for (final SoundHandle h in _loops.values) {
      try {
        _soloud.stop(h);
      } catch (_) {
        
      }
    }
    _loops.clear();
  }

  bool get looping => _loops.isNotEmpty;

  int setVolume(int v) {
    _volume = v.clamp(volMin, volMax);
    if (_ready) {
      _soloud.setGlobalVolume(volumeCurve(_volume));
    }
    if (muted) stopLoops();
    return _volume;
  }

  void dispose() {
    stopLoops();
    if (_ready) {
      try {
        _soloud.deinit();
      } catch (_) {
        
      }
    }
    _ready = false;
  }
}
