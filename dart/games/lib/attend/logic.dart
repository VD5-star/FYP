import 'dart:math' as math;

import '../core/paint.dart';
import '../core/synth.dart';
import '../core/zones.dart';
import 'session.dart';

const String pageMenu = 'menu';
const String pageRun = 'run';
const String pageOver = 'over';
const String pageSettings = 'settings';

const double overFade = 1.4;

class AttendLogic {
  AttendLogic({
    this.w = 1100,
    this.h = 700,
    int minutes = defaultMinutes,
    this.seed,
    int volume = volDefault,
  })  : minutes = clampMinutes(minutes),
        _volume = volume.clamp(volMin, volMax) {
    menuZones();
  }

  double w;
  double h;
  int minutes;
  final int? seed;
  int _volume;

  final Zones zones = Zones();
  String page = pageMenu;
  String prevPage = pageMenu;
  Session? sess;
  bool ready = false;
  bool quitting = false;
  double overAt = 0.0;
  bool sliding = false;
  String? pressKey;
  double pressX = 0;
  double pressY = 0;

  int get volume => _volume;
  bool get muted => _volume <= volMin;
  bool get paused =>
      page == pageRun && sess != null && sess!.paused;
  bool get inStopMenu => page == pageRun && zones.has('panel');
  bool get soundShouldPlay =>
      page == pageRun && !paused && !muted && sess != null;

  void resize(double width, double height) {
    w = width;
    h = height;
    rebuild();
  }

  void rebuild() {
    if (page == pageMenu) {
      menuZones();
    } else if (page == pageSettings) {
      settingsZones();
    } else if (page == pageOver) {
      overZones();
    } else if (inStopMenu) {
      pauseZones();
    } else {
      runZones();
    }
  }

  void menuZones() {
    zones.clear();
    final double sw = math.min(460.0, w - 120);
    final double y = h * 0.42;
    zones.add('minutes', Zone((w - sw) / 2, y, sw, 56));
    zones.add('begin', Zone((w - 220) / 2, y + 128, 220, 62));
    zones.add('settings', Zone(w - 158, 28, 130, 46));
  }

  void runZones() {
    zones.clear();
    zones.add('open', stopZone(w, h));
  }

  void pauseZones() {
    zones.clear();
    final PauseLayout p = PauseLayout(w, h);
    zones.add('panel', p.panel);
    zones.add('volume', p.volume);
    zones.add('resume', p.topLeft);
    zones.add('again', p.topRight);
    zones.add('home', p.bottomLeft);
    zones.add('quit', p.bottomRight);
  }

  void settingsZones() {
    zones.clear();
    final double pw = math.min(520.0, w - 80);
    const double ph = 260;
    final double px = (w - pw) / 2;
    final double py = (h - ph) / 2;
    zones.add('panel', Zone(px, py, pw, ph));
    zones.add('volume', Zone(px + 30, py + 112, pw - 60, 56));
    zones.add('close', Zone(w / 2 - 85, py + ph - 76, 170, 54));
  }

  void overZones() {
    zones.clear();
    zones.add('again', Zone(w / 2 - 190, h / 2 + 30, 170, 56));
    zones.add('stop', Zone(w / 2 + 20, h / 2 + 30, 170, 56));
  }

  List<PauseButton> pauseButtons() {
    return <PauseButton>[
      PauseButton('resume', 'keep going', zones['resume']!, lead: true),
      PauseButton('again', 'start over', zones['again']!),
      PauseButton('home', 'back', zones['home']!),
      PauseButton('quit', 'close', zones['quit']!),
    ];
  }

  void begin() {
    if (!ready) return;
    sess = Session(minutes: minutes, seed: seed);
    page = pageRun;
    overAt = 0.0;
    sliding = false;
    runZones();
  }

  void toMenu() {
    sess = null;
    page = pageMenu;
    overAt = 0.0;
    sliding = false;
    menuZones();
  }

  void openPause() {
    if (page != pageRun || sess == null || sess!.finished) return;
    if (!sess!.paused) sess!.togglePause();
    sliding = false;
    pauseZones();
  }

  void closePause() {
    if (sess != null && sess!.paused) sess!.togglePause();
    sliding = false;
    runZones();
  }

  void openSettings() {
    if (page == pageSettings) return;
    prevPage = page;
    page = pageSettings;
    settingsZones();
  }

  void closeSettings() {
    sliding = false;
    page = prevPage;
    if (page == pageOver) {
      overZones();
    } else {
      menuZones();
    }
  }

  void setMinutes(double x) {
    final Zone? r = zones['minutes'];
    if (r == null) return;
    minutes = sliderValue(r, x, lo: minMinutes, hi: maxMinutes);
  }

  void setVolume(double x) {
    final Zone? r = zones['volume'];
    if (r == null) return;
    _volume = sliderValue(r, x, lo: volMin, hi: volMax);
  }

  void setVolumeValue(int v) {
    _volume = v.clamp(volMin, volMax);
  }

  void tap(String key) {
    if (page == pageSettings) {
      if (key == 'close') closeSettings();
      return;
    }
    if (page == pageMenu) {
      if (key == 'begin') {
        begin();
      } else if (key == 'settings') {
        openSettings();
      }
      return;
    }
    if (page == pageRun) {
      if (key == 'open') {
        openPause();
      } else if (key == 'resume') {
        closePause();
      } else if (key == 'again') {
        begin();
      } else if (key == 'home') {
        toMenu();
      } else if (key == 'quit') {
        quitting = true;
      }
      return;
    }
    if (key == 'again') {
      begin();
    } else if (key == 'stop') {
      toMenu();
    }
  }

  void pointerDown(double x, double y) {
    pressX = x;
    pressY = y;
    pressKey = zones.at(x, y);
    if (pressKey == 'volume') {
      sliding = true;
      setVolume(x);
    } else if (pressKey == 'minutes' && page == pageMenu) {
      sliding = true;
      setMinutes(x);
    }
  }

  void pointerMove(double x, double y) {
    if (!sliding) return;
    if (pressKey == 'volume') {
      setVolume(x);
    } else if (pressKey == 'minutes') {
      setMinutes(x);
    }
  }

  void pointerUp(double x, double y) {
    if (sliding) {
      if (pressKey == 'volume') {
        setVolume(x);
      } else if (pressKey == 'minutes') {
        setMinutes(x);
      }
      sliding = false;
      pressKey = null;
      return;
    }
    final double moved = (x - pressX).abs() + (y - pressY).abs();
    final String? key = zones.at(x, y);
    if (moved <= tapSlop &&
        key != null &&
        key == pressKey &&
        key != 'panel') {
      tap(key);
    }
    pressKey = null;
  }

  void step(double dt, double now) {
    if (page != pageRun || sess == null) return;
    if (inStopMenu) return;
    sess!.advance(dt);
    if (sess!.finished && page != pageOver) {
      page = pageOver;
      overAt = now;
      overZones();
    }
  }

  ShapeBand get shape => shapeFor(minutes);
}
