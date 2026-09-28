import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import '../core/adaptive.dart';
import '../core/hardware_volume.dart';
import 'audio.dart';
import 'draw.dart';
import 'engine.dart';
import 'logic.dart';
import 'mlkit_pose.dart';
import 'pose_source.dart';

const String desktopNote =
    'body tracking needs a phone, this screen has no camera pose';

class ReachGame extends StatefulWidget {
  const ReachGame({
    super.key,
    this.seed,
    this.sound = true,
    this.poseSource,
    this.onClose,
  });

  final int? seed;
  final bool sound;
  final PoseSource? poseSource;
  final VoidCallback? onClose;

  @override
  State<ReachGame> createState() => _ReachGameState();
}

class _ReachGameState extends State<ReachGame>
    with SingleTickerProviderStateMixin {
  late final ReachLogic logic;
  late final PoseSource pose;
  late final Ticker _ticker;
  final ValueNotifier<int> _tick = ValueNotifier<int>(0);

  ReachAudio? audio;
  double _last = 0.0;
  Size _size = Size.zero;

  @override
  void initState() {
    super.initState();
    pose = widget.poseSource ?? createPoseSource();
    logic = ReachLogic(seed: widget.seed);
    logic.trackingNote = pose.supported
        ? 'stand where the camera can see all of you'
        : desktopNote;
    logic.onVolume = _applyVolume;
    logic.onHitSound = _playHit;
    logic.onDoneSound = _playDone;
    logic.onClose = _handleClose;
    logic.onPickCamera = _pickCamera;
    _ticker = createTicker(_frame)..start();
    if (widget.sound) _initAudio();
    _startPose();
  }

  void _applyVolume(int v) {
    audio?.setVolume(v);
    if (mounted) setState(() {});
  }

  void _hardwareVolume(int delta) {
    final int next = (logic.volume + delta).clamp(0, 100);
    logic.setVolume(next);
    _applyVolume(next);
  }

  void _playHit(Hit hit) {
    final ReachAudio? a = audio;
    if (a == null) return;
    if (hit.bonus) {
      a.reward();
    } else {
      a.hit();
    }
  }

  void _playDone() {
    audio?.done();
  }

  void _handleClose() {
    final VoidCallback? f = widget.onClose;
    if (f != null) f();
  }

  Future<void> _initAudio() async {
    final ReachAudio a = ReachAudio(seed: widget.seed ?? 0);
    await a.load();
    a.setVolume(logic.volume);
    if (!mounted) {
      a.dispose();
      return;
    }
    setState(() => audio = a);
  }

  Future<void> _startPose() async {
    if (!pose.supported) return;
    final bool ok = await pose.start();
    if (!mounted) return;
    setState(() {
      if (!ok) logic.trackingNote = pose.status;
      logic.setCameras(pose.cameras, pose.cameraIndex);
    });
  }

  Future<bool> _pickCamera(int index) async {
    final bool ok = await pose.useCamera(index);
    if (!mounted) return ok;
    setState(() {
      if (!ok) logic.trackingNote = pose.status;
    });
    return ok;
  }

  @override
  void dispose() {
    _ticker.dispose();
    _tick.dispose();
    audio?.dispose();
    if (widget.poseSource == null) pose.dispose();
    super.dispose();
  }

  void _frame(Duration d) {
    final double now = d.inMicroseconds / 1000000.0;
    _last = now;
    final String before = logic.engine.state.phase;
    final bool wasMenu = logic.inMenu;
    logic.step(pose.latest, now);
    if (logic.engine.state.phase != before || logic.inMenu != wasMenu) {
      setState(() {});
    }
    _tick.value = _tick.value + 1;
  }

  void _down(PointerDownEvent e) {
    setState(() =>
        logic.pointerDown(e.localPosition.dx, e.localPosition.dy));
  }

  void _move(PointerMoveEvent e) {
    logic.pointerMove(e.localPosition.dx, e.localPosition.dy);
  }

  void _up(PointerUpEvent e) {
    setState(() => logic.pointerUp(e.localPosition.dx, e.localPosition.dy));
  }

  void _cancel(PointerCancelEvent e) {
    logic.pointerUp(e.localPosition.dx, e.localPosition.dy);
  }

  @override
  Widget build(BuildContext context) {
    return HardwareVolume(
      onVolume: _hardwareVolume,
      child: LayoutBuilder(
        builder: (BuildContext context, BoxConstraints c) {
          final Size s = Size(c.maxWidth, c.maxHeight);
          if (s != _size && s.width > 0 && s.height > 0) {
            _size = s;
            WidgetsBinding.instance.addPostFrameCallback((_) {
              if (!mounted) return;
              setState(() => logic.resize(s.width, s.height));
            });
          }
          final double pad = Adaptive.safePad(context).top;
          return MediaQuery(
            data: MediaQuery.of(context).copyWith(
              textScaler: TextScaler.linear(
                  (Adaptive.scale(context) * 0.95).clamp(0.85, 1.15)),
            ),
            child: Padding(
              padding: EdgeInsets.only(top: pad > 0 ? 0 : 0),
              child: Listener(
                behavior: HitTestBehavior.opaque,
                onPointerDown: _down,
                onPointerMove: _move,
                onPointerUp: _up,
                onPointerCancel: _cancel,
                child: CustomPaint(
                  size: s,
                  painter: ReachPainter(
                    logic: logic,
                    now: _last,
                    cameraReady: pose.supported,
                    repaint: _tick,
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class ReachApp extends StatelessWidget {
  const ReachApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'reach',
      debugShowCheckedModeBanner: false,
      home: const Scaffold(
        backgroundColor: reachBg,
        body: ReachGame(),
      ),
    );
  }
}

const Color reachBg = Color(0xFF1D1F22);
