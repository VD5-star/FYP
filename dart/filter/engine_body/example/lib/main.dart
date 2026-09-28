import 'dart:async';

import 'package:engine_body/engine_body.dart';
import 'package:flutter/foundation.dart' show ValueListenable;
import 'package:flutter/material.dart';

void main() => runApp(const BodyEngineDemo());

class BodyEngineDemo extends StatelessWidget {
  const BodyEngineDemo({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'PsyBot body engine',
    theme: ThemeData.dark(useMaterial3: true),
    home: const TrackerScreen(),
  );
}

class TrackerScreen extends StatefulWidget {
  const TrackerScreen({super.key});

  @override
  State<TrackerScreen> createState() => _TrackerScreenState();
}

class _TrackerScreenState extends State<TrackerScreen> {
  static const _autoStart = bool.fromEnvironment('AUTOSTART');

  final _tracker = BodyTracker();
  final _analyser = MovementAnalyser();
  final _smoother = PoseSmoother();

  final _frameNotifier = ValueNotifier<BodyFrame>(BodyFrame.empty);
  final _signalsNotifier = ValueNotifier<MovementSignals>(
    MovementSignals.unavailable,
  );

  StreamSubscription<BodyFrame>? _subscription;
  SessionRecorder? _recorder;
  Timer? _panelTimer;

  PreviewInfo? _preview;
  String? _error;
  bool _running = false;
  int _frameCount = 0;
  double _fps = 0;
  DateTime? _firstFrameAt;

  @override
  void initState() {
    super.initState();
    _log('app started, autoStart=$_autoStart supported=${BodyTracker.isSupported}');
    if (_autoStart && BodyTracker.isSupported) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _start());
    }
  }

  @override
  void dispose() {
    _panelTimer?.cancel();
    _subscription?.cancel();
    _tracker.stop();
    _frameNotifier.dispose();
    _signalsNotifier.dispose();
    super.dispose();
  }

  Future<void> _start() async {
    setState(() => _error = null);
    try {
      if (!await _tracker.hasPermission()) {
        _log('requesting camera permission');
        if (!await _tracker.requestPermission()) {
          _log('camera permission denied');
          if (mounted) {
            setState(() => _error = 'Camera permission is required');
          }
          return;
        }
      }

      final preview = await _tracker.start();
      _log('preview: $preview');

      _subscription = _tracker.frames.listen(
        _onFrame,
        onError: (Object e) {
          _log('STREAM ERROR: $e');
          if (mounted) setState(() => _error = '$e');
        },
      );

      _panelTimer = Timer.periodic(
        const Duration(milliseconds: 400),
        (_) => _signalsNotifier.value = _analyser.signals,
      );

      setState(() {
        _preview = preview;
        _running = true;
      });
    } on BodyTrackerException catch (e) {
      _log('START FAILED: ${e.error.name}: ${e.message}');
      setState(() => _error = e.message);
    }
  }

  Future<void> _stop() async {
    _panelTimer?.cancel();
    _panelTimer = null;
    await _subscription?.cancel();
    _subscription = null;
    await _tracker.stop();
    _smoother.reset();
    _analyser.reset();
    setState(() {
      _running = false;
      _preview = null;
    });
  }

  void _onFrame(BodyFrame rawFrame) {
    _analyser.add(rawFrame);

    if (!rawFrame.framing.framing.hasTorso) _smoother.reset();

    final frame = _smoother.smooth(rawFrame);
    _frameNotifier.value = frame;

    _frameCount++;
    _firstFrameAt ??= DateTime.now();
    final elapsed = DateTime.now().difference(_firstFrameAt!).inMilliseconds;
    if (elapsed > 0) _fps = _frameCount * 1000 / elapsed;

    if (_frameCount % 15 == 1) {
      final signals = _analyser.signals;
      _log(
        'frame $_frameCount  ${rawFrame.framing.framing.name}  '
        'fps=${_fps.toStringAsFixed(1)}  '
        'landmarks=${rawFrame.landmarks.length}  '
        'quality=${rawFrame.framing.upperBodyQuality.toStringAsFixed(2)}  '
        'energy=${signals.movementEnergy.toStringAsFixed(3)}  '
        'lElbow=${rawFrame.leftElbowAngle?.round()}  '
        'rElbow=${rawFrame.rightElbowAngle?.round()}',
      );
    }

    _recorder?.add(rawFrame);
  }

  void _log(String message) => debugPrint('[engine_body] $message');

  void _toggleRecording() => setState(() {
    if (_recorder != null) {
      _log('recording stopped, ${_recorder!.frameCount} frames');
      _recorder = null;
    } else {
      _recorder = SessionRecorder(sessionId: DateTime.now().toIso8601String());
      _log('recording started');
    }
  });

  @override
  Widget build(BuildContext context) {
    if (!BodyTracker.isSupported) return const _UnsupportedPlatformNotice();

    final preview = _preview;

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (preview != null)
                    BodyCameraPreview(
                      preview: preview,
                      overlay: ValueListenableBuilder<BodyFrame>(
                        valueListenable: _frameNotifier,
                        builder: (context, frame, _) => CustomPaint(
                          painter: SkeletonPainter(frame: frame),
                        ),
                      ),
                    )
                  else
                    const ColoredBox(color: Colors.black),

                  if (_running)
                    const Positioned(
                      top: 12,
                      right: 12,
                      child: _CameraIndicator(),
                    ),

                  Positioned(
                    bottom: 12,
                    left: 12,
                    right: 12,
                    child: ValueListenableBuilder<BodyFrame>(
                      valueListenable: _frameNotifier,
                      builder: (context, frame, _) {
                        final advice = frame.framing.advice;
                        if (advice == FramingAdvice.none) {
                          return const SizedBox.shrink();
                        }
                        return _AdviceBanner(text: framingAdviceText(advice));
                      },
                    ),
                  ),
                ],
              ),
            ),
            _SignalPanel(
              frames: _frameNotifier,
              signals: _signalsNotifier,
              error: _error,
              running: _running,
              recording: _recorder != null,
              onToggleRun: _running ? _stop : _start,
              onToggleRecord: _running ? _toggleRecording : null,
            ),
          ],
        ),
      ),
    );
  }
}

class _SignalPanel extends StatelessWidget {
  const _SignalPanel({
    required this.frames,
    required this.signals,
    required this.error,
    required this.running,
    required this.recording,
    required this.onToggleRun,
    required this.onToggleRecord,
  });

  final ValueListenable<BodyFrame> frames;
  final ValueListenable<MovementSignals> signals;
  final String? error;
  final bool running;
  final bool recording;
  final VoidCallback onToggleRun;
  final VoidCallback? onToggleRecord;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 12),
      color: const Color(0xFF101214),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (error != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                error!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ),
          ValueListenableBuilder<BodyFrame>(
            valueListenable: frames,
            builder: (context, frame, _) => Text(
              '${frame.framing.framing.name}   '
              'quality ${frame.framing.upperBodyQuality.toStringAsFixed(2)}',
              style: Theme.of(context).textTheme.labelLarge,
            ),
          ),
          const SizedBox(height: 6),
          ValueListenableBuilder<MovementSignals>(
            valueListenable: signals,
            builder: (context, s, _) => Wrap(
              spacing: 18,
              runSpacing: 2,
              children: [
                _Stat('energy', s.movementEnergy.toStringAsFixed(3)),
                _Stat('still', s.stillness.toStringAsFixed(2)),
                _Stat('lean', '${s.trunkLeanDegrees.toStringAsFixed(0)}°'),
                _Stat('conf', s.confidence.toStringAsFixed(2)),
                _Stat('n', '${s.sampleCount}'),
              ],
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              FilledButton.icon(
                onPressed: onToggleRun,
                icon: Icon(running ? Icons.stop : Icons.play_arrow),
                label: Text(running ? 'Stop' : 'Start'),
              ),
              const SizedBox(width: 12),
              OutlinedButton.icon(
                onPressed: onToggleRecord,
                icon: Icon(
                  recording ? Icons.stop_circle : Icons.fiber_manual_record,
                ),
                label: Text(recording ? 'Recording' : 'Record'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat(this.label, this.value);

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Text(
        '$label ',
        style: Theme.of(
          context,
        ).textTheme.bodySmall?.copyWith(color: Colors.white54),
      ),
      Text(value, style: Theme.of(context).textTheme.bodyMedium),
    ],
  );
}

class _CameraIndicator extends StatelessWidget {
  const _CameraIndicator();

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
    decoration: BoxDecoration(
      color: Colors.red.withValues(alpha: 0.85),
      borderRadius: BorderRadius.circular(20),
    ),
    child: const Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.videocam, size: 16),
        SizedBox(width: 6),
        Text('Camera on', style: TextStyle(fontSize: 12)),
      ],
    ),
  );
}

class _AdviceBanner extends StatelessWidget {
  const _AdviceBanner({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
    decoration: BoxDecoration(
      color: Colors.black.withValues(alpha: 0.7),
      borderRadius: BorderRadius.circular(12),
    ),
    child: Text(text, textAlign: TextAlign.center),
  );
}

class _UnsupportedPlatformNotice extends StatelessWidget {
  const _UnsupportedPlatformNotice();

  @override
  Widget build(BuildContext context) => const Scaffold(
    body: Center(
      child: Padding(
        padding: EdgeInsets.all(32),
        child: Text(
          'Live tracking requires Android.\n\n'
          'ML Kit has no Windows implementation, so the laptop cannot produce '
          'a skeleton from a camera. Use SessionReplay to analyse a session '
          'recorded on a device: the analysis code is identical.',
          textAlign: TextAlign.center,
        ),
      ),
    ),
  );
}
