import 'dart:convert';

import 'body_frame.dart';
import 'framing.dart';

class SessionRecorder {
  SessionRecorder({required this.sessionId, DateTime? startedAt})
    : startedAt = startedAt ?? DateTime.now();

  final String sessionId;
  final DateTime startedAt;

  final List<String> _lines = [];

  int get frameCount => _lines.length;

  void add(BodyFrame frame) => _lines.add(jsonEncode(frame.toMap()));

  String encode() {
    final header = jsonEncode({
      'schema': schemaVersion,
      'sessionId': sessionId,
      'startedAtMicros': startedAt.microsecondsSinceEpoch,
      'frameCount': _lines.length,
    });
    return [header, ..._lines].join('\n');
  }

  void clear() => _lines.clear();

  static const schemaVersion = 1;
}

class SessionReplay {
  const SessionReplay._({
    required this.sessionId,
    required this.startedAt,
    required this.frames,
  });

  final String sessionId;
  final DateTime startedAt;
  final List<BodyFrame> frames;

  static SessionReplay decode(
    String content, {
    FramingGate gate = const FramingGate(),
  }) {
    final lines = const LineSplitter().convert(content).where(
      (l) => l.trim().isNotEmpty,
    );
    if (lines.isEmpty) {
      throw const FormatException('Empty session recording');
    }

    final header = jsonDecode(lines.first);
    if (header is! Map<String, Object?>) {
      throw const FormatException('Session header is not an object');
    }

    final version = header['schema'];
    if (version != SessionRecorder.schemaVersion) {
      throw FormatException(
        'Unsupported session schema $version, '
        'expected ${SessionRecorder.schemaVersion}',
      );
    }

    final frames = <BodyFrame>[];
    for (final line in lines.skip(1)) {
      try {
        final decoded = jsonDecode(line);
        if (decoded is Map<String, Object?>) {
          frames.add(BodyFrame.fromMap(decoded, gate: gate));
        }
      } on FormatException {
        continue;
      }
    }

    final startedMicros = header['startedAtMicros'];
    return SessionReplay._(
      sessionId: header['sessionId'] as String? ?? 'unknown',
      startedAt: startedMicros is num
          ? DateTime.fromMicrosecondsSinceEpoch(startedMicros.toInt())
          : DateTime.fromMillisecondsSinceEpoch(0),
      frames: frames,
    );
  }

  Stream<BodyFrame> stream({bool realTime = false}) async* {
    DateTime? previous;
    for (final frame in frames) {
      if (realTime && previous != null) {
        final gap = frame.timestamp.difference(previous);
        if (gap > Duration.zero) await Future<void>.delayed(gap);
      }
      previous = frame.timestamp;
      yield frame;
    }
  }

  double? get measuredFrameRate {
    if (frames.length < 2) return null;
    final span = frames.last.timestamp
        .difference(frames.first.timestamp)
        .inMicroseconds;
    if (span <= 0) return null;
    return (frames.length - 1) / (span / 1e6);
  }
}
