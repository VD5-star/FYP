import 'dart:convert';

import 'rep_counter.dart';

class UserProfile {
  UserProfile({Map<String, CalibratedThresholds>? exercises})
      : exercises = exercises ?? {};

  final Map<String, CalibratedThresholds> exercises;

  bool store(String exercise, HysteresisConfig? config,
      {double? observedMin, double? observedMax}) {
    if (config == null) return false;
    exercises[exercise] = CalibratedThresholds(
      downBelow: config.downBelow,
      upAbove: config.upAbove,
      observedMin: observedMin,
      observedMax: observedMax,
    );
    return true;
  }

  HysteresisConfig configFor(String exercise, HysteresisConfig fallback) {
    final stored = exercises[exercise];
    if (stored == null) return fallback;
    try {
      return HysteresisConfig(
        name: exercise,
        downBelow: stored.downBelow,
        upAbove: stored.upAbove,
        confirmFrames: fallback.confirmFrames,
        minDwell: fallback.minDwell,
      );
    } on ArgumentError {
      return fallback;
    }
  }

  String toJson() => jsonEncode({
        'exercises': {
          for (final entry in exercises.entries) entry.key: entry.value.toMap(),
        },
      });

  static UserProfile fromJson(String? source) {
    if (source == null || source.isEmpty) return UserProfile();
    try {
      final decoded = jsonDecode(source);
      if (decoded is! Map) return UserProfile();
      final raw = decoded['exercises'];
      if (raw is! Map) return UserProfile();

      final exercises = <String, CalibratedThresholds>{};
      raw.forEach((key, value) {
        if (key is! String || value is! Map) return;
        final thresholds = CalibratedThresholds.fromMap(value);
        if (thresholds != null) exercises[key] = thresholds;
      });
      return UserProfile(exercises: exercises);
    } on FormatException {
      return UserProfile();
    }
  }
}

class CalibratedThresholds {
  const CalibratedThresholds({
    required this.downBelow,
    required this.upAbove,
    this.observedMin,
    this.observedMax,
  });

  final double downBelow;
  final double upAbove;

  final double? observedMin;
  final double? observedMax;

  Map<String, dynamic> toMap() => {
        'downBelow': downBelow,
        'upAbove': upAbove,
        if (observedMin != null) 'observedMin': observedMin,
        if (observedMax != null) 'observedMax': observedMax,
      };

  static CalibratedThresholds? fromMap(Map<dynamic, dynamic> map) {
    final down = map['downBelow'];
    final up = map['upAbove'];
    if (down is! num || up is! num) return null;
    if (!down.toDouble().isFinite || !up.toDouble().isFinite) return null;

    final min = map['observedMin'];
    final max = map['observedMax'];
    return CalibratedThresholds(
      downBelow: down.toDouble(),
      upAbove: up.toDouble(),
      observedMin: min is num ? min.toDouble() : null,
      observedMax: max is num ? max.toDouble() : null,
    );
  }
}
