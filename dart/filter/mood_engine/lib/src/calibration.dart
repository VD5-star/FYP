import 'baseline.dart';
import 'maths.dart';

class CalibrationStep {
  const CalibrationStep({
    required this.key,
    required this.frames,
    required this.en,
    required this.ar,
  });

  final String key;
  final int frames;
  final String en;
  final String ar;
}

const List<CalibrationStep> calibrationSteps = <CalibrationStep>[
  CalibrationStep(
    key: 'rest',
    frames: 340,
    en: 'Look at the camera and relax your face',
    ar: 'انظر إلى الكاميرا وأرخِ ملامحك',
  ),
  CalibrationStep(
    key: 'smile',
    frames: 170,
    en: 'Smile naturally, then relax',
    ar: 'ابتسم بطبيعية ثم استرخِ',
  ),
  CalibrationStep(
    key: 'brows',
    frames: 150,
    en: 'Raise your eyebrows, then relax',
    ar: 'ارفع حاجبيك ثم استرخِ',
  ),
  CalibrationStep(
    key: 'turn',
    frames: 170,
    en: 'Turn your head slowly left, then right',
    ar: 'أدِر رأسك ببطء يساراً ثم يميناً',
  ),
  CalibrationStep(
    key: 'settle',
    frames: 170,
    en: 'Relax again and look ahead',
    ar: 'استرخِ مجدداً وانظر للأمام',
  ),
];

final int totalFrames = calibrationSteps.fold<int>(
  0,
  (int sum, CalibrationStep step) => sum + step.frames,
);

final int restFrames = calibrationSteps
    .where((CalibrationStep step) =>
        step.key == 'rest' || step.key == 'settle')
    .fold<int>(0, (int sum, CalibrationStep step) => sum + step.frames);

const String rejectNoFace = 'no_face';
const String rejectQuality = 'low_quality';
const String rejectAngle = 'extreme_angle';

class CalibrationSession {
  CalibrationSession({this.personId, this.personName, double? startedAt})
    : startedAt = startedAt ?? _nowSeconds();

  static const double minQuality = 0.35;

  static const double maxYaw = 32.0;
  static const double maxPitch = 26.0;

  final int? personId;
  final String? personName;
  final double startedAt;

  int stepIndex = 0;
  final Map<String, int> accepted = <String, int>{};
  final Map<String, int> rejected = <String, int>{};
  double? finishedAt;

  static double _nowSeconds() =>
      DateTime.now().millisecondsSinceEpoch / 1000.0;

  CalibrationStep? get step {
    if (stepIndex >= calibrationSteps.length) return null;
    return calibrationSteps[stepIndex];
  }

  bool get complete => stepIndex >= calibrationSteps.length;

  int get totalAccepted {
    var total = 0;
    for (final value in accepted.values) {
      total += value;
    }
    return total;
  }

  double get progress {
    final denominator = totalFrames > 1 ? totalFrames : 1;
    final value = totalAccepted / denominator;
    return value < 1.0 ? value : 1.0;
  }

  double stepProgress() {
    final current = step;
    if (current == null) return 1.0;
    final got = accepted[current.key] ?? 0;
    final denominator = current.frames > 1 ? current.frames : 1;
    final value = got / denominator;
    return value < 1.0 ? value : 1.0;
  }

  String? offer({
    required bool hasFace,
    required double quality,
    double yaw = 0.0,
    double pitch = 0.0,
  }) {
    if (complete) return null;

    final current = step!;

    if (!hasFace) return _reject(rejectNoFace);
    if (quality < minQuality) return _reject(rejectQuality);
    if (current.key != 'turn') {
      if (yaw.abs() > maxYaw || pitch.abs() > maxPitch) {
        return _reject(rejectAngle);
      }
    }

    final key = current.key;
    accepted[key] = (accepted[key] ?? 0) + 1;
    if (accepted[key]! >= current.frames) {
      stepIndex += 1;
      if (complete) finishedAt = _nowSeconds();
    }
    return null;
  }

  String _reject(String reason) {
    rejected[reason] = (rejected[reason] ?? 0) + 1;
    return reason;
  }

  String? hint() {
    if (rejected.isEmpty) return null;
    var rejectedTotal = 0;
    for (final value in rejected.values) {
      rejectedTotal += value;
    }
    final total = rejectedTotal + totalAccepted;
    if (total < 30) return null;
    final top = rankedByCount(rejected).first;
    final denominator = total > 1 ? total : 1;
    return top.value / denominator > 0.25 ? top.key : null;
  }

  Map<String, Object?> toMap({double? now}) {
    final current = step;
    final stamp = now ?? _nowSeconds();
    return <String, Object?>{
      'active': !complete,
      'person_id': personId,
      'person_name': personName,
      'step_index': stepIndex,
      'step_count': calibrationSteps.length,
      'step_key': current?.key,
      'step_en': current?.en,
      'step_ar': current?.ar,
      'step_progress': roundTo(stepProgress(), 4),
      'progress': roundTo(progress, 4),
      'accepted': totalAccepted,
      'required': totalFrames,
      'elapsed_s': roundTo(stamp - startedAt, 1),
      'hint': hint(),
      'complete': complete,
    };
  }
}

class CalibrationManager {
  CalibrationManager({this.enabled = true});

  static const int identitySwitchFrames = 45;

  final bool enabled;

  CalibrationSession? session;

  final Set<int?> _skipped = <int?>{};
  int? _pendingPerson;
  int _pendingFrames = 0;

  bool neededFor(PersonBaseline? baseline) {
    if (!enabled) return false;
    if (baseline == null) return true;
    return !baseline.ready || baseline.expired;
  }

  CalibrationSession begin(int? personId, [String? personName]) {
    final created = CalibrationSession(
      personId: personId,
      personName: personName,
    );
    session = created;
    return created;
  }

  CalibrationSession? ensure(
    int? personId,
    String? personName,
    PersonBaseline? baseline,
  ) {
    if (_skipped.contains(personId) || !neededFor(baseline)) return null;

    final active = session;
    if (active != null && !active.complete) {
      if (active.personId == personId) {
        _pendingPerson = null;
        _pendingFrames = 0;
        return active;
      }

      if (personId == _pendingPerson) {
        _pendingFrames += 1;
      } else {
        _pendingPerson = personId;
        _pendingFrames = 1;
      }

      if (_pendingFrames < identitySwitchFrames) return active;

      if (personId == null) {
        _pendingPerson = null;
        _pendingFrames = 0;
        return active;
      }

      _pendingPerson = null;
      _pendingFrames = 0;
      return begin(personId, personName);
    }

    _pendingPerson = null;
    _pendingFrames = 0;
    return begin(personId, personName);
  }

  void skip() {
    final active = session;
    if (active != null) _skipped.add(active.personId);
    session = null;
  }

  void clear() => session = null;

  Map<String, Object?> toMap() {
    final active = session;
    if (active == null) return <String, Object?>{'active': false};
    return active.toMap();
  }
}

void assertCalibrationCanFinish() {
  if (PersonBaseline.required > restFrames) {
    throw StateError(
      'PersonBaseline.required (${PersonBaseline.required}) exceeds the '
      '$restFrames resting frames guided calibration collects, so it '
      'could never finish. Lower required or lengthen the rest steps.',
    );
  }
}

bool restFramesOnly(CalibrationSession? session, List<double> probs) {
  if (session == null || session.complete) return true;
  final step = session.step;
  return step != null && (step.key == 'rest' || step.key == 'settle');
}
