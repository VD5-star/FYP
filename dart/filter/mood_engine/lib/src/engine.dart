import 'affect.dart';
import 'attributes.dart';
import 'baseline.dart';
import 'calibration.dart';
import 'config.dart';
import 'database.dart';
import 'detector.dart';
import 'emotion.dart';
import 'geometry.dart';
import 'maths.dart';
import 'recognizer.dart';
import 'session.dart';

class FrameResult {
  FrameResult({
    required this.ok,
    required this.frameIndex,
    required this.timestamp,
    required this.elapsedMs,
    this.faceCount = 0,
    this.bbox,
    this.detScore = 0.0,
    this.personId,
    this.personName,
    this.isNewPerson = false,
    this.isProvisional = false,
    this.matchScore = 0.0,
    this.matchUncertain = false,
    this.emotion,
    this.pose,
    this.spoof,
    this.affect,
    this.mood,
    this.age,
    this.gender,
    this.quality = 0.0,
    this.observationId,
    this.message = '',
  });

  final bool ok;
  final int frameIndex;
  final double timestamp;
  final double elapsedMs;
  final int faceCount;
  final List<int>? bbox;
  final double detScore;
  final int? personId;
  final String? personName;
  final bool isNewPerson;
  final bool isProvisional;
  final double matchScore;
  final bool matchUncertain;
  final EmotionResult? emotion;
  final PoseResult? pose;
  final SpoofResult? spoof;
  final AffectReading? affect;
  final MoodSummary? mood;
  final double? age;
  final String? gender;
  final double quality;
  int? observationId;
  final String message;

  Map<String, Object?> toMap() => <String, Object?>{
    'ok': ok,
    'frame_index': frameIndex,
    'timestamp': timestamp,
    'elapsed_ms': roundTo(elapsedMs, 2),
    'face_count': faceCount,
    'bbox': bbox,
    'det_score': roundTo(detScore, 4),
    'person': <String, Object?>{
      'id': personId,
      'name': personName,
      'is_new': isNewPerson,
      'is_provisional': isProvisional,
      'match_score': roundTo(matchScore, 4),
      'uncertain': matchUncertain,
    },
    'emotion': emotion?.toMap(),
    'pose': pose?.toMap(),
    'spoof': spoof?.toMap(),
    'affect': affect?.toMap(),
    'mood': mood?.toMap(),
    'age': age,
    'gender': gender,
    'quality': roundTo(quality, 4),
    'observation_id': observationId,
    'message': message,
  };
}

class FrameInput {
  const FrameInput({
    required this.width,
    required this.height,
    this.faces = const <DetectedFace>[],
    this.crop,
    this.spoofSignals = const <String, double>{},
    this.sharpness = 0.0,
    this.now,
  });

  final int width;
  final int height;
  final List<DetectedFace> faces;
  final FaceImage? crop;
  final Map<String, double> spoofSignals;
  final double sharpness;
  final double? now;
}

class MoodEngine {
  MoodEngine({
    required this.db,
    EngineConfig? engineConfig,
    EmotionModel? emotionModel,
    this.autoEnrolUnknown = true,
  }) : config = engineConfig ?? defaultEngineConfig,
       detector = FaceDetector(
         config: (engineConfig ?? defaultEngineConfig).detection,
         tracking: (engineConfig ?? defaultEngineConfig).tracking,
       ),
       recognizer = FaceRecognizer(
         db,
         config: (engineConfig ?? defaultEngineConfig).recognition,
       ),
       emotion = EmotionAnalyzer(
         config: (engineConfig ?? defaultEngineConfig).emotion,
         model: emotionModel,
       ),
       pose = PoseEstimator(),
       spoof = SpoofDetector(
         config: (engineConfig ?? defaultEngineConfig).spoof,
       ),
       affect = AffectAnalyzer(),
       baseline = BaselineTracker(),
       calibration = CalibrationManager(
         enabled: (engineConfig ?? defaultEngineConfig).guidedCalibration,
       ),
       session = SessionAnalyser(
         periodS: (engineConfig ?? defaultEngineConfig).sessionPeriodS,
       );

  final MoodStore db;
  final EngineConfig config;
  final FaceDetector detector;
  final FaceRecognizer recognizer;
  final EmotionAnalyzer emotion;
  final PoseEstimator pose;
  final SpoofDetector spoof;
  final AffectAnalyzer affect;
  final BaselineTracker baseline;
  final CalibrationManager calibration;
  final SessionAnalyser session;

  final bool autoEnrolUnknown;

  final List<SessionReport> recentReports = <SessionReport>[];

  int frameIndex = 0;
  int? sessionId;
  int? _currentPerson;
  bool _loaded = false;

  int? get currentPerson => _currentPerson;

  Future<void> load() async {
    if (_loaded) return;
    try {
      final baselines = await db.allBaselines();
      for (final entry in baselines.entries) {
        baseline.load(entry.key, entry.value);
      }
    } on Object {
      _loaded = true;
    }
    _loaded = true;
  }

  Future<int> startSession([String? note]) async {
    sessionId = await db.startSession(note);
    frameIndex = 0;
    return sessionId!;
  }

  Future<void> endSession() async {
    final id = sessionId;
    if (id != null) {
      await db.endSession(id, frameCount: frameIndex);
      sessionId = null;
    }
  }

  void reset() {
    detector.resetTracking();
    emotion.reset();
    affect.reset();
    baseline.reset();
    pose.reset();
    _currentPerson = null;
  }

  Future<FrameResult> analyseFrame(
    FrameInput input, {
    bool persist = true,
  }) async {
    await load();
    final started = DateTime.now().microsecondsSinceEpoch;
    frameIndex += 1;
    final now = input.now ?? DateTime.now().millisecondsSinceEpoch / 1000.0;

    double elapsed() =>
        (DateTime.now().microsecondsSinceEpoch - started) / 1000.0;

    final faces = detector.filterDetections(
      input.faces,
      input.width,
      input.height,
      input.height,
    );
    final subject = detector.selectSubject(faces, input.width, input.height);

    if (subject == null) {
      _currentPerson = null;
      return FrameResult(
        ok: false,
        frameIndex: frameIndex,
        timestamp: now,
        elapsedMs: elapsed(),
        faceCount: faces.length,
        message: 'no_face',
      );
    }

    final quality = FaceDetector.qualityScore(
      faceHeight: subject.height,
      frameHeight: input.height,
      detScore: subject.detScore,
      pose: subject.pose,
      sharpness: input.sharpness,
    );
    final spoofResult = spoof.combine(input.spoofSignals);
    final poseResult = pose.estimate(
      landmarks: subject.landmarks2d,
      pose: subject.pose,
    );

    final match = await recognizer.match(subject.embedding ?? <double>[]);
    var personId = match.personId;
    var personName = match.name;
    var isNew = false;
    var isProvisional = false;

    if (!spoofResult.isSpoof) {
      if (match.isNew) {
        if (autoEnrolUnknown && quality >= 0.45) {
          final enrolled = await recognizer.enrolProvisional(
            subject.embedding ?? <double>[],
            quality: quality,
          );
          personId = enrolled.$1;
          personName = enrolled.$2;
          isNew = true;
          isProvisional = true;
        }
      } else if (personId != null) {
        await recognizer.maybeAutoEnrol(
          personId,
          subject.embedding ?? <double>[],
          match.score,
          quality,
        );
        final record = await db.getPerson(personId);
        isProvisional = record != null && record['is_provisional'] == 1;
      }
    }

    if (personId != null && personId != _currentPerson) {
      if (_currentPerson != null) emotion.reset();
      _currentPerson = personId;
      await db.touchPerson(personId);
    }

    final calSession = calibration.ensure(
      personId,
      personName,
      baseline.baselineFor(personId),
    );
    if (calSession != null && !calSession.complete) {
      calSession.offer(
        hasFace: true,
        quality: quality,
        yaw: poseResult.yaw,
        pitch: poseResult.pitch,
      );
    }

    final emotionResult = emotion.analyse(
      crop: input.crop,
      landmarks: subject.landmarks2d,
      quality: quality,
      rebalance: (List<double> p, Map<String, double> aus) => baseline
          .adjust(p, aus, personId, learn: restFramesOnly(calSession, p))
          .$1,
    );

    final affectReading = affect.update(
      AffectSignals(
        label: emotionResult.label,
        valence: emotionResult.valence,
        arousal: emotionResult.arousal,
        probs: emotionResult.probs,
      ),
      AffectPose(
        eyeOpenness: poseResult.eyeOpenness,
        pitch: poseResult.pitch,
        attention: poseResult.attention,
        isBlinking: poseResult.isBlinking,
      ),
      emotionResult.actionUnits,
      now: now,
    );

    final mood = baseline.updateMood(
      emotionResult.valence,
      emotionResult.arousal,
      affectReading.tension,
      affectReading.volatility,
      personId,
    );

    final report = session.observe(
      personId: personId,
      personName: personName,
      valence: emotionResult.valence,
      arousal: emotionResult.arousal,
      moodState: mood.state,
      emotion: emotionResult.label,
      confidence: emotionResult.confidence,
      units: <String, double>{
        'engagement': affectReading.engagement,
        'fatigue': affectReading.fatigue,
        'tension': affectReading.tension,
      },
      now: now,
    );
    if (report != null) await _storeSession(report);

    final result = FrameResult(
      ok: true,
      frameIndex: frameIndex,
      timestamp: now,
      elapsedMs: elapsed(),
      faceCount: faces.length,
      bbox: <int>[
        subject.bbox.x1.toInt(),
        subject.bbox.y1.toInt(),
        subject.bbox.x2.toInt(),
        subject.bbox.y2.toInt(),
      ],
      detScore: subject.detScore,
      personId: personId,
      personName: personName,
      isNewPerson: isNew,
      isProvisional: isProvisional,
      matchScore: match.score,
      matchUncertain: match.isUncertain,
      emotion: emotionResult,
      pose: poseResult,
      spoof: spoofResult,
      affect: affectReading,
      mood: mood,
      age: subject.age,
      gender: subject.gender,
      quality: quality,
      message: spoofResult.isSpoof ? 'spoof_suspected' : 'ok',
    );

    if (persist) {
      result.observationId = await db.addObservation(
        observationPayload(result, null),
      );
    }

    return result;
  }

  Map<String, Object?> observationPayload(
    FrameResult r,
    String? snapshot,
  ) => <String, Object?>{
    'person_id': r.personId,
    'session_id': sessionId,
    'emotion': r.emotion!.label,
    'emotion_conf': r.emotion!.confidence,
    'valence': r.emotion!.valence,
    'arousal': r.emotion!.arousal,
    'probs': r.emotion!.probs,
    'age': r.age,
    'gender': r.gender,
    'yaw': r.pose?.yaw,
    'pitch': r.pose?.pitch,
    'roll': r.pose?.roll,
    'gaze_x': r.pose?.gazeX,
    'gaze_y': r.pose?.gazeY,
    'attention': r.pose?.attention,
    'liveness': r.spoof?.liveness,
    'is_spoof': (r.spoof?.isSpoof ?? false) ? 1 : 0,
    'match_score': r.matchScore,
    'det_score': r.detScore,
    'snapshot_path': snapshot,
    'duchenne': r.affect?.duchenne,
    'smile_type': r.affect?.smileType,
    'compound': r.affect?.compound,
    'engagement': r.affect?.engagement,
    'fatigue': r.affect?.fatigue,
    'tension': r.affect?.tension,
    'volatility': r.affect?.volatility,
    'blink_rate': r.affect?.blinkRate,
    'expressiveness': r.affect?.expressiveness,
    'mood_state': r.mood?.state,
  };

  Future<void> _storeSession(SessionReport report) async {
    recentReports.insert(0, report);
    if (recentReports.length > 12) {
      recentReports.removeRange(12, recentReports.length);
    }
    await db.saveSessionReport(report.toMap(), sessionId: sessionId);
  }

  Future<SessionReport?> endPeriod({double? now}) async {
    final report = session.flush(now: now);
    if (report != null) await _storeSession(report);
    return report;
  }

  Future<void> renamePerson(int personId, String newName) async {
    final existing = await db.getPersonByName(newName);
    if (existing != null && existing['id'] != personId) {
      await db.mergePersons(personId, existing['id']! as int);
      return;
    }
    await db.renamePerson(personId, newName);
  }

  Future<List<Map<String, Object?>>> listPersons() => db.listPersons();

  Map<String, Object?> trend() => emotion.trend();

  Future<int> saveBaselines() async {
    var saved = 0;
    for (final personId in baseline.knownPersonIds.toList()) {
      final payload = baseline.export(personId);
      final samples = (payload?['samples'] as num?)?.toInt() ?? 0;
      if (payload != null && samples >= 20) {
        try {
          await db.saveBaseline(personId, payload);
          saved += 1;
        } on Object {
          continue;
        }
      }
    }
    return saved;
  }

  Future<Map<String, Object?>> stats() async => <String, Object?>{
    ...await db.stats(),
    'emotion_backend': emotion.backend,
    'emotion_model': emotion.modelName,
  };

  Future<void> close() async {
    try {
      await saveBaselines();
    } on Object {
      await endSession();
    }
    await endSession();
    await db.close();
  }
}

Point2 pointOf(double x, double y) => Point2(x, y);
