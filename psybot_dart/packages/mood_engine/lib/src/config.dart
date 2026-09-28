const List<String> emotions = <String>[
  'neutral',
  'happy',
  'sad',
  'surprise',
  'fear',
  'disgust',
  'anger',
];

const Map<String, String> emotionsAr = <String, String>{
  'neutral': 'محايد',
  'happy': 'سعيد',
  'sad': 'حزين',
  'surprise': 'مندهش',
  'fear': 'خائف',
  'disgust': 'مشمئز',
  'anger': 'غاضب',
};

const Map<String, List<double>> emotionVa = <String, List<double>>{
  'neutral': <double>[0.00, 0.20],
  'happy': <double>[0.80, 0.65],
  'sad': <double>[-0.70, 0.25],
  'surprise': <double>[0.25, 0.90],
  'fear': <double>[-0.65, 0.85],
  'disgust': <double>[-0.60, 0.55],
  'anger': <double>[-0.75, 0.85],
};

int emotionIndex(String name) => emotions.indexOf(name);

class DetectionConfig {
  const DetectionConfig({
    this.detSize = const <int>[320, 320],
    this.minDetScore = 0.45,
    this.minFaceRatio = 0.05,
    this.detectMaxWidth = 640,
    this.modelPack = 'buffalo_s',
    this.intraOpThreads = 6,
    this.modules = const <String>[
      'detection',
      'recognition',
      'genderage',
      'landmark_2d_106',
    ],
    this.identityEveryNFrames = 1,
    this.liveIdentityEveryNFrames = 5,
    this.retryWithPadding = true,
    this.paddingRatio = 0.25,
    this.alignSize = 224,
    this.alignEyeRatio = 0.34,
  });

  final List<int> detSize;
  final double minDetScore;
  final double minFaceRatio;
  final int detectMaxWidth;
  final String modelPack;
  final int intraOpThreads;
  final List<String> modules;
  final int identityEveryNFrames;
  final int liveIdentityEveryNFrames;
  final bool retryWithPadding;
  final double paddingRatio;
  final int alignSize;
  final double alignEyeRatio;
}

class RecognitionConfig {
  const RecognitionConfig({
    this.matchThreshold = 0.42,
    this.newThreshold = 0.32,
    this.maxEmbeddingsPerPerson = 40,
    this.autoEnrolLow = 0.45,
    this.autoEnrolHigh = 0.92,
  });

  final double matchThreshold;
  final double newThreshold;
  final int maxEmbeddingsPerPerson;
  final double autoEnrolLow;
  final double autoEnrolHigh;
}

class EmotionConfig {
  const EmotionConfig({
    this.smoothingAlpha = 0.35,
    this.smoothingAlphaMin = 0.12,
    this.historyLen = 30,
    this.minConfidence = 0.25,
    this.switchMargin = 0.06,
    this.switchFrames = 2,
  });

  final double smoothingAlpha;
  final double smoothingAlphaMin;
  final int historyLen;
  final double minConfidence;
  final double switchMargin;
  final int switchFrames;
}

class SpoofConfig {
  const SpoofConfig({
    this.enabled = true,
    this.threshold = 0.55,
    this.historyLen = 12,
  });

  final bool enabled;
  final double threshold;
  final int historyLen;
}

class TrackingConfig {
  const TrackingConfig({
    this.maxCentreDrift = 0.18,
    this.maxMissingFrames = 15,
  });

  final double maxCentreDrift;
  final int maxMissingFrames;
}

class EngineConfig {
  const EngineConfig({
    this.detection = const DetectionConfig(),
    this.recognition = const RecognitionConfig(),
    this.emotion = const EmotionConfig(),
    this.spoof = const SpoofConfig(),
    this.tracking = const TrackingConfig(),
    this.cameraIndex = 0,
    this.cameraSizes = const <List<int>>[
      <int>[1280, 720],
      <int>[1024, 768],
      <int>[960, 540],
      <int>[800, 600],
      <int>[640, 480],
    ],
    this.cameraFourcc = 'MJPG',
    this.analyseEveryNFrames = 1,
    this.guidedCalibration = true,
    this.sessionPeriodS = 300.0,
    this.saveSnapshots = true,
    this.snapshotMinConfidence = 0.80,
    this.snapshotMinIntervalS = 20.0,
  });

  final DetectionConfig detection;
  final RecognitionConfig recognition;
  final EmotionConfig emotion;
  final SpoofConfig spoof;
  final TrackingConfig tracking;
  final int cameraIndex;
  final List<List<int>> cameraSizes;
  final String cameraFourcc;
  final int analyseEveryNFrames;
  final bool guidedCalibration;
  final double sessionPeriodS;
  final bool saveSnapshots;
  final double snapshotMinConfidence;
  final double snapshotMinIntervalS;
}

const EngineConfig defaultEngineConfig = EngineConfig();
