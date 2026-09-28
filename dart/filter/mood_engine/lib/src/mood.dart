import 'dart:collection';

import 'maths.dart';

class MoodReading {
  MoodReading({
    this.moodType = 'neutral',
    this.moodAr = 'محايد',
    this.ageGroup = 'adult',
    this.ageGroupAr = 'بالغ',
    this.gender = 'male',
    this.genderAr = 'ذكر',
    this.expressionStable = 1.0,
    this.recentMoodChanges = 0,
    this.summary = '',
    this.summaryAr = '',
  });

  String moodType;
  String moodAr;
  String ageGroup;
  String ageGroupAr;
  String gender;
  String genderAr;
  double expressionStable;
  int recentMoodChanges;
  String summary;
  String summaryAr;

  Map<String, Object?> toMap() => <String, Object?>{
    'mood_type': moodType,
    'mood_ar': moodAr,
    'age_group': ageGroup,
    'age_group_ar': ageGroupAr,
    'gender': gender,
    'gender_ar': genderAr,
    'expression_stable': roundTo(expressionStable, 4),
    'recent_mood_changes': recentMoodChanges,
    'summary': summary,
    'summary_ar': summaryAr,
  };
}

class MoodInput {
  const MoodInput({
    this.label = 'neutral',
    this.valence = 0.0,
    this.arousal = 0.0,
    this.probs = const <String, double>{},
  });

  final String label;
  final double valence;
  final double arousal;
  final Map<String, double> probs;
}

class MoodPose {
  const MoodPose({this.age, this.gender, this.arousal});

  final double? age;
  final String? gender;
  final double? arousal;
}

class MoodAnalyzer {
  MoodAnalyzer({this.window = 30});

  static const double positiveMin = 0.3;
  static const double negativeMax = -0.2;

  final int window;

  final ListQueue<double> _valences = ListQueue<double>();
  final ListQueue<double> _arousals = ListQueue<double>();
  final ListQueue<String> _labels = ListQueue<String>();
  final ListQueue<String> _ageVotes = ListQueue<String>();
  final ListQueue<String> _genderVotes = ListQueue<String>();

  void _push<T>(ListQueue<T> buffer, T value, int limit) {
    buffer.addLast(value);
    while (buffer.length > limit) {
      buffer.removeFirst();
    }
  }

  MoodReading update(
    MoodInput emotion, {
    MoodPose? pose,
    double quality = 1.0,
  }) {
    final reading = MoodReading();

    final probs = emotion.probs;
    final valence = emotion.valence;
    final arousal = emotion.arousal;
    final label = emotion.label;

    String mood;
    String moodAr;
    if (valence >= positiveMin) {
      mood = 'positive';
      moodAr = 'إيجابي';
    } else if (valence <= negativeMax) {
      mood = 'negative';
      moodAr = 'سلبي';
    } else if ((probs['neutral'] ?? 0.0) > 0.6) {
      mood = 'neutral';
      moodAr = 'محايد';
    } else {
      final top = probs.isEmpty ? 'neutral' : rankedDescending(probs).first.key;
      if (top == 'happy' || top == 'surprise') {
        mood = 'positive';
        moodAr = 'إيجابي';
      } else if (top == 'sad' ||
          top == 'fear' ||
          top == 'disgust' ||
          top == 'anger') {
        mood = 'negative';
        moodAr = 'سلبي';
      } else {
        mood = 'neutral';
        moodAr = 'محايد';
      }
    }

    reading.moodType = mood;
    reading.moodAr = moodAr;

    for (final vote in _voteAge(probs, pose, quality)) {
      _push(_ageVotes, vote, 10);
    }
    if (_ageVotes.isNotEmpty) {
      reading.ageGroup = mostCommon(_ageVotes)!;
      reading.ageGroupAr = _ageAr(reading.ageGroup);
    }

    final gender = pose?.gender;
    if (gender == 'male' || gender == 'female') {
      _push(_genderVotes, gender!, 10);
    }
    if (_genderVotes.isNotEmpty) {
      reading.gender = mostCommon(_genderVotes)!;
      reading.genderAr = _genderAr(reading.gender);
    }

    _push(_valences, valence, window);
    _push(_arousals, arousal, window);
    _push(_labels, label, window);

    if (_labels.length >= 2) {
      final labels = _labels.toList();
      var changes = 0;
      for (var i = 1; i < labels.length; i++) {
        if (labels[i - 1] != labels[i]) changes++;
      }
      reading.recentMoodChanges = changes < 5 ? changes : 5;
      final denominator = labels.length - 1 > 1 ? labels.length - 1 : 1;
      reading.expressionStable = 1.0 - (changes / denominator);
    }

    reading.summary = _buildSummary(reading, mood, label);
    reading.summaryAr = _buildSummaryAr(reading, moodAr, label);

    return reading;
  }

  List<String> _voteAge(
    Map<String, double> probs,
    MoodPose? pose,
    double quality,
  ) {
    final votes = <String>[];

    final age = pose?.age;
    if (age != null) {
      if (age < 12) {
        votes.add('child');
      } else if (age < 25) {
        votes.add('young');
      } else if (age < 60) {
        votes.add('adult');
      } else {
        votes.add('senior');
      }
    }

    if ((probs['surprise'] ?? 0.0) > 0.5) votes.add('child');

    final poseArousal = pose?.arousal;
    if (poseArousal != null && poseArousal > 0.6) votes.add('young');

    return votes.length <= 3 ? votes : votes.sublist(0, 3);
  }

  static String _ageAr(String age) => const <String, String>{
    'child': 'طفل',
    'young': 'شاب',
    'adult': 'بالغ',
    'senior': 'مسن',
  }[age] ??
      age;

  static String _genderAr(String gender) => const <String, String>{
    'male': 'ذكر',
    'female': 'أنثى',
    'unknown': 'غير معروف',
  }[gender] ??
      gender;

  String _buildSummary(MoodReading reading, String mood, String label) =>
      <String>[reading.ageGroup, reading.gender, mood, label]
          .where((String part) => part.isNotEmpty)
          .join(' ');

  String _buildSummaryAr(MoodReading reading, String moodAr, String label) =>
      <String>[reading.ageGroupAr, reading.genderAr, moodAr, label]
          .where((String part) => part.isNotEmpty)
          .join(' ');

  void reset() {
    _valences.clear();
    _arousals.clear();
    _labels.clear();
    _ageVotes.clear();
    _genderVotes.clear();
  }

  Map<String, Object?> summary() {
    if (_labels.isEmpty) return <String, Object?>{'samples': 0};
    final labels = _labels.toList();
    final dominant = mostCommon(labels)!;
    return <String, Object?>{
      'samples': labels.length,
      'dominant_mood': dominant,
      'dominant_emotion': dominant,
      'stability': countOf(labels, dominant) / labels.length,
      'mean_valence': mean(_valences),
      'mean_arousal': mean(_arousals),
    };
  }
}
