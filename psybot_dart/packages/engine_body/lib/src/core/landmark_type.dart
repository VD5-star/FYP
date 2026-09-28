enum LandmarkType {
  nose,
  leftEyeInner,
  leftEye,
  leftEyeOuter,
  rightEyeInner,
  rightEye,
  rightEyeOuter,
  leftEar,
  rightEar,
  leftMouth,
  rightMouth,
  leftShoulder,
  rightShoulder,
  leftElbow,
  rightElbow,
  leftWrist,
  rightWrist,
  leftPinky,
  rightPinky,
  leftIndex,
  rightIndex,
  leftThumb,
  rightThumb,
  leftHip,
  rightHip,
  leftKnee,
  rightKnee,
  leftAnkle,
  rightAnkle,
  leftHeel,
  rightHeel,
  leftFootIndex,
  rightFootIndex;

  static LandmarkType? fromId(int id) =>
      (id >= 0 && id < values.length) ? values[id] : null;

  int get id => index;
}

const torsoLandmarks = <LandmarkType>{
  LandmarkType.leftShoulder,
  LandmarkType.rightShoulder,
  LandmarkType.leftHip,
  LandmarkType.rightHip,
};

const upperBodyLandmarks = <LandmarkType>{
  LandmarkType.nose,
  LandmarkType.leftShoulder,
  LandmarkType.rightShoulder,
  LandmarkType.leftElbow,
  LandmarkType.rightElbow,
  LandmarkType.leftWrist,
  LandmarkType.rightWrist,
  LandmarkType.leftHip,
  LandmarkType.rightHip,
};

const lowerBodyLandmarks = <LandmarkType>{
  LandmarkType.leftKnee,
  LandmarkType.rightKnee,
  LandmarkType.leftAnkle,
  LandmarkType.rightAnkle,
};

const skeletonBones = <(LandmarkType, LandmarkType)>[
  (LandmarkType.leftShoulder, LandmarkType.rightShoulder),
  (LandmarkType.leftHip, LandmarkType.rightHip),
  (LandmarkType.leftShoulder, LandmarkType.leftHip),
  (LandmarkType.rightShoulder, LandmarkType.rightHip),
  (LandmarkType.leftShoulder, LandmarkType.leftElbow),
  (LandmarkType.leftElbow, LandmarkType.leftWrist),
  (LandmarkType.rightShoulder, LandmarkType.rightElbow),
  (LandmarkType.rightElbow, LandmarkType.rightWrist),
  (LandmarkType.leftWrist, LandmarkType.leftThumb),
  (LandmarkType.leftWrist, LandmarkType.leftIndex),
  (LandmarkType.leftWrist, LandmarkType.leftPinky),
  (LandmarkType.rightWrist, LandmarkType.rightThumb),
  (LandmarkType.rightWrist, LandmarkType.rightIndex),
  (LandmarkType.rightWrist, LandmarkType.rightPinky),
  (LandmarkType.leftHip, LandmarkType.leftKnee),
  (LandmarkType.leftKnee, LandmarkType.leftAnkle),
  (LandmarkType.rightHip, LandmarkType.rightKnee),
  (LandmarkType.rightKnee, LandmarkType.rightAnkle),
  (LandmarkType.leftAnkle, LandmarkType.leftHeel),
  (LandmarkType.leftHeel, LandmarkType.leftFootIndex),
  (LandmarkType.rightAnkle, LandmarkType.rightHeel),
  (LandmarkType.rightHeel, LandmarkType.rightFootIndex),
  (LandmarkType.leftEar, LandmarkType.leftEye),
  (LandmarkType.leftEye, LandmarkType.nose),
  (LandmarkType.nose, LandmarkType.rightEye),
  (LandmarkType.rightEye, LandmarkType.rightEar),
  (LandmarkType.leftMouth, LandmarkType.rightMouth),
];
