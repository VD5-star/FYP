from __future__ import annotations

LANDMARK_NAMES = [
    "nose", "leftEyeInner", "leftEye", "leftEyeOuter", "rightEyeInner",
    "rightEye", "rightEyeOuter", "leftEar", "rightEar", "leftMouth",
    "rightMouth", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist", "leftPinky", "rightPinky", "leftIndex",
    "rightIndex", "leftThumb", "rightThumb", "leftHip", "rightHip",
    "leftKnee", "rightKnee", "leftAnkle", "rightAnkle", "leftHeel",
    "rightHeel", "leftFootIndex", "rightFootIndex",
]

IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}
