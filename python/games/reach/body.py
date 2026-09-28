from __future__ import annotations
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from anchor import JointAnchor
from angles import IDX
from occlusion import OcclusionTracker
from skeleton import SkeletonCalibrator, apply_model, torso_length
MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"
BONES = [
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("leftHip", "leftKnee"), ("leftKnee", "leftAnkle"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
]
HEAD = ("nose", "leftEye", "rightEye", "leftEar", "rightEar")
BELIEVE_THRESHOLD = 0.2
DRAW_THRESHOLD = 0.15
MIN_TORSO_PX = 40.0
DETECT_WIDTH = 960
class BodyTracker:
    def __init__(self, believe_threshold: float = BELIEVE_THRESHOLD,
                 detect_width: int = DETECT_WIDTH) -> None:
        self.available = MODEL_PATH.exists()
        self.detect_width = detect_width
        self._landmarker = None
        if self.available:
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.skeleton = SkeletonCalibrator()
        self.occlusion = OcclusionTracker(believe_threshold=believe_threshold)
        self.anchor = JointAnchor()
        self.points: np.ndarray | None = None
        self.visibility: np.ndarray | None = None
        self.discarded: np.ndarray | None = None
        self.inferred: np.ndarray | None = None
        self.torso: float | None = None
        self._frame = 0
    def update(self, image: np.ndarray, timestamp: float):
        self.points = None
        self.visibility = None
        self.discarded = None
        self.inferred = None
        self.torso = None
        if not self.available or self._landmarker is None:
            return None
        h, w = image.shape[:2]
        self._frame += 1
        source = image
        if 0 < self.detect_width < w:
            dh = max(1, int(round(h * self.detect_width / w)))
            source = cv2.resize(image, (self.detect_width, dh),
                                interpolation=cv2.INTER_LINEAR)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(source, cv2.COLOR_BGR2RGB)),
            int(timestamp * 1000))
        if not result.pose_landmarks:
            return None
        pose = result.pose_landmarks[0]
        points = np.array([[p.x * w, p.y * h] for p in pose])
        visibility = np.array([p.visibility for p in pose])
        model = self.skeleton.update(points, visibility, (w, h))
        points = apply_model(points, model)
        report = self.occlusion.update(points, visibility, model,
                                       timestamp, w, h)
        torso = torso_length(report.points)
        points = self.anchor.update(report.points, torso,
                                    ~report.discarded, (w, h))
        self.points = points
        self.visibility = report.confidence
        self.discarded = report.discarded
        self.inferred = report.inferred
        self.torso = torso if torso >= MIN_TORSO_PX else None
        return self.points
    def wrists(self) -> list[np.ndarray]:
        if self.points is None or self.visibility is None:
            return []
        out = []
        for name in ("leftWrist", "rightWrist"):
            i = IDX[name]
            if self.discarded is not None and self.discarded[i]:
                continue
            if self.visibility[i] < DRAW_THRESHOLD:
                continue
            out.append(self.points[i])
        return out
    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
