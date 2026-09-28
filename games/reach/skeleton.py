from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from angles import IDX
TREE: list[tuple[str, str | None]] = [
    ("leftHip", None),
    ("rightHip", None),
    ("leftShoulder", "leftHip"),
    ("rightShoulder", "rightHip"),
    ("leftElbow", "leftShoulder"),
    ("rightElbow", "rightShoulder"),
    ("leftWrist", "leftElbow"),
    ("rightWrist", "rightElbow"),
    ("leftKnee", "leftHip"),
    ("rightKnee", "rightHip"),
    ("leftAnkle", "leftKnee"),
    ("rightAnkle", "rightKnee"),
]
MIRRORED: list[tuple[str, str]] = [
    ("leftShoulder", "rightShoulder"),
    ("leftElbow", "rightElbow"),
    ("leftWrist", "rightWrist"),
    ("leftKnee", "rightKnee"),
    ("leftAnkle", "rightAnkle"),
]
EXTREMITIES: list[tuple[str, str]] = [
    ("leftThumb", "leftWrist"), ("leftIndex", "leftWrist"),
    ("leftPinky", "leftWrist"),
    ("rightThumb", "rightWrist"), ("rightIndex", "rightWrist"),
    ("rightPinky", "rightWrist"),
    ("leftHeel", "leftAnkle"), ("leftFootIndex", "leftAnkle"),
    ("rightHeel", "rightAnkle"), ("rightFootIndex", "rightAnkle"),
]
CALIBRATION_VISIBILITY = 0.2
TORSO_VISIBILITY = 0.2
MIN_SAMPLES = 10
def torso_length(points: np.ndarray) -> float:
    mid_shoulder = (points[IDX["leftShoulder"]] + points[IDX["rightShoulder"]]) / 2
    mid_hip = (points[IDX["leftHip"]] + points[IDX["rightHip"]]) / 2
    return float(np.linalg.norm(mid_shoulder - mid_hip))
@dataclass
class BodyModel:
    lengths: dict[str, float] = field(default_factory=dict)
    samples: int = 0
    @property
    def ready(self) -> bool:
        return len(self.lengths) >= 8
class SkeletonCalibrator:
    def __init__(self, frames: int = 30, refresh_every: int = 300) -> None:
        self.frames = frames
        """How many frames to learn from. 30 is about one second at 30 fps.
        Was 60. Measured across six real clips, learning from 30 frames gives a
        *better* bone spread than 60 (2.4% against 3.5%) as well as being ready
        twice as fast. A longer window is not a better median here - it spans
        more of the subject's movement, and a bone seen at many angles has more
        foreshortened samples dragging the median down.
            learn  refresh   bone spread   ready
               60      300          3.5%     79%
               45      150          3.2%     77%
             **30      300          2.4%     79%**
               30      150          8.1%     75%
               60       90         19.5%     76%
        """
        self.refresh_every = refresh_every
        """Re-measure this often, in frames, after the model is built.
        The body does not change, but the *estimate* can improve, and a subject
        who calibrated side-on will have poor lengths until they turn.
        Measured: refreshing faster is clearly worse, not better. At 90 frames
        the spread is 19.5% against 3.5% at 300 - a refresh throws away a
        settled median and rebuilds it from whatever the subject happens to be
        doing, so frequent refreshes keep the model permanently half-learned.
        300 frames is ten seconds, slow enough not to cause visible drift.
        """
        self._samples: dict[str, list[float]] = {}
        self._count = 0
        self._since_refresh = 0
        self.model = BodyModel()
    def update(self, points: np.ndarray, visibility: np.ndarray,
               frame_size: tuple[int, int] | None = None) -> BodyModel:
        if self._count >= self.frames:
            self._since_refresh += 1
            if self._since_refresh < self.refresh_every:
                return self.model
            self._since_refresh = 0
            self._count = 0
            self._samples = {}
        torso = torso_length(points)
        if torso < 1e-6:
            return self.model
        for name in ("leftShoulder", "rightShoulder", "leftHip", "rightHip"):
            i = IDX[name]
            if visibility[i] < TORSO_VISIBILITY:
                return self.model
            if not np.all(np.isfinite(points[i])):
                return self.model
            if frame_size is not None:
                w, h = frame_size
                mx, my = w * 0.002, h * 0.002
                if not (-mx <= points[i][0] <= w + mx
                        and -my <= points[i][1] <= h + my):
                    return self.model
        for child, parent in TREE:
            if parent is None:
                continue
            ci, pi = IDX[child], IDX[parent]
            if (visibility[ci] < CALIBRATION_VISIBILITY
                    or visibility[pi] < CALIBRATION_VISIBILITY):
                continue
            length = float(np.linalg.norm(points[ci] - points[pi])) / torso
            if np.isfinite(length) and length > 1e-6:
                self._samples.setdefault(child, []).append(length)
        self._count += 1
        self._rebuild()
        return self.model
    def _rebuild(self) -> None:
        lengths: dict[str, float] = {}
        for child, values in self._samples.items():
            if len(values) >= MIN_SAMPLES:
                lengths[child] = float(np.median(values))
        for a, b in MIRRORED:
            if a in lengths and b in lengths:
                mean = (lengths[a] + lengths[b]) / 2
                lengths[a] = lengths[b] = mean
        self.model = BodyModel(lengths=lengths, samples=self._count)
    def reset(self) -> None:
        self._samples = {}
        self._count = 0
        self._since_refresh = 0
        self.model = BodyModel()
def apply_model(points: np.ndarray, model: BodyModel) -> np.ndarray:
    if not model.ready:
        return points
    torso = torso_length(points)
    if torso < 1e-6 or not np.isfinite(torso):
        return points
    out = points.copy()
    for child, parent in TREE:
        if parent is None or child not in model.lengths:
            continue
        ci, pi = IDX[child], IDX[parent]
        direction = points[ci] - points[pi]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6 or not np.isfinite(norm):
            continue
        out[ci] = out[pi] + direction / norm * (model.lengths[child] * torso)
    for name, parent in EXTREMITIES:
        ni, pi = IDX[name], IDX[parent]
        out[ni] = points[ni] + (out[pi] - points[pi])
    return out
