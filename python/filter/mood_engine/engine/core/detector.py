from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import CONFIG, MODELS_DIR, DetectionConfig, TrackingConfig

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("insightface").setLevel(logging.ERROR)

os.environ.setdefault("INSIGHTFACE_HOME", str(MODELS_DIR / "insightface"))


@dataclass
class DetectedFace:
    bbox: np.ndarray
    det_score: float
    embedding: np.ndarray
    keypoints: np.ndarray
    landmarks_2d: np.ndarray | None = None
    landmarks_3d: np.ndarray | None = None
    age: float | None = None
    gender: str | None = None
    pose: np.ndarray | None = None
    raw: Any = field(default=None, repr=False)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centre(self) -> tuple[float, float]:
        return ((float(self.bbox[0]) + float(self.bbox[2])) / 2.0,
                (float(self.bbox[1]) + float(self.bbox[3])) / 2.0)

    def crop(self, frame: np.ndarray, margin: float = 0.2) -> np.ndarray:
        h, w = frame.shape[:2]
        mx, my = self.width * margin, self.height * margin
        x1 = max(0, int(self.bbox[0] - mx))
        y1 = max(0, int(self.bbox[1] - my))
        x2 = min(w, int(self.bbox[2] + mx))
        y2 = min(h, int(self.bbox[3] + my))
        if x2 <= x1 or y2 <= y1:
            return frame[0:1, 0:1]
        return frame[y1:y2, x1:x2]


class FaceDetector:


    def __init__(self, config: DetectionConfig | None = None,
                 tracking: TrackingConfig | None = None,
                 providers: Sequence[str] | None = None) -> None:
        self.config = config or CONFIG.detection
        self.tracking = tracking or CONFIG.tracking
        self._providers = list(providers or ["CPUExecutionProvider"])
        self._app = None
        self._identity_every = int(
            getattr(self.config, "identity_every_n_frames", 1))
        self._identity_counter = 0
        self._cached_identity: tuple[np.ndarray, float | None, str | None] | None = None
        self._last_centre: tuple[float, float] | None = None
        self._missing = 0

    def load(self) -> None:
        if self._app is not None:
            return
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        ort.set_default_logger_severity(3)
        modules = list(self.config.modules) if self.config.modules else None
        app = FaceAnalysis(name=self.config.model_pack,
                           providers=self._providers,
                           allowed_modules=modules)
        app.prepare(ctx_id=-1, det_size=self.config.det_size)

        threads = int(self.config.intra_op_threads)
        if threads > 0:
            for model in app.models.values():
                path = getattr(model, "model_file", None)
                if not path:
                    continue
                so = ort.SessionOptions()
                so.log_severity_level = 3
                so.intra_op_num_threads = threads
                so.inter_op_num_threads = 1
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                so.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
                try:
                    model.session = ort.InferenceSession(
                        path, sess_options=so, providers=self._providers)
                except Exception:
                    pass
            app.prepare(ctx_id=-1, det_size=self.config.det_size)

        self._app = app

    @property
    def app(self):
        if self._app is None:
            self.load()
        return self._app

    @property
    def model_names(self) -> list[str]:
        return sorted(self.app.models.keys())

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        if frame is None or frame.size == 0:
            return []

        limit = int(getattr(self.config, "detect_max_width", 0) or 0)
        if limit and frame.shape[1] > limit:
            ratio = limit / frame.shape[1]
            small = cv2.resize(frame, (limit, round(frame.shape[0] * ratio)),
                               interpolation=cv2.INTER_AREA)
            faces = self._detect_pass(small, small.shape[0])
            if faces:
                self._rescale_faces(faces, 1.0 / ratio,
                                    frame.shape[1], frame.shape[0])
                return faces
        else:
            faces = self._detect_pass(frame, frame.shape[0])

        if faces or not self.config.retry_with_padding:
            return faces

        if not self._may_be_cropped_portrait(frame):
            return faces

        pad = int(max(frame.shape[:2]) * self.config.padding_ratio)
        if pad <= 0:
            return faces
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad,
                                    cv2.BORDER_REPLICATE)
        h, w = frame.shape[:2]
        recovered = []
        for face in self._detect_pass(padded, padded.shape[0]):
            shifted = np.asarray(face.bbox, dtype=np.float32) - pad
            face.bbox = np.array([
                float(np.clip(shifted[0], 0.0, w - 1.0)),
                float(np.clip(shifted[1], 0.0, h - 1.0)),
                float(np.clip(shifted[2], 0.0, w - 1.0)),
                float(np.clip(shifted[3], 0.0, h - 1.0)),
            ], dtype=np.float32)
            if face.keypoints is not None and face.keypoints.size:
                face.keypoints = face.keypoints - pad
            if face.landmarks_2d is not None:
                face.landmarks_2d = face.landmarks_2d - pad
            if face.width > 1 and face.height > 1:
                recovered.append(face)
        return recovered

    def _may_be_cropped_portrait(self, frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return False

        aspect = max(w / h, h / w)
        if aspect > 1.7:
            return False

        thumb = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        if float(np.std(thumb)) < 12.0:
            return False

        return True

    def enable_identity_cache(self, every_n_frames: int) -> None:
        self._identity_every = max(1, int(every_n_frames))
        self._identity_counter = 0
        self._cached_identity = None

    def _wants_identity(self) -> bool:
        if self._identity_every <= 1 or self._cached_identity is None:
            return True
        return (self._identity_counter % self._identity_every) == 0

    @staticmethod
    def _rescale_faces(faces: list[DetectedFace], scale: float,
                       width: int, height: int) -> None:
        if scale == 1.0:
            return
        for face in faces:
            box = np.asarray(face.bbox, dtype=np.float32) * scale
            face.bbox = np.array([
                float(np.clip(box[0], 0.0, width - 1.0)),
                float(np.clip(box[1], 0.0, height - 1.0)),
                float(np.clip(box[2], 0.0, width - 1.0)),
                float(np.clip(box[3], 0.0, height - 1.0)),
            ], dtype=np.float32)
            for name in ("keypoints", "landmarks_2d", "landmarks_3d"):
                points = getattr(face, name, None)
                if points is not None and len(points):
                    setattr(face, name,
                            np.asarray(points, dtype=np.float32) * scale)

    def _run_models(self, frame: np.ndarray) -> list:
        app = self.app
        if self._wants_identity():
            return app.get(frame)

        skipped = ("recognition",)
        held = {name: app.models.pop(name)
                for name in skipped if name in app.models}
        try:
            return app.get(frame)
        finally:
            app.models.update(held)

    def _detect_pass(self, frame: np.ndarray,
                     reference_height: int) -> list[DetectedFace]:
        h, w = frame.shape[:2]
        faces: list[DetectedFace] = []
        for f in self._run_models(frame):
            score = float(getattr(f, "det_score", 0.0))
            if score < self.config.min_det_score:
                continue
            bbox = np.asarray(f.bbox, dtype=np.float32)
            if (bbox[3] - bbox[1]) < (self.config.min_face_ratio
                                      * reference_height):
                continue
            bbox = np.array([
                float(np.clip(bbox[0], 0.0, w - 1.0)),
                float(np.clip(bbox[1], 0.0, h - 1.0)),
                float(np.clip(bbox[2], 0.0, w - 1.0)),
                float(np.clip(bbox[3], 0.0, h - 1.0)),
            ], dtype=np.float32)
            raw_embedding = getattr(f, "embedding", None)
            if raw_embedding is not None:
                emb = np.asarray(raw_embedding, dtype=np.float32).ravel()
                norm = float(np.linalg.norm(emb))
                emb = emb / norm if norm > 0 else emb
                age = float(f.age) if getattr(f, "age", None) is not None else None
                gender = self._gender_label(f)
            elif self._cached_identity is not None:
                emb, age, gender = self._cached_identity
            else:
                continue

            faces.append(DetectedFace(
                bbox=bbox,
                det_score=score,
                embedding=emb,
                keypoints=np.asarray(getattr(f, "kps", np.zeros((5, 2))),
                                     dtype=np.float32),
                landmarks_2d=self._as_array(getattr(f, "landmark_2d_106", None)),
                landmarks_3d=self._as_array(getattr(f, "landmark_3d_68", None)),
                age=age,
                gender=gender,
                pose=self._as_array(getattr(f, "pose", None)),
                raw=f,
            ))

        if faces:
            subject = max(faces, key=lambda x: x.area)
            if getattr(faces[0], "embedding", None) is not None:
                self._cached_identity = (subject.embedding, subject.age,
                                         subject.gender)
            self._identity_counter += 1
        else:
            self._identity_counter = 0
            self._cached_identity = None

        return faces

    def align(self, face: DetectedFace, frame: np.ndarray,
              size: int | None = None) -> np.ndarray:
        size = size or self.config.align_size
        kps = face.keypoints
        if kps is None or len(kps) < 2:
            crop = face.crop(frame, margin=0.15)
            return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

        left_eye = np.asarray(kps[0], dtype=np.float32)
        right_eye = np.asarray(kps[1], dtype=np.float32)
        centre = (left_eye + right_eye) / 2.0
        delta = right_eye - left_eye
        distance = float(np.hypot(delta[0], delta[1]))
        if distance < 1e-3:
            crop = face.crop(frame, margin=0.15)
            return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

        angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
        desired = self.config.align_eye_ratio * size
        scale = desired / distance

        matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])),
                                         angle, scale)
        matrix[0, 2] += size * 0.5 - centre[0]
        matrix[1, 2] += size * 0.42 - centre[1]

        return cv2.warpAffine(frame, matrix, (size, size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _as_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _gender_label(face: Any) -> str | None:
        sex = getattr(face, "sex", None)
        if sex in ("M", "F"):
            return sex
        g = getattr(face, "gender", None)
        if g is None:
            return None
        return "M" if int(g) == 1 else "F"

    def select_subject(self, faces: Sequence[DetectedFace],
                       frame_shape: tuple[int, ...]) -> DetectedFace | None:
        if not faces:
            self._missing += 1
            if self._missing > self.tracking.max_missing_frames:
                self._last_centre = None
                self._missing = 0
            return None

        self._missing = 0
        largest = max(faces, key=lambda f: f.area)

        if self._last_centre is not None and len(faces) > 1:
            h, w = frame_shape[:2]
            diag = math.hypot(w, h)
            limit = self.tracking.max_centre_drift * diag
            lx, ly = self._last_centre
            near = [f for f in faces
                    if math.hypot(f.centre[0] - lx, f.centre[1] - ly) <= limit]
            if near:
                tracked = max(near, key=lambda f: f.area)
                if largest.area <= tracked.area * 1.35:
                    largest = tracked

        self._last_centre = largest.centre
        return largest

    def detect_subject(self, frame: np.ndarray
                       ) -> tuple[DetectedFace | None, list[DetectedFace]]:
        faces = self.detect(frame)
        return self.select_subject(faces, frame.shape), faces

    def reset_tracking(self) -> None:
        self._last_centre = None
        self._missing = 0
        self._identity_counter = 0
        self._cached_identity = None

    @staticmethod
    def quality_score(face: DetectedFace, frame: np.ndarray) -> float:
        import cv2

        h = frame.shape[0]
        size = min(1.0, face.height / (0.45 * h))
        conf = min(1.0, face.det_score)

        frontality = 1.0
        if face.pose is not None and len(face.pose) >= 2:
            pitch, yaw = float(face.pose[0]), float(face.pose[1])
            frontality = max(0.0, 1.0 - (abs(yaw) / 45.0) * 0.7
                             - (abs(pitch) / 45.0) * 0.3)

        crop = face.crop(frame, margin=0.0)
        sharp = 0.0
        if crop.size:
            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = min(1.0, cv2.Laplacian(grey, cv2.CV_64F).var() / 220.0)

        return float(np.clip(
            0.30 * conf + 0.25 * size + 0.25 * frontality + 0.20 * sharp,
            0.0, 1.0))
