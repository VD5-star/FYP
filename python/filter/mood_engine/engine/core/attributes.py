from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import CONFIG, SpoofConfig

_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)

_IDX_106 = {"nose": 86, "chin": 0, "eye_l": 35, "eye_r": 93,
            "mouth_l": 52, "mouth_r": 61}


@dataclass
class PoseResult:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    attention: float = 0.0
    eye_openness: float = 0.0
    is_blinking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class SpoofResult:
    liveness: float = 1.0
    is_spoof: bool = False
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "liveness": round(self.liveness, 4),
            "is_spoof": self.is_spoof,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
        }


class PoseEstimator:


    def __init__(self) -> None:
        self._blink_history: deque[float] = deque(maxlen=90)
        self.blink_count = 0
        self._was_closed = False

    def estimate(self, face: Any, frame: np.ndarray) -> PoseResult:
        result = PoseResult()
        lm = getattr(face, "landmarks_2d", None)

        pose = getattr(face, "pose", None)
        if pose is not None and len(pose) >= 3:
            result.pitch, result.yaw, result.roll = (float(pose[0]),
                                                     float(pose[1]),
                                                     float(pose[2]))
        elif lm is not None and len(lm) >= 106:
            result.yaw, result.pitch, result.roll = self._solve_pnp(lm, frame)

        result.roll = self._wrap_180(result.roll)
        result.pitch = self._wrap_180(result.pitch)
        result.yaw = self._wrap_180(result.yaw)
        if abs(result.roll) > 90.0:
            result.roll = self._wrap_180(result.roll - 180.0)
            result.pitch = self._wrap_180(-result.pitch)
            result.yaw = self._wrap_180(-result.yaw)

        if lm is not None and len(lm) >= 106:
            result.eye_openness = self._eye_openness(lm)
            result.gaze_x, result.gaze_y = self._gaze(lm, frame, result)

        self._blink_history.append(result.eye_openness)
        if result.eye_openness < 0.18 and not self._was_closed:
            self._was_closed = True
            self.blink_count += 1
            result.is_blinking = True
        elif result.eye_openness > 0.24:
            self._was_closed = False

        result.attention = self._attention(result)
        return result

    @staticmethod
    def _wrap_180(angle: float) -> float:
        return float((angle + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _solve_pnp(lm: np.ndarray,
                   frame: np.ndarray) -> tuple[float, float, float]:
        try:
            pts = np.array([lm[_IDX_106[k]] for k in
                            ("nose", "chin", "eye_l", "eye_r",
                             "mouth_l", "mouth_r")], dtype=np.float64)
            h, w = frame.shape[:2]
            f = float(w)
            cam = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]],
                           dtype=np.float64)
            ok, rvec, _ = cv2.solvePnP(_MODEL_3D, pts, cam,
                                       np.zeros((4, 1)),
                                       flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return 0.0, 0.0, 0.0
            rmat, _ = cv2.Rodrigues(rvec)
            sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
            if sy > 1e-6:
                pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
                yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
                roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
            else:
                pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
                yaw = 0.0
                roll = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            return float(yaw), float(pitch), float(roll)
        except Exception:
            return 0.0, 0.0, 0.0

    @staticmethod
    def _eye_openness(lm: np.ndarray) -> float:
        left = lm[[35, 36, 37, 38, 39, 40, 41, 42]]
        right = lm[[89, 90, 91, 92, 93, 94, 95, 96]]
        lc, rc = left.mean(axis=0), right.mean(axis=0)
        iod = float(np.linalg.norm(rc - lc)) or 1.0
        h = ((left[:, 1].max() - left[:, 1].min())
             + (right[:, 1].max() - right[:, 1].min())) / 2.0
        return float(np.clip(h / iod / 0.35, 0.0, 1.5))

    @staticmethod
    def _gaze(lm: np.ndarray, frame: np.ndarray,
              pose: PoseResult) -> tuple[float, float]:
        try:
            left = lm[[35, 36, 37, 38, 39, 40, 41, 42]]
            right = lm[[89, 90, 91, 92, 93, 94, 95, 96]]
            gx_parts, gy_parts = [], []
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = grey.shape[:2]

            for eye in (left, right):
                x1, y1 = eye[:, 0].min(), eye[:, 1].min()
                x2, y2 = eye[:, 0].max(), eye[:, 1].max()
                pad = (x2 - x1) * 0.15
                x1i, y1i = int(max(0, x1 - pad)), int(max(0, y1 - pad))
                x2i, y2i = int(min(w, x2 + pad)), int(min(h, y2 + pad))
                if x2i - x1i < 4 or y2i - y1i < 3:
                    continue
                roi = grey[y1i:y2i, x1i:x2i]
                roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
                _, thr = cv2.threshold(roi_blur, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                m = cv2.moments(thr)
                if m["m00"] <= 0:
                    continue
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                gx_parts.append((cx / max(1, roi.shape[1])) * 2.0 - 1.0)
                gy_parts.append(1.0 - (cy / max(1, roi.shape[0])) * 2.0)

            if not gx_parts:
                return 0.0, 0.0
            gx = float(np.mean(gx_parts)) + pose.yaw / 60.0
            gy = float(np.mean(gy_parts)) + pose.pitch / 60.0
            return float(np.clip(gx, -1, 1)), float(np.clip(gy, -1, 1))
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _attention(p: PoseResult) -> float:
        head = max(0.0, 1.0 - (abs(p.yaw) / 40.0) * 0.6
                   - (abs(p.pitch) / 35.0) * 0.4)
        gaze = max(0.0, 1.0 - (abs(p.gaze_x) * 0.6 + abs(p.gaze_y) * 0.4))
        eyes = 1.0 if p.eye_openness > 0.25 else p.eye_openness / 0.25
        return float(np.clip(0.5 * head + 0.35 * gaze + 0.15 * eyes, 0.0, 1.0))


class SpoofDetector:


    def __init__(self, config: SpoofConfig | None = None) -> None:
        self.config = config or CONFIG.spoof
        self._crops: deque[np.ndarray] = deque(maxlen=self.config.history_len)

    def check(self, face: Any, frame: np.ndarray) -> SpoofResult:
        if not self.config.enabled:
            return SpoofResult(liveness=1.0, is_spoof=False)

        crop = face.crop(frame, margin=0.05)
        if crop.size == 0 or min(crop.shape[:2]) < 24:
            return SpoofResult(liveness=0.5, is_spoof=False,
                               signals={"too_small": 1.0})

        signals: dict[str, float] = {}
        signals["depth"] = self._depth_signal(face)
        signals["texture"] = self._texture_signal(crop)
        signals["colour"] = self._colour_signal(crop)
        signals["moire"] = self._moire_signal(crop)
        signals["motion"] = self._motion_signal(crop)

        weights = {"depth": 0.28, "texture": 0.24, "colour": 0.14,
                   "moire": 0.16, "motion": 0.18}
        live = sum(signals[k] * w for k, w in weights.items())
        live = float(np.clip(live, 0.0, 1.0))
        return SpoofResult(liveness=live,
                           is_spoof=live < self.config.threshold,
                           signals=signals)

    @staticmethod
    def _depth_signal(face: Any) -> float:
        lm3d = getattr(face, "landmarks_3d", None)
        if lm3d is not None and np.asarray(lm3d).shape[-1] >= 3:
            z = np.asarray(lm3d)[:, 2].astype(np.float32)
            spread = float(np.std(z))
            scale = float(face.height) or 1.0
            return float(np.clip((spread / scale) / 0.045, 0.0, 1.0))

        pose = getattr(face, "pose", None)
        if pose is not None and len(pose) >= 2:
            yaw, pitch = abs(float(pose[1])), abs(float(pose[0]))
            return float(np.clip(0.45 + (yaw / 30.0) * 0.35
                                 + (pitch / 30.0) * 0.20, 0.0, 1.0))
        return 0.6

    @staticmethod
    def _texture_signal(crop: np.ndarray) -> float:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        grey = cv2.resize(grey, (128, 128), interpolation=cv2.INTER_AREA)
        lap = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        hf = float(np.mean(np.abs(grey.astype(np.float32)
                                  - cv2.GaussianBlur(grey, (9, 9), 0))))
        return float(np.clip(0.6 * min(1.0, lap / 260.0)
                             + 0.4 * min(1.0, hf / 9.0), 0.0, 1.0))

    @staticmethod
    def _colour_signal(crop: np.ndarray) -> float:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s_mean = float(np.mean(hsv[..., 1])) / 255.0
        v_std = float(np.std(hsv[..., 2])) / 255.0
        s_ok = 1.0 - min(1.0, abs(s_mean - 0.34) / 0.34)
        v_ok = min(1.0, v_std / 0.16)
        return float(np.clip(0.5 * s_ok + 0.5 * v_ok, 0.0, 1.0))

    @staticmethod
    def _moire_signal(crop: np.ndarray) -> float:
        try:
            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            grey = cv2.resize(grey, (128, 128), interpolation=cv2.INTER_AREA)
            f = np.fft.fftshift(np.fft.fft2(grey.astype(np.float32)))
            mag = np.log1p(np.abs(f))
            c = mag.shape[0] // 2
            mid = mag[c - 40:c + 40, c - 40:c + 40]
            centre = mag[c - 6:c + 6, c - 6:c + 6]
            peak_ratio = float(mid.max() / (mid.mean() + 1e-6))
            energy = float(mid.mean() / (centre.mean() + 1e-6))
            score = 1.0 - np.clip((peak_ratio - 2.6) / 3.0, 0.0, 1.0)
            score = 0.7 * score + 0.3 * np.clip(energy / 0.65, 0.0, 1.0)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.6

    def _motion_signal(self, crop: np.ndarray) -> float:
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        self._crops.append(small)
        if len(self._crops) < 4:
            return 0.7
        diffs = [float(np.mean(np.abs(self._crops[i] - self._crops[i - 1])))
                 for i in range(1, len(self._crops))]
        motion = float(np.mean(diffs))
        return float(np.clip(motion / 2.2, 0.0, 1.0))

    def reset(self) -> None:
        self._crops.clear()
