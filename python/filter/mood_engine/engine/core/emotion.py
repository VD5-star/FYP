from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..config import (CONFIG, EMOTIONS, EMOTIONS_AR, EMOTION_VA, MODELS_DIR,
                      EmotionConfig)

EMOTION_MODEL_DIR = MODELS_DIR / "emotion"
EMOTION_HEAD_PATH = MODELS_DIR / "emotion_head.json"

_HSE_8 = ("anger", "contempt", "disgust", "fear", "happy", "neutral", "sad",
          "surprise")
_HSE_7 = ("anger", "disgust", "fear", "happy", "neutral", "sad", "surprise")
_CONTEMPT_TARGET = "disgust"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class EmotionResult:
    label: str
    label_ar: str
    confidence: float
    probs: dict[str, float]
    valence: float
    arousal: float
    source: str
    action_units: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "label_ar": self.label_ar,
            "confidence": round(self.confidence, 4),
            "probs": {k: round(v, 4) for k, v in self.probs.items()},
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "source": self.source,
            "action_units": {k: round(v, 4)
                             for k, v in self.action_units.items()},
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.max(x)
    e = np.exp(x)
    total = e.sum()
    return e / total if total > 0 else np.full_like(e, 1.0 / e.size)


class EmotionAnalyzer:


    _PREFERRED = ("enet_b0_8_best_vgaf.onnx", "enet_b0_8_best_afew.onnx",
                  "enet_b2_8.onnx")

    def __init__(self, config: EmotionConfig | None = None,
                 model_path: Path | str | None = None) -> None:
        self.config = config or CONFIG.emotion
        self._session = None
        self._input_name: str | None = None
        self._input_size = (224, 224)
        self._input_channels = 3
        self._class_order: tuple[str, ...] = _HSE_8
        self._model_path = Path(model_path) if model_path else None
        self._smoothed: np.ndarray | None = None
        self.history: deque[tuple[str, float, float]] = deque(
            maxlen=self.config.history_len)
        self._sticky_label: str | None = None
        self._challenger: str | None = None
        self._challenger_frames = 0
        self._head: tuple[np.ndarray, np.ndarray, list[str]] | None = None
        self._head_metrics: dict[str, Any] = {}
        self._try_load_model()
        self._try_load_head()

    def _discover_model(self) -> Path | None:
        if self._model_path and self._model_path.exists():
            return self._model_path
        if not EMOTION_MODEL_DIR.exists():
            return None
        for name in self._PREFERRED:
            candidate = EMOTION_MODEL_DIR / name
            if candidate.exists():
                return candidate
        found = sorted(EMOTION_MODEL_DIR.glob("*.onnx"))
        return found[0] if found else None

    def _try_load_model(self) -> None:
        path = self._discover_model()
        if path is None:
            return
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.intra_op_num_threads = 4
            sess = ort.InferenceSession(str(path), sess_options=so,
                                        providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            shape = list(inp.shape)
            if len(shape) == 4:
                c = shape[1] if isinstance(shape[1], int) else 3
                h = shape[2] if isinstance(shape[2], int) else 224
                w = shape[3] if isinstance(shape[3], int) else 224
                self._input_channels = int(c)
                self._input_size = (int(w), int(h))
            out_dim = sess.get_outputs()[0].shape[-1]
            if isinstance(out_dim, int):
                self._class_order = _HSE_8 if out_dim == 8 else _HSE_7
            self._session = sess
            self._input_name = inp.name
            self._model_path = path
        except Exception:
            self._session = None

    def _try_load_head(self) -> None:
        if not EMOTION_HEAD_PATH.exists():
            return
        try:
            import json
            payload = json.loads(EMOTION_HEAD_PATH.read_text(encoding="utf-8"))
            coef = np.asarray(payload["coef"], dtype=np.float64)
            intercept = np.asarray(payload["intercept"], dtype=np.float64)
            classes = list(payload["classes"])
            expected = len(EMOTIONS) + 6
            if coef.shape[1] != expected or coef.shape[0] != len(classes):
                return
            self._head = (coef, intercept, classes)
            self._head_metrics = payload.get("metrics", {})
        except Exception:
            self._head = None

    def _apply_head(self, probs: np.ndarray,
                    aus: dict[str, float]) -> np.ndarray:
        if self._head is None:
            return probs
        coef, intercept, classes = self._head
        features = np.concatenate([
            np.log(np.clip(probs, 1e-6, 1.0)),
            np.array([aus.get(k, 0.0) for k in
                      ("smile", "eye_open", "brow_raise", "brow_knit",
                       "mouth_open", "mouth_width")], dtype=np.float64),
        ])
        scores = _softmax(coef @ features + intercept)

        out = np.zeros(len(EMOTIONS), dtype=np.float64)
        for name, value in zip(classes, scores):
            if name in EMOTIONS:
                out[EMOTIONS.index(name)] = value
        total = out.sum()
        return out / total if total > 0 else probs

    @property
    def backend(self) -> str:
        if self._session is None:
            return "geometric"
        return "onnx+head" if self._head is not None else "onnx"

    @property
    def head_metrics(self) -> dict[str, Any]:
        return dict(self._head_metrics)

    @property
    def model_path(self) -> str | None:
        return str(self._model_path) if self._session is not None else None

    def analyse(self, face_crop: np.ndarray,
                landmarks: np.ndarray | None = None,
                *, smooth: bool = True,
                quality: float = 1.0,
                rebalance: Callable[[np.ndarray, dict[str, float]],
                                    np.ndarray] | None = None) -> EmotionResult:
        aus: dict[str, float] = {}
        if self._session is not None:
            probs = self._infer_onnx(face_crop)
            source = "onnx"
            if landmarks is not None:
                aus = self.action_units(landmarks)
            if self._head is not None:
                probs = self._apply_head(probs, aus)
                source = "onnx+head"
        else:
            aus = self.action_units(landmarks) if landmarks is not None else {}
            probs = self._infer_geometric(aus)
            source = "geometric"

        if rebalance is not None:
            try:
                adjusted = np.asarray(rebalance(probs, aus), dtype=np.float64)
                if (adjusted.shape == probs.shape
                        and np.all(np.isfinite(adjusted))
                        and adjusted.sum() > 0):
                    probs = adjusted / adjusted.sum()
                    source += "+baseline"
            except Exception:
                pass

        if smooth:
            probs = self._smooth(probs, quality)
            label = self._stable_label(probs)
        else:
            label = EMOTIONS[int(np.argmax(probs))]

        confidence = float(probs[EMOTIONS.index(label)])
        valence, arousal = self._valence_arousal(probs, aus)

        self.history.append((label, valence, arousal))
        return EmotionResult(
            label=label,
            label_ar=EMOTIONS_AR[label],
            confidence=confidence,
            probs={e: float(p) for e, p in zip(EMOTIONS, probs)},
            valence=valence,
            arousal=arousal,
            source=source,
            action_units=aus,
        )

    def probabilities(self, face_crop: np.ndarray) -> np.ndarray:
        if self._session is not None:
            return self._infer_onnx(face_crop)
        return self._infer_geometric({})

    def _infer_onnx(self, face_crop: np.ndarray) -> np.ndarray:
        import cv2

        w, h = self._input_size
        img = cv2.resize(face_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        if self._input_channels == 1:
            x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
            x = x.astype(np.float32) / 255.0
        else:
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
        logits = np.asarray(
            self._session.run(None, {self._input_name: x})[0]).ravel()
        raw = _softmax(logits)
        order = self._class_order

        winner_idx = int(np.argmax(raw[:len(order)]))
        winner = order[winner_idx]
        winner = _CONTEMPT_TARGET if winner == "contempt" else winner

        out = np.zeros(len(EMOTIONS), dtype=np.float64)
        for i, cls in enumerate(order):
            if i >= raw.size:
                break
            target = _CONTEMPT_TARGET if cls == "contempt" else cls
            if target in EMOTIONS:
                out[EMOTIONS.index(target)] += float(raw[i])

        total = out.sum()
        out = (out / total if total > 0
               else np.full(len(EMOTIONS), 1.0 / len(EMOTIONS)))

        w = EMOTIONS.index(winner)
        if int(np.argmax(out)) != w:
            out[w] = float(np.max(out)) + 1e-6
            out = out / out.sum()
        return out

    @staticmethod
    def action_units(landmarks: np.ndarray | None) -> dict[str, float]:
        if landmarks is None or len(landmarks) < 106:
            return {}
        p = np.asarray(landmarks, dtype=np.float32)

        left_eye = p[[35, 36, 37, 38, 39, 40, 41, 42]]
        right_eye = p[[89, 90, 91, 92, 93, 94, 95, 96]]
        left_brow = p[[43, 44, 45, 46, 47]]
        right_brow = p[[97, 98, 99, 100, 101]]
        mouth_outer = p[[52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64]]

        lc, rc = left_eye.mean(axis=0), right_eye.mean(axis=0)
        iod = float(np.linalg.norm(rc - lc)) or 1.0

        def spread(pts: np.ndarray) -> float:
            return float(pts[:, 1].max() - pts[:, 1].min())

        eye_open = (spread(left_eye) + spread(right_eye)) / 2.0 / iod
        brow_y = (left_brow[:, 1].mean() + right_brow[:, 1].mean()) / 2.0
        eye_y = (lc[1] + rc[1]) / 2.0
        brow_raise = float(eye_y - brow_y) / iod

        mouth_h = spread(mouth_outer) / iod
        mouth_w = float(mouth_outer[:, 0].max() - mouth_outer[:, 0].min()) / iod

        corners_y = float((p[52][1] + p[61][1]) / 2.0)
        lips_y = float(mouth_outer[:, 1].mean())
        smile = (lips_y - corners_y) / iod

        knit = float(np.linalg.norm(left_brow[-1] - right_brow[-1])) / iod

        return {
            "eye_open": eye_open,
            "brow_raise": brow_raise,
            "brow_knit": knit,
            "mouth_open": mouth_h,
            "mouth_width": mouth_w,
            "smile": smile,
        }

    @staticmethod
    def _infer_geometric(aus: dict[str, float]) -> np.ndarray:
        if not aus:
            probs = np.zeros(len(EMOTIONS))
            probs[EMOTIONS.index("neutral")] = 1.0
            return probs

        smile = aus.get("smile", 0.0)
        mouth_open = aus.get("mouth_open", 0.0)
        brow_raise = aus.get("brow_raise", 0.0)
        brow_knit = aus.get("brow_knit", 0.0)
        eye_open = aus.get("eye_open", 0.0)

        s = {e: 0.0 for e in EMOTIONS}
        s["neutral"] = 1.0
        s["happy"] = 6.0 * max(0.0, smile - 0.015)
        s["sad"] = 5.0 * max(0.0, -smile - 0.010) + 1.5 * max(0.0, 0.30 - brow_raise)
        s["surprise"] = 4.0 * max(0.0, mouth_open - 0.28) + 3.0 * max(0.0, brow_raise - 0.42)
        s["fear"] = 3.0 * max(0.0, eye_open - 0.34) + 2.0 * max(0.0, brow_raise - 0.40)
        s["anger"] = 4.0 * max(0.0, 0.62 - brow_knit) + 2.0 * max(0.0, 0.26 - brow_raise)
        s["disgust"] = 3.0 * max(0.0, 0.60 - brow_knit) + 2.0 * max(0.0, -smile - 0.02)

        return _softmax(np.array([s[e] for e in EMOTIONS]) * 2.2)

    def _smooth(self, probs: np.ndarray, quality: float = 1.0) -> np.ndarray:
        cfg = self.config
        q = float(np.clip(quality, 0.0, 1.0))
        a = cfg.smoothing_alpha_min + (
            cfg.smoothing_alpha - cfg.smoothing_alpha_min) * q
        if self._smoothed is None:
            self._smoothed = probs.copy()
        else:
            self._smoothed = a * probs + (1.0 - a) * self._smoothed
        total = self._smoothed.sum()
        return self._smoothed / total if total > 0 else self._smoothed

    def _stable_label(self, probs: np.ndarray) -> str:
        cfg = self.config
        top = EMOTIONS[int(np.argmax(probs))]

        if self._sticky_label is None:
            self._sticky_label = top
            self._challenger = None
            self._challenger_frames = 0
            return top

        if top == self._sticky_label:
            self._challenger = None
            self._challenger_frames = 0
            return self._sticky_label

        lead = float(probs[EMOTIONS.index(top)]
                     - probs[EMOTIONS.index(self._sticky_label)])
        if lead < cfg.switch_margin:
            return self._sticky_label

        if top == self._challenger:
            self._challenger_frames += 1
        else:
            self._challenger = top
            self._challenger_frames = 1

        if self._challenger_frames >= cfg.switch_frames:
            self._sticky_label = top
            self._challenger = None
            self._challenger_frames = 0
        return self._sticky_label

    @staticmethod
    def _valence_arousal(probs: np.ndarray,
                         aus: dict[str, float]) -> tuple[float, float]:
        v = float(sum(p * EMOTION_VA[e][0] for e, p in zip(EMOTIONS, probs)))
        a = float(sum(p * EMOTION_VA[e][1] for e, p in zip(EMOTIONS, probs)))
        if aus:
            a += 0.18 * np.clip(aus.get("mouth_open", 0.0) - 0.25, 0.0, 0.6)
            a += 0.12 * np.clip(aus.get("eye_open", 0.0) - 0.30, 0.0, 0.4)
            v += 0.20 * np.clip(aus.get("smile", 0.0), -0.10, 0.10) * 5.0
        return float(np.clip(v, -1.0, 1.0)), float(np.clip(a, 0.0, 1.0))

    def reset(self) -> None:
        self._smoothed = None
        self._sticky_label = None
        self._challenger = None
        self._challenger_frames = 0
        self.history.clear()

    def trend(self) -> dict[str, Any]:
        if not self.history:
            return {"dominant": None, "avg_valence": 0.0, "avg_arousal": 0.0,
                    "samples": 0, "stability": 0.0}
        labels = [h[0] for h in self.history]
        dominant = max(set(labels), key=labels.count)
        return {
            "dominant": dominant,
            "dominant_ar": EMOTIONS_AR[dominant],
            "avg_valence": float(np.mean([h[1] for h in self.history])),
            "avg_arousal": float(np.mean([h[2] for h in self.history])),
            "samples": len(self.history),
            "stability": labels.count(dominant) / len(labels),
        }
