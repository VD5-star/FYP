from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import EMOTIONS_AR

COMPOUND_NAMES = {
    frozenset({"happy", "surprise"}): ("delighted", "مبتهج"),
    frozenset({"happy", "fear"}): ("nervous joy", "فرح متوتر"),
    frozenset({"happy", "sad"}): ("bittersweet", "فرح ممزوج بحزن"),
    frozenset({"sad", "anger"}): ("frustrated", "محبط"),
    frozenset({"sad", "fear"}): ("anxious", "قلق"),
    frozenset({"anger", "disgust"}): ("contempt", "ازدراء"),
    frozenset({"fear", "surprise"}): ("alarmed", "مفزوع"),
    frozenset({"anger", "fear"}): ("threatened", "متوتر ومتوجس"),
    frozenset({"neutral", "sad"}): ("subdued", "فاتر"),
    frozenset({"neutral", "happy"}): ("content", "مرتاح"),
}


@dataclass
class AffectReading:
    duchenne: float = 0.0
    smile_type: str = "none"
    compound: str | None = None
    compound_ar: str | None = None
    mixture: list[tuple[str, float]] = field(default_factory=list)
    engagement: float = 0.0
    fatigue: float = 0.0
    tension: float = 0.0
    volatility: float = 0.0
    blink_rate: float = 0.0
    expressiveness: float = 0.0
    transition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duchenne": round(self.duchenne, 4),
            "smile_type": self.smile_type,
            "compound": self.compound,
            "compound_ar": self.compound_ar,
            "mixture": [[name, round(value, 4)] for name, value in self.mixture],
            "engagement": round(self.engagement, 4),
            "fatigue": round(self.fatigue, 4),
            "tension": round(self.tension, 4),
            "volatility": round(self.volatility, 4),
            "blink_rate": round(self.blink_rate, 2),
            "expressiveness": round(self.expressiveness, 4),
            "transition": self.transition,
        }


class AffectAnalyzer:


    def __init__(self, *, window: int = 150) -> None:
        self.window = window
        self._valence: deque[float] = deque(maxlen=window)
        self._arousal: deque[float] = deque(maxlen=window)
        self._labels: deque[str] = deque(maxlen=window)
        self._blinks: deque[float] = deque(maxlen=60)
        self._eye_open: deque[float] = deque(maxlen=window)
        self._pitch: deque[float] = deque(maxlen=window)
        self._attention: deque[float] = deque(maxlen=window)
        self._last_label: str | None = None
        self._was_blinking = False

    def update(self, emotion: Any, pose: Any,
               action_units: dict[str, float]) -> AffectReading:
        reading = AffectReading()
        now = time.time()

        probs = getattr(emotion, "probs", {}) or {}
        label = getattr(emotion, "label", "neutral")

        self._valence.append(float(getattr(emotion, "valence", 0.0)))
        self._arousal.append(float(getattr(emotion, "arousal", 0.0)))
        self._labels.append(label)

        if pose is not None:
            self._eye_open.append(float(getattr(pose, "eye_openness", 0.0)))
            self._pitch.append(float(getattr(pose, "pitch", 0.0)))
            self._attention.append(float(getattr(pose, "attention", 0.0)))
            blinking = bool(getattr(pose, "is_blinking", False))
            if blinking and not self._was_blinking:
                self._blinks.append(now)
            self._was_blinking = blinking

        reading.duchenne, reading.smile_type = self._duchenne(
            probs, action_units)
        reading.mixture, reading.compound, reading.compound_ar = \
            self._compound(probs)
        reading.expressiveness = self._expressiveness(probs)
        reading.engagement = self._engagement(reading.expressiveness)
        reading.blink_rate = self._blink_rate(now)
        reading.fatigue = self._fatigue(reading.blink_rate)
        reading.tension = self._tension(action_units, probs)
        reading.volatility = self._volatility()
        reading.transition = self._transition(label)

        return reading

    @staticmethod
    def _duchenne(probs: dict[str, float],
                  units: dict[str, float]) -> tuple[float, str]:
        happy = float(probs.get("happy", 0.0))
        smile = float(units.get("smile", 0.0))
        if happy < 0.30 or smile <= 0.005:
            return 0.0, "none"

        eye_open = float(units.get("eye_open", 0.30))
        narrowing = np.clip((0.30 - eye_open) / 0.12, 0.0, 1.0)
        mouth = np.clip(smile / 0.045, 0.0, 1.0)

        score = float(np.clip(0.45 * mouth + 0.55 * narrowing, 0.0, 1.0))
        if score >= 0.55:
            return score, "genuine"
        return score, "social"

    @staticmethod
    def _compound(probs: dict[str, float]
                  ) -> tuple[list[tuple[str, float]], str | None, str | None]:
        if not probs:
            return [], None, None
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top, second = ranked[0], ranked[1] if len(ranked) > 1 else (None, 0.0)

        mixture = [(name, value) for name, value in ranked[:3] if value >= 0.12]
        if second[0] is None or second[1] < 0.22 or top[1] > 0.80:
            return mixture, None, None

        key = frozenset({top[0], second[0]})
        named = COMPOUND_NAMES.get(key)
        if named is None:
            return mixture, None, None
        return mixture, named[0], named[1]

    @staticmethod
    def _expressiveness(probs: dict[str, float]) -> float:
        if not probs:
            return 0.0
        return float(np.clip(1.0 - probs.get("neutral", 0.0), 0.0, 1.0))

    def _engagement(self, expressiveness: float) -> float:
        if not self._attention:
            return 0.0
        attention = float(np.mean(list(self._attention)[-30:]))
        return float(np.clip(0.65 * attention + 0.35 * expressiveness, 0.0, 1.0))

    def _blink_rate(self, now: float) -> float:
        cutoff = now - 60.0
        while self._blinks and self._blinks[0] < cutoff:
            self._blinks.popleft()
        if not self._blinks:
            return 0.0
        span = max(now - self._blinks[0], 1.0)
        return float(len(self._blinks) * 60.0 / span)

    def _fatigue(self, blink_rate: float) -> float:
        if len(self._eye_open) < 10:
            return 0.0
        eye = float(np.mean(list(self._eye_open)[-60:]))
        droop = 0.0
        if self._pitch:
            pitch = float(np.mean(list(self._pitch)[-60:]))
            droop = np.clip((-pitch - 8.0) / 22.0, 0.0, 1.0)

        closed = np.clip((0.42 - eye) / 0.22, 0.0, 1.0)
        excess_blink = np.clip((blink_rate - 22.0) / 26.0, 0.0, 1.0)
        return float(np.clip(0.45 * closed + 0.30 * excess_blink
                             + 0.25 * droop, 0.0, 1.0))

    @staticmethod
    def _tension(units: dict[str, float], probs: dict[str, float]) -> float:
        if not units:
            return 0.0
        knit = np.clip((0.95 - units.get("brow_knit", 0.95)) / 0.35, 0.0, 1.0)
        lips = np.clip((0.28 - units.get("mouth_open", 0.28)) / 0.20, 0.0, 1.0)
        negative = sum(probs.get(k, 0.0)
                       for k in ("anger", "fear", "disgust", "sad"))
        return float(np.clip(0.40 * knit + 0.25 * lips
                             + 0.35 * negative, 0.0, 1.0))

    def _volatility(self) -> float:
        if len(self._valence) < 8:
            return 0.0
        recent_v = np.array(list(self._valence)[-45:])
        recent_a = np.array(list(self._arousal)[-45:])
        movement = float(np.mean(np.abs(np.diff(recent_v)))
                         + np.mean(np.abs(np.diff(recent_a))))
        return float(np.clip(movement / 0.16, 0.0, 1.0))

    def _transition(self, label: str) -> str | None:
        if self._last_label is None:
            self._last_label = label
            return None
        if label == self._last_label:
            return None
        previous, self._last_label = self._last_label, label
        return f"{previous} -> {label}"

    def summary(self) -> dict[str, Any]:
        if not self._labels:
            return {"samples": 0}
        labels = list(self._labels)
        dominant = max(set(labels), key=labels.count)
        return {
            "samples": len(labels),
            "dominant": dominant,
            "dominant_ar": EMOTIONS_AR.get(dominant, dominant),
            "stability": labels.count(dominant) / len(labels),
            "mean_valence": float(np.mean(self._valence)),
            "mean_arousal": float(np.mean(self._arousal)),
            "valence_range": float(np.ptp(self._valence))
            if len(self._valence) > 1 else 0.0,
            "distinct_emotions": len(set(labels)),
        }

    def reset(self) -> None:
        for buffer in (self._valence, self._arousal, self._labels,
                       self._eye_open, self._pitch, self._attention):
            buffer.clear()
        self._blinks.clear()
        self._last_label = None
        self._was_blinking = False
