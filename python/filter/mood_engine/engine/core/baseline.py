from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import EMOTIONS

MOOD_STATES = {
    "positive": ("Positive", "إيجابي"),
    "content": ("Content", "مرتاح"),
    "neutral": ("Neutral", "محايد"),
    "withdrawn": ("Withdrawn", "منسحب"),
    "tense": ("Tense", "متوتر"),
    "negative": ("Negative", "سلبي"),
    "volatile": ("Volatile", "متقلّب"),
}


@dataclass
class MoodSummary:
    state: str = "neutral"
    state_en: str = "Neutral"
    state_ar: str = "محايد"
    confidence: float = 0.0
    valence: float = 0.0
    energy: float = 0.0
    stability: float = 1.0
    samples: int = 0
    calibrated: bool = False
    baseline_progress: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "state_en": self.state_en,
            "state_ar": self.state_ar,
            "confidence": round(self.confidence, 4),
            "valence": round(self.valence, 4),
            "energy": round(self.energy, 4),
            "stability": round(self.stability, 4),
            "samples": self.samples,
            "calibrated": self.calibrated,
            "baseline_progress": round(self.baseline_progress, 3),
        }


@dataclass
class PersonBaseline:
    probs: np.ndarray = field(
        default_factory=lambda: np.zeros(len(EMOTIONS), dtype=np.float64))
    action_units: dict[str, float] = field(default_factory=dict)
    samples: int = 0
    updated_at: float = field(default_factory=time.time)

    REQUIRED = 500

    MAX_AGE_S = 7 * 24 * 3600.0

    EMA_CAP = 400

    @property
    def ready(self) -> bool:
        return self.samples >= self.REQUIRED

    @property
    def progress(self) -> float:
        return min(1.0, self.samples / self.REQUIRED)

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.updated_at)

    @property
    def expired(self) -> bool:
        return self.ready and self.age_s > self.MAX_AGE_S

    def observe(self, probs: np.ndarray, units: dict[str, float]) -> None:
        alpha = 1.0 / min(self.samples + 1, self.EMA_CAP)
        self.probs = (1 - alpha) * self.probs + alpha * probs
        for key, value in units.items():
            previous = self.action_units.get(key, value)
            self.action_units[key] = (1 - alpha) * previous + alpha * value
        self.samples += 1
        self.updated_at = time.time()

    def reset(self) -> None:
        self.probs = np.zeros(len(EMOTIONS), dtype=np.float64)
        self.action_units = {}
        self.samples = 0
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "probs": {name: round(float(v), 5)
                      for name, v in zip(EMOTIONS, self.probs)},
            "action_units": {k: round(float(v), 5)
                             for k, v in self.action_units.items()},
            "samples": self.samples,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PersonBaseline:
        probs = np.array([payload.get("probs", {}).get(n, 0.0)
                          for n in EMOTIONS], dtype=np.float64)
        return cls(
            probs=probs,
            action_units=dict(payload.get("action_units", {})),
            samples=int(payload.get("samples", 0)),
            updated_at=float(payload.get("updated_at", time.time())),
        )


class BaselineTracker:


    EXCLUDE_ABOVE_CONFIDENCE = 0.80

    def __init__(self, *, window: int = 240) -> None:
        self._baselines: dict[int, PersonBaseline] = {}
        self._anonymous = PersonBaseline()
        self._current: int | None = None
        self._valence: deque[float] = deque(maxlen=window)
        self._arousal: deque[float] = deque(maxlen=window)
        self._states: deque[str] = deque(maxlen=window)

    def baseline_for(self, person_id: int | None) -> PersonBaseline:
        if person_id is None:
            return self._anonymous
        return self._baselines.setdefault(person_id, PersonBaseline())

    def load(self, person_id: int, payload: dict[str, Any]) -> None:
        try:
            self._baselines[person_id] = PersonBaseline.from_dict(payload)
        except Exception:
            pass

    def export(self, person_id: int) -> dict[str, Any] | None:
        baseline = self._baselines.get(person_id)
        return baseline.to_dict() if baseline else None

    def switch_person(self, person_id: int | None) -> None:
        if person_id == self._current:
            return
        self._current = person_id
        self._valence.clear()
        self._arousal.clear()
        self._states.clear()

    def adjust(self, probs: np.ndarray, units: dict[str, float],
               person_id: int | None,
               learn: bool = True) -> tuple[np.ndarray, PersonBaseline]:
        baseline = self.baseline_for(person_id)

        if learn and (not baseline.ready
                      or float(probs.max()) < self.EXCLUDE_ABOVE_CONFIDENCE):
            baseline.observe(probs, units)

        if not baseline.ready:
            return probs, baseline

        neutral_index = EMOTIONS.index("neutral")

        floor = 0.06

        expressive = np.ones(len(EMOTIONS), dtype=bool)
        expressive[neutral_index] = False

        neutral_mass = float(probs[neutral_index])
        expressive_mass = float(probs[expressive].sum())
        if expressive_mass <= 0 or not np.isfinite(expressive_mass):
            return probs, baseline

        ratio = probs[expressive] / np.maximum(baseline.probs[expressive],
                                               floor)
        ratio_total = float(ratio.sum())
        if ratio_total <= 0 or not np.isfinite(ratio_total):
            return probs, baseline

        adjusted = np.empty_like(probs)
        adjusted[neutral_index] = neutral_mass
        adjusted[expressive] = ratio / ratio_total * expressive_mass

        strength = 0.5
        blended = strength * adjusted + (1.0 - strength) * probs
        total = float(blended.sum())
        if total <= 0 or not np.isfinite(total):
            return probs, baseline
        return blended / total, baseline

    def update_mood(self, valence: float, arousal: float,
                    tension: float, volatility: float,
                    person_id: int | None) -> MoodSummary:
        self.switch_person(person_id)
        self._valence.append(float(valence))
        self._arousal.append(float(arousal))

        baseline = self.baseline_for(person_id)
        summary = MoodSummary(
            samples=len(self._valence),
            calibrated=baseline.ready,
            baseline_progress=baseline.progress,
        )
        if len(self._valence) < 10:
            summary.state_en, summary.state_ar = MOOD_STATES["neutral"]
            return summary

        mean_valence = float(np.mean(self._valence))
        mean_arousal = float(np.mean(self._arousal))
        swing = float(np.std(self._valence))

        summary.valence = mean_valence
        summary.energy = mean_arousal
        summary.stability = float(np.clip(1.0 - swing / 0.5, 0.0, 1.0))

        state = self._classify(mean_valence, mean_arousal, swing,
                               tension, volatility)
        self._states.append(state)

        dominant = max(set(self._states), key=self._states.count)
        summary.state = dominant
        summary.state_en, summary.state_ar = MOOD_STATES[dominant]
        summary.confidence = self._states.count(dominant) / len(self._states)
        return summary

    @staticmethod
    def _classify(valence: float, arousal: float, swing: float,
                  tension: float, volatility: float) -> str:
        if swing > 0.34 or volatility > 0.72:
            return "volatile"
        if valence >= 0.30:
            return "positive" if arousal >= 0.45 else "content"
        if valence <= -0.28:
            if tension >= 0.55 or arousal >= 0.55:
                return "tense"
            return "negative"
        if valence <= -0.10 and arousal < 0.35:
            return "withdrawn"
        if tension >= 0.62:
            return "tense"
        return "neutral"

    def reset(self) -> None:
        self._valence.clear()
        self._arousal.clear()
        self._states.clear()
        self._current = None

    def forget(self, person_id: int | None = None) -> None:
        if person_id is None:
            self._baselines.clear()
            self._anonymous = PersonBaseline()
        else:
            self._baselines.pop(person_id, None)
