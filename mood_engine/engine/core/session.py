from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SESSION_STATES: dict[str, tuple[str, str]] = {
    "positive": ("Positive", "\u0625\u064a\u062c\u0627\u0628\u064a"),
    "content": ("Content", "\u0645\u0631\u062a\u0627\u062d"),
    "calm": ("Calm", "\u0647\u0627\u062f\u0626"),
    "neutral": ("Neutral", "\u0645\u062d\u0627\u064a\u062f"),
    "tired": ("Tired", "\u0645\u062a\u0639\u0628"),
    "withdrawn": ("Withdrawn", "\u0645\u0646\u0633\u062d\u0628"),
    "tense": ("Tense", "\u0645\u062a\u0648\u062a\u0631"),
    "restless": ("Restless", "\u0645\u0636\u0637\u0631\u0628"),
    "distressed": ("Distressed", "\u0642\u0644\u0642"),
    "low": ("Low", "\u062d\u0632\u064a\u0646"),
}


@dataclass(frozen=True)
class SessionReport:


    person_id: int | None
    person_name: str | None
    started_at: float
    ended_at: float
    samples: int
    state: str
    confidence: float
    dominance: float
    valence: float
    arousal: float
    stability: float
    engagement: float
    fatigue: float
    tension: float
    emotion_mix: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def duration_s(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    def label(self, arabic: bool = False) -> str:
        names = SESSION_STATES.get(self.state)
        if not names:
            return self.state
        return names[1] if arabic else names[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "person_name": self.person_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": round(self.duration_s, 1),
            "samples": self.samples,
            "state": self.state,
            "label_en": self.label(False),
            "label_ar": self.label(True),
            "confidence": round(self.confidence, 4),
            "dominance": round(self.dominance, 4),
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "stability": round(self.stability, 4),
            "engagement": round(self.engagement, 4),
            "fatigue": round(self.fatigue, 4),
            "tension": round(self.tension, 4),
            "emotion_mix": {k: round(v, 4) for k, v in self.emotion_mix.items()},
            "notes": list(self.notes),
        }


class SessionAnalyser:


    DEFAULT_PERIOD_S = 300.0

    MIN_SAMPLES = 60

    MIN_SAMPLE_CONFIDENCE = 0.35

    def __init__(self, *, period_s: float = DEFAULT_PERIOD_S,
                 min_samples: int = MIN_SAMPLES) -> None:
        self.period_s = max(30.0, float(period_s))
        self.min_samples = max(5, int(min_samples))
        self._person: int | None = None
        self._name: str | None = None
        self._started = time.time()
        self._valence: deque[float] = deque()
        self._arousal: deque[float] = deque()
        self._states: Counter[str] = Counter()
        self._emotions: Counter[str] = Counter()
        self._units: dict[str, list[float]] = {}
        self._confidences: list[float] = []
        self._weak = 0

    def observe(self, *, person_id: int | None, person_name: str | None,
                valence: float, arousal: float, mood_state: str | None,
                emotion: str | None, confidence: float,
                units: dict[str, float] | None = None,
                now: float | None = None) -> SessionReport | None:
        now = time.time() if now is None else now
        report: SessionReport | None = None

        if person_id != self._person:
            report = self._finalise(now)
            self._reset(person_id, person_name, now)
        elif person_name and not self._name:
            self._name = person_name

        if confidence >= self.MIN_SAMPLE_CONFIDENCE:
            self._valence.append(float(valence))
            self._arousal.append(float(arousal))
            if mood_state:
                self._states[mood_state] += 1
            if emotion:
                self._emotions[emotion] += 1
            for key, value in (units or {}).items():
                self._units.setdefault(key, []).append(float(value))
        else:
            self._weak += 1
        self._confidences.append(float(confidence))

        if report is None and now - self._started >= self.period_s:
            report = self._finalise(now)
            self._reset(person_id, person_name, now)
        return report

    def flush(self, now: float | None = None) -> SessionReport | None:
        now = time.time() if now is None else now
        report = self._finalise(now)
        self._reset(self._person, self._name, now)
        return report

    def reset(self) -> None:
        self._reset(None, None, time.time())

    def progress(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        samples = len(self._valence)
        elapsed = max(0.0, now - self._started)
        return {
            "person_id": self._person,
            "person_name": self._name,
            "elapsed_s": round(elapsed, 1),
            "period_s": self.period_s,
            "samples": samples,
            "min_samples": self.min_samples,
            "progress": round(min(elapsed / self.period_s,
                                  samples / max(1, self.min_samples)), 4),
            "will_report": samples >= self.min_samples,
        }

    def _reset(self, person_id: int | None, name: str | None,
               now: float) -> None:
        self._person = person_id
        self._name = name
        self._started = now
        self._valence.clear()
        self._arousal.clear()
        self._states.clear()
        self._emotions.clear()
        self._units.clear()
        self._confidences.clear()
        self._weak = 0

    def _unit(self, key: str) -> float:
        values = self._units.get(key)
        return float(np.mean(values)) if values else 0.0

    def _finalise(self, now: float) -> SessionReport | None:
        samples = len(self._valence)
        if samples < self.min_samples:
            return None

        valence = float(np.mean(self._valence))
        arousal = float(np.mean(self._arousal))
        swing = float(np.std(self._valence))
        stability = float(np.clip(1.0 - swing * 2.2, 0.0, 1.0))
        engagement = self._unit("engagement")
        fatigue = self._unit("fatigue")
        tension = self._unit("tension")

        state, dominance = self._verdict(
            valence, arousal, stability, fatigue, tension)

        total = float(sum(self._emotions.values())) or 1.0
        mix = {k: v / total for k, v in self._emotions.most_common()}

        coverage = min(1.0, samples / max(1.0, self.min_samples * 3))
        quality = float(np.mean(self._confidences)) if self._confidences else 0.0
        confidence = float(np.clip(
            0.45 * coverage + 0.30 * dominance + 0.25 * quality, 0.0, 1.0))

        return SessionReport(
            person_id=self._person,
            person_name=self._name,
            started_at=self._started,
            ended_at=now,
            samples=samples,
            state=state,
            confidence=confidence,
            dominance=dominance,
            valence=valence,
            arousal=arousal,
            stability=stability,
            engagement=engagement,
            fatigue=fatigue,
            tension=tension,
            emotion_mix=mix,
            notes=self._notes(samples, stability, fatigue, tension, engagement),
        )

    def _verdict(self, valence: float, arousal: float, stability: float,
                 fatigue: float, tension: float) -> tuple[str, float]:
        votes = sum(self._states.values())
        if votes:
            top, count = self._states.most_common(1)[0]
            share = count / votes
            if share >= 0.55:
                return self._map_state(top, fatigue, tension), share

        dominance = 0.0
        if votes:
            dominance = self._states.most_common(1)[0][1] / votes

        if stability < 0.35:
            return "restless", max(dominance, 1.0 - stability)
        if tension > 0.60 and valence < 0.05:
            return "distressed", max(dominance, tension)
        if fatigue > 0.60 and arousal < 0.40:
            return "tired", max(dominance, fatigue)
        if valence <= -0.30:
            return "low", max(dominance, min(1.0, abs(valence)))
        if valence < -0.08:
            return "withdrawn" if arousal < 0.45 else "tense", \
                max(dominance, min(1.0, abs(valence) * 2))
        if valence >= 0.30:
            return "positive", max(dominance, min(1.0, valence))
        if valence > 0.08:
            return "content", max(dominance, min(1.0, valence * 2))
        if arousal < 0.35:
            return "calm", max(dominance, 1.0 - arousal)
        return "neutral", max(dominance, 0.5)

    @staticmethod
    def _map_state(frame_state: str, fatigue: float, tension: float) -> str:
        if frame_state == "volatile":
            return "restless"
        if frame_state == "negative":
            return "low"
        if frame_state == "tense":
            return "distressed" if tension > 0.65 else "tense"
        if frame_state == "neutral" and fatigue > 0.60:
            return "tired"
        return frame_state if frame_state in SESSION_STATES else "neutral"

    def _notes(self, samples: int, stability: float, fatigue: float,
               tension: float, engagement: float) -> tuple[str, ...]:
        notes: list[str] = []
        if self._weak > samples:
            notes.append("face often unclear")
        if stability < 0.40:
            notes.append("mood shifted repeatedly")
        elif stability > 0.85:
            notes.append("steady throughout")
        if fatigue > 0.60:
            notes.append("signs of tiredness")
        if tension > 0.60:
            notes.append("sustained tension")
        if engagement < 0.30:
            notes.append("low engagement")
        return tuple(notes)
