from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MoodReading:
    mood_type: str = "neutral"
    mood_ar: str = "محايد"
    age_group: str = "adult"
    age_group_ar: str = "بالغ"
    gender: str = "male"
    gender_ar: str = "ذكر"

    expression_stable: float = 1.0
    recent_mood_changes: int = 0

    summary: str = ""
    summary_ar: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood_type": self.mood_type,
            "mood_ar": self.mood_ar,
            "age_group": self.age_group,
            "age_group_ar": self.age_group_ar,
            "gender": self.gender,
            "gender_ar": self.gender_ar,
            "expression_stable": round(self.expression_stable, 4),
            "recent_mood_changes": self.recent_mood_changes,
            "summary": self.summary,
            "summary_ar": self.summary_ar,
        }


class MoodAnalyzer:


    POSITIVE_MIN = 0.3
    NEGATIVE_MAX = -0.2

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self._valences: deque[float] = deque(maxlen=window)
        self._arousals: deque[float] = deque(maxlen=window)
        self._labels: deque[str] = deque(maxlen=window)
        self._age_votes: deque[str] = deque(maxlen=10)
        self._gender_votes: deque[str] = deque(maxlen=10)
        self._last_mood = "neutral"
        self._mood_changes = 0
        self._last_change_time = time.time()

    def update(self, emotion: Any, pose: Any = None,
               quality: float = 1.0) -> MoodReading:
        reading = MoodReading()

        probs = getattr(emotion, "probs", {}) or {}
        valence = float(getattr(emotion, "valence", 0.0))
        arousal = float(getattr(emotion, "arousal", 0.0))
        label = getattr(emotion, "label", "neutral")

        if valence >= self.POSITIVE_MIN:
            mood = "positive"
            mood_ar = "إيجابي"
        elif valence <= self.NEGATIVE_MAX:
            mood = "negative"
            mood_ar = "سلبي"
        else:
            if probs.get("neutral", 0) > 0.6:
                mood = "neutral"
                mood_ar = "محايد"
            else:
                top = max(probs.items(), key=lambda kv: kv[1]) if probs else ("neutral", 0)
                if top[0] in ("happy", "surprise"):
                    mood = "positive"
                    mood_ar = "إيجابي"
                elif top[0] in ("sad", "fear", "disgust", "anger"):
                    mood = "negative"
                    mood_ar = "سلبي"
                else:
                    mood = "neutral"
                    mood_ar = "محايد"

        reading.mood_type = mood
        reading.mood_ar = mood_ar

        age_votes = self._vote_age(probs, pose, quality)
        if age_votes:
            self._age_votes.extend(age_votes)
        age_counts = {}
        for v in self._age_votes:
            age_counts[v] = age_counts.get(v, 0) + 1
        if age_counts:
            reading.age_group = max(age_counts, key=age_counts.get)
            reading.age_group_ar = self._age_ar(reading.age_group)

        if pose and hasattr(pose, "gender"):
            gender = getattr(pose, "gender", "unknown")
            if gender in ("male", "female"):
                self._gender_votes.append(gender)
        if self._gender_votes:
            gender_counts = {}
            for g in self._gender_votes:
                gender_counts[g] = gender_counts.get(g, 0) + 1
            reading.gender = max(gender_counts, key=gender_counts.get)
            reading.gender_ar = self._gender_ar(reading.gender)

        self._valences.append(valence)
        self._arousals.append(arousal)
        self._labels.append(label)

        if len(self._labels) >= 2:
            changes = sum(1 for a, b in zip(self._labels, list(self._labels)[1:]) if a != b)
            reading.recent_mood_changes = min(changes, 5)
            reading.expression_stable = 1.0 - (changes / max(len(self._labels) - 1, 1))

        reading.summary = self._build_summary(reading, mood, label)
        reading.summary_ar = self._build_summary_ar(reading, mood_ar, label)

        return reading

    def _vote_age(self, probs: dict, pose: Any, quality: float) -> list[str]:
        votes = []

        if pose and hasattr(pose, "age"):
            age = getattr(pose, "age", 25)
            if age < 12:
                votes.append("child")
            elif age < 25:
                votes.append("young")
            elif age < 60:
                votes.append("adult")
            else:
                votes.append("senior")

        if probs.get("surprise", 0) > 0.5:
            votes.append("child")

        if hasattr(pose, "arousal") and getattr(pose, "arousal", 0) > 0.6:
            votes.append("young")

        return votes[:3]

    @staticmethod
    def _age_ar(age: str) -> str:
        return {"child": "طفل", "young": "شاب", "adult": "بالغ", "senior": "مسن"}.get(age, age)

    @staticmethod
    def _gender_ar(gender: str) -> str:
        return {"male": "ذكر", "female": "أنثى", "unknown": "غير معروف"}.get(gender, gender)

    def _build_summary(self, reading: MoodReading, mood: str, label: str) -> str:
        parts = [reading.age_group, reading.gender, mood, label]
        return " ".join(p for p in parts if p)

    def _build_summary_ar(self, reading: MoodReading, mood_ar: str, label: str) -> str:
        parts = [
            reading.age_group_ar,
            reading.gender_ar,
            mood_ar,
            label
        ]
        return " ".join(p for p in parts if p)

    def reset(self) -> None:
        self._valences.clear()
        self._arousals.clear()
        self._labels.clear()
        self._age_votes.clear()
        self._gender_votes.clear()
        self._last_mood = "neutral"
        self._mood_changes = 0

    def summary(self) -> dict[str, Any]:
        if not self._labels:
            return {"samples": 0}
        labels = list(self._labels)
        dominant = max(set(labels), key=labels.count)
        return {
            "samples": len(labels),
            "dominant_mood": max(set([l for l in labels]), key=labels.count),
            "dominant_emotion": dominant,
            "stability": labels.count(dominant) / len(labels),
            "mean_valence": float(np.mean(self._valences)),
            "mean_arousal": float(np.mean(self._arousals)),
        }
