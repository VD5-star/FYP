from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CALIBRATION_STEPS: tuple[dict[str, Any], ...] = (
    {
        "key": "rest",
        "frames": 340,
        "en": "Look at the camera and relax your face",
        "ar": "\u0627\u0646\u0638\u0631 \u0625\u0644\u0649 \u0627\u0644\u0643\u0627\u0645\u064a\u0631\u0627 "
              "\u0648\u0623\u0631\u062e\u0650 \u0645\u0644\u0627\u0645\u062d\u0643",
    },
    {
        "key": "smile",
        "frames": 170,
        "en": "Smile naturally, then relax",
        "ar": "\u0627\u0628\u062a\u0633\u0645 \u0628\u0637\u0628\u064a\u0639\u064a\u0629 "
              "\u062b\u0645 \u0627\u0633\u062a\u0631\u062e\u0650",
    },
    {
        "key": "brows",
        "frames": 150,
        "en": "Raise your eyebrows, then relax",
        "ar": "\u0627\u0631\u0641\u0639 \u062d\u0627\u062c\u0628\u064a\u0643 "
              "\u062b\u0645 \u0627\u0633\u062a\u0631\u062e\u0650",
    },
    {
        "key": "turn",
        "frames": 170,
        "en": "Turn your head slowly left, then right",
        "ar": "\u0623\u062f\u0650\u0631 \u0631\u0623\u0633\u0643 \u0628\u0628\u0637\u0621 "
              "\u064a\u0633\u0627\u0631\u0627\u064b \u062b\u0645 \u064a\u0645\u064a\u0646\u0627\u064b",
    },
    {
        "key": "settle",
        "frames": 170,
        "en": "Relax again and look ahead",
        "ar": "\u0627\u0633\u062a\u0631\u062e\u0650 \u0645\u062c\u062f\u062f\u0627\u064b "
              "\u0648\u0627\u0646\u0638\u0631 \u0644\u0644\u0623\u0645\u0627\u0645",
    },
)

TOTAL_FRAMES = sum(step["frames"] for step in CALIBRATION_STEPS)

REST_FRAMES = sum(step["frames"] for step in CALIBRATION_STEPS
                  if step["key"] in ("rest", "settle"))

REJECT_NO_FACE = "no_face"
REJECT_QUALITY = "low_quality"
REJECT_ANGLE = "extreme_angle"


@dataclass
class CalibrationSession:


    person_id: int | None = None
    person_name: str | None = None
    started_at: float = field(default_factory=time.time)
    step_index: int = 0
    accepted: dict[str, int] = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)
    finished_at: float | None = None

    MIN_QUALITY = 0.35

    MAX_YAW = 32.0
    MAX_PITCH = 26.0

    @property
    def step(self) -> dict[str, Any] | None:
        if self.step_index >= len(CALIBRATION_STEPS):
            return None
        return CALIBRATION_STEPS[self.step_index]

    @property
    def complete(self) -> bool:
        return self.step_index >= len(CALIBRATION_STEPS)

    @property
    def total_accepted(self) -> int:
        return sum(self.accepted.values())

    @property
    def progress(self) -> float:
        return min(1.0, self.total_accepted / max(1, TOTAL_FRAMES))

    def step_progress(self) -> float:
        step = self.step
        if step is None:
            return 1.0
        got = self.accepted.get(step["key"], 0)
        return min(1.0, got / max(1, step["frames"]))

    def offer(self, *, has_face: bool, quality: float,
              yaw: float = 0.0, pitch: float = 0.0) -> str | None:
        if self.complete:
            return None

        step = self.step
        assert step is not None

        if not has_face:
            return self._reject(REJECT_NO_FACE)
        if quality < self.MIN_QUALITY:
            return self._reject(REJECT_QUALITY)
        if step["key"] != "turn":
            if abs(yaw) > self.MAX_YAW or abs(pitch) > self.MAX_PITCH:
                return self._reject(REJECT_ANGLE)

        key = step["key"]
        self.accepted[key] = self.accepted.get(key, 0) + 1
        if self.accepted[key] >= step["frames"]:
            self.step_index += 1
            if self.complete:
                self.finished_at = time.time()
        return None

    def _reject(self, reason: str) -> str:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        return reason

    def hint(self) -> str | None:
        if not self.rejected:
            return None
        total = sum(self.rejected.values()) + self.total_accepted
        if total < 30:
            return None
        reason, count = max(self.rejected.items(), key=lambda kv: kv[1])
        return reason if count / max(1, total) > 0.25 else None

    def to_dict(self) -> dict[str, Any]:
        step = self.step
        return {
            "active": not self.complete,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "step_index": self.step_index,
            "step_count": len(CALIBRATION_STEPS),
            "step_key": step["key"] if step else None,
            "step_en": step["en"] if step else None,
            "step_ar": step["ar"] if step else None,
            "step_progress": round(self.step_progress(), 4),
            "progress": round(self.progress, 4),
            "accepted": self.total_accepted,
            "required": TOTAL_FRAMES,
            "elapsed_s": round(time.time() - self.started_at, 1),
            "hint": self.hint(),
            "complete": self.complete,
        }


class CalibrationManager:


    IDENTITY_SWITCH_FRAMES = 45

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.session: CalibrationSession | None = None
        self._skipped: set[int | None] = set()
        self._pending_person: int | None = None
        self._pending_frames = 0

    def needed_for(self, baseline: Any) -> bool:
        if not self.enabled:
            return False
        if baseline is None:
            return True
        return (not getattr(baseline, "ready", False)
                or getattr(baseline, "expired", False))

    def begin(self, person_id: int | None,
              person_name: str | None = None) -> CalibrationSession:
        self.session = CalibrationSession(
            person_id=person_id, person_name=person_name)
        return self.session

    def ensure(self, person_id: int | None, person_name: str | None,
               baseline: Any) -> CalibrationSession | None:
        if person_id in self._skipped or not self.needed_for(baseline):
            return None

        if self.session is not None and not self.session.complete:
            if self.session.person_id == person_id:
                self._pending_person = None
                self._pending_frames = 0
                return self.session

            if person_id == self._pending_person:
                self._pending_frames += 1
            else:
                self._pending_person = person_id
                self._pending_frames = 1

            if self._pending_frames < self.IDENTITY_SWITCH_FRAMES:
                return self.session

            if person_id is None:
                self._pending_person = None
                self._pending_frames = 0
                return self.session

            self._pending_person = None
            self._pending_frames = 0
            return self.begin(person_id, person_name)

        self._pending_person = None
        self._pending_frames = 0
        return self.begin(person_id, person_name)

    def skip(self) -> None:
        if self.session is not None:
            self._skipped.add(self.session.person_id)
        self.session = None

    def clear(self) -> None:
        self.session = None

    def to_dict(self) -> dict[str, Any]:
        if self.session is None:
            return {"active": False}
        return self.session.to_dict()


def _assert_calibration_can_finish() -> None:
    from .baseline import PersonBaseline

    if PersonBaseline.REQUIRED > REST_FRAMES:
        raise AssertionError(
            f"PersonBaseline.REQUIRED ({PersonBaseline.REQUIRED}) exceeds the "
            f"{REST_FRAMES} resting frames guided calibration collects, so it "
            f"could never finish. Lower REQUIRED or lengthen the rest steps.")


_assert_calibration_can_finish()


def rest_frames_only(session: CalibrationSession | None,
                     probs: np.ndarray) -> bool:
    if session is None or session.complete:
        return True
    step = session.step
    return step is not None and step["key"] in ("rest", "settle")
