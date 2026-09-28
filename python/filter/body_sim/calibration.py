from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from movement import HysteresisConfig

MIN_USABLE_RANGE = 35.0

DEAD_ZONE_FRACTION = 0.5


@dataclass
class CalibrationResult:
    """The outcome of a calibration attempt."""

    ok: bool
    reason: str = ""
    observed_min: float | None = None
    observed_max: float | None = None
    config: HysteresisConfig | None = None
    samples: int = 0

    @property
    def observed_range(self) -> float | None:
        if self.observed_min is None or self.observed_max is None:
            return None
        return self.observed_max - self.observed_min


class RangeCalibrator:
    """Learns one joint's range of motion from a guided movement.

    Usage: ask the user to perform the movement slowly two or three times, feed
    every frame's angle, then call `result()`.


    The extremes of a tracked movement are the least reliable samples in it -
    the deepest point of a squat is where the thighs occlude the knees, and a
    single bad frame there would set the threshold for every future session.

    The 5th and 95th percentiles discard that tail. This costs a little range
    and buys immunity to exactly the failure that would be most damaging,
    because a calibration error is permanent in a way a per-frame error is not.
    """

    def __init__(self, name: str, inverted: bool = False,
                 min_samples: int = 60) -> None:
        self.name = name
        self.inverted = inverted
        self.min_samples = min_samples
        """At 30 fps this is two seconds of usable frames.

        Enough to span a slow repetition; low enough not to punish a user who
        moves briskly.
        """
        self._samples: list[float] = []

    def add(self, angle: float | None) -> None:
        """Feed one frame. None (an unreliable angle) is ignored, not stored."""
        if angle is not None:
            self._samples.append(float(angle))

    @property
    def count(self) -> int:
        return len(self._samples)

    def result(self) -> CalibrationResult:
        if len(self._samples) < self.min_samples:
            return CalibrationResult(
                ok=False,
                reason=(f"only {len(self._samples)} usable frames, need "
                        f"{self.min_samples} - move into full view and try again"),
                samples=len(self._samples),
            )

        data = np.array(self._samples)
        low = float(np.percentile(data, 5))
        high = float(np.percentile(data, 95))
        span = high - low

        if span < MIN_USABLE_RANGE:
            return CalibrationResult(
                ok=False,
                reason=(f"range of motion was only {span:.0f} degrees, which is "
                        "too small to set a reliable threshold - the movement "
                        "may not have been performed, or the camera may not "
                        "have seen it clearly"),
                observed_min=low, observed_max=high, samples=len(data),
            )

        margin = span * (1 - DEAD_ZONE_FRACTION) / 2
        down_below = low + margin
        up_above = high - margin

        try:
            config = HysteresisConfig(
                name=self.name,
                down_below=down_below,
                up_above=up_above,
            )
        except ValueError as exc:
            return CalibrationResult(
                ok=False, reason=str(exc),
                observed_min=low, observed_max=high, samples=len(data),
            )

        return CalibrationResult(
            ok=True, observed_min=low, observed_max=high,
            config=config, samples=len(data),
        )

    def reset(self) -> None:
        self._samples.clear()


@dataclass
class UserProfile:
    """Calibrated thresholds for one person, persisted between sessions.

    Stores only joint angles in degrees. No images, no landmark coordinates, no
    identifying information - a body-tracking app should keep the least it can,
    and angles are the least that is useful.
    """

    exercises: dict[str, dict[str, float]] = field(default_factory=dict)

    def store(self, exercise: str, result: CalibrationResult) -> bool:
        """Records a calibration. Returns False if it was not usable."""
        if not result.ok or result.config is None:
            return False
        self.exercises[exercise] = {
            "down_below": result.config.down_below,
            "up_above": result.config.up_above,
            "observed_min": result.observed_min or 0.0,
            "observed_max": result.observed_max or 0.0,
        }
        return True

    def config_for(self, exercise: str,
                   default: HysteresisConfig) -> HysteresisConfig:
        """The calibrated config, or the supplied default.

        Falling back to the default is the correct behaviour for an uncalibrated
        user, and the caller does not need to special-case it.
        """
        stored = self.exercises.get(exercise)
        if stored is None:
            return default
        try:
            return HysteresisConfig(
                name=exercise,
                down_below=stored["down_below"],
                up_above=stored["up_above"],
                confirm_frames=default.confirm_frames,
                min_dwell_seconds=default.min_dwell_seconds,
            )
        except (KeyError, ValueError):
            return default

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"exercises": self.exercises}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "UserProfile":
        """Loads a profile, returning an empty one if anything is wrong.

        A corrupt profile means an uncalibrated user, which is a state the rest
        of the system already handles. It does not mean a crash.
        """
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            exercises = raw.get("exercises", {})
            if not isinstance(exercises, dict):
                return cls()
            return cls(exercises=exercises)
        except (OSError, json.JSONDecodeError, AttributeError):
            return cls()