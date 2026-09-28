from __future__ import annotations

from dataclasses import dataclass

from reliability import SignedVelocity


@dataclass
class Turnaround:
    """A detected direction reversal."""

    timestamp: float
    at_maximum: bool
    """True when the signal reversed at a peak, false at a trough.

    For a knee angle: a trough is the bottom of a squat, a peak is standing.
    """

    value: float
    """The angle at the moment of reversal — the depth reached."""

    travelled: float
    """Degrees travelled since the previous reversal.

    This is what separates a repetition from a twitch, without needing to know
    the user's absolute range.
    """


@dataclass
class VelocityRepResult:
    """What the detector saw on this frame."""

    completed: bool = False
    count: int = 0
    turnaround: Turnaround | None = None
    depth_reached: float | None = None
    """The extreme value of the repetition that just completed."""

    velocity: float | None = None
    rejected: str | None = None
    """Why a reversal was not counted, when one was seen but discarded."""


class VelocityRepCounter:
    """Counts repetitions from direction reversals in a joint angle.


    A repetition is *two* reversals: down then up. Counting single reversals
    would double-count. So a completed repetition is recorded when the signal
    returns to the same kind of extreme it started from, having passed through
    the opposite one.


    Three guards, each addressing a different failure:

    - **`min_travel`** — the signal must have moved a meaningful distance since
      the last reversal. Expressed in degrees but applied to *relative* travel,
      so it does not encode any assumption about the user's range.
    - **`min_speed`** — the velocity must have been genuinely non-zero before
      reversing. A signal hovering at zero crosses sign endlessly through noise.
    - **`min_interval`** — two reversals cannot be closer together than a human
      can move.
    """

    def __init__(self, min_travel: float = 25.0, min_speed: float = 15.0,
                 min_interval: float = 0.25) -> None:
        self.min_travel = min_travel
        """Degrees the angle must travel between reversals.

        25 degrees is below the ~19 degree published limits of agreement plus
        margin, so it is above the noise but far below the 60 degree dead zone
        the threshold counter needs — which is precisely the advantage of
        working in *relative* rather than absolute terms.
        """

        self.min_speed = min_speed
        """Degrees per second the signal must reach before a reversal counts.

        A resting joint jitters at a few degrees per second; deliberate movement
        is an order of magnitude faster.
        """

        self.min_interval = min_interval
        """Seconds between reversals. Matches the dwell time used elsewhere."""

        self.count = 0
        self.turnarounds: list[Turnaround] = []

        self._velocity = SignedVelocity()
        self._last_reversal_value: float | None = None
        self._last_reversal_time: float | None = None
        self._last_was_maximum: bool | None = None
        self._peak_speed = 0.0
        self._prev_sign = 0
        self._pending_extreme: float | None = None
        self._resting_extreme: bool | None = None
        self._first_value: float | None = None

    def update(self, angle: float | None, timestamp: float) -> VelocityRepResult:
        """Feed one frame's joint angle."""
        rate = self._velocity.update(angle, timestamp)

        if angle is None or rate is None:
            return VelocityRepResult(count=self.count, velocity=rate)

        self._peak_speed = max(self._peak_speed, abs(rate))

        sign = 0 if abs(rate) < 1e-9 else (1 if rate > 0 else -1)
        if sign == 0:
            return VelocityRepResult(count=self.count, velocity=rate)

        if self._first_value is None:
            self._first_value = angle

        if self._prev_sign == 0:
            if self._resting_extreme is None:
                self._resting_extreme = sign < 0
            self._prev_sign = sign
            self._pending_extreme = angle
            return VelocityRepResult(count=self.count, velocity=rate)

        if sign == self._prev_sign:
            self._pending_extreme = angle
            return VelocityRepResult(count=self.count, velocity=rate)

        reversed_at_maximum = self._prev_sign > 0
        self._prev_sign = sign
        extreme = self._pending_extreme if self._pending_extreme is not None else angle
        self._pending_extreme = angle

        peak_speed = self._peak_speed
        self._peak_speed = 0.0

        if peak_speed < self.min_speed:
            return VelocityRepResult(
                count=self.count, velocity=rate,
                rejected='movement too slow to be deliberate',
            )

        if self._last_reversal_time is not None:
            if timestamp - self._last_reversal_time < self.min_interval:
                return VelocityRepResult(
                    count=self.count, velocity=rate,
                    rejected='reversals too close together',
                )

        reference = self._last_reversal_value
        if reference is None:
            reference = self._first_value
        if reference is None:
            return VelocityRepResult(
                count=self.count, velocity=rate,
                rejected='no reference to measure travel from',
            )

        travelled = abs(extreme - reference)
        if travelled < self.min_travel:
            return VelocityRepResult(
                count=self.count, velocity=rate,
                rejected='not enough movement between turns',
            )

        turnaround = Turnaround(
            timestamp=timestamp,
            at_maximum=reversed_at_maximum,
            value=extreme,
            travelled=travelled,
        )
        self.turnarounds.append(turnaround)

        completed = False
        depth = None

        if self._resting_extreme is None:  # pragma: no cover - set on first motion
            self._resting_extreme = reversed_at_maximum
        elif reversed_at_maximum != self._resting_extreme:
            self.count += 1
            completed = True
            depth = extreme

        self._last_was_maximum = reversed_at_maximum
        self._last_reversal_value = extreme
        self._last_reversal_time = timestamp

        return VelocityRepResult(
            completed=completed,
            count=self.count,
            turnaround=turnaround,
            depth_reached=depth,
            velocity=rate,
        )

    def reset(self, keep_counts: bool = True) -> None:
        self._velocity.reset()
        self._last_reversal_value = None
        self._last_reversal_time = None
        self._last_was_maximum = None
        self._peak_speed = 0.0
        self._prev_sign = 0
        self._pending_extreme = None
        self._resting_extreme = None
        self._first_value = None
        if not keep_counts:
            self.count = 0
            self.turnarounds.clear()