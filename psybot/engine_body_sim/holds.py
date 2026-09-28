from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reliability import VelocityTracker


@dataclass
class HoldEvent:
    """A completed hold."""

    duration: float
    """Seconds the position was held."""

    mean_angle: float | None
    """Average tracked joint angle during the hold, when one was supplied."""

    steadiness: float
    """0 to 1, where 1 means perfectly motionless.

    Reported because *how* still someone was is more informative than the bare
    duration: thirty seconds of trembling is a different event from thirty
    seconds of calm, and the engine can say which without interpreting it.
    """


@dataclass
class HoldState:
    """What the detector currently sees."""

    holding: bool
    elapsed: float = 0.0
    """Seconds held so far, zero when not holding."""

    completed: HoldEvent | None = None
    """Set on the frame a hold ends having met the minimum duration."""

    reason: str = ""
    """Why a hold is not in progress, when it is not."""


class HoldDetector:
    """Times how long a position is held still.


    A momentary twitch should not reset a twenty-five second hold. Real holds
    include small corrections — a wobble, a rebalancing — and a detector that
    abandons on the first one would be unusable in practice.

    So movement above the threshold starts a countdown rather than ending the
    hold immediately. Only sustained movement ends it.
    """

    def __init__(self, max_speed: float = 0.85,
                 min_duration: float = 3.0,
                 grace: float = 0.5) -> None:
        self.max_speed = max_speed
        """Landmark speed below which the subject counts as still, in torso
        lengths per second.

        **Measured, after a first attempt was badly wrong.** The initial value
        of 2.0 was reasoned from the noise floor and was far too permissive: a
        person performing continuous full squats was recorded as holding a
        position for eight unbroken seconds.

        Measuring the actual speeds settled it. Median landmark speed, with
        3 px of synthetic noise:

            standing still    0.73    p90 0.78    max 0.85
            small wobble      0.74    p90 0.79    max 0.83
            slow squat        0.86    p90 1.00    max 1.05
            normal squat      1.27    p90 1.67    max 1.77

        0.85 sits at the top of the still distribution and below the bottom of
        the moving one. The gap is narrower than intuition suggests, which is
        precisely why reasoning about it produced a threshold twice as large as
        it should have been.

        Note the still figure is dominated by *measurement noise*, not by the
        person - the same reason the noise floor matters so much elsewhere. On
        a cleaner camera this threshold should be re-measured and will likely
        come down.
        """

        self.min_duration = min_duration
        """Shortest hold worth reporting. Below this it is a pause, not a hold."""

        self.grace = grace
        """Seconds of movement tolerated before a hold is abandoned."""

        self.max_drift = 0.12
        """How far the body may drift from the anchor pose and still count as
        held, in torso lengths.

        This is the threshold that actually decides a hold once one has begun;
        `max_speed` only decides when one *starts*.

        0.12 torso lengths is roughly 6 cm on an adult. Generous enough for
        breathing and postural sway, far below the excursion of any deliberate
        movement - a squat moves the hips by more than a whole torso length.

        Unlike a speed threshold this does not degrade with camera noise,
        because zero-mean noise cancels in the displacement rather than
        accumulating.
        """

        self.velocity = VelocityTracker()
        self.holds: list[HoldEvent] = []

        self._started: float | None = None
        self._anchor: np.ndarray | None = None
        self._anchor_time: float | None = None
        self._last_still: float | None = None
        self._angles: list[float] = []
        self._speeds: list[float] = []

    def update(self, points: np.ndarray, visibility: np.ndarray,
               torso: float | None, timestamp: float,
               angle: float | None = None) -> HoldState:
        speeds = self.velocity.update(points, torso, timestamp)

        if speeds is None:
            return HoldState(holding=self._started is not None,
                             elapsed=self._elapsed(timestamp),
                             reason='waiting for tracking')

        visible = speeds[visibility >= 0.5]
        if visible.size == 0:
            return HoldState(holding=self._started is not None,
                             elapsed=self._elapsed(timestamp),
                             reason='cannot see you clearly')

        speed = float(np.median(visible))

        if self._anchor is None:
            self._anchor = points.copy()
            self._anchor_time = timestamp

        drift = np.linalg.norm(points - self._anchor, axis=1)
        visible_drift = drift[visibility >= 0.5]
        still = (visible_drift.size > 0
                 and float(np.median(visible_drift)) / max(torso or 1.0, 1e-6)
                 <= self.max_drift)

        if still:
            if self._started is None:
                self._started = self._anchor_time or timestamp
                self._angles = []
                self._speeds = []
            self._last_still = timestamp
            if angle is not None:
                self._angles.append(angle)
            self._speeds.append(speed)
            return HoldState(holding=True, elapsed=timestamp - self._started)

        if self._started is None:
            self._anchor = points.copy()
            self._anchor_time = timestamp
            return HoldState(holding=False, reason='moving')

        if self._last_still is not None and timestamp - self._last_still <= self.grace:
            return HoldState(holding=True, elapsed=timestamp - self._started)

        completed = self._finish(self._last_still or timestamp)
        return HoldState(holding=False, completed=completed, reason='moving')

    def _finish(self, end_time: float) -> HoldEvent | None:
        started = self._started
        self._started = None
        self._anchor = None
        self._anchor_time = None
        self._last_still = None

        if started is None:
            return None

        duration = end_time - started
        angles = self._angles
        speeds = self._speeds
        self._angles = []
        self._speeds = []

        if duration < self.min_duration:
            return None

        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        steadiness = float(np.clip(1.0 - mean_speed / self.max_speed, 0.0, 1.0))

        event = HoldEvent(
            duration=duration,
            mean_angle=float(np.mean(angles)) if angles else None,
            steadiness=steadiness,
        )
        self.holds.append(event)
        return event

    def finish(self, timestamp: float) -> HoldEvent | None:
        """Closes an in-progress hold, for when a session ends mid-hold."""
        return self._finish(timestamp)

    def _elapsed(self, timestamp: float) -> float:
        return 0.0 if self._started is None else timestamp - self._started

    @property
    def total_seconds(self) -> float:
        return sum(h.duration for h in self.holds)

    def reset(self, keep_holds: bool = True) -> None:
        self.velocity.reset()
        self._started = None
        self._anchor = None
        self._anchor_time = None
        self._last_still = None
        self._angles = []
        self._speeds = []
        if not keep_holds:
            self.holds.clear()

@dataclass
class Exercise:
    """Everything the engine needs to track one movement.

    Presets exist so that adding an exercise is a data change rather than a
    code change. The set below is deliberately small: each entry names a joint
    that is reliably visible from a phone placed in front of the user, which
    rules out most of what a gym app would offer.
    """

    name: str
    joint: str
    """Which joint to track, as used by `angles.better_side` - so `Knee`
    rather than `leftKnee`."""

    kind: str
    """`reps` or `hold`."""

    inverted: bool = False
    """True when the working position is the *high* angle, as in an arm raise."""

    down_below: float = 100.0
    up_above: float = 160.0
    description: str = ''

    def config(self):
        """The hysteresis thresholds, for `reps` exercises."""
        from movement import HysteresisConfig
        return HysteresisConfig(
            name=self.name,
            down_below=self.down_below,
            up_above=self.up_above,
        )


EXERCISES: dict[str, Exercise] = {
    'squat': Exercise(
        'squat', 'Knee', 'reps',
        down_below=100, up_above=160,
        description='stand side-on or facing the camera',
    ),
    'pushup': Exercise(
        'pushup', 'Elbow', 'reps',
        down_below=100, up_above=155,
        description='place the phone at floor level, side-on',
    ),
    'armraise': Exercise(
        'armraise', 'Shoulder', 'reps',
        inverted=True, down_below=60, up_above=140,
        description='face the camera',
    ),
    'situp': Exercise(
        'situp', 'Hip', 'reps',
        down_below=80, up_above=140,
        description='lie side-on to the camera',
    ),
    'plank': Exercise(
        'plank', 'Hip', 'hold',
        description='side-on, whole body in frame',
    ),
    'stretch': Exercise(
        'stretch', 'Knee', 'hold',
        description='hold the position still',
    ),
    'balance': Exercise(
        'balance', 'Knee', 'hold',
        description='stand on one leg, facing the camera',
    ),
}
