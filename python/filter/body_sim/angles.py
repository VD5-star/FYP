from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from pose_sim import IDX
except ImportError:  # pragma: no cover
    IDX = {}


@dataclass(frozen=True)
class JointAngle:
    """An interior joint angle with the confidence of its weakest landmark.

    The confidence is the *minimum* of the three landmark visibilities, not the
    mean. A mean would let two confident landmarks disguise one the model is
    guessing at, and the guessed one moves the angle just as much as the others.
    """

    degrees: float
    confidence: float

    RELIABLE = 0.5

    @property
    def reliable(self) -> bool:
        return self.confidence >= self.RELIABLE

    def or_none(self) -> float | None:
        """The angle, or None when it should not be trusted.

        This is the form `RepCounter.update` expects, because it treats None as
        *absence of information* rather than as a state - which is why
        intermittent occlusion was measured to have no effect on counting.
        """
        return self.degrees if self.reliable else None


def joint_angle(points: np.ndarray, visibility: np.ndarray,
                a: str, vertex: str, c: str) -> JointAngle | None:
    """Interior angle at `vertex`, between the segments to `a` and `c`.

    Returns None only when a landmark name is unknown or the points coincide -
    a *structural* failure. Low confidence is reported through the result
    rather than by returning None, so the caller can decide.
    """
    try:
        ia, iv, ic = IDX[a], IDX[vertex], IDX[c]
    except KeyError:  # pragma: no cover
        return None

    if not (np.isfinite(points[ia]).all() and np.isfinite(points[iv]).all()
            and np.isfinite(points[ic]).all()):
        return None

    v1 = points[ia] - points[iv]
    v2 = points[ic] - points[iv]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if not math.isfinite(n1) or not math.isfinite(n2) or n1 < 1e-6 or n2 < 1e-6:
        return None

    cosine = float(np.dot(v1, v2)) / (n1 * n2)
    degrees = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    confidence = float(min(visibility[ia], visibility[iv], visibility[ic]))
    return JointAngle(degrees, confidence)


JOINTS: dict[str, tuple[str, str, str]] = {
    "leftKnee": ("leftHip", "leftKnee", "leftAnkle"),
    "rightKnee": ("rightHip", "rightKnee", "rightAnkle"),
    "leftHip": ("leftShoulder", "leftHip", "leftKnee"),
    "rightHip": ("rightShoulder", "rightHip", "rightKnee"),
    "leftElbow": ("leftShoulder", "leftElbow", "leftWrist"),
    "rightElbow": ("rightShoulder", "rightElbow", "rightWrist"),
    "leftShoulder": ("leftElbow", "leftShoulder", "leftHip"),
    "rightShoulder": ("rightElbow", "rightShoulder", "rightHip"),
}


def all_angles(points: np.ndarray,
               visibility: np.ndarray) -> dict[str, JointAngle]:
    """Every tracked joint angle. Structurally impossible ones are omitted."""
    out: dict[str, JointAngle] = {}
    for name, (a, vertex, c) in JOINTS.items():
        angle = joint_angle(points, visibility, a, vertex, c)
        if angle is not None:
            out[name] = angle
    return out


MAX_ANGLE_RATE = 500.0


class AngleRateGuard:
    """Rejects a joint angle that changed faster than a joint can move.

    Stateful per joint. Rejects rather than corrects: a rejected angle becomes
    "no information", which every consumer in this engine already handles,
    whereas a corrected one would be a number nobody measured.
    """

    def __init__(self, max_rate: float = MAX_ANGLE_RATE,
                 reset_gap: float = 0.5) -> None:
        self.max_rate = max_rate
        self.reset_gap = reset_gap
        """Seconds after which history is discarded.

        A gap in detection is a discontinuity, not fast movement. Judging a
        post-gap angle against a pre-gap one would reject a perfectly good
        measurement for having been separated by seconds.
        """
        self._last: dict[str, tuple[float, float]] = {}

    def check(self, name: str, degrees: float, timestamp: float) -> bool:
        """Whether this angle is believable. Records it if so."""
        previous = self._last.get(name)
        if previous is not None:
            dt = timestamp - previous[0]
            if dt <= 0:
                return True
            if dt <= self.reset_gap:
                if abs(degrees - previous[1]) / dt > self.max_rate:
                    return False
        self._last[name] = (timestamp, degrees)
        return True

    def filter(self, angles: dict[str, JointAngle],
               timestamp: float) -> dict[str, JointAngle]:
        """Drop every angle that changed impossibly fast."""
        return {n: a for n, a in angles.items()
                if self.check(n, a.degrees, timestamp)}

    def reset(self) -> None:
        self._last.clear()


def better_side(angles: dict[str, JointAngle], joint: str) -> JointAngle | None:
    """The more reliable of the left and right versions of a joint.


    A phone sits to one side of the user more often than not, and the far leg
    is then partly hidden behind the near one. Measuring a fixed side means
    measuring the occluded one roughly half the time.

    Picking the more confident side each frame is better - but it introduces a
    hazard: **switching sides mid-repetition changes the measured value**, and
    a state machine cannot distinguish that from the user moving. `SideTracker`
    exists to handle that; this function is the raw per-frame choice.
    """
    left = angles.get(f"left{joint}")
    right = angles.get(f"right{joint}")
    if left is None:
        return right
    if right is None:
        return left
    return left if left.confidence >= right.confidence else right


class SideTracker:
    """Chooses a body side and sticks to it unless the other is clearly better.


    Choosing the more confident side per frame sounds obviously right, and it is
    wrong. Left and right knee angles differ by several degrees even in
    symmetric movement, so switching sides mid-descent produces a step change in
    the measured angle that looks exactly like the user moving. Near a
    threshold, alternating sides produces alternating states.

    So a side is held once chosen, and given up only when the other side is
    **clearly** better for **several consecutive frames** - the same hysteresis
    reasoning `movement.py` applies to states, applied to the choice of which
    limb to watch.
    """

    def __init__(self, margin: float = 0.25, patience: int = 5) -> None:
        self.margin = margin
        """How much more confident the other side must be before switching.

        0.25 on MediaPipe's 0-1 visibility scale: a clear difference, not a
        marginal one.
        """

        self.patience = patience
        """Consecutive frames the other side must be better before switching.

        Five frames is about 170 ms at 30 fps - long enough that a brief
        occlusion of the tracked side does not cause a switch.
        """

        self._side: str | None = None
        self._pressure = 0

    @property
    def side(self) -> str | None:
        return self._side

    def update(self, angles: dict[str, JointAngle],
               joint: str) -> JointAngle | None:
        left = angles.get(f"left{joint}")
        right = angles.get(f"right{joint}")

        if left is None and right is None:
            return None
        if left is None:
            self._side, self._pressure = "right", 0
            return right
        if right is None:
            self._side, self._pressure = "left", 0
            return left

        if self._side is None:
            self._side = "left" if left.confidence >= right.confidence else "right"
            self._pressure = 0

        current = left if self._side == "left" else right
        other = right if self._side == "left" else left

        if other.confidence > current.confidence + self.margin:
            self._pressure += 1
            if self._pressure >= self.patience:
                self._side = "right" if self._side == "left" else "left"
                self._pressure = 0
                return other
        else:
            self._pressure = 0

        return current

    def reset(self) -> None:
        self._side = None
        self._pressure = 0


def symmetry(angles: dict[str, JointAngle], joint: str) -> float | None:
    """Absolute left-right difference in degrees, or None if either is unusable.

    Useful as a **form** signal rather than a movement one: a large asymmetry in
    a squat means the user is favouring one leg.

    It must not be read as a clinical sign. It is also the first thing to
    degrade when the subject turns, because the far limb foreshortens - so it
    should only be consulted when `viewpoint_offset_degrees` is small.
    """
    left = angles.get(f"left{joint}")
    right = angles.get(f"right{joint}")
    if left is None or right is None:
        return None
    if not (left.reliable and right.reliable):
        return None
    return abs(left.degrees - right.degrees)