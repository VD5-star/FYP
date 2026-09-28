from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:
    from pose_sim import IDX
except ImportError:  # pragma: no cover
    IDX = {}

class VelocityTracker:
    """Filtered per-landmark velocity, in torso lengths per second.


    Differencing is a high-pass operation: its transfer function is
    ``2·sin(ωT/2)``, so it amplifies exactly the high-frequency noise that
    position filtering was meant to suppress. Differentiating an already-smooth
    position signal still produces a noisy velocity.

    The cutoff used here matches MediaPipe's own choice for the same job.
    `pose_landmark_filtering.pbtxt` sets ``derivate_cutoff: 1.0`` and its comment
    records the resulting coefficient:

        "Derivative cutoff 1.0 results into ~0.17 alpha in landmark EMA filter"

    So 0.17 is not a guess — it is the value Google tuned for smoothing the
    derivative of these same landmarks.


    A velocity in pixels per second means nothing without knowing the distance
    to the camera and the resolution. Dividing by torso length removes both, so
    one threshold works at any distance — which is the same reason torso length
    is used for every other normalisation in this engine.
    """

    ALPHA = 0.17

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._prev_t: float | None = None
        self._velocity: np.ndarray | None = None

    def update(self, points: np.ndarray, torso: float | None,
               timestamp: float) -> np.ndarray | None:
        """Returns filtered per-landmark speed, or None until it has two frames."""
        if torso is None or torso <= 0:
            self.reset()
            return None

        if self._prev is None or self._prev_t is None:
            self._prev = points.copy()
            self._prev_t = timestamp
            return None

        dt = timestamp - self._prev_t
        if dt <= 1e-6 or dt > 0.100:
            self._prev = points.copy()
            self._prev_t = timestamp
            self._velocity = None
            return None

        if dt <= 0:
            return None
        raw = np.linalg.norm(points - self._prev, axis=1) / torso / dt
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if self._velocity is None:
            self._velocity = raw
        else:
            self._velocity = self._velocity + self.ALPHA * (raw - self._velocity)

        self._prev = points.copy()
        self._prev_t = timestamp
        return self._velocity

    def reset(self) -> None:
        self._prev = None
        self._prev_t = None
        self._velocity = None


class SignedVelocity:
    """Filtered signed rate of change of one scalar, such as a joint angle.

    Separate from [VelocityTracker] because a *signed* rate is what reveals a
    turnaround, and a magnitude cannot. Used by the zero-crossing rep detector.
    """

    ALPHA = 0.17

    def __init__(self) -> None:
        self._prev: float | None = None
        self._prev_t: float | None = None
        self._rate: float | None = None

    @property
    def rate(self) -> float | None:
        return self._rate

    def update(self, value: float | None, timestamp: float) -> float | None:
        if value is None or not math.isfinite(value):
            self._prev = None
            self._prev_t = None
            return self._rate

        if self._prev is None or self._prev_t is None:
            self._prev = value
            self._prev_t = timestamp
            return self._rate

        dt = timestamp - self._prev_t
        if dt <= 1e-6 or dt > 0.100:
            self._prev = value
            self._prev_t = timestamp
            self._rate = None
            return None

        raw = (value - self._prev) / dt
        self._rate = raw if self._rate is None else (
            self._rate + self.ALPHA * (raw - self._rate)
        )
        self._prev = value
        self._prev_t = timestamp
        return self._rate

    def reset(self) -> None:
        self._prev = None
        self._prev_t = None
        self._rate = None

BONES: list[tuple[str, str]] = [
    ("leftShoulder", "leftElbow"),
    ("leftElbow", "leftWrist"),
    ("rightShoulder", "rightElbow"),
    ("rightElbow", "rightWrist"),
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),
    ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"),
    ("leftHip", "rightHip"),
]


class BoneLengthMonitor:
    """Flags landmarks whose bone lengths have become impossible.


    A femur does not change length. When the knee landmark snaps onto the other
    leg — the classic pose-estimation failure — the thigh segment changes length
    immediately and unmistakably, while the *velocity* of the jump may be small
    if the legs are close together.

    So this catches the failure that velocity gating misses, and it needs no
    extra inference.


    A limb pointing toward the camera projects shorter. That is correct
    behaviour, not a fault, and it can legitimately halve a segment's apparent
    length.

    Two consequences:

    1. The tolerance must be generous. 35% is chosen to sit above ordinary
       foreshortening during in-plane exercise while still catching a snap.
    2. A **running median** is the reference, not a single calibration frame.
       The median tracks the user's actual working range over recent seconds,
       so a change of posture does not raise a permanent false alarm.
    """

    def __init__(self, tolerance: float = 0.35, window: int = 45,
                 min_samples: int = 15) -> None:
        self.tolerance = tolerance
        """Fractional deviation from the running median that counts as a fault."""

        self.window = window
        """Frames of history for the median. 45 is 1.5 s at 30 fps — long enough
        to be stable, short enough to follow a genuine change of pose."""

        self.min_samples = min_samples
        """Until this many frames are seen, no judgement is made. Reporting a
        fault from a two-frame history would be inventing information."""

        self._history: dict[tuple[str, str], deque[float]] = {
            bone: deque(maxlen=window) for bone in BONES
        }

    def update(self, points: np.ndarray, visibility: np.ndarray,
               torso: float | None) -> set[str]:
        """Returns the names of landmarks implicated in an impossible bone.

        Both endpoints are implicated, because the measurement cannot say which
        of the two moved.
        """
        suspect: set[str] = set()
        if torso is None or torso <= 0 or not IDX:
            return suspect

        for bone in BONES:
            a, b = bone
            if a not in IDX or b not in IDX:  # pragma: no cover
                continue
            ia, ib = IDX[a], IDX[b]

            if visibility[ia] < 0.5 or visibility[ib] < 0.5:
                continue

            length = float(np.linalg.norm(points[ia] - points[ib])) / torso
            history = self._history[bone]

            if len(history) >= self.min_samples:
                reference = float(np.median(history))
                if reference > 1e-6:
                    deviation = abs(length - reference) / reference
                    if deviation > self.tolerance:
                        suspect.add(a)
                        suspect.add(b)
                        continue

            history.append(length)

        return suspect

    def reset(self) -> None:
        for history in self._history.values():
            history.clear()

def angle_3d(world_points: np.ndarray, a: str, vertex: str, c: str) -> float | None:
    """Interior joint angle computed in 3D world space.

    MediaPipe's world landmarks are metric and body-centred, so this angle does
    not change when the subject rotates relative to the camera — unlike the
    projected 2D angle.

    That invariance is real but it is bought with a weakly-supervised depth
    estimate. The depth is a plausible completion from a body model, not a
    measurement, so this is **not** treated as more accurate than the 2D angle.
    It is used only for [angle_disagreement].
    """
    try:
        ia, iv, ic = IDX[a], IDX[vertex], IDX[c]
    except KeyError:  # pragma: no cover
        return None

    v1 = world_points[ia] - world_points[iv]
    v2 = world_points[ic] - world_points[iv]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None

    cosine = float(np.dot(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def angle_disagreement(angle_2d: float | None,
                       angle_3d_value: float | None) -> float | None:
    """How far the projected and 3D angles differ, in degrees.


    Neither angle is ground truth. The 2D angle is wrong when the limb is
    foreshortened; the 3D angle is uncertain when depth is hard to infer — and
    those are the *same* situations. So their disagreement is a usable proxy for
    "this joint is badly positioned relative to the camera".

    Google's own guidance supports treating 2D as primary: their pose
    classification documentation states that 2D angles "vary according to the
    angle between the subject and the camera" and that the best results come
    from a head-on view, then suggests trying the z coordinate to *see if it
    performs better* — a hedge, not a recommendation.

    So this is used to **suppress** feedback, never to correct an angle.
    Showing nothing beats showing a confidently wrong number.
    """
    if angle_2d is None or angle_3d_value is None:
        return None
    return abs(angle_2d - angle_3d_value)

@dataclass
class TrustReport:
    """Why this frame should or should not be believed."""

    trustworthy: bool
    reasons: list[str] = field(default_factory=list)
    suspect_landmarks: set[str] = field(default_factory=set)
    max_speed: float | None = None
    disagreement: float | None = None


class ReliabilityMonitor:
    """Combines the fast checks into one per-frame verdict.

    Deliberately *not* a replacement for the visibility gate — the two catch
    different failures. This adds the fast half that visibility, lagged 233 ms
    by MediaPipe's own smoothing, cannot provide.
    """

    def __init__(self, max_speed: float = 12.0,
                 max_disagreement: float = 30.0) -> None:
        self.max_speed = max_speed
        """Maximum plausible landmark speed, in torso lengths per second.

        A wrist in a fast punch moves roughly 2 torso lengths in 0.2 s, so about
        10/s. 12 leaves headroom above the fastest real movement while still
        rejecting a teleport, which registers in the hundreds.
        """

        self.max_disagreement = max_disagreement
        """Degrees of 2D/3D divergence above which the joint is not trusted.

        Generous, because the published limits of agreement for pose-derived
        joint angles span about 19 degrees even against laboratory equipment.
        A threshold near that width would fire constantly.
        """

        self.velocity = VelocityTracker()
        self.bones = BoneLengthMonitor()

    def update(self, points: np.ndarray, visibility: np.ndarray,
               torso: float | None, timestamp: float,
               angle_2d: float | None = None,
               angle_3d_value: float | None = None) -> TrustReport:
        reasons: list[str] = []

        speeds = self.velocity.update(points, torso, timestamp)
        peak = None
        if speeds is not None:
            visible = speeds[visibility >= 0.5]
            if visible.size:
                peak = float(visible.max())
                if peak > self.max_speed:
                    reasons.append('landmark moved impossibly fast')

        suspect = self.bones.update(points, visibility, torso)
        if suspect:
            reasons.append('limb length changed impossibly')

        disagreement = angle_disagreement(angle_2d, angle_3d_value)
        if disagreement is not None and disagreement > self.max_disagreement:
            reasons.append('2D and 3D angles disagree - check camera angle')

        return TrustReport(
            trustworthy=not reasons,
            reasons=reasons,
            suspect_landmarks=suspect,
            max_speed=peak,
            disagreement=disagreement,
        )

    def reset(self) -> None:
        self.velocity.reset()
        self.bones.reset()
