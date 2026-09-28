from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    from pose_sim import IDX, LANDMARK_NAMES
except ImportError:  # pragma: no cover - only when used standalone
    LANDMARK_NAMES = []
    IDX = {}

GAP_RESET_SECONDS = 0.100


def sanitise(points: np.ndarray) -> np.ndarray:
    """Replaces non-finite coordinates with zero.


    Non-finite coordinates arrive from real sources: a tracker that cannot
    place a landmark, or an arithmetic edge case upstream. They are far more
    damaging than they look.

    ``inf - inf`` is NaN, so one infinite coordinate becomes NaN at the first
    subtraction. A NaN then enters an exponential filter and **stays there for
    ever**, since ``NaN + alpha * (x - NaN)`` is NaN. Every later comparison
    against it is False, so no direction is established and the counters stop
    counting *silently*.

    Measured before this guard: one NaN landmark mid-session took repetition
    counting from 7 to 4 out of 7, and the three lost repetitions were every one
    that followed.

    Zero is substituted rather than the row dropped, so the array keeps its
    shape and its indices still match the landmark list. A landmark at the
    origin is then rejected by the ordinary confidence and geometry checks,
    which is the correct outcome: it is a landmark we do not have.
    """
    if np.all(np.isfinite(points)):
        return points
    return np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)


class MedianFilter:
    """Median filter over a short window, per coordinate.

    Median rather than mean because the failure being targeted is not Gaussian
    noise - it is **spikes and left/right limb swaps**, documented in Aderinola
    et al. 2023 as a real and frequent pose-estimation failure. An order
    statistic rejects an outlier completely; any linear filter blends it in and
    spreads it across neighbouring frames.

    Width 5 follows Pose Trainer (Chen & Yang, arXiv:2006.11718), which applies
    it twice for the same reason.
    """

    def __init__(self, width: int = 5) -> None:
        if width % 2 == 0:
            raise ValueError("median window must be odd")
        self.width = width
        self._buf: deque[np.ndarray] = deque(maxlen=width)
        self._last_t: float | None = None

    def __call__(self, points: np.ndarray, timestamp: float) -> np.ndarray:
        if not np.all(np.isfinite(points)):
            points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

        if self._last_t is not None and timestamp - self._last_t > GAP_RESET_SECONDS:
            self._buf.clear()
        self._last_t = timestamp

        self._buf.append(points.copy())
        if len(self._buf) < self.width:
            return points
        return np.median(np.stack(self._buf), axis=0)

    def reset(self) -> None:
        self._buf.clear()
        self._last_t = None


def viewpoint_offset_degrees(points: np.ndarray, visibility: np.ndarray) -> float | None:
    """How far the subject is turned away from the camera, in degrees.

    Measured as the difference between the two nose-to-shoulder distances,
    expressed as an angle.


    LearnOpenCV's `offset_angle` is the angle at the nose subtended by the two
    shoulders. The first version here assumed that angle is widest when facing
    the camera and shrinks in profile, so it reported ``abs(180 - angle)``.

    Measured on a front-facing pose, that angle is about **63 degrees**, not
    180 - the nose sits above and between the shoulders, so it subtends a
    modest angle. The gate therefore reported every front-facing subject as
    117 degrees off and refused to analyse anything. The test caught it.

    What actually distinguishes facing from profile is **symmetry**: face-on,
    the nose is equidistant from both shoulders; in profile, the near shoulder
    is much closer than the far one. So the asymmetry of those two distances is
    the signal, and it is converted to degrees so the 35 degree threshold from
    LearnOpenCV remains meaningful.

    Declining beyond that threshold is the point: **produce no number rather
    than a wrong one.** Torso normalisation degrades as the subject turns, and
    no filter recovers a foreshortened limb.

    Returns None when the landmarks needed are unreliable.
    """
    try:
        nose = IDX["nose"]
        ls, rs = IDX["leftShoulder"], IDX["rightShoulder"]
    except KeyError:  # pragma: no cover
        return None

    for i in (nose, ls, rs):
        if visibility[i] < 0.5:
            return None

    points = sanitise(points)

    h_left = abs(float(points[ls][0] - points[nose][0]))
    h_right = abs(float(points[rs][0] - points[nose][0]))
    total = h_left + h_right
    if total < 1e-6:
        return None

    asymmetry = abs(h_left - h_right) / total

    return asymmetry * 90.0


def torso_yaw_degrees(world_points: np.ndarray) -> float | None:
    """True torso yaw in degrees, from 3D world landmarks.

    0 is square to the camera, 90 is full profile.


    MediaPipe emits `pose_world_landmarks` alongside the screen landmarks:
    metric coordinates in a body-centred frame, including depth. The angle
    between the shoulder line and the image plane is then a direct
    trigonometric result rather than something inferred from foreshortening.

    Validated on a real photograph where the 2D nose-based estimate reported
    64 degrees and refused the frame: this returns **6.7 degrees**, which
    matches the visible pose.

    The shoulder line is used rather than the hips because shoulders are the
    most reliably detected pair in the set, and because it is the *torso*
    orientation that determines whether limb angles are foreshortened.

    Depth accuracy from a single camera is the weakest part of any monocular
    pose estimate - flagged in round 3 - but yaw only needs the *ratio* of
    depth to width, which is far more robust than absolute depth.
    """
    try:
        ls, rs = IDX["leftShoulder"], IDX["rightShoulder"]
    except KeyError:  # pragma: no cover
        return None

    dx = float(world_points[ls][0] - world_points[rs][0])
    dz = float(world_points[ls][2] - world_points[rs][2])
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return None

    return abs(math.degrees(math.atan2(dz, dx)))


def shoulder_torso_ratio(points: np.ndarray, visibility: np.ndarray) -> float | None:
    """Shoulder width divided by torso length - a yaw indicator.

    Round 3 notes that shoulder width is the *worst* choice for scale
    normalisation, because it collapses toward zero at 90 degrees of yaw. That
    same property makes it an excellent **detector** of yaw.

    Roughly 0.6-0.8 facing front; it falls sharply in profile.
    """
    try:
        ls, rs = IDX["leftShoulder"], IDX["rightShoulder"]
        lh, rh = IDX["leftHip"], IDX["rightHip"]
    except KeyError:  # pragma: no cover
        return None

    for i in (ls, rs, lh, rh):
        if visibility[i] < 0.5:
            return None

    width = float(np.linalg.norm(points[ls] - points[rs]))
    mid_sh = (points[ls] + points[rs]) / 2
    mid_hip = (points[lh] + points[rh]) / 2
    torso = float(np.linalg.norm(mid_sh - mid_hip))
    if torso < 1e-6:
        return None
    return width / torso


class SubjectSwitchDetector:
    """Detects the tracker jumping to a different person.

    MediaPipe tracks one person and will switch subjects without warning if
    another enters the frame. A switch is physically impossible to distinguish
    from teleportation, which is exactly what makes it detectable: a real torso
    cannot change length by 20% in one frame, and real hips cannot move half a
    torso-length between frames.

    On detection every downstream state machine must be reset and any
    in-progress repetition discarded - otherwise one person's descent is paired
    with another person's ascent.


    The first version compared torso length against the previous frame using a
    fixed 20% threshold. It fired constantly during squats, because a real
    squat changes the *apparent* torso length in the image by over 100% as the
    body folds and foreshortens. The movement is large; what makes it a
    movement rather than a substitution is that it is **gradual**.

    So the thresholds are per-second rates. At 30 fps a limit of 250% per second
    allows an 8% change between adjacent frames - far more than any real body
    motion produces, far less than a subject switch.


    The `hip_rate` limit of 9.0 was justified as leaving "generous headroom for
    a jump, which moves the hips faster than a squat does". That was reasoned,
    not measured, and measurement contradicted it: an explosive countermovement
    jump reaches **13.9 torso-lengths per second** at push-off. The guard fired
    mid-jump, reset the counters and discarded the repetition - the glitch the
    user reported.

    The obvious fix - raise the number - is wrong, and measuring the other
    population shows why:

        explosive jump, push-off            13.92 /s
        switch to same-build person,
          half a torso away                 15.00 /s

    **The populations overlap.** No threshold on hip speed can separate them.
    Direction does not help either: a real crouch moves the hips backwards, so
    horizontal motion is not exclusive to a switch. Nor does predicting the
    hip's position, because a jump is precisely an acceleration.

    So the thresholds are moved above every measured real movement instead of
    sitting inside them.


    The first attempt at a fix required the anomaly to hold for two consecutive
    frames, on the reasoning that a jump's peak is transient while a switch
    persists. **A test disproved it.** A genuine switch is anomalous on exactly
    one frame - the transition - and the substituted person is then perfectly
    steady, so "two consecutive anomalous frames" never occurs and the guard
    stopped detecting switches entirely. The reasoning confused *the person
    persisting* with *the rate staying high*. Only the first is true.


    Measured over 399 physiologically-timed movements at 30 fps - jumps of
    5-75 cm with push-off 133-300 ms and landing absorption 100-200 ms, per the
    countermovement-jump literature, plus squats at five depths and three
    tempos - with landmark noise at the 1.09 px measured on a real human
    (`docs/measurements-real-human.md`):

        worst real movement       torso 2.66 /s    hip 17.50 /s
        switch, 15% build change  torso 4.50 /s
        switch, one torso away                     hip 30.00 /s

    False positives over those 399 movements:

        torso 7.0, hip 9.0  (before)    186 / 399
        torso 4.0, hip 20.0 (now)         0 / 399


    A switch to a person of **the same build standing within half a torso** of
    the previous subject is not detectable from these quantities. It gives a
    hip rate of 15 /s, inside the jump population, and no torso change at all.
    That is a limit of the measurement, not a defect to be thresholded away -
    lowering the number until that case is caught is exactly what made the
    guard fire on jumps.

    Of 35 sampled switches, 26 are caught. The nine missed are all
    small-build-difference, small-displacement cases.
    """

    def __init__(self, torso_rate: float = 4.0,
                 hip_rate: float = 20.0) -> None:
        self.torso_rate = torso_rate
        """Maximum fractional change in torso length per second.

        Worst real movement over 399 physiological sequences, with 1.09 px of
        landmark noise: 2.66 /s. A switch to a person 15% different in build
        gives 4.50 /s. 4.0 sits in that gap.

        The previous value of 7.0 was above every build change under 23%, so it
        contributed almost nothing; the hip test was doing all the work, and
        firing on jumps while it did.
        """

        self.hip_rate = hip_rate
        """Maximum hip displacement per second, in torso lengths.

        Worst real movement: 17.50 /s, at an explosive push-off. 20.0 clears it
        with 1.14x margin.

        The previous value of 9.0 sat *inside* the jump population - an
        explosive push-off reaches 13.92 /s - which is what caused the guard to
        fire mid-jump, reset the counters, and discard the repetition.
        """

        self._torso: float | None = None
        self._hip: np.ndarray | None = None
        self._last_t: float | None = None

    def __call__(self, points: np.ndarray, visibility: np.ndarray,
                 timestamp: float) -> bool:
        """Returns True when the subject appears to have changed."""
        try:
            ls, rs = IDX["leftShoulder"], IDX["rightShoulder"]
            lh, rh = IDX["leftHip"], IDX["rightHip"]
        except KeyError:  # pragma: no cover
            return False

        if any(visibility[i] < 0.5 for i in (ls, rs, lh, rh)):
            return False

        mid_sh = (points[ls] + points[rs]) / 2
        mid_hip = (points[lh] + points[rh]) / 2
        torso = float(np.linalg.norm(mid_sh - mid_hip))
        if torso < 1e-6:
            return False

        gapped = (self._last_t is not None
                  and timestamp - self._last_t > GAP_RESET_SECONDS * 5)
        dt = 0.0 if self._last_t is None else timestamp - self._last_t
        self._last_t = timestamp

        switched = False
        if (self._torso is not None and self._hip is not None
                and not gapped and dt > 1e-6 and self._torso > 1e-6):
            torso_rate = abs(torso - self._torso) / self._torso / dt
            hip_rate = float(np.linalg.norm(mid_hip - self._hip)) / torso / dt
            switched = torso_rate > self.torso_rate or hip_rate > self.hip_rate

        self._torso = torso
        self._hip = mid_hip
        return switched

    def reset(self) -> None:
        self._torso = None
        self._hip = None
        self._last_t = None

DEMPSTER_SEGMENTS: list[tuple[float, tuple[str, ...]]] = [
    (0.430, ("leftShoulder", "rightShoulder", "leftHip", "rightHip")),
    (0.100, ("leftHip", "leftKnee")),
    (0.100, ("rightHip", "rightKnee")),
    (0.0465, ("leftKnee", "leftAnkle")),
    (0.0465, ("rightKnee", "rightAnkle")),
    (0.015, ("leftAnkle", "leftFootIndex")),
    (0.015, ("rightAnkle", "rightFootIndex")),
    (0.028, ("leftShoulder", "leftElbow")),
    (0.028, ("rightShoulder", "rightElbow")),
    (0.022, ("leftElbow", "leftWrist")),
    (0.022, ("rightElbow", "rightWrist")),
]

MIN_MASS_FRACTION = 0.60


def centre_of_mass(points: np.ndarray, visibility: np.ndarray,
                   threshold: float = 0.5) -> np.ndarray | None:
    """Mass-weighted centre of mass, or None when too much of the body is
    unreliable.

    Refusing is important. A CoM computed from three visible landmarks is not a
    worse CoM - it is a different quantity, and reporting it as the same one
    would be inventing data.
    """
    if not IDX:  # pragma: no cover
        return None

    points = sanitise(points)
    total = 0.0
    acc = np.zeros(2, dtype=np.float64)
    for mass, names in DEMPSTER_SEGMENTS:
        idxs = [IDX[n] for n in names if n in IDX]
        if len(idxs) != len(names):  # pragma: no cover
            continue
        if any(visibility[i] < threshold for i in idxs):
            continue
        acc += mass * points[idxs].mean(axis=0)
        total += mass

    if total < MIN_MASS_FRACTION:
        return None
    return acc / total


def torso_length(points: np.ndarray, visibility: np.ndarray,
                 threshold: float = 0.5) -> float | None:
    """Mid-hip to mid-shoulder distance: the scale for every normalised value.

    Chosen over shoulder width because it is mostly vertical regardless of yaw,
    so it degrades gracefully as the subject turns. This is also what ML Kit's
    `PoseEmbedding.getPoseSize` uses.
    """
    try:
        ls, rs = IDX["leftShoulder"], IDX["rightShoulder"]
        lh, rh = IDX["leftHip"], IDX["rightHip"]
    except KeyError:  # pragma: no cover
        return None

    if any(visibility[i] < threshold for i in (ls, rs, lh, rh)):
        return None

    points = sanitise(points)
    mid_sh = (points[ls] + points[rs]) / 2
    mid_hip = (points[lh] + points[rh]) / 2
    length = float(np.linalg.norm(mid_sh - mid_hip))
    return length if math.isfinite(length) and length > 1e-6 else None

class RepState(Enum):
    """Where in a repetition the subject is."""

    UP = "up"
    """The resting or extended position - standing, arms down."""

    DOWN = "down"
    """The working position - squat bottom, push-up bottom, arms raised."""

    BETWEEN = "between"
    """Inside the dead zone. Deliberately a state of its own rather than a
    forced binary choice: it degrades to 'hold the previous state', which is
    what makes narrow-band jitter harmless."""


@dataclass
class HysteresisConfig:
    """Thresholds for one exercise.


    Two independent lines of evidence converge on a floor of about 20 degrees:

    * **Measured.** Knee-angle limits of agreement are -6.7 to +11.9 degrees
      against a laboratory system. A narrower band sits inside the noise.
    * **Shipped.** ML Kit's `RepetitionCounter` enters at 6/10 and exits at
      4/10 - a dead zone of 20% of the signal's range. A squat's knee angle
      spans roughly 170 to 80 degrees, so 20% of that range is 18 degrees.

    The defaults below double that floor, because our users are in bedrooms with
    a propped-up phone rather than in a laboratory.


    LearnOpenCV's widely-copied thresholds are **angles to the vertical**, not
    interior joint angles - their "95 degrees" means a thigh near horizontal.
    Everything here is an **interior** angle: 180 is a straight limb.
    """

    name: str
    down_below: float
    """Enter DOWN when the angle falls below this."""

    up_above: float
    """Enter UP when the angle rises above this."""

    confirm_frames: int = 3
    """Consecutive frames the condition must hold before the state changes.

    Kills single-frame spikes that hysteresis alone cannot: a spike wide enough
    to cross a 60-degree dead zone is still only one frame long.
    """

    min_dwell_seconds: float = 0.30
    """Minimum time in a state before it may change.

    A human squat bottoms out for longer than 300 ms. A faster transition is
    noise, not movement.
    """

    def __post_init__(self) -> None:
        if self.up_above <= self.down_below:
            raise ValueError(
                f"{self.name}: up_above must exceed down_below, or there is no "
                "dead zone and the state will chatter"
            )
        if self.dead_zone < 20.0:
            raise ValueError(
                f"{self.name}: dead zone of {self.dead_zone:.0f} deg is inside "
                "the measured noise band (published LoA is -6.7 to +11.9 deg "
                "on knee angle). Widen it to at least 20 deg."
            )

    @property
    def dead_zone(self) -> float:
        return self.up_above - self.down_below


SQUAT = HysteresisConfig("squat", down_below=100.0, up_above=160.0)
PUSHUP = HysteresisConfig("pushup", down_below=100.0, up_above=155.0)

ARM_RAISE = HysteresisConfig("arm_raise", down_below=60.0, up_above=140.0)


@dataclass
class RepResult:
    """What changed on this frame."""

    state: RepState
    completed: bool = False
    """A full, valid repetition finished on this frame."""

    partial: bool = False
    """The subject went partway down and came back up without reaching the
    bottom. Reported rather than ignored, because 'you did not go deep enough'
    is useful feedback and silence is not."""

    count: int = 0
    """Total valid repetitions so far."""

    partial_count: int = 0


class RepCounter:
    """Counts repetitions from a single joint angle.


    Counting on a rising edge can be fooled by any excursion that crosses the
    threshold. This requires the full ``UP -> BETWEEN -> DOWN -> BETWEEN -> UP``
    journey, following LearnOpenCV's `state_seq` design.

    The payoff is the distinction between a partial repetition and no
    repetition: going partway down and returning produces a sequence that
    reached BETWEEN but never DOWN, which is counted as **improper** rather than
    discarded. That difference is the whole value of the approach.


    For an arm raise the "working" position is the high one. Pass
    ``inverted=True`` and the same machinery applies with the comparisons
    flipped, rather than duplicating the state logic.
    """

    def __init__(self, config: HysteresisConfig, inverted: bool = False) -> None:
        self.config = config
        self.inverted = inverted
        self.count = 0
        self.partial_count = 0

        self._state = RepState.BETWEEN
        self._pending: RepState | None = None
        self._pending_frames = 0
        self._state_entered_at: float | None = None
        self._reached_down = False
        self._down_entered_at: float | None = None
        self._left_up = False
        self._last_t: float | None = None
        self._blocked_transitions = 0
        self._blocked_target: RepState | None = None

    @property
    def state(self) -> RepState:
        return self._state

    def _classify(self, angle: float) -> RepState:
        low, high = self.config.down_below, self.config.up_above
        if self.inverted:
            if angle > high:
                return RepState.DOWN
            if angle < low:
                return RepState.UP
            return RepState.BETWEEN
        if angle < low:
            return RepState.DOWN
        if angle > high:
            return RepState.UP
        return RepState.BETWEEN

    def update(self, angle: float | None, timestamp: float) -> RepResult:
        """Feed one frame's joint angle.

        ``angle`` may be None when the landmarks were unreliable. That is not
        treated as a state - it is treated as no information, which keeps a
        brief occlusion from being read as a movement.
        """
        if self._last_t is not None and timestamp - self._last_t > GAP_RESET_SECONDS:
            self._pending = None
            self._pending_frames = 0
        self._last_t = timestamp

        if angle is None:
            return RepResult(self._state, count=self.count,
                             partial_count=self.partial_count)

        observed = self._classify(angle)

        if observed == self._state:
            self._pending = None
            self._pending_frames = 0
            if self._blocked_target is not None and self._note_abandoned_attempt():
                self.partial_count += 1
                return RepResult(self._state, partial=True, count=self.count,
                                 partial_count=self.partial_count)
            return RepResult(self._state, count=self.count,
                             partial_count=self.partial_count)

        if observed == self._pending:
            self._pending_frames += 1
        else:
            self._pending = observed
            self._pending_frames = 1

        if self._pending_frames < self.config.confirm_frames:
            return RepResult(self._state, count=self.count,
                             partial_count=self.partial_count)

        if (self._state_entered_at is not None
                and self._state is not RepState.BETWEEN
                and timestamp - self._state_entered_at < self.config.min_dwell_seconds):
            self._blocked_target = observed
            return RepResult(self._state, count=self.count,
                             partial_count=self.partial_count)

        self._blocked_target = None
        self._blocked_transitions = 0
        return self._transition(observed, timestamp)

    def _note_abandoned_attempt(self) -> bool:
        """Records a movement that was started too fast and then given up.

        Returns True when enough have accumulated to report a partial.
        """
        self._blocked_target = None
        self._blocked_transitions += 1
        if self._blocked_transitions >= self._BLOCKED_BEFORE_PARTIAL:
            self._blocked_transitions = 0
            return True
        return False

    _BLOCKED_BEFORE_PARTIAL = 3

    def _transition(self, new_state: RepState, timestamp: float) -> RepResult:
        previous = self._state
        self._state = new_state
        self._state_entered_at = timestamp
        self._pending = None
        self._pending_frames = 0

        completed = partial = False

        if new_state is RepState.DOWN:
            self._reached_down = True
            self._down_entered_at = timestamp
        elif new_state is RepState.UP:
            held = (
                self._reached_down
                and self._down_entered_at is not None
                and timestamp - self._down_entered_at >= self.config.min_dwell_seconds
            )
            if held and self._left_up:
                self.count += 1
                completed = True
            elif self._left_up:
                self.partial_count += 1
                partial = True
            self._reached_down = False
            self._down_entered_at = None
            self._left_up = False

        if previous is RepState.UP:
            self._left_up = True

        return RepResult(self._state, completed=completed, partial=partial,
                         count=self.count, partial_count=self.partial_count)

    def reset(self, keep_counts: bool = True) -> None:
        """Clear in-progress state.

        Counts are kept by default: a subject switch or a dropped-frame gap
        invalidates the *current* repetition, not the ones already completed.
        """
        self._state = RepState.BETWEEN
        self._pending = None
        self._pending_frames = 0
        self._state_entered_at = None
        self._reached_down = False
        self._down_entered_at = None
        self._left_up = False
        self._last_t = None
        self._blocked_transitions = 0
        self._blocked_target = None
        if not keep_counts:
            self.count = 0
            self.partial_count = 0

GRAVITY = 9.81


@dataclass
class JumpEvent:
    """A detected jump."""

    takeoff_time: float
    landing_time: float
    flight_seconds: float
    peak_rise_torso_units: float
    """How far the centre of mass rose, in torso lengths.

    Deliberately **not** reported in centimetres. At 40 fps a one-frame error at
    each of take-off and landing is about 9% of flight time, and since
    ``h proportional to T squared`` that becomes roughly 18% of height. The
    published validation also found flight time biased by +61 ms even at 240
    fps. A relative score is defensible; a number in centimetres is not.
    """

    fitted_gravity_ratio: float
    """Fitted vertical acceleration during flight, divided by g.

    Near 1.0 for a real jump. This is what separates a jump from rising onto
    tiptoes, which has no free-fall phase at all.
    """


class JumpDetector:
    """Detects jumps from the vertical motion of the centre of mass.


    Measured, from the validation study: CoM vertical position RMSE **21 mm**,
    r = 0.999, bias 0.000 m. Joint angles: RMSE 5.4 to 8.0 degrees with limits
    of agreement spanning 15 to 20 degrees.

    Position beats angle decisively, because CoM is a mass-weighted average over
    many landmarks and independent per-keypoint noise cancels in the average.


    Flight time is biased +61 ms in published measurement, and height goes as
    time squared. Take-off and landing are the hardest frames for any detector:
    motion blur, foot occlusion, an ill-defined toe point. So the **peak rise**
    of the CoM is used as the magnitude instead, and even that is reported in
    torso units rather than centimetres.


    The hardest discrimination is a jump against rising onto tiptoes, and the
    literature barely covers it. Physics does: during flight the only force is
    gravity, so the CoM traces a parabola with vertical acceleration equal to g.
    Tiptoes has no such phase - the body rises and stops.

    Fitting a quadratic to the airborne CoM trajectory and comparing the implied
    acceleration against g is therefore a direct test of whether the subject was
    actually airborne. **This is inferred reasoning, not a measured result**, and
    is flagged as such in round 3.
    """

    def __init__(self, rise_threshold: float = 0.08,
                 min_flight_frames: int = 5,
                 gravity_tolerance: float = 0.20,
                 history_seconds: float = 3.0) -> None:
        self.rise_threshold = rise_threshold
        """Minimum CoM rise, in torso lengths, to consider a jump.

        A torso is roughly 50 cm, so 0.08 is about 4 cm - low enough to catch a
        weak jump, high enough to reject postural sway. Published bilateral
        jumps average 21 cm; tiptoe rise is 5-12 cm, so this threshold alone
        cannot separate them, which is why the free-fall check exists.
        """

        self.min_flight_frames = min_flight_frames
        self.gravity_tolerance = gravity_tolerance
        """How far the fitted acceleration may deviate from g, as a fraction.

        **Measured, not assumed.** Twelve seeds per condition at 30 fps with
        3 px landmark noise, fitting the parabola to the airborne trajectory:

            real jump  0.10 m    0.93 - 1.07
            real jump  0.20 m    0.97 - 1.05
            real jump  0.30 m    0.98 - 1.04
            tiptoe     0.10 m    0.32 - 0.37
            tiptoe     0.15 m    0.48 - 0.54
            tiptoe     0.20 m    0.65 - 0.71

        Real jumps cluster tightly at 1.0 because free fall genuinely is
        parabolic. A tiptoe rise is a sine arc, whose best-fit parabola gives a
        much smaller acceleration - but not a near-zero one, which was the
        original mistaken assumption.

        The first value here was 0.45, chosen on the theory that the fit would
        be too noisy to allow anything tighter. Measurement shows the opposite:
        the fit is tight, and 0.45 accepted everything above 0.55 - so a 20 cm
        tiptoe rise at 0.68 was read as a jump in **eight runs out of eight**.

        0.20 separates the two populations cleanly with margin on both sides.
        """

        self._t: deque[float] = deque()
        self._y: deque[float] = deque()
        self._history_seconds = history_seconds
        self._baseline: float | None = None
        self._airborne = False
        self._takeoff_t: float | None = None
        self._peak: float = 0.0
        self.jumps: list[JumpEvent] = []

    def update(self, com: np.ndarray | None, torso: float | None,
               timestamp: float) -> JumpEvent | None:
        """Feed one frame. Returns a JumpEvent on the frame a jump completes."""
        if com is None or torso is None or torso <= 0:
            self._abort()
            return None

        if self._t and timestamp - self._t[-1] > GAP_RESET_SECONDS:
            self._abort()

        height = -float(com[1]) / torso

        self._t.append(timestamp)
        self._y.append(height)
        while self._t and timestamp - self._t[0] > self._history_seconds:
            self._t.popleft()
            self._y.popleft()

        if len(self._y) >= 5:
            self._baseline = float(np.median(list(self._y)))

        if self._baseline is None:
            return None

        rise = height - self._baseline

        if not self._airborne:
            if rise > self.rise_threshold:
                self._airborne = True
                self._takeoff_t = timestamp
                self._peak = rise
            return None

        self._peak = max(self._peak, rise)

        if rise > self.rise_threshold * 0.5:
            return None

        event = self._finish(timestamp)
        self._airborne = False
        self._takeoff_t = None
        self._peak = 0.0
        return event

    def _finish(self, landing_t: float) -> JumpEvent | None:
        if self._takeoff_t is None:
            return None

        flight = landing_t - self._takeoff_t
        times = np.array(self._t)
        heights = np.array(self._y)
        mask = (times >= self._takeoff_t) & (times <= landing_t)
        if mask.sum() < self.min_flight_frames:
            return None

        t_rel = times[mask] - self._takeoff_t
        try:
            coeffs = np.polyfit(t_rel, heights[mask], 2)
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover
            return None

        accel = 2.0 * coeffs[0] * 0.5
        ratio = abs(accel) / GRAVITY

        if coeffs[0] >= 0 or abs(ratio - 1.0) > self.gravity_tolerance:
            return None

        expected_rise_m = GRAVITY * flight * flight / 8.0
        expected_rise_torso = expected_rise_m / 0.5
        if expected_rise_torso > 1e-6:
            rise_ratio = self._peak / expected_rise_torso
            if not 0.4 < rise_ratio < 2.5:
                return None

        event = JumpEvent(
            takeoff_time=self._takeoff_t,
            landing_time=landing_t,
            flight_seconds=flight,
            peak_rise_torso_units=self._peak,
            fitted_gravity_ratio=ratio,
        )
        self.jumps.append(event)
        return event

    def _abort(self) -> None:
        self._airborne = False
        self._takeoff_t = None
        self._peak = 0.0

    def reset(self) -> None:
        self._t.clear()
        self._y.clear()
        self._baseline = None
        self._abort()

@dataclass
class MovementReport:
    """What the movement layer knows about the current frame."""

    usable: bool
    reason: str = ""
    rep_state: RepState = RepState.BETWEEN
    rep_count: int = 0
    partial_count: int = 0
    rep_completed: bool = False
    rep_partial: bool = False
    jump: JumpEvent | None = None
    jump_count: int = 0
    viewpoint_offset: float | None = None
    subject_switched: bool = False

    velocity_count: int = 0
    """Repetitions counted from velocity reversal, which needs no calibration."""

    velocity_completed: bool = False

    trust_reasons: list[str] = field(default_factory=list)
    """Why the frame was refused, when the fast reliability gates rejected it."""

    form: object | None = None
    """A `FormReport` on the frame a repetition completes, otherwise None.

    Typed loosely to avoid a circular import; it is always a
    `form_score.FormReport`.
    """


class MovementSession:
    """Combines the robustness guards, the rep counter and the jump detector.

    The order of operations matters and is deliberate:

    1. **Subject switch** first - if the person changed, nothing else on this
       frame is comparable to the last, so state is reset before it is used.
    2. **Viewpoint gate** next - if the subject is turned away, decline rather
       than compute wrong numbers.
    3. **Median filter** - reject spikes and limb swaps before measuring.
    4. Only then measure.

    Declining is a first-class outcome. Round 3 records a **20% outright
    failure rate** on unilateral jumps in a laboratory with cooperative
    subjects; in a bedroom it will be higher. "I could not read that clearly" is
    a feature, and far better than a confidently wrong count.
    """

    def __init__(self, rep_config: HysteresisConfig = SQUAT,
                 inverted: bool = False,
                 max_viewpoint_offset: float = 35.0,
                 use_median: bool = True,
                 use_velocity: bool = True,
                 check_reliability: bool = True) -> None:
        self.rep_counter = RepCounter(rep_config, inverted=inverted)
        self.jump_detector = JumpDetector()
        self.subject_switch = SubjectSwitchDetector()
        self.median = MedianFilter(5) if use_median else None
        self.max_viewpoint_offset = max_viewpoint_offset
        self.jump_count = 0

        from reliability import ReliabilityMonitor
        from velocity_reps import VelocityRepCounter

        self.velocity_counter = VelocityRepCounter() if use_velocity else None
        """Counts the same repetitions without needing calibration.

        Measured against the threshold counter on identical movement, neither
        calibrated:

            bottoms at  80 deg   threshold 5/5   velocity 5/5
            bottoms at 107 deg   threshold 0/5   velocity 5/5
            bottoms at 138 deg   threshold 0/5   velocity 5/5

        Both are kept rather than one replacing the other, because they answer
        different questions. The threshold counter knows *how deep* the user
        went and can say "not deep enough"; the velocity counter only knows
        that they turned around. Coaching needs the first, counting needs the
        second, and running both costs a subtraction per frame.
        """

        from form_score import FormAnalyser, RepetitionCollector

        self.form = FormAnalyser()
        """Scores each completed repetition against the user's own first one.

        Runs *after* a repetition completes, so it is off the critical path
        entirely. Measured at 0.35 ms per comparison, once per repetition -
        against 13.81 ms of inference per frame.

        The reference is the user's own first repetition by default, so the
        comparison avoids every problem with comparing them to a population:
        body proportions, mobility, injury history, camera placement.
        """

        self.collector = RepetitionCollector()

        self.reliability = ReliabilityMonitor() if check_reliability else None
        """Catches bad tracking in one frame rather than in seven.

        MediaPipe smooths its visibility scores with alpha 0.1, so a landmark
        that becomes wrong takes 233 ms to be reported as low-confidence. These
        checks compare consecutive frames.
        """

    def update(self, points: np.ndarray, visibility: np.ndarray,
               angle: float | None, timestamp: float,
               world_points: np.ndarray | None = None,
               angle_3d: float | None = None) -> MovementReport:
        """Feed one frame.

        `world_points` are MediaPipe's 3D world landmarks. Supplying them makes
        the viewpoint gate accurate rather than approximate: validated on a real
        photograph, the 2D estimate reported 64 degrees of yaw and refused a
        subject whose true yaw was 6.7 degrees.
        """
        points = sanitise(points)
        switched = self.subject_switch(points, visibility, timestamp)
        if switched:
            self.rep_counter.reset(keep_counts=True)
            self.jump_detector.reset()
            if self.median:
                self.median.reset()
            return MovementReport(
                usable=False, reason="subject changed",
                rep_count=self.rep_counter.count,
                partial_count=self.rep_counter.partial_count,
                jump_count=self.jump_count, subject_switched=True,
            )

        offset = None
        if world_points is not None:
            offset = torso_yaw_degrees(world_points)
        if offset is None:
            offset = viewpoint_offset_degrees(points, visibility)

        if offset is not None and offset > self.max_viewpoint_offset:
            return MovementReport(
                usable=False, reason="turn to face the camera",
                rep_count=self.rep_counter.count,
                partial_count=self.rep_counter.partial_count,
                jump_count=self.jump_count, viewpoint_offset=offset,
            )

        filtered = self.median(points, timestamp) if self.median else points
        torso = torso_length(filtered, visibility)

        trust = None
        if self.reliability is not None:
            trust = self.reliability.update(filtered, visibility, torso,
                                            timestamp, angle_2d=angle,
                                            angle_3d_value=angle_3d)
            if not trust.trustworthy:
                return MovementReport(
                    usable=False, reason='; '.join(trust.reasons),
                    rep_count=self.rep_counter.count,
                    partial_count=self.rep_counter.partial_count,
                    jump_count=self.jump_count,
                    viewpoint_offset=offset,
                    velocity_count=(self.velocity_counter.count
                                    if self.velocity_counter else 0),
                    trust_reasons=trust.reasons,
                )

        rep = self.rep_counter.update(angle, timestamp)

        velocity_count = 0
        velocity_completed = False
        form_report = None
        self.collector.add(angle, timestamp)

        if self.velocity_counter is not None:
            velocity = self.velocity_counter.update(angle, timestamp)
            velocity_count = velocity.count
            velocity_completed = velocity.completed
            if velocity.completed:
                record = self.collector.close(timestamp)
                if record is not None:
                    form_report = self.form.add(record)

        com = centre_of_mass(filtered, visibility)
        jump = self.jump_detector.update(com, torso, timestamp)
        if jump is not None:
            self.jump_count += 1

        return MovementReport(
            usable=True,
            rep_state=rep.state,
            rep_count=rep.count,
            partial_count=rep.partial_count,
            rep_completed=rep.completed,
            rep_partial=rep.partial,
            jump=jump,
            jump_count=self.jump_count,
            viewpoint_offset=offset,
            velocity_count=velocity_count,
            velocity_completed=velocity_completed,
            form=form_report,
        )

    def reset(self, keep_counts: bool = False) -> None:
        self.rep_counter.reset(keep_counts=keep_counts)
        self.jump_detector.reset()
        self.subject_switch.reset()
        if self.median:
            self.median.reset()
        if self.velocity_counter:
            self.velocity_counter.reset(keep_counts=keep_counts)
        self.collector.reset()
        if not keep_counts:
            self.form.reset()
        if self.reliability:
            self.reliability.reset()
        if not keep_counts:
            self.jump_count = 0
