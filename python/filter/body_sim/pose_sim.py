from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_VARIANT = "lite"
MODEL_PATH = (Path(__file__).parent / "models"
              / f"pose_landmarker_{MODEL_VARIANT}.task")

LANDMARK_NAMES = [
    "nose", "leftEyeInner", "leftEye", "leftEyeOuter", "rightEyeInner",
    "rightEye", "rightEyeOuter", "leftEar", "rightEar", "leftMouth",
    "rightMouth", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist", "leftPinky", "rightPinky", "leftIndex",
    "rightIndex", "leftThumb", "rightThumb", "leftHip", "rightHip",
    "leftKnee", "rightKnee", "leftAnkle", "rightAnkle", "leftHeel",
    "rightHeel", "leftFootIndex", "rightFootIndex",
]
IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}

HEAD = {IDX[n] for n in (
    "nose", "leftEyeInner", "leftEye", "leftEyeOuter", "rightEyeInner",
    "rightEye", "rightEyeOuter", "leftEar", "rightEar", "leftMouth",
    "rightMouth",
)}
TORSO = [IDX["leftShoulder"], IDX["rightShoulder"], IDX["leftHip"], IDX["rightHip"]]
UPPER_BODY = [IDX[n] for n in (
    "nose", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist", "leftHip", "rightHip",
)]
LOWER_BODY = [IDX[n] for n in ("leftKnee", "rightKnee", "leftAnkle", "rightAnkle")]

BONES: list[tuple[int, int, str]] = [
    (IDX["leftShoulder"], IDX["leftElbow"], "limb"),
    (IDX["leftElbow"], IDX["leftWrist"], "limb"),
    (IDX["rightShoulder"], IDX["rightElbow"], "limb"),
    (IDX["rightElbow"], IDX["rightWrist"], "limb"),
    (IDX["leftHip"], IDX["leftKnee"], "limb"),
    (IDX["leftKnee"], IDX["leftAnkle"], "limb"),
    (IDX["rightHip"], IDX["rightKnee"], "limb"),
    (IDX["rightKnee"], IDX["rightAnkle"], "limb"),
]

HAND_GROUPS = [
    (IDX["leftWrist"], [IDX["leftThumb"], IDX["leftIndex"], IDX["leftPinky"]]),
    (IDX["rightWrist"], [IDX["rightThumb"], IDX["rightIndex"], IDX["rightPinky"]]),
]
FOOT_GROUPS = [
    (IDX["leftAnkle"], [IDX["leftHeel"], IDX["leftFootIndex"]]),
    (IDX["rightAnkle"], [IDX["rightHeel"], IDX["rightFootIndex"]]),
]

MAJOR_JOINTS = [IDX[n] for n in (
    "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist", "leftHip", "rightHip",
    "leftKnee", "rightKnee", "leftAnkle", "rightAnkle",
)]

ANGLE_JOINTS = [
    (IDX["leftElbow"], IDX["leftShoulder"], IDX["leftWrist"]),
    (IDX["rightElbow"], IDX["rightShoulder"], IDX["rightWrist"]),
    (IDX["leftShoulder"], IDX["leftHip"], IDX["leftElbow"]),
    (IDX["rightShoulder"], IDX["rightHip"], IDX["rightElbow"]),
    (IDX["leftKnee"], IDX["leftHip"], IDX["leftAnkle"]),
    (IDX["rightKnee"], IDX["rightHip"], IDX["rightAnkle"]),
]

COLOR_OK = (194, 211, 76)
COLOR_WEAK = (38, 167, 255)
COLOR_BENT = (79, 213, 255)
COLOR_TEXT = (255, 255, 255)

VISIBILITY_THRESHOLD = 0.5


class OneEuro:
    """One-Euro filter for a single scalar.

    Pose landmarks jitter by a few pixels even on a motionless subject, and that
    jitter is indistinguishable from small real movement. A fixed low-pass
    filter forces an unwinnable choice: smooth enough to kill the jitter and
    fast motion visibly lags; fast enough to track motion and the skeleton
    shivers when still.

    This adapts its cutoff to the observed speed, so it is heavy when slow and
    nearly transparent when fast.

    Casiez, Roussel and Vogel, *1 Euro Filter*, CHI 2012.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        dt = 1.0 / rate
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: float, timestamp: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t, self._dx = value, timestamp, 0.0
            return value

        dt = timestamp - self._t
        if dt <= 0:
            dt = 1 / 30
        rate = 1.0 / dt

        dx = (value - self._x) * rate
        a_d = self._alpha(rate, self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(rate, cutoff)
        self._x = a * value + (1 - a) * self._x
        self._t = timestamp
        return self._x

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = None


@dataclass
class PoseSmoother:
    """Smooths the drawn skeleton without inventing motion.

    Deliberately does *not* interpolate between detections. Interpolated
    positions are fabricated data, and this overlay is the only view anyone has
    of what the tracker really sees. If detection is slow the honest
    presentation is a slow skeleton: that visible sluggishness is a true report
    of the frame rate, and hiding it would hide the problem worth finding.


    MediaPipe already applies a One-Euro filter internally, with published
    constants (screen landmarks: min_cutoff 0.05, beta 80.0). This filter is
    therefore a *second* filter on an already-filtered signal, and it was
    measured:

        MediaPipe only          0.837 px wrist jitter on a still clip
        this filter             0.702 px      (a 16% improvement)
        gentler settings        0.702 px      (identical - the knob does little)
        MediaPipe's own values  0.839 px      (worse than no filter at all)

    A 16% jitter reduction is weak justification for a filter that adds lag,
    and the user reports lag. The lag half of that trade could not be measured,
    because the synthetic clip provides no tracked signal to measure lag
    against - so rather than delete the filter on an unfinished measurement or
    keep it on an assumption, it is **configurable and defaults to off**.

    Turn it on with --smoothing to compare directly once a camera works.
    """

    min_cutoff: float = 1.2
    beta: float = 0.02
    enabled: bool = False
    _fx: dict[int, OneEuro] = field(default_factory=dict)
    _fy: dict[int, OneEuro] = field(default_factory=dict)
    _last_t: float | None = None

    gap_reset_seconds: float = 0.100

    def __call__(self, points: np.ndarray, visibility: np.ndarray,
                 timestamp: float) -> np.ndarray:
        if not self.enabled:
            return points

        if (self._last_t is not None
                and timestamp - self._last_t > self.gap_reset_seconds):
            self.reset()
        self._last_t = timestamp

        out = points.copy()
        for i in range(len(points)):
            if visibility[i] < 0.3:
                continue
            if i not in self._fx:
                self._fx[i] = OneEuro(self.min_cutoff, self.beta)
                self._fy[i] = OneEuro(self.min_cutoff, self.beta)
            out[i, 0] = self._fx[i](float(points[i, 0]), timestamp)
            out[i, 1] = self._fy[i](float(points[i, 1]), timestamp)
        return out

    def reset(self) -> None:
        self._last_t = None
        for f in self._fx.values():
            f.reset()
        for f in self._fy.values():
            f.reset()


def mean_confidence(visibility: np.ndarray, group: list[int]) -> float:
    """Mean visibility across a landmark group."""
    if not group:
        return 0.0
    return float(np.mean([visibility[i] for i in group]))


def assess_framing(visibility: np.ndarray | None) -> tuple[str, str, float, float, float]:
    """Decide how much of the subject is usable.

    MediaPipe, like ML Kit, never reports failure by omission: it returns all 33
    landmarks whether or not it can see them, lowering `visibility` for the ones
    it inferred. Code that ignores that treats invented coordinates as
    measurements.
    """
    if visibility is None:
        return "noSubject", "Step into view of the camera", 0.0, 0.0, 0.0

    torso = mean_confidence(visibility, TORSO)
    upper = mean_confidence(visibility, UPPER_BODY)
    lower = mean_confidence(visibility, LOWER_BODY)

    if torso < 0.6:
        advice = "Turn to face the camera" if upper > 0.3 else "Step into view of the camera"
        return "partial", advice, torso, upper, lower

    if lower >= 0.6:
        return "full", "", torso, upper, lower
    return "upperBodyOnly", "Move back so your legs are in view", torso, upper, lower


def torso_frame(points: np.ndarray, visibility: np.ndarray):
    """Build the body-relative coordinate system, or None if unusable.

    Image coordinates answer "where on screen", which is the wrong question: the
    same gesture yields different numbers when the subject steps closer or is
    simply taller. Re-expressing everything relative to the body itself removes
    both effects.

    Returns None rather than a default frame - a fabricated coordinate system
    would produce plausible-looking numbers from unusable input.
    """
    for i in TORSO:
        if visibility[i] < VISIBILITY_THRESHOLD:
            return None

    ls, rs = points[IDX["leftShoulder"]], points[IDX["rightShoulder"]]
    lh, rh = points[IDX["leftHip"]], points[IDX["rightHip"]]

    mid_hip = (lh + rh) / 2
    mid_shoulder = (ls + rs) / 2

    spine = np.array([mid_shoulder[0] - mid_hip[0], -(mid_shoulder[1] - mid_hip[1])])
    scale = float(np.linalg.norm(spine))
    if scale < 1e-4:
        return None

    tilt = math.degrees(math.atan2(spine[0], spine[1]))
    return mid_hip, scale, tilt


def joint_angle(points: np.ndarray, visibility: np.ndarray,
                vertex: int, a: int, b: int) -> float | None:
    """Interior angle at `vertex`, in degrees, or None if unmeasurable.

    Angles are the most transferable body measurement available: they do not
    change with distance from the camera, with the subject's size, or with where
    they stand in the frame.

    Note the angle is relative to the *adjoining segment*, never to the world. A
    horizontal arm reads about 98 degrees at the shoulder, not 90, because the
    side of the trunk is not vertical - shoulders are wider than hips.
    """
    for i in (vertex, a, b):
        if visibility[i] < VISIBILITY_THRESHOLD:
            return None

    v1 = points[a] - points[vertex]
    v2 = points[b] - points[vertex]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def draw_extremity_box(image: np.ndarray, points: np.ndarray,
                       visibility: np.ndarray, anchor: int, members: list[int],
                       unit: float, is_hand: bool) -> None:
    """Draw a hand or foot as an oriented box fitted to it.

    The box is aligned to the limb's own direction rather than to the screen,
    so it stays correct when the hand is turned or the foot is at an angle. An
    axis-aligned box would balloon whenever the limb pointed diagonally, which
    would misrepresent the size of what was actually detected.

    Sizing comes from the landmarks themselves, so the box grows and shrinks
    with the real hand as the subject moves towards or away from the camera.
    A fixed-size box would be right at one distance and wrong at every other.
    """
    visible = [i for i in members if visibility[i] >= 0.25]
    if visibility[anchor] < 0.25 or not visible:
        return

    anchor_pt = points[anchor]
    pts = np.vstack([anchor_pt, points[visible]])

    tip = points[visible].mean(axis=0)
    axis = tip - anchor_pt
    length = float(np.linalg.norm(axis))
    if length < 1e-3:
        return
    axis = axis / length

    normal = np.array([-axis[1], axis[0]])

    rel = pts - anchor_pt
    along = rel @ axis
    across = rel @ normal

    a0, a1 = float(along.min()), float(along.max())
    c0, c1 = float(across.min()), float(across.max())

    span = max(a1 - a0, 1e-3)
    if is_hand:
        a0 -= span * 0.30
        a1 += span * 0.55
        half = max(span * 0.55, unit * 1.6)
    else:
        a0 -= span * 0.18
        a1 += span * 0.22
        half = max(span * 0.34, unit * 1.3)

    c_mid = (c0 + c1) / 2
    c0, c1 = c_mid - half, c_mid + half

    corners = np.array([
        anchor_pt + axis * a0 + normal * c0,
        anchor_pt + axis * a1 + normal * c0,
        anchor_pt + axis * a1 + normal * c1,
        anchor_pt + axis * a0 + normal * c1,
    ], dtype=np.int32)

    strong = all(visibility[i] >= VISIBILITY_THRESHOLD for i in members)
    colour = COLOR_OK if strong else COLOR_WEAK

    cv2.polylines(image, [corners], True, (0, 0, 0),
                  max(2, int(unit * 0.9)), cv2.LINE_AA)
    cv2.polylines(image, [corners], True, colour,
                  max(1, int(unit * 0.5)), cv2.LINE_AA)

    cv2.line(image, tuple(anchor_pt.astype(int)),
             tuple((anchor_pt + axis * a0).astype(int)),
             colour, max(1, int(unit * 0.5)), cv2.LINE_AA)


def draw_trunk(image: np.ndarray, points: np.ndarray,
               visibility: np.ndarray, unit: float,
               discarded: np.ndarray | None = None) -> bool:
    """Draw the torso as a stick figure: one spine, two short cross-bars.

    The trunk used to be four bones forming a quadrilateral - shoulder to
    shoulder, hip to hip, and both sides. That is an accurate rendering of the
    four landmarks and a poor rendering of a body: it reads as a crate, and it
    puts a joint dot at each corner of the chest, where no joint exists. There
    is no shoulder-to-hip bone in a skeleton; the ribcage is not a hinge.

    A stick figure says what the four landmarks actually establish - where the
    shoulder line is, where the hip line is, and the axis between them - and
    nothing more.

    The spine is drawn from mid-hip to mid-shoulder because that is the axis
    every measurement in this engine already uses: the torso frame's origin and
    unit vector. Drawing it makes the coordinate system visible, so a viewer
    can see when it is wrong.

    Returns True if it drew, so the caller knows whether the trunk landmarks
    were usable.
    """
    ls, rs = IDX["leftShoulder"], IDX["rightShoulder"]
    lh, rh = IDX["leftHip"], IDX["rightHip"]
    if any(visibility[i] < 0.2 for i in (ls, rs, lh, rh)):
        return False
    if discarded is not None and any(discarded[i] for i in (ls, rs, lh, rh)):
        return False

    mid_shoulder = (points[ls] + points[rs]) / 2
    mid_hip = (points[lh] + points[rh]) / 2

    strong = all(visibility[i] >= VISIBILITY_THRESHOLD
                 for i in (ls, rs, lh, rh))
    colour = COLOR_OK if strong else COLOR_WEAK

    spine_w = max(1, int(1.7 * unit))
    bar_w = max(1, int(1.3 * unit))
    casing = max(2, int(unit))

    def line(a, b, width):
        draw_bone(image, a, b, colour, width, casing=casing)

    line(mid_hip, mid_shoulder, spine_w)
    line(points[ls], points[rs], bar_w)
    line(points[lh], points[rh], bar_w)

    spine = mid_shoulder - mid_hip
    length = float(np.linalg.norm(spine))
    if length > 1e-6:
        neck = mid_shoulder + spine / length * (length * 0.16)
        line(mid_shoulder, neck, bar_w)
    return True


def draw_dashed_line(image: np.ndarray, a: tuple[int, int], b: tuple[int, int],
                     colour: tuple[int, int, int], thickness: int,
                     dash: int = 9) -> None:
    """A dashed line, for a limb whose position is inferred rather than seen.

    Drawn differently on purpose. A held position is a claim about where a limb
    probably still is, and rendering it identically to a measured one invites
    the viewer to believe it equally. Users understand "the system cannot see
    my arm"; they do not understand a solid limb in the wrong place.
    """
    pa, pb = np.array(a, dtype=float), np.array(b, dtype=float)
    span = float(np.linalg.norm(pb - pa))
    if span < 1e-6:
        return
    steps = max(2, int(span / dash))
    for i in range(0, steps, 2):
        s = pa + (pb - pa) * (i / steps)
        e = pa + (pb - pa) * (min(i + 1, steps) / steps)
        cv2.line(image, tuple(s.astype(int)), tuple(e.astype(int)),
                 colour, thickness, cv2.LINE_AA)


def clip_to_frame(a: np.ndarray, b: np.ndarray, width: int,
                  height: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Trim a segment to the part inside the image, or None if it misses.

    A bone with one end outside the picture is drawn only as far as the edge,
    rather than either vanishing or being drawn to a coordinate that is not in
    the image at all. Liang-Barsky, which is exact and needs no iteration.
    """
    direction = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-direction[0], a[0]), (direction[0], width - a[0]),
                 (-direction[1], a[1]), (direction[1], height - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return a + direction * t0, a + direction * t1


def draw_bone(image: np.ndarray, a: np.ndarray, b: np.ndarray,
              colour: tuple[int, int, int], thickness: int,
              dashed: bool = False, casing: int = 0) -> None:
    """Draw one bone, trimmed to the image.

    Centred on the segment between the two joints, which is the limb's own
    axis - a line drawn to an unclipped endpoint outside the picture skews
    towards whichever side the invented coordinate landed on, which is what
    made the skeleton sit beside the body rather than along it.
    """
    h, w = image.shape[:2]
    clipped = clip_to_frame(np.asarray(a, dtype=float),
                            np.asarray(b, dtype=float), w, h)
    if clipped is None:
        return
    pa, pb = clipped
    ia = (int(round(pa[0])), int(round(pa[1])))
    ib = (int(round(pb[0])), int(round(pb[1])))
    if dashed:
        draw_dashed_line(image, ia, ib, colour, thickness)
        return
    if casing:
        cv2.line(image, ia, ib, (0, 0, 0), thickness + casing, cv2.LINE_AA)
    cv2.line(image, ia, ib, colour, thickness, cv2.LINE_AA)


def draw_skeleton(image: np.ndarray, points: np.ndarray,
                  visibility: np.ndarray, framing: str,
                  inferred: np.ndarray | None = None,
                  discarded: np.ndarray | None = None) -> None:
    """Draw the figure over the camera image.

    Confidence is drawn, never hidden. A landmark the model inferred appears in
    a warning colour and faded: a skeleton at uniform opacity presents a guessed
    ankle exactly like a measured shoulder, and whoever is watching will believe
    it.

    `inferred` marks landmarks held from an earlier frame because the model's
    own confidence could not be trusted - measured at 0.97 on a wrist a full
    torso length out of place. Those bones are drawn dashed.

    `discarded` marks landmarks that must not be drawn at all: outside the
    picture, never properly observed, or held past the limit. Measured, the
    model places off-frame landmarks at coordinates outside the image - the
    worst at x = 1354 in a 1280-wide frame - while still reporting 0.82
    confidence. Drawing those is what produced random marks when jumping or
    stepping off the edge.
    """
    h, w = image.shape[:2]
    unit = max(1.0, min(w, h) / 100.0)
    if inferred is None:
        inferred = np.zeros(len(points), dtype=bool)
    if discarded is None:
        discarded = np.array([
            not np.all(np.isfinite(points[i]))
            or points[i, 0] < -w * 0.02 or points[i, 0] > w * 1.02
            or points[i, 1] < -h * 0.02 or points[i, 1] > h * 1.02
            for i in range(len(points))
        ])

    def px(i: int) -> tuple[int, int]:
        return int(round(points[i, 0])), int(round(points[i, 1]))

    draw_trunk(image, points, visibility, unit, discarded)

    widths = {"trunk": 1.6, "limb": 1.2, "tip": 0.7}
    for a, b, kind in BONES:
        if discarded[a] or discarded[b]:
            continue
        if visibility[a] < 0.2 or visibility[b] < 0.2:
            continue
        ok = visibility[a] >= VISIBILITY_THRESHOLD and visibility[b] >= VISIBILITY_THRESHOLD
        colour = COLOR_OK if ok else COLOR_WEAK
        thickness = max(1, int(widths[kind] * unit))
        draw_bone(image, points[a], points[b],
                  COLOR_WEAK if (inferred[a] or inferred[b]) else colour,
                  thickness, dashed=bool(inferred[a] or inferred[b]),
                  casing=max(2, int(unit)))

    head = [i for i in HEAD if visibility[i] >= 0.2 and not discarded[i]]
    if len(head) >= 3:
        xs = points[head, 0]
        ys = points[head, 1]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        pad_x = (x1 - x0) * 0.42 + unit
        pad_y = (y1 - y0) * 0.75 + unit
        p0 = (int(x0 - pad_x), int(y0 - pad_y))
        p1 = (int(x1 + pad_x), int(y1 + pad_y * 0.5))
        cv2.rectangle(image, p0, p1, (0, 0, 0), max(2, int(unit * 0.9)), cv2.LINE_AA)
        cv2.rectangle(image, p0, p1, COLOR_OK, max(1, int(unit * 0.5)), cv2.LINE_AA)

    for anchor, members in HAND_GROUPS:
        if discarded[anchor]:
            continue
        live = [i for i in members if not discarded[i]]
        if live:
            draw_extremity_box(image, points, visibility, anchor, live, unit,
                               True)
    for anchor, members in FOOT_GROUPS:
        if discarded[anchor]:
            continue
        live = [i for i in members if not discarded[i]]
        if live:
            draw_extremity_box(image, points, visibility, anchor, live, unit,
                               False)

    extremity_points = {i for _, m in HAND_GROUPS + FOOT_GROUPS for i in m}
    for i in range(len(points)):
        if i in HEAD or i in extremity_points or visibility[i] < 0.15:
            continue
        if discarded[i]:
            continue
        major = i in MAJOR_JOINTS
        ok = visibility[i] >= VISIBILITY_THRESHOLD
        colour = COLOR_OK if ok else COLOR_WEAK
        r = int((1.6 if major else 0.9) * unit)
        if inferred[i]:
            cv2.circle(image, px(i), r, COLOR_WEAK,
                       max(1, int(unit * 0.35)), cv2.LINE_AA)
            continue
        cv2.circle(image, px(i), r + max(1, int(unit * 0.3)), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, px(i), r, colour, -1, cv2.LINE_AA)
        if major:
            cv2.circle(image, px(i), max(1, int(r * 0.4)), (255, 255, 255), -1, cv2.LINE_AA)

    for vertex, a, b in ANGLE_JOINTS:
        if discarded[vertex] or discarded[a] or discarded[b]:
            continue
        angle = joint_angle(points, visibility, vertex, a, b)
        if angle is None:
            continue
        bent = angle < 160
        if bent:
            cv2.circle(image, px(vertex), int(unit * 2.8), COLOR_BENT,
                       max(1, int(unit * 0.4)), cv2.LINE_AA)
        label = f"{angle:.0f}"
        origin = (px(vertex)[0] + int(unit * 2.2), px(vertex)[1] - int(unit * 1.4))
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    unit * 0.055, (0, 0, 0), max(3, int(unit * 0.7)), cv2.LINE_AA)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    unit * 0.055, COLOR_BENT if bent else COLOR_TEXT,
                    max(1, int(unit * 0.25)), cv2.LINE_AA)

    strong = [i for i in range(len(points))
              if visibility[i] >= VISIBILITY_THRESHOLD and not discarded[i]]
    if len(strong) >= 4:
        xs, ys = points[strong, 0], points[strong, 1]
        x0, y0 = int(xs.min() - unit * 2), int(ys.min() - unit * 2)
        x1, y1 = int(xs.max() + unit * 2), int(ys.max() + unit * 2)
        colour = COLOR_OK if framing == "full" else COLOR_WEAK
        seg = int(min(x1 - x0, y1 - y0) * 0.16)
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                 (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(image, (cx, cy), (cx + seg * dx, cy), colour, max(1, int(unit * 0.5)), cv2.LINE_AA)
            cv2.line(image, (cx, cy), (cx, cy + seg * dy), colour, max(1, int(unit * 0.5)), cv2.LINE_AA)


HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (0, 9), (9, 10), (10, 11), (11, 12),       # middle
    (0, 13), (13, 14), (14, 15), (15, 16),     # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # knuckle line
]


def draw_hands(image: np.ndarray, hands, unit: float) -> None:
    """Draw each detected hand, with extended fingers highlighted.

    An extended finger is drawn in the strong colour and a folded one faded, so
    the reading the gesture logic is acting on is visible. If the program
    refuses a gesture, the user can see which finger it disagreed about rather
    than guessing.
    """
    from hands import FINGER_JOINTS

    finger_of = {}
    for name, (tip, mid, base) in FINGER_JOINTS.items():
        for j in (tip, mid, base):
            finger_of[j] = name

    for hand in hands:
        pts = hand.points
        for a, b in HAND_BONES:
            if a >= len(pts) or b >= len(pts):
                continue
            name = finger_of.get(b) or finger_of.get(a)
            up = hand.states.get(name, False) if name else False
            colour = COLOR_OK if up else COLOR_WEAK
            pa = tuple(np.round(pts[a]).astype(int))
            pb = tuple(np.round(pts[b]).astype(int))
            cv2.line(image, pa, pb, (0, 0, 0), max(2, int(unit * 0.7)),
                     cv2.LINE_AA)
            cv2.line(image, pa, pb, colour, max(1, int(unit * 0.35)),
                     cv2.LINE_AA)

        for name, (tip, _, _) in FINGER_JOINTS.items():
            if tip >= len(pts):
                continue
            up = hand.states.get(name, False)
            p = tuple(np.round(pts[tip]).astype(int))
            cv2.circle(image, p, max(2, int(unit * 0.5)),
                       COLOR_OK if up else COLOR_WEAK, -1, cv2.LINE_AA)

        numbers = hand.numbers
        label = "".join(str(n) for n in numbers) if numbers else "-"
        origin = (int(pts[0][0]) - int(unit * 2), int(pts[0][1]) + int(unit * 4))
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    unit * 0.06, (0, 0, 0), max(3, int(unit * 0.8)),
                    cv2.LINE_AA)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    unit * 0.06, COLOR_TEXT, max(1, int(unit * 0.3)),
                    cv2.LINE_AA)


def draw_stop_progress(image: np.ndarray, progress: float) -> None:
    """A ring that fills while the stop gesture is held.

    Visible feedback is what makes a 1.5 second hold usable rather than
    mysterious: the user can see the command being accepted and can abandon it
    by opening their hand before it completes.
    """
    h, w = image.shape[:2]
    centre = (w // 2, 70)
    radius = 34
    cv2.circle(image, centre, radius + 3, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, centre, radius, (60, 60, 60), 4, cv2.LINE_AA)
    cv2.ellipse(image, centre, (radius, radius), -90, 0,
                int(360 * max(0.0, min(1.0, progress))), COLOR_BENT, 5,
                cv2.LINE_AA)
    text = "STOP"
    cv2.putText(image, text, (centre[0] - 26, centre[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (centre[0] - 26, centre[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


class _Landmark:
    """One landmark, shaped like MediaPipe's own.

    A re-acquired detection has to be handed downstream in exactly the form a
    normal detection takes, so that nothing after this point needs to know
    whether the subject was found on the full frame or in a search crop.
    """

    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float, y: float, visibility: float) -> None:
        self.x = x
        self.y = y
        self.z = 0.0
        self.visibility = visibility


class _ReacquiredResult:
    """A detection found inside a search crop, in full-frame coordinates."""

    def __init__(self, normalised: np.ndarray, visibility: list[float],
                 world) -> None:
        self.pose_landmarks = [[
            _Landmark(float(p[0]), float(p[1]), float(v))
            for p, v in zip(normalised, visibility)
        ]]
        self.pose_world_landmarks = world


def draw_panel(image: np.ndarray, lines: list[str]) -> None:
    """A readable stats panel over the image."""
    pad = 10
    y = pad + 18
    for line in lines:
        cv2.putText(image, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_TEXT, 1, cv2.LINE_AA)
        y += 22


CAPTURE_BACKENDS = ([cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
                    if sys.platform == "win32" else [cv2.CAP_ANY])


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a webcam on the fastest backend that works.

    No resolution is requested. This camera ignores every size asked of it, in
    both YUY2 and MJPG and on both backends, and always returns 2560x1440:

        asked  640x480  -> got 2560x1440
        asked 1280x720  -> got 2560x1440
        asked 1920x1080 -> got 2560x1440

    So the frame is resized after capture instead, by `working_size`. Asking for
    a size the driver silently refuses only makes the code look like it did
    something.
    """
    for backend in CAPTURE_BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise SystemExit(f"Could not open camera {index}. Run without --camera to "
                     f"list the cameras that actually exist.")


DEFAULT_WORKING_HEIGHT = 720


def working_size(frame_width: int, frame_height: int,
                 target_height: int) -> tuple[int, int] | None:
    """The size to reduce a frame to, or None if it is already small enough.

    Only ever shrinks. Enlarging a frame to hit a target would invent pixels
    and slow everything down to process them.
    """
    if target_height <= 0 or frame_height <= target_height:
        return None
    scale = target_height / frame_height
    return max(1, int(round(frame_width * scale))), target_height


def screen_size() -> tuple[int, int]:
    """Usable screen area in real pixels, falling back to 1280x720.

    Two Windows-specific problems are handled here.

    The process must declare DPI awareness *before* asking, or Windows reports
    the scaled size and then stretches every window by the scale factor - so on
    a 150% display a window sized to the reported height still overflows.

    And the *work area* is asked for rather than the screen, because it
    excludes the taskbar. Sizing to the full screen height puts the bottom of
    the image behind it.
    """
    if sys.platform != "win32":
        return 1280, 720
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass
    try:
        class _Rect(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        rect = _Rect()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):
            return rect.right - rect.left, rect.bottom - rect.top
        return (ctypes.windll.user32.GetSystemMetrics(0),
                ctypes.windll.user32.GetSystemMetrics(1))
    except (AttributeError, OSError):
        return 1280, 720


def fit_to_screen(frame_width: int, frame_height: int,
                  margin: float = 0.90) -> tuple[int, int]:
    """Window size that fits on screen, keeping the aspect ratio.

    Limited by width *and* height. Fitting width alone is the common mistake
    and it fails on exactly this camera: 2560x1440 scaled to a 1920-wide screen
    is 1080 tall, which is taller than the 1032 px work area, so the bottom of
    the image would sit behind the taskbar.

    The margin leaves room for the title bar and borders, which are outside the
    size OpenCV is given.
    """
    screen_w, screen_h = screen_size()
    scale = min(screen_w * margin / frame_width,
                screen_h * margin / frame_height,
                1.0)  # never enlarge a small frame
    return max(160, int(frame_width * scale)), max(120, int(frame_height * scale))


CAMERA_PROBE_LIMIT = 8


@dataclass
class CameraInfo:
    """A camera that was opened *and* delivered a frame."""

    index: int
    width: int
    height: int
    name: str = ""

    def label(self) -> str:
        size = f"{self.width}x{self.height}"
        return f"{size:>10}  {self.name}" if self.name else f"{size:>10}"


def camera_names() -> list[str]:
    """Friendly device names from Windows, best effort.

    Cosmetic only. OpenCV's index order and the order Windows enumerates
    devices in are not guaranteed to match, so a name here is a hint rather
    than an identification - which is why the chooser also prints the
    resolution, a property read back from the device that really opened.

    Returns an empty list on any failure, including on non-Windows.
    """
    if sys.platform != "win32":
        return []
    import subprocess
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_PnPEntity | "
             "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
             "Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def probe_cameras(limit: int = CAMERA_PROBE_LIMIT) -> list[CameraInfo]:
    """Find every camera that opens and returns a frame.

    The second half of that matters. On Windows a device can report itself
    opened and then never deliver an image - a camera already held by another
    application behaves exactly this way. Listing it would send the user to a
    black window with no explanation, so a device joins the list only after it
    has produced a real frame.
    """
    previous_log_level = None
    try:
        previous_log_level = cv2.getLogLevel()
        cv2.setLogLevel(0)
    except AttributeError:
        pass

    names = camera_names()
    found: list[CameraInfo] = []
    backends = CAPTURE_BACKENDS[:1]
    try:
        for index in range(limit):
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                try:
                    if not cap.isOpened():
                        continue
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    h, w = frame.shape[:2]
                    name = names[len(found)] if len(found) < len(names) else ""
                    found.append(CameraInfo(index, w, h, name))
                    break
                finally:
                    cap.release()
    finally:
        if previous_log_level is not None:
            cv2.setLogLevel(previous_log_level)
    return found


def choose_camera(cameras: list[CameraInfo]) -> int:
    """Ask which camera to use, before anything heavy starts.

    Asking beats defaulting to index 0. On a laptop with an external webcam the
    built-in one is usually index 0 and usually the worse choice, and until now
    the only way to find the other was to guess indices on the command line.

    A single camera is taken without a prompt: a question with one possible
    answer is just an extra keystroke.
    """
    if not cameras:
        raise SystemExit(
            "No camera found.\n"
            "Close anything that might be holding it (Teams, Zoom, Camera), "
            "check Settings > Privacy > Camera, then try again."
        )

    if len(cameras) == 1:
        only = cameras[0]
        print(f"Using the only camera found: index {only.index} {only.label()}")
        return only.index

    print("\nCameras found:\n")
    for position, cam in enumerate(cameras, start=1):
        print(f"  {position}. index {cam.index} {cam.label()}")
    print()

    while True:
        try:
            answer = input(f"Choose a camera [1-{len(cameras)}, Enter for 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit("No camera chosen.")
        if not answer:
            return cameras[0].index
        if answer.isdigit() and 1 <= int(answer) <= len(cameras):
            return cameras[int(answer) - 1].index
        print(f"Enter a number between 1 and {len(cameras)}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="PsyBot body engine, laptop simulator")
    parser.add_argument("--camera", type=int, default=None,
                        help="camera index; omit to scan and choose")
    parser.add_argument("--list-cameras", action="store_true",
                        help="show the cameras that work and exit")
    parser.add_argument("--height", type=int, default=DEFAULT_WORKING_HEIGHT,
                        help=f"reduce frames to this height before processing "
                             f"(default {DEFAULT_WORKING_HEIGHT}; measured to "
                             f"cost no angle accuracy down to 360). 0 keeps "
                             f"the camera's own size")
    parser.add_argument("--full-size", action="store_true",
                        help="process at the camera's native size and do not "
                             "shrink the window")
    parser.add_argument("--record", type=str, default=None,
                        help="write a JSONL session the Dart engine can replay")
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not mirror the preview")
    parser.add_argument("--smoothing", action="store_true",
                        help="enable the extra One-Euro filter (off by "
                             "default: measured to cut jitter only 16%% while "
                             "MediaPipe already filters internally)")
    parser.add_argument("--exercise", default="squat",
                        help="which movement to track (see --list-exercises)")
    parser.add_argument("--list-exercises", action="store_true",
                        help="show the available exercises and exit")
    parser.add_argument("--calibrate", action="store_true",
                        help="learn this user's range of motion before "
                             "counting, instead of using population defaults")
    parser.add_argument("--debug", action="store_true",
                        help="show the diagnostic overlay: frame rate, "
                             "framing, landmark quality, joint angle, tracked "
                             "side and viewpoint yaw")
    parser.add_argument("--model", choices=("lite", "full", "heavy"),
                        default=MODEL_VARIANT,
                        help="pose model. Measured end to end at 1280x720: "
                             "lite 24.7 fps, full 19.6 fps, heavy 6.7 fps. "
                             "On angle steadiness across 12 comparisons heavy "
                             "won 9, lite 2, full 1 - so full buys about half "
                             "a degree for a quarter of the frame rate")
    parser.add_argument("--no-anchor", action="store_true",
                        help="do not hold quiet joints still (the anchor is "
                             "measured to cut 83%% of residual shiver for "
                             "0.146 deg of knee error)")
    parser.add_argument("--no-skeleton-model", action="store_true",
                        help="do not enforce calibrated bone lengths "
                             "(measured to cut bone variation from 0.72%% to "
                             "0.20%% while changing joint angles by 0.000 deg)")
    parser.add_argument("--no-hands", action="store_true",
                        help="disable finger reading and the stop gesture")
    parser.add_argument("--no-reacquire", action="store_true",
                        help="do not search for a lost subject by cropping "
                             "and upscaling (measured: fresh detection fails "
                             "below a 400px subject, cropping recovers to 60px)")
    parser.add_argument("--hand-every", type=int, default=4,
                        help="sample hands every N frames (default 4: the "
                             "gesture is held for 1.5s, so ~13 samples)")
    parser.add_argument("--profile", type=str, default=None,
                        help="load and save calibrated thresholds here")
    args = parser.parse_args()

    if args.list_cameras:
        for cam in probe_cameras():
            print(f"index {cam.index} {cam.label()}")
        return 0

    import angles as ang_mod
    import calibration as cal_mod
    import movement as mv
    from anchor import JointAnchor
    from hands import HandReader, HeldGesture
    from holds import EXERCISES, HoldDetector
    from occlusion import OcclusionTracker
    from skeleton import SkeletonCalibrator, apply_model
    from tracker import SubjectLocator, crop_for_search

    if args.list_exercises:
        print("Available exercises:\n")
        for key, item in EXERCISES.items():
            print(f"  {key:<10} {item.kind:<5} {item.joint:<9} {item.description}")
        return 0

    if args.exercise not in EXERCISES:
        print(f"Unknown exercise {args.exercise!r}. "
              f"Try: {', '.join(EXERCISES)}", file=sys.stderr)
        return 1

    exercise = EXERCISES[args.exercise]
    joint_name = exercise.joint
    inverted = exercise.inverted
    is_hold = exercise.kind == "hold"
    default_config = exercise.config() if not is_hold else mv.SQUAT
    hold_detector = HoldDetector() if is_hold else None

    profile = cal_mod.UserProfile()
    if args.profile and Path(args.profile).exists():
        profile = cal_mod.UserProfile.load(args.profile)
        print(f"Loaded profile from {args.profile}")

    config = profile.config_for(args.exercise, default_config)
    calibrator = cal_mod.RangeCalibrator(args.exercise) if args.calibrate else None
    session = mv.MovementSession(rep_config=config, inverted=inverted)
    side_tracker = ang_mod.SideTracker()
    angle_guard = ang_mod.AngleRateGuard()

    model_path = (Path(__file__).parent / "models"
                  / f"pose_landmarker_{args.model}.task")
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1
    if args.model != MODEL_VARIANT:
        print(f"Using the {args.model} model.")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if args.camera is None:
        print("Scanning for cameras...")
        camera_index = choose_camera(probe_cameras())
    else:
        camera_index = args.camera

    cap = open_camera(camera_index)
    smoother = PoseSmoother(enabled=args.smoothing)

    skeleton = SkeletonCalibrator()
    occlusion = OcclusionTracker()
    anchor = JointAnchor()
    use_anchor = not args.no_anchor
    use_skeleton = not args.no_skeleton_model

    locator = SubjectLocator()
    reacquire = not args.no_reacquire
    reacquisitions = 0

    hand_reader = None
    stop_gesture = HeldGesture(seconds=1.5)
    if not args.no_hands:
        hand_reader = HandReader(every_n_frames=max(1, args.hand_every))
        if not hand_reader.available:
            print("Hand model not found; finger reading is off. "
                  "Expected models/hand_landmarker.task")
            hand_reader = None
    last_hands: list = []

    target_height = 0 if args.full_size else max(0, args.height)
    resize_to: tuple[int, int] | None = None
    window = "PsyBot body engine - laptop simulator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_ASPECT_RATIO,
                          cv2.WINDOW_KEEPRATIO)
    window_sized = False

    recorder = None
    if args.record:
        recorder = open(args.record, "w", encoding="utf-8")
        recorder.write(json.dumps({
            "schema": 1,
            "sessionId": time.strftime("%Y-%m-%dT%H-%M-%S"),
            "startedAtMicros": int(time.time() * 1e6),
            "frameCount": 0,
        }) + "\n")

    frames = 0
    recorded = 0
    fps = 0.0
    last = time.time()
    started = time.time()

    print("Running. Keys: q quit, r toggle recording, s save a still.")

    search_options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker, \
            vision.PoseLandmarker.create_from_options(
                search_options) as searcher:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed", file=sys.stderr)
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            if resize_to is None and target_height:
                resize_to = working_size(frame.shape[1], frame.shape[0],
                                         target_height)
                if resize_to:
                    print(f"Camera gives {frame.shape[1]}x{frame.shape[0]}; "
                          f"processing at {resize_to[0]}x{resize_to[1]}")
            if resize_to:
                frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)

            if not window_sized:
                win_w, win_h = fit_to_screen(frame.shape[1], frame.shape[0])
                cv2.resizeWindow(window, win_w, win_h)
                window_sized = True

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - started) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            now = time.time()

            # region the subject was last in, upscaled - the detector needs a
            if (reacquire and not result.pose_landmarks
                    and locator.should_search(now)):
                locator.searched(now)
                for attempt in range(locator.attempts()):
                    region = locator.region(attempt, w, h, now)
                    if region is None:
                        break
                    crop = crop_for_search(frame, region)
                    if crop is None:
                        continue
                    found = searcher.detect(mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    if not found.pose_landmarks:
                        continue
                    ch, cw = crop.shape[:2]
                    local = np.array([[p.x * cw, p.y * ch]
                                      for p in found.pose_landmarks[0]])
                    mapped = region.to_frame(local, cw, ch)
                    result = _ReacquiredResult(
                        mapped / np.array([w, h]),
                        [p.visibility for p in found.pose_landmarks[0]],
                        found.pose_world_landmarks)
                    reacquisitions += 1
                    break
            frames += 1
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                norm = np.array([[p.x, p.y] for p in lm], dtype=np.float64)
                visibility = np.array([p.visibility for p in lm], dtype=np.float64)

                framing, advice, torso_q, upper_q, lower_q = assess_framing(visibility)

                pixels = norm * np.array([w, h])

                inferred = np.zeros(len(pixels), dtype=bool)
                discarded = np.zeros(len(pixels), dtype=bool)
                if use_skeleton:
                    body = skeleton.update(pixels, visibility, (w, h))
                    pixels = apply_model(pixels, body)

                    report = occlusion.update(pixels, visibility, body, now,
                                              w, h)
                    pixels = report.points
                    visibility = report.confidence
                    inferred = report.inferred
                    discarded = report.discarded

                    if use_anchor:
                        pixels = anchor.update(
                            pixels,
                            mv.torso_length(pixels, visibility) or 0.0,
                            ~discarded, (w, h))

                locator.seen(pixels, visibility, now)

                out_of_frame_advice = ""
                measured = [IDX[f"{side}{joint_name}"]
                            for side in ("left", "right")
                            if f"{side}{joint_name}" in IDX]
                if measured and all(discarded[i] for i in measured):
                    out_of_frame_advice = "step back - I cannot see your " \
                                          f"{joint_name.lower()}"

                drawn = smoother(pixels, visibility, now)
                draw_skeleton(frame, drawn, visibility, framing, inferred,
                              discarded)

                if hand_reader is not None:
                    if hand_reader.tick():
                        wrists = [pixels[i] for i in
                                  (IDX["leftWrist"], IDX["rightWrist"])
                                  if visibility[i] >= 0.3]
                        last_hands = hand_reader.read(
                            frame, wrists,
                            mv.torso_length(pixels, visibility) or 100.0)
                    if last_hands:
                        draw_hands(frame, last_hands, max(1.0, min(w, h) / 100.0))
                    gesture = stop_gesture.update(
                        any(hand.stop for hand in last_hands), now)
                    if gesture.progress > 0:
                        draw_stop_progress(frame, gesture.progress)
                    if gesture.fired:
                        print("Stop gesture held - quitting.")
                        cv2.imshow(window, frame)
                        cv2.waitKey(400)
                        break

                tf = torso_frame(norm, visibility)
                tilt = f"{tf[2]:+.0f}" if tf else "--"

                all_ang = angle_guard.filter(
                    ang_mod.all_angles(pixels, visibility), now)
                tracked = side_tracker.update(all_ang, joint_name)
                angle_value = tracked.or_none() if tracked else None

                if out_of_frame_advice:
                    angle_value = None

                world = None
                if result.pose_world_landmarks:
                    world = np.array(
                        [[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]],
                        dtype=np.float64,
                    )

                hold_state = None
                if hold_detector is not None:
                    hold_state = hold_detector.update(
                        pixels, visibility, mv.torso_length(pixels, visibility),
                        now, angle=angle_value,
                    )

                if calibrator is not None:
                    calibrator.add(angle_value)
                    report = None
                elif is_hold:
                    report = None
                else:
                    angle_3d_value = None
                    if world is not None and side_tracker.side:
                        import reliability as rel_mod
                        joint_key = f"{side_tracker.side}{joint_name}"
                        definition = ang_mod.JOINTS.get(joint_key)
                        if definition:
                            angle_3d_value = rel_mod.angle_3d(world, *definition)

                    report = session.update(pixels, visibility, angle_value,
                                            now, world_points=world,
                                            angle_3d=angle_3d_value)

                lines = [
                    f"fps {fps:4.1f}   {framing}",
                    f"torso {torso_q:.2f}  upper {upper_q:.2f}  lower {lower_q:.2f}",
                ] if args.debug else []

                if calibrator is not None:
                    need = max(0, calibrator.min_samples - calibrator.count)
                    lines.append(
                        f"CALIBRATING {args.exercise}: {calibrator.count} frames"
                        + (f", {need} more" if need else " - press c to finish")
                    )
                    lines.append(f"angle {'--' if angle_value is None else f'{angle_value:.0f}'}")
                elif hold_state is not None:
                    total = hold_detector.total_seconds
                    if hold_state.holding:
                        lines.append(
                            f"{args.exercise}  HOLDING {hold_state.elapsed:4.1f}s"
                            f"   total {total:.0f}s"
                        )
                    else:
                        lines.append(
                            f"{args.exercise}  {hold_state.reason or 'get into position'}"
                            f"   total {total:.0f}s"
                        )
                    if hold_detector.holds:
                        last_hold = hold_detector.holds[-1]
                        lines.append(
                            f"last {last_hold.duration:.1f}s"
                            f"   steadiness {last_hold.steadiness:.2f}"
                            f"   holds {len(hold_detector.holds)}"
                        )
                    else:
                        lines.append(exercise.description)
                elif report is not None:
                    if report.usable:
                        state = report.rep_state.value
                        lines.append(
                            f"{args.exercise}  reps {report.rep_count}"
                            f"  (vel {report.velocity_count})"
                            f"   partial {report.partial_count}"
                            f"   jumps {report.jump_count}"
                        )
                        if args.debug:
                            lines.append(
                                f"state {state:<7} angle "
                                f"{'--' if angle_value is None else f'{angle_value:.0f}'}"
                                f"  side {side_tracker.side or '--'}"
                                f"  yaw {'--' if report.viewpoint_offset is None else f'{report.viewpoint_offset:.0f}'}"
                            )
                    else:
                        lines.append(f"PAUSED: {report.reason}")
                        lines.append(
                            f"reps {report.rep_count} held"
                        )

                if reacquisitions and args.debug:
                    lines.append(f"re-acquired {reacquisitions}x")
                if recorder:
                    lines.append(f"REC {recorded}")
                draw_panel(frame, lines)

                if hold_state is not None and hold_state.completed is not None:
                    done = hold_state.completed
                    banner = f"HELD {done.duration:.0f}s"
                    cv2.putText(frame, banner, (w // 2 - 110, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                (0, 0, 0), 8, cv2.LINE_AA)
                    cv2.putText(frame, banner, (w // 2 - 110, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                COLOR_OK, 3, cv2.LINE_AA)

                if report is not None and report.usable:
                    if report.rep_completed:
                        banner = f"REP {report.rep_count}"
                        cv2.putText(frame, banner, (w // 2 - 90, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                                    (0, 0, 0), 8, cv2.LINE_AA)
                        cv2.putText(frame, banner, (w // 2 - 90, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                                    COLOR_OK, 3, cv2.LINE_AA)
                    elif report.form is not None and report.form.notes:
                        note = report.form.notes[0]
                        cv2.putText(frame, note, (w // 2 - 200, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 0), 6, cv2.LINE_AA)
                        cv2.putText(frame, note, (w // 2 - 200, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    COLOR_WEAK, 2, cv2.LINE_AA)
                    elif report.rep_partial:
                        cv2.putText(frame, "not deep enough", (w // 2 - 130, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 0, 0), 6, cv2.LINE_AA)
                        cv2.putText(frame, "not deep enough", (w // 2 - 130, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    COLOR_WEAK, 2, cv2.LINE_AA)

                message = out_of_frame_advice or advice
                if message:
                    cv2.putText(frame, message, (10, h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, message, (10, h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WEAK, 1, cv2.LINE_AA)

                if recorder:
                    recorder.write(json.dumps({
                        "timestampMicros": int(now * 1e6),
                        "isMirrored": not args.no_mirror,
                        "landmarks": [
                            {
                                "type": i,
                                "x": float(norm[i, 0]),
                                "y": float(norm[i, 1]),
                                "z": 0.0,
                                "likelihood": float(visibility[i]),
                                "inFrameLikelihood": float(visibility[i]),
                            }
                            for i in range(len(norm))
                        ],
                    }) + "\n")
                    recorded += 1
            else:
                smoother.reset()
                idle = [f"fps {fps:4.1f}   noSubject"] if args.debug \
                    else ["I cannot see you"]
                if recorder:
                    idle.append(f"REC {recorded}")
                draw_panel(frame, idle)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                name = f"still_{int(now)}.png"
                cv2.imwrite(name, frame)
                print("saved", name)
            if key == ord("c") and calibrator is not None:
                outcome = calibrator.result()
                if not outcome.ok:
                    print(f"Calibration failed: {outcome.reason}")
                else:
                    print(
                        f"Calibrated {args.exercise}: range "
                        f"{outcome.observed_min:.0f}-{outcome.observed_max:.0f} deg, "
                        f"thresholds {outcome.config.down_below:.0f}/"
                        f"{outcome.config.up_above:.0f}"
                    )
                    profile.store(args.exercise, outcome)
                    if args.profile:
                        profile.save(args.profile)
                        print(f"Saved to {args.profile}")
                    session = mv.MovementSession(rep_config=outcome.config,
                                                 inverted=inverted)
                    calibrator = None

    cap.release()
    cv2.destroyAllWindows()
    if recorder:
        recorder.close()
        print(f"Recorded {recorded} frames to {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())