from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from angles import IDX
from skeleton import TREE, BodyModel, torso_length
BONE_TOLERANCE = 0.35
TELEPORT_RATE = 45.0
MAX_HOLD_SECONDS = 1.5
FRAMES_BEFORE_TRUSTED = 20
EDGE_MARGIN = 0.002
BELIEVE_THRESHOLD = 0.5
@dataclass
class LandmarkState:
    position: np.ndarray
    confidence: float
    observations: int = 0
    inferred: bool = False
    held_for: float = 0.0
    anchor_offset: np.ndarray | None = None
    @property
    def trusted(self) -> bool:
        return self.observations >= FRAMES_BEFORE_TRUSTED
@dataclass
class OcclusionReport:
    points: np.ndarray
    confidence: np.ndarray
    inferred: np.ndarray
    discarded: np.ndarray
    @property
    def any_inferred(self) -> bool:
        return bool(self.inferred.any())
    def usable(self, index: int) -> bool:
        return (not self.discarded[index]
                and self.confidence[index] >= 0.5)
    def drawable(self, index: int) -> bool:
        return not self.discarded[index]
def outside_frame(point: np.ndarray, width: int, height: int,
                  margin: float = EDGE_MARGIN) -> bool:
    if not np.all(np.isfinite(point)):
        return True
    mx = width * margin
    my = height * margin
    return bool(point[0] < -mx or point[0] > width + mx
                or point[1] < -my or point[1] > height + my)
class OcclusionTracker:
    def __init__(self, bone_tolerance: float = BONE_TOLERANCE,
                 teleport_rate: float = TELEPORT_RATE,
                 max_hold_seconds: float = MAX_HOLD_SECONDS,
                 frames_before_trusted: int = FRAMES_BEFORE_TRUSTED,
                 believe_threshold: float = BELIEVE_THRESHOLD) -> None:
        self.bone_tolerance = bone_tolerance
        self.teleport_rate = teleport_rate
        self.believe_threshold = believe_threshold
        self.max_hold_seconds = max_hold_seconds
        self.frames_before_trusted = frames_before_trusted
        self._state: dict[int, LandmarkState] = {}
        self._last_t: float | None = None
    def update(self, points: np.ndarray, visibility: np.ndarray,
               model: BodyModel, timestamp: float,
               frame_width: int, frame_height: int) -> OcclusionReport:
        count = len(points)
        out = points.copy().astype(np.float64)
        conf = visibility.copy().astype(np.float64)
        inferred = np.zeros(count, dtype=bool)
        discarded = np.zeros(count, dtype=bool)
        dt = 0.0 if self._last_t is None else max(0.0, timestamp - self._last_t)
        if dt > 1.0:
            self._state.clear()
            dt = 0.0
        self._last_t = timestamp
        torso = torso_length(points)
        suspect = self._suspect_landmarks(points, visibility, model, torso, dt)
        mid_hip = (points[IDX["leftHip"]] + points[IDX["rightHip"]]) / 2
        can_rebase = (torso > 1e-6 and np.isfinite(torso)
                      and np.all(np.isfinite(mid_hip))
                      and np.all(np.isfinite(points[IDX["leftHip"]]))
                      and np.all(np.isfinite(points[IDX["rightHip"]])))
        for i in range(count):
            if outside_frame(points[i], frame_width, frame_height):
                self._state.pop(i, None)
                conf[i] = 0.0
                discarded[i] = True
                continue
            state = self._state.get(i)
            believable = visibility[i] >= self.believe_threshold and i not in suspect
            if believable:
                observations = (state.observations + 1) if state else 1
                self._state[i] = LandmarkState(
                    position=points[i].copy(),
                    confidence=float(visibility[i]),
                    observations=observations,
                    inferred=False,
                    held_for=0.0,
                    anchor_offset=((points[i] - mid_hip) / torso
                                   if can_rebase else None),
                )
                continue
            if state is None or not state.trusted:
                if state is not None:
                    self._state[i] = LandmarkState(
                        position=state.position,
                        confidence=state.confidence,
                        observations=state.observations,
                        inferred=True,
                        held_for=state.held_for + dt,
                        anchor_offset=state.anchor_offset,
                    )
                conf[i] = 0.0
                discarded[i] = True
                continue
            held = state.held_for + dt
            position = state.position
            if state.anchor_offset is not None and can_rebase:
                rebased = mid_hip + state.anchor_offset * torso
                if not outside_frame(rebased, frame_width, frame_height):
                    position = rebased
            out[i] = position
            inferred[i] = True
            if held >= self.max_hold_seconds:
                conf[i] = 0.0
                discarded[i] = True
            else:
                conf[i] = state.confidence * (1.0 - held / self.max_hold_seconds)
            self._state[i] = LandmarkState(
                position=position,
                confidence=state.confidence,
                observations=state.observations,
                inferred=True,
                held_for=held,
                anchor_offset=state.anchor_offset,
            )
        return OcclusionReport(points=out, confidence=conf,
                               inferred=inferred, discarded=discarded)
    def _suspect_landmarks(self, points: np.ndarray, visibility: np.ndarray,
                           model: BodyModel, torso: float,
                           dt: float) -> set[int]:
        suspect: set[int] = set()
        if torso < 1e-6 or not np.isfinite(torso):
            return suspect
        if model.ready:
            for child, parent in TREE:
                if parent is None or child not in model.lengths:
                    continue
                ci, pi = IDX[child], IDX[parent]
                expected = model.lengths[child] * torso
                if expected < 1e-6:
                    continue
                actual = float(np.linalg.norm(points[ci] - points[pi]))
                if abs(actual - expected) / expected > self.bone_tolerance:
                    suspect.add(ci)
        if dt > 1e-6:
            limit = self.teleport_rate * torso * dt
            for i, state in self._state.items():
                if state.inferred or i >= len(points):
                    continue
                if float(np.linalg.norm(points[i] - state.position)) > limit:
                    suspect.add(i)
        return suspect
    def reset(self) -> None:
        self._state.clear()
        self._last_t = None
