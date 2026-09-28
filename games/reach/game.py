from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from targets import (BODY_GAP, BONUS, BONUS_EVERY, BONUS_SECONDS, LIFETIME,
                     Target, TargetSpawner)
RUNNING = "running"
PAUSED = "paused"
ENDING = "ending"
FINISHED = "finished"
ENDING_SECONDS = 1.1
TIMED = "timed"
CALM = "calm"
LIMBS = (
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("leftHip", "leftKnee"), ("leftKnee", "leftAnkle"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip"),
    ("leftShoulder", "leftHip"), ("rightShoulder", "rightHip"),
    ("leftWrist", "leftIndex"), ("rightWrist", "rightIndex"),
    ("leftAnkle", "leftFootIndex"), ("rightAnkle", "rightFootIndex"),
)
LIMB_STEPS = 4
TOUCH_CONFIDENCE = 0.15
ANCHOR_CONFIDENCE = 0.15
@dataclass
class Hit:
    band: str
    points: int
    at: float
    bonus: bool = False
@dataclass
class GameState:
    mode: str = TIMED
    phase: str = RUNNING
    score: int = 0
    hits: list[Hit] = field(default_factory=list)
    misses: int = 0
    streak: int = 0
    bonus_time: float = 0.0
    bonus_taken: int = 0
    started: float | None = None
    ended: float | None = None
    paused_at: float | None = None
    paused_total: float = 0.0
    last_hit: float | None = None
    last_band: str | None = None
    ending_at: float | None = None
class ReachGame:
    def __init__(self, mode: str = TIMED, duration: float = 60.0,
                 lifetime: float = LIFETIME, body_gap: float = BODY_GAP,
                 seed: int | None = None) -> None:
        self.mode = mode
        self.duration = duration
        self.spawner = TargetSpawner(lifetime=lifetime, body_gap=body_gap,
                                     seed=seed)
        self.state = GameState(mode=mode)
        self.target: Target | None = None
        self._bonus_due = False
    @property
    def targets(self) -> list[Target]:
        return [self.target] if self.target is not None else []
    def elapsed(self, now: float) -> float:
        if self.state.started is None:
            return 0.0
        end = self.state.ended if self.state.ended is not None else now
        if self.state.phase == PAUSED and self.state.paused_at is not None:
            end = self.state.paused_at
        spent = max(0.0, end - self.state.started - self.state.paused_total)
        if self.mode == TIMED:
            return min(spent, self.duration + self.state.bonus_time)
        return spent
    def total_time(self) -> float:
        return self.duration + self.state.bonus_time
    def remaining(self, now: float) -> float:
        if self.mode != TIMED:
            return float("inf")
        return max(0.0, self.total_time() - self.elapsed(now))
    def pause(self, now: float) -> None:
        if self.state.phase != RUNNING:
            return
        self.state.phase = PAUSED
        self.state.paused_at = now
    def resume(self, now: float) -> None:
        if self.state.phase != PAUSED:
            return
        if self.state.paused_at is not None:
            shift = now - self.state.paused_at
            self.state.paused_total += shift
            if self.target is not None:
                self.target.born += shift
        self.state.paused_at = None
        self.state.phase = RUNNING
    def begin_ending(self, now: float) -> None:
        if self.state.phase in (ENDING, FINISHED):
            return
        if self.target is not None:
            self.state.misses += 1
        self.state.phase = ENDING
        self.state.ending_at = now
        self.state.ended = now
    def finish(self, now: float) -> None:
        if self.state.phase == FINISHED:
            return
        if self.state.ended is None:
            self.state.ended = now
        self.state.phase = FINISHED
        self.target = None
    def ending_progress(self, now: float) -> float:
        if self.state.ending_at is None:
            return 1.0
        return max(0.0, min(1.0, (now - self.state.ending_at) / ENDING_SECONDS))
    def update(self, points: np.ndarray | None, visibility: np.ndarray | None,
               index: dict[str, int], torso: float | None, now: float,
               width: int, height: int) -> list[Hit]:
        if self.state.phase == ENDING:
            if self.ending_progress(now) >= 1.0:
                self.finish(now)
            return []
        if self.state.phase in (FINISHED, PAUSED):
            return []
        if points is None or visibility is None or torso is None or torso <= 1e-6:
            return []
        if self.state.started is None:
            self.state.started = now
        if self.mode == TIMED and self.remaining(now) <= 0.0:
            self.begin_ending(now)
            return []
        anchor = self._anchor(points, visibility, index)
        if anchor is None:
            return []
        body = self._body(points, visibility, index)
        scored: list[Hit] = []
        if self.target is not None and self.target.expired(now):
            if not self.target.bonus:
                self.state.misses += 1
                self.state.streak = 0
            self.target = None
        if self.target is None:
            self._spawn(anchor, torso, now, width, height, body)
        if self.target is None:
            return scored
        touched = self._touch_points(points, visibility, index)
        if self.target.contains_any(touched, now):
            target = self.target
            hit = Hit(band=target.band, points=target.points, at=now,
                      bonus=target.bonus)
            if target.bonus:
                self.state.bonus_time += BONUS_SECONDS
                self.state.bonus_taken += 1
            else:
                self.state.score += target.points
                self.state.streak += 1
                if self.state.streak % BONUS_EVERY == 0 \
                        and self.mode == TIMED:
                    self._bonus_due = True
            self.state.hits.append(hit)
            self.state.last_hit = now
            self.state.last_band = target.band
            scored.append(hit)
            self.target = None
            self._spawn(anchor, torso, now, width, height, body)
        return scored
    def _spawn(self, anchor: np.ndarray, torso: float, now: float,
               width: int, height: int, body: np.ndarray | None) -> None:
        bonus = self._bonus_due and self.mode == TIMED
        target = self.spawner.spawn(anchor, torso, now, width, height,
                                    body=body, bonus=bonus)
        if target is None:
            return
        if bonus:
            self._bonus_due = False
        self.target = target
    def _body(self, points: np.ndarray, visibility: np.ndarray,
              index: dict[str, int]) -> np.ndarray:
        keep = []
        for i in range(len(points)):
            if visibility[i] < TOUCH_CONFIDENCE:
                continue
            if not np.all(np.isfinite(points[i])):
                continue
            keep.append(points[i])
        return np.array(keep) if keep else np.empty((0, 2))
    def _anchor(self, points: np.ndarray, visibility: np.ndarray,
                index: dict[str, int]) -> np.ndarray | None:
        names = ("leftShoulder", "rightShoulder", "leftHip", "rightHip")
        chosen = []
        for name in names:
            i = index.get(name)
            if i is None:
                continue
            if visibility[i] < ANCHOR_CONFIDENCE:
                continue
            if not np.all(np.isfinite(points[i])):
                continue
            chosen.append(points[i])
        if len(chosen) < 2:
            return None
        return np.mean(np.array(chosen), axis=0)
    def _touch_points(self, points: np.ndarray, visibility: np.ndarray,
                      index: dict[str, int]) -> np.ndarray:
        out = []
        usable = np.zeros(len(points), dtype=bool)
        for i in range(len(points)):
            if visibility[i] < TOUCH_CONFIDENCE:
                continue
            if not np.all(np.isfinite(points[i])):
                continue
            usable[i] = True
            out.append(points[i])
        for a, b in LIMBS:
            ia, ib = index.get(a), index.get(b)
            if ia is None or ib is None:
                continue
            if not (usable[ia] and usable[ib]):
                continue
            pa, pb = points[ia], points[ib]
            for k in range(1, LIMB_STEPS):
                out.append(pa + (pb - pa) * (k / LIMB_STEPS))
        return np.array(out) if out else np.empty((0, 2))
    def summary(self, now: float) -> dict:
        bands: dict[str, int] = {}
        for hit in self.state.hits:
            bands[hit.band] = bands.get(hit.band, 0) + 1
        return {
            "score": self.state.score,
            "hits": sum(1 for h in self.state.hits if not h.bonus),
            "misses": self.state.misses,
            "bonus": self.state.bonus_taken,
            "seconds": self.elapsed(now),
            "bands": bands,
        }
    def reset(self) -> None:
        self.state = GameState(mode=self.mode)
        self.target = None
        self._bonus_due = False
        self.spawner.reset()
