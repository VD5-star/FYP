from __future__ import annotations
import random
from dataclasses import dataclass
import numpy as np
NEAR = "near"
FAR = "far"
VERY_FAR = "very_far"
BONUS = "bonus"
ARM_LENGTH = 0.97
BODY_GAP = 0.95
BANDS: dict[str, tuple[float, float, int]] = {
    NEAR: (0.95, 1.15, 1),
    FAR: (1.15, 1.40, 2),
    VERY_FAR: (1.40, 99.0, 3),
}
REACH_SIDE = 2.30
REACH_UP = 2.50
REACH_DOWN = 1.80
EDGE_STEP = 0.25
BASE_SIZE = 0.42
MIN_SIZE_PX = 22.0
LIFETIME = 1.5
SHRINK_FLOOR = 0.35
BONUS_EVERY = 5
BONUS_SECONDS = 3.0
BONUS_SIZE = 1.25
@dataclass
class Target:
    x: float
    y: float
    radius: float
    band: str
    points: int
    born: float
    lifetime: float
    @property
    def bonus(self) -> bool:
        return self.band == BONUS
    def age(self, now: float) -> float:
        return now - self.born
    def remaining(self, now: float) -> float:
        return max(0.0, self.lifetime - self.age(now))
    def life(self, now: float) -> float:
        return max(0.0, min(1.0, self.remaining(now) / max(self.lifetime, 1e-6)))
    def scale(self, now: float) -> float:
        return SHRINK_FLOOR + (1.0 - SHRINK_FLOOR) * self.life(now)
    def current_radius(self, now: float) -> float:
        return self.radius * self.scale(now)
    def contains(self, point: np.ndarray, now: float) -> bool:
        if not np.all(np.isfinite(point)):
            return False
        r = self.current_radius(now)
        return abs(point[0] - self.x) <= r and abs(point[1] - self.y) <= r
    def contains_any(self, points: np.ndarray, now: float) -> bool:
        if points is None or len(points) == 0:
            return False
        pts = np.asarray(points, dtype=float)
        r = self.current_radius(now)
        inside = ((np.abs(pts[:, 0] - self.x) <= r)
                  & (np.abs(pts[:, 1] - self.y) <= r))
        return bool(inside.any())
    def expired(self, now: float) -> bool:
        return self.age(now) >= self.lifetime
def edge_points(body: np.ndarray, torso: float, width: int, height: int,
                margin: float = 10.0) -> np.ndarray:
    if body.size == 0 or torso <= 1e-6:
        return np.empty((0, 2))
    step = max(8.0, EDGE_STEP * torso)
    x0, x1 = float(body[:, 0].min()), float(body[:, 0].max())
    y0, y1 = float(body[:, 1].min()), float(body[:, 1].max())
    pad = torso * 0.5
    out: list[list[float]] = []
    if y0 <= margin:
        for x in np.arange(x0 - pad, x1 + pad + step, step):
            out.append([float(x), 0.0])
    if y1 >= height - margin:
        for x in np.arange(x0 - pad, x1 + pad + step, step):
            out.append([float(x), float(height)])
    if x0 <= margin:
        for y in np.arange(y0 - pad, y1 + pad + step, step):
            out.append([0.0, float(y)])
    if x1 >= width - margin:
        for y in np.arange(y0 - pad, y1 + pad + step, step):
            out.append([float(width), float(y)])
    return np.array(out) if out else np.empty((0, 2))
class TargetSpawner:
    def __init__(self, lifetime: float = LIFETIME,
                 body_gap: float = BODY_GAP,
                 seed: int | None = None) -> None:
        self.lifetime = lifetime
        self.body_gap = body_gap
        self._seed = seed
        self._random = random.Random(seed)
        self.rejected = 0
        self.relaxed = 0
    def reset(self) -> None:
        self._random = random.Random(self._seed)
        self.rejected = 0
        self.relaxed = 0
    def _band_for(self, gap: float) -> tuple[str, int]:
        for name, (low, high, points) in BANDS.items():
            if low <= gap < high:
                return name, points
        return VERY_FAR, BANDS[VERY_FAR][2]
    def _limit(self, dy: float) -> float:
        if dy > 0:
            return REACH_SIDE * (1 - dy) + REACH_UP * dy
        return REACH_SIDE * (1 + dy) + REACH_DOWN * -dy
    def spawn(self, anchor: np.ndarray, torso: float, now: float,
              width: int, height: int,
              body: np.ndarray | None = None,
              bonus: bool = False) -> Target | None:
        if torso <= 1e-6 or not np.all(np.isfinite(anchor)):
            return None
        base = max(MIN_SIZE_PX, torso * BASE_SIZE * 0.5)
        radius = base * BONUS_SIZE if bonus else base
        margin = radius * 1.05
        if body is None or len(body) == 0:
            avoid = np.array([anchor])
        else:
            avoid = np.asarray(body, dtype=float)
            avoid = avoid[np.all(np.isfinite(avoid), axis=1)]
            if avoid.size == 0:
                avoid = np.array([anchor])
            else:
                fence = edge_points(avoid, torso, width, height)
                if fence.size:
                    avoid = np.vstack([avoid, fence])
        best = None
        best_gap = -1.0
        for _ in range(220):
            fraction = self._random.random()
            angle = self._random.uniform(0.0, 2.0 * np.pi)
            dx = float(np.cos(angle))
            dy = float(np.sin(angle))
            span = self._limit(dy) * torso
            near = self.body_gap * torso
            if span <= near:
                continue
            distance = near + fraction * (span - near)
            x = anchor[0] + distance * dx
            y = anchor[1] - distance * dy
            if not (margin <= x <= width - margin):
                continue
            if not (margin <= y <= height - margin):
                continue
            gap = float(np.min(np.hypot(avoid[:, 0] - x, avoid[:, 1] - y)))
            units = gap / torso
            if gap >= self.body_gap * torso:
                band, points = self._band_for(units)
                if bonus:
                    band, points = BONUS, 0
                return Target(x=float(x), y=float(y), radius=float(radius),
                              band=band, points=points, born=now,
                              lifetime=self.lifetime)
            if gap > best_gap:
                best_gap = gap
                band, points = self._band_for(units)
                if bonus:
                    band, points = BONUS, 0
                best = Target(x=float(x), y=float(y), radius=float(radius),
                              band=band, points=points, born=now,
                              lifetime=self.lifetime)
        if best is not None and best_gap >= self.body_gap * torso * 0.6:
            self.relaxed += 1
            best.born = now
            return best
        self.rejected += 1
        return None
