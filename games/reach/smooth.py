from __future__ import annotations
import math
import numpy as np
MIN_CUTOFF = 1.2
BETA = 0.02
D_CUTOFF = 1.0
GAP_RESET_SECONDS = 0.100
SMOOTH_VISIBILITY = 0.15
class OneEuro:
    def __init__(self, min_cutoff: float = MIN_CUTOFF, beta: float = BETA,
                 d_cutoff: float = D_CUTOFF) -> None:
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
class PointSmoother:
    def __init__(self, min_cutoff: float = MIN_CUTOFF, beta: float = BETA,
                 gap_reset: float = GAP_RESET_SECONDS) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.gap_reset = gap_reset
        self._fx: dict[int, OneEuro] = {}
        self._fy: dict[int, OneEuro] = {}
        self._last_t: float | None = None
    def __call__(self, points: np.ndarray, visibility: np.ndarray | None,
                 timestamp: float) -> np.ndarray:
        if points is None:
            return points
        if (self._last_t is not None
                and timestamp - self._last_t > self.gap_reset):
            self.reset()
        self._last_t = timestamp
        out = points.copy()
        for i in range(len(points)):
            if visibility is not None and visibility[i] < SMOOTH_VISIBILITY:
                self._fx.pop(i, None)
                self._fy.pop(i, None)
                continue
            if not np.all(np.isfinite(points[i])):
                continue
            if i not in self._fx:
                self._fx[i] = OneEuro(self.min_cutoff, self.beta)
                self._fy[i] = OneEuro(self.min_cutoff, self.beta)
            out[i, 0] = self._fx[i](float(points[i, 0]), timestamp)
            out[i, 1] = self._fy[i](float(points[i, 1]), timestamp)
        return out
    def reset(self) -> None:
        self._last_t = None
        self._fx.clear()
        self._fy.clear()
