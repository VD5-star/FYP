from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
SPARK_LIFE = 0.62
SPARK_COUNT = 14
SPARK_SPEED = 170.0
SPARK_DRAG = 3.1
GRAVITY = 210.0
RING_LIFE = 0.46
RING_GROW = 2.1
MAX_RINGS = 6
DUST_COUNT = 5
DUST_LIFE = 0.9
MAX_PARTICLES = 260
@dataclass
class Particles:
    cap: int = MAX_PARTICLES
    n: int = 0
    px: np.ndarray = field(default=None)
    py: np.ndarray = field(default=None)
    vx: np.ndarray = field(default=None)
    vy: np.ndarray = field(default=None)
    life: np.ndarray = field(default=None)
    full: np.ndarray = field(default=None)
    size: np.ndarray = field(default=None)
    col: np.ndarray = field(default=None)
    grav: np.ndarray = field(default=None)
    def __post_init__(self):
        c = self.cap
        self.px = np.zeros(c, np.float32)
        self.py = np.zeros(c, np.float32)
        self.vx = np.zeros(c, np.float32)
        self.vy = np.zeros(c, np.float32)
        self.life = np.zeros(c, np.float32)
        self.full = np.ones(c, np.float32)
        self.size = np.ones(c, np.float32)
        self.col = np.zeros((c, 3), np.float32)
        self.grav = np.zeros(c, np.float32)
    def clear(self) -> None:
        self.n = 0
    def _room(self, k: int) -> tuple[int, int]:
        if self.n + k > self.cap:
            drop = min(self.n, self.n + k - self.cap)
            keep = self.n - drop
            for a in (self.px, self.py, self.vx, self.vy, self.life,
                      self.full, self.size, self.grav):
                a[:keep] = a[drop:self.n]
            self.col[:keep] = self.col[drop:self.n]
            self.n = keep
        k = min(k, self.cap - self.n)
        s = self.n
        self.n += k
        return s, s + k
    def emit(self, x: float, y: float, count: int, rng, colour,
             speed: float = SPARK_SPEED, life: float = SPARK_LIFE,
             gravity: float = GRAVITY, size: float = 2.2,
             spread: float = 1.0) -> None:
        if count <= 0:
            return
        s, e = self._room(count)
        k = e - s
        if k <= 0:
            return
        ang = rng.uniform(0, 2 * np.pi, k)
        mag = rng.uniform(0.25, 1.0, k) ** 0.6 * speed * spread
        jr = rng.uniform(0.0, 1.0, k) ** 0.5 * max(1.0, size * 2.2)
        ja = rng.uniform(0, 2 * np.pi, k)
        self.px[s:e] = x + np.cos(ja) * jr
        self.py[s:e] = y + np.sin(ja) * jr
        self.vx[s:e] = np.cos(ang) * mag
        self.vy[s:e] = np.sin(ang) * mag - speed * 0.25
        lf = life * rng.uniform(0.65, 1.25, k)
        self.life[s:e] = lf
        self.full[s:e] = lf
        self.size[s:e] = size * rng.uniform(0.6, 1.5, k)
        self.grav[s:e] = gravity
        base = np.array(colour, np.float32)[None, :]
        tint = rng.uniform(0.82, 1.18, (k, 1)).astype(np.float32)
        self.col[s:e] = np.clip(base * tint, 0, 255)
    def step(self, dt: float) -> None:
        if self.n == 0:
            return
        n = self.n
        dt = float(np.clip(dt, 0.0, 0.05))
        d = float(np.exp(-SPARK_DRAG * dt))
        self.vx[:n] *= d
        self.vy[:n] *= d
        self.vy[:n] += self.grav[:n] * dt
        self.px[:n] += self.vx[:n] * dt
        self.py[:n] += self.vy[:n] * dt
        self.life[:n] -= dt
        alive = self.life[:n] > 0.0
        if alive.all():
            return
        idx = np.flatnonzero(alive)
        k = idx.size
        for a in (self.px, self.py, self.vx, self.vy, self.life,
                  self.full, self.size, self.grav):
            a[:k] = a[idx]
        self.col[:k] = self.col[idx]
        self.n = k
    def view(self):
        n = self.n
        t = np.clip(self.life[:n] / np.maximum(self.full[:n], 1e-6), 0, 1)
        return (self.px[:n], self.py[:n], self.size[:n] * (0.35 + t * 0.65),
                self.col[:n], t)
@dataclass
class Ring:
    x: float
    y: float
    r0: float
    life: float
    full: float
    colour: tuple
@dataclass
class Effects:
    rng: object = None
    parts: Particles = field(default_factory=Particles)
    rings: list = field(default_factory=list)
    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(7)
    def clear(self) -> None:
        self.parts.clear()
        self.rings.clear()
    def place(self, x: float, y: float, radius: float,
              colour=(196, 224, 236), strength: float = 1.0) -> None:
        self.parts.emit(x, y, int(SPARK_COUNT * strength), self.rng, colour,
                        speed=SPARK_SPEED * (0.8 + strength * 0.3),
                        size=2.1 * strength)
        self.rings.append(Ring(x, y, radius * 0.45, RING_LIFE, RING_LIFE,
                               colour))
        if len(self.rings) > MAX_RINGS:
            del self.rings[:-MAX_RINGS]
    def lift(self, x: float, y: float) -> None:
        self.parts.emit(x, y, DUST_COUNT, self.rng, (150, 150, 146),
                        speed=52.0, life=DUST_LIFE, gravity=-26.0,
                        size=1.5, spread=0.7)
    def finish(self, w: int, h: int, n: int = 90) -> None:
        spots = 7
        for i in range(spots):
            x = float(self.rng.uniform(w * 0.18, w * 0.82))
            y = float(self.rng.uniform(h * 0.22, h * 0.66))
            self.parts.emit(x, y, max(4, n // spots), self.rng,
                            (188, 206, 214), speed=190.0, life=1.6,
                            gravity=110.0, size=2.0)
    def step(self, dt: float) -> None:
        self.parts.step(dt)
        if self.rings:
            live = []
            for r in self.rings:
                r.life -= dt
                if r.life > 0:
                    live.append(r)
            self.rings = live
    @property
    def busy(self) -> bool:
        return self.parts.n > 0 or bool(self.rings)
