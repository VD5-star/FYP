from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
import numpy as np
from pieces import INTERLOCK, Cut, Piece, cut_image, make_cut
SNAP_RATIO = 0.34
SNAP_MIN = 14.0
SETTLE_SECONDS = 0.22
def snap_radius(cut: Cut) -> float:
    return max(SNAP_MIN, min(cut.cell_w, cut.cell_h) * SNAP_RATIO)
@dataclass
class Placement:
    piece: Piece
    at: float
@dataclass
class Board:
    cut: Cut
    pieces: list[Piece]
    origin: tuple[int, int]
    tray: tuple[int, int, int, int]
    order: list[int] = field(default_factory=list)
    held: int | None = None
    grab: tuple[int, int] = (0, 0)
    settled: list[Placement] = field(default_factory=list)
    moves: int = 0
    last_move: float = 0.0
    gaps: list[float] = field(default_factory=list)
    @property
    def radius(self) -> float:
        return snap_radius(self.cut)
    @property
    def total(self) -> int:
        return len(self.pieces)
    @property
    def done(self) -> int:
        return sum(1 for p in self.pieces if p.placed)
    @property
    def solved(self) -> bool:
        return self.done == self.total
    def target(self, piece: Piece) -> tuple[int, int]:
        return (self.origin[0] + piece.home[0], self.origin[1] + piece.home[1])
    def scatter(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        tx, ty, tw, th = self.tray
        for p in self.pieces:
            x = rng.randint(tx, max(tx, tx + tw - p.w))
            y = rng.randint(ty, max(ty, ty + th - p.h))
            p.pos = [x, y]
            p.placed = False
        self.order = list(range(len(self.pieces)))
        rng.shuffle(self.order)
    def reshuffle(self, seed: int | None = None) -> int:
        rng = random.Random(seed)
        tx, ty, tw, th = self.tray
        moved = 0
        self.held = None
        for i, p in enumerate(self.pieces):
            if p.placed:
                continue
            p.pos = [rng.randint(tx, max(tx, tx + tw - p.w)),
                     rng.randint(ty, max(ty, ty + th - p.h))]
            moved += 1
        loose = [i for i in self.order if not self.pieces[i].placed]
        rng.shuffle(loose)
        it = iter(loose)
        self.order = [i if self.pieces[i].placed else next(it)
                      for i in self.order]
        return moved
    def pick(self, x: int, y: int, now: float | None = None) -> bool:
        for idx in reversed(self.order):
            p = self.pieces[idx]
            if p.placed:
                continue
            if p.hits(x, y):
                self.held = idx
                self.grab = (x - p.pos[0], y - p.pos[1])
                self.order.remove(idx)
                self.order.append(idx)
                return True
        return False
    def drag(self, x: int, y: int) -> None:
        if self.held is None:
            return
        p = self.pieces[self.held]
        p.pos[0] = x - self.grab[0]
        p.pos[1] = y - self.grab[1]
    def drop(self, now: float | None = None) -> bool:
        if self.held is None:
            return False
        now = time.monotonic() if now is None else now
        idx = self.held
        p = self.pieces[idx]
        self.held = None
        self.moves += 1
        if self.last_move:
            self.gaps.append(now - self.last_move)
        self.last_move = now
        tgt = self.target(p)
        dist = float(np.hypot(p.pos[0] - tgt[0], p.pos[1] - tgt[1]))
        if dist <= self.radius:
            p.pos = [tgt[0], tgt[1]]
            p.placed = True
            self.settled.append(Placement(piece=p, at=now))
            if idx in self.order:
                self.order.remove(idx)
            return True
        return False
    def settle_age(self, piece: Piece, now: float) -> float:
        for s in self.settled:
            if s.piece is piece:
                return now - s.at
        return 999.0
def build(image: np.ndarray, rows: int, cols: int, origin: tuple[int, int],
          tray: tuple[int, int, int, int], style: str = INTERLOCK,
          seed: int | None = None) -> Board:
    h, w = image.shape[:2]
    cut = make_cut(rows, cols, w, h, style=style, seed=seed)
    ps = cut_image(image, cut)
    b = Board(cut=cut, pieces=ps, origin=origin, tray=tray)
    b.scatter(seed=seed)
    return b
