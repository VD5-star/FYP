from __future__ import annotations
import random
from dataclasses import dataclass, field
import cv2
import numpy as np
SQUARE = "square"
INTERLOCK = "interlock"
TAB_RATIO = 0.22
CURVE_STEPS = 22
SS = 4
_TAB_PATH = (
    ((0.00, 0.00), (0.20, 0.00), (0.33, 0.00), (0.40, 0.00)),
    ((0.46, 0.00), (0.28, 0.62), (0.40, 0.78)),
    ((0.52, 0.94), (0.48, 0.94), (0.60, 0.78)),
    ((0.72, 0.62), (0.54, 0.00), (0.60, 0.00)),
    ((0.67, 0.00), (0.80, 0.00), (1.00, 0.00)),
)
def _cubic(p0, c1, c2, p3, steps):
    t = np.linspace(0.0, 1.0, steps, endpoint=False)[:, None]
    p0 = np.array(p0, float)
    c1 = np.array(c1, float)
    c2 = np.array(c2, float)
    p3 = np.array(p3, float)
    u = 1.0 - t
    return (u ** 3 * p0 + 3 * u ** 2 * t * c1
            + 3 * u * t ** 2 * c2 + t ** 3 * p3)
def tab_profile(steps: int = CURVE_STEPS) -> np.ndarray:
    pts = []
    current = np.array(_TAB_PATH[0][0], float)
    for seg in _TAB_PATH:
        if len(seg) == 4:
            p0, c1, c2, p3 = seg
        else:
            c1, c2, p3 = seg
            p0 = current
        pts.append(_cubic(p0, c1, c2, p3, steps))
        current = np.array(p3, float)
    pts.append(np.array([[1.0, 0.0]]))
    return np.vstack(pts)
@dataclass
class Cut:
    rows: int
    cols: int
    width: int
    height: int
    style: str
    horizontal: np.ndarray
    vertical: np.ndarray
    amp: float
    @property
    def cell_w(self) -> float:
        return self.width / self.cols
    @property
    def cell_h(self) -> float:
        return self.height / self.rows
def make_cut(rows: int, cols: int, width: int, height: int,
             style: str = INTERLOCK, seed: int | None = None) -> Cut:
    rng = random.Random(seed)
    horizontal = np.zeros((rows + 1, cols), dtype=np.int8)
    vertical = np.zeros((rows, cols + 1), dtype=np.int8)
    if style == INTERLOCK:
        for r in range(1, rows):
            for c in range(cols):
                horizontal[r, c] = rng.choice((-1, 1))
        for r in range(rows):
            for c in range(1, cols):
                vertical[r, c] = rng.choice((-1, 1))
    amp = min(width / cols, height / rows) * TAB_RATIO
    return Cut(rows=rows, cols=cols, width=width, height=height, style=style,
               horizontal=horizontal, vertical=vertical, amp=amp)
def _h_edge(cut: Cut, r: int, c: int, profile: np.ndarray) -> np.ndarray:
    x0 = c * cut.cell_w
    x1 = (c + 1) * cut.cell_w
    y = r * cut.cell_h
    sign = cut.horizontal[r, c]
    if sign == 0:
        return np.array([[x0, y], [x1, y]])
    xs = x0 + profile[:, 0] * (x1 - x0)
    ys = y + profile[:, 1] * cut.amp * sign
    return np.stack([xs, ys], axis=1)
def _v_edge(cut: Cut, r: int, c: int, profile: np.ndarray) -> np.ndarray:
    y0 = r * cut.cell_h
    y1 = (r + 1) * cut.cell_h
    x = c * cut.cell_w
    sign = cut.vertical[r, c]
    if sign == 0:
        return np.array([[x, y0], [x, y1]])
    ys = y0 + profile[:, 0] * (y1 - y0)
    xs = x + profile[:, 1] * cut.amp * sign
    return np.stack([xs, ys], axis=1)
def piece_outline(cut: Cut, r: int, c: int,
                  profile: np.ndarray | None = None) -> np.ndarray:
    if profile is None:
        profile = tab_profile()
    top = _h_edge(cut, r, c, profile)
    right = _v_edge(cut, r, c + 1, profile)
    bottom = _h_edge(cut, r + 1, c, profile)[::-1]
    left = _v_edge(cut, r, c, profile)[::-1]
    return np.vstack([top, right[1:], bottom[1:], left[1:-1]])
@dataclass
class Piece:
    row: int
    col: int
    sprite: np.ndarray
    alpha: np.ndarray
    home: tuple[int, int]
    pos: list[int] = field(default_factory=list)
    placed: bool = False
    solid: tuple[int, int, int, int] = (0, 0, 0, 0)
    af: np.ndarray | None = None
    core: np.ndarray | None = None
    eidx: tuple | None = None
    ea: np.ndarray | None = None
    ergb: np.ndarray | None = None
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)
    ccore: np.ndarray | None = None
    csprite: np.ndarray | None = None
    ceidx: tuple | None = None
    ccore3: np.ndarray | None = None
    @property
    def w(self) -> int:
        return self.sprite.shape[1]
    @property
    def h(self) -> int:
        return self.sprite.shape[0]
    def hits(self, x: int, y: int) -> bool:
        lx = x - self.pos[0]
        ly = y - self.pos[1]
        if lx < 0 or ly < 0 or lx >= self.w or ly >= self.h:
            return False
        return bool(self.alpha[ly, lx] > 96)
def cut_image(image: np.ndarray, cut: Cut) -> list[Piece]:
    h, w = image.shape[:2]
    profile = tab_profile()
    pad = int(np.ceil(cut.amp)) + 2
    canvas_w = w + 2 * pad
    canvas_h = h + 2 * pad
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                                cv2.BORDER_REPLICATE)
    out = []
    for r in range(cut.rows):
        for c in range(cut.cols):
            poly = piece_outline(cut, r, c, profile) + pad
            ip = np.round(poly).astype(np.int32)
            x0 = max(0, int(ip[:, 0].min()))
            y0 = max(0, int(ip[:, 1].min()))
            x1 = min(canvas_w, int(ip[:, 0].max()) + 1)
            y1 = min(canvas_h, int(ip[:, 1].max()) + 1)
            hi = np.zeros(((y1 - y0) * SS, (x1 - x0) * SS), np.uint8)
            fine = np.round((poly - [x0, y0]) * SS).astype(np.int32)
            cv2.fillPoly(hi, [fine], 255, cv2.LINE_8)
            mask = cv2.resize(hi, (x1 - x0, y1 - y0),
                              interpolation=cv2.INTER_AREA)
            sprite = padded[y0:y1, x0:x1].copy()
            ys, xs = np.nonzero(mask)
            solid = (int(xs.min()), int(ys.min()),
                     int(xs.max()) + 1, int(ys.max()) + 1)
            af = (mask.astype(np.float32) / 255.0)[:, :, None]
            core = mask >= 250
            edge = (mask > 0) & ~core
            ei = np.nonzero(edge)
            cy0, cy1 = solid[1], solid[3]
            cx0, cx1 = solid[0], solid[2]
            out.append(Piece(
                row=r, col=c, sprite=sprite, alpha=mask,
                home=(x0 - pad, y0 - pad), pos=[x0 - pad, y0 - pad],
                solid=solid, af=af, core=core, eidx=ei,
                ea=af[ei[0], ei[1], :],
                ergb=sprite[ei[0], ei[1], :].astype(np.float32),
                crop=(cx0, cy0, cx1, cy1),
                ccore=np.ascontiguousarray(core[cy0:cy1, cx0:cx1]),
                csprite=np.ascontiguousarray(sprite[cy0:cy1, cx0:cx1]),
                ceidx=(np.ascontiguousarray(ei[0] - cy0),
                       np.ascontiguousarray(ei[1] - cx0)),
                ccore3=np.ascontiguousarray(
                    np.repeat(core[cy0:cy1, cx0:cx1, None], 3, axis=2))))
    return out
def coverage(cut: Cut) -> np.ndarray:
    profile = tab_profile()
    count = np.zeros((cut.height, cut.width), np.int32)
    for r in range(cut.rows):
        for c in range(cut.cols):
            poly = np.round(piece_outline(cut, r, c, profile)).astype(np.int32)
            layer = np.zeros((cut.height, cut.width), np.uint8)
            cv2.fillPoly(layer, [poly], 1, cv2.LINE_8)
            count += layer
    return count
