from __future__ import annotations
import cv2
import numpy as np
SIZE = (1100, 700)
MAX_LUMA = 96.0
def _bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.clip(rgb, 0, 255).astype(np.uint8)[:, :, ::-1])
def _noise(w: int, h: int, seed: int, cells: int) -> np.ndarray:
    cells = max(2, int(cells))
    cy = max(2, int(round(cells * h / w)))
    rng = np.random.default_rng(seed)
    g = rng.random((cy + 1, cells + 1)).astype(np.float32)
    return cv2.resize(g, (w + 1, h + 1),
                      interpolation=cv2.INTER_CUBIC)[:h, :w]
def _fbm(w: int, h: int, seed: int, octaves: int = 5,
         cells: float = 4.0) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, tot, c = 1.0, 0.0, cells
    for i in range(octaves):
        out += _noise(w, h, seed + i * 97, int(c)) * amp
        tot += amp
        amp *= 0.5
        c *= 2
    return out / tot
def _vgrad(w: int, h: int, top, bottom) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    a = np.array(top, np.float32)[None, None, :]
    b = np.array(bottom, np.float32)[None, None, :]
    return np.repeat(a * (1 - t) + b * t, w, axis=1)
def _vignette(img: np.ndarray, amount: float = 0.30) -> None:
    h, w = img.shape[:2]
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    r = np.sqrt(x * x + y * y) / 1.414
    img *= (1.0 - amount * (r ** 2))[:, :, None]
def rain(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    img = _vgrad(w, h, (38, 46, 56), (22, 28, 36))
    mist = _fbm(w, h, 5, 5, 3.0)
    img += (mist[:, :, None] - 0.5) * np.array([16, 18, 22], np.float32)
    rng = np.random.default_rng(9)
    layer = np.zeros((h, w), np.float32)
    for _ in range(int(w * h / 5200)):
        x0 = int(rng.uniform(0, w))
        y0 = int(rng.uniform(0, h))
        ln = int(rng.uniform(h * 0.04, h * 0.13))
        lean = int(ln * 0.14)
        cv2.line(layer, (x0, y0), (x0 + lean, y0 + ln),
                 float(rng.uniform(0.25, 0.75)), 1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.7)
    img += layer[:, :, None] * np.array([34, 40, 48], np.float32)
    glow = np.exp(-((np.linspace(-1, 1, w)[None, :] * 1.6) ** 2
                    + (np.linspace(-1.4, 0.6, h)[:, None] * 1.3) ** 2) * 1.6)
    img += glow[:, :, None] * np.array([18, 22, 28], np.float32)
    _vignette(img, 0.34)
    return _bgr(img)
def wood(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    img = _vgrad(w, h, (62, 46, 34), (36, 26, 19))
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    warp = _fbm(w, h, 21, 4, 3.0)
    rings = np.sin((yy * 13.0 + warp * 2.6) * np.pi * 2.0)
    img += (rings * 0.5)[:, :, None] * np.array([20, 14, 9], np.float32)
    grain = _fbm(w, h, 23, 5, 40.0)
    img += (grain[:, :, None] - 0.5) * np.array([16, 12, 8], np.float32)
    knot = np.exp(-(((xx - 0.72) * 5.0) ** 2
                    + ((yy - 0.36) * 6.2) ** 2) * 2.0)
    img -= knot[:, :, None] * np.array([26, 20, 14], np.float32)
    lamp = np.exp(-(((xx - 0.30) * 1.5) ** 2
                    + ((yy - 0.24) * 1.8) ** 2) * 1.5)
    img += lamp[:, :, None] * np.array([30, 22, 12], np.float32)
    _vignette(img, 0.36)
    return _bgr(img)
def wind(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    img = _vgrad(w, h, (44, 52, 62), (30, 40, 44))
    streak = _fbm(w, h, 31, 5, 9.0)
    stretch = cv2.resize(streak, (w, max(2, h // 8)),
                         interpolation=cv2.INTER_AREA)
    stretch = cv2.resize(stretch, (w, h), interpolation=cv2.INTER_CUBIC)
    img += (stretch[:, :, None] - 0.5) * np.array([26, 28, 30], np.float32)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    field = _fbm(w, h, 33, 4, 5.0)
    grass = np.clip((yy - 0.62) * 3.2, 0, 1) * (0.4 + field * 0.6)
    img -= grass[:, :, None] * np.array([18, 16, 14], np.float32)
    rng = np.random.default_rng(35)
    blades = np.zeros((h, w), np.float32)
    for _ in range(int(w / 7)):
        x0 = int(rng.uniform(0, w))
        base = int(rng.uniform(h * 0.72, h * 1.02))
        ln = int(rng.uniform(h * 0.08, h * 0.22))
        bend = int(rng.uniform(6, 26))
        pts = np.array([[x0, base],
                        [x0 + bend // 2, base - ln // 2],
                        [x0 + bend, base - ln]], np.int32)
        cv2.polylines(blades, [pts], False,
                      float(rng.uniform(0.2, 0.55)), 1, cv2.LINE_AA)
    img += blades[:, :, None] * np.array([22, 26, 22], np.float32)
    _vignette(img, 0.32)
    return _bgr(img)
def chime(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    img = _vgrad(w, h, (30, 34, 54), (46, 40, 52))
    glow = np.exp(-((np.linspace(-1, 1, w)[None, :] * 1.2) ** 2
                    + (np.linspace(-0.4, 1.8, h)[:, None] * 1.1) ** 2) * 1.4)
    img += glow[:, :, None] * np.array([40, 34, 30], np.float32)
    rng = np.random.default_rng(41)
    spec = np.zeros((h, w), np.float32)
    for _ in range(90):
        x = int(rng.uniform(0, w))
        y = int(rng.uniform(0, h * 0.8))
        r = int(rng.uniform(1, 3))
        cv2.circle(spec, (x, y), r, float(rng.uniform(0.3, 1.0)), -1,
                   cv2.LINE_AA)
    spec = cv2.GaussianBlur(spec, (0, 0), 1.6)
    img += spec[:, :, None] * np.array([46, 44, 50], np.float32)
    haze = _fbm(w, h, 43, 5, 4.0)
    img += (haze[:, :, None] - 0.5) * np.array([14, 13, 16], np.float32)
    _vignette(img, 0.38)
    return _bgr(img)
def hum(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    img = _vgrad(w, h, (26, 30, 38), (34, 30, 30))
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    depth = _fbm(w, h, 51, 5, 2.5)
    img += (depth[:, :, None] - 0.5) * np.array([18, 18, 20], np.float32)
    for cx, cy, rad, amp in ((0.5, 0.52, 0.55, 1.0),
                             (0.5, 0.52, 0.30, 0.6)):
        d = np.sqrt(((xx - cx) * (w / h)) ** 2 + (yy - cy) ** 2)
        ring = np.exp(-((d - rad) ** 2) / 0.010) * amp
        img += ring[:, :, None] * np.array([16, 15, 18], np.float32)
    core = np.exp(-(((xx - 0.5) * (w / h)) ** 2
                    + ((yy - 0.52)) ** 2) * 9.0)
    img += core[:, :, None] * np.array([24, 22, 26], np.float32)
    _vignette(img, 0.40)
    return _bgr(img)
GALLERY = {
    "stream": rain,
    "bowl": chime,
    "breeze": wind,
    "strings": wood,
    "deep": hum,
}
NAMES = list(GALLERY)
def _cap(img: np.ndarray, ceiling: float = MAX_LUMA) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = float(g.mean())
    if m <= ceiling:
        return img
    return np.clip(img.astype(np.float32) * (ceiling / m), 0,
                   255).astype(np.uint8)
def render(name: str, w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    return _cap(GALLERY[name](w, h))
def render_all(w: int = SIZE[0], h: int = SIZE[1]) -> dict:
    return {n: render(n, w, h) for n in NAMES}
