from __future__ import annotations
import cv2
import numpy as np
SIZE = (1200, 800)
def _bgr(rgb: np.ndarray) -> np.ndarray:
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out[:, :, ::-1])
def _smooth(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)
def _noise(w: int, h: int, seed: int, cx: int, cy: int | None = None):
    cx = max(2, int(cx))
    if cy is None:
        cy = max(2, int(round(cx * h / w)))
    cy = max(2, int(cy))
    rng = np.random.default_rng(seed)
    g = rng.random((cy + 1, cx + 1)).astype(np.float32)
    return cv2.resize(g[:-1, :-1] if False else g, (w + 1, h + 1),
                      interpolation=cv2.INTER_CUBIC)[:h, :w]
FBM_WORK = 460
def _fbm(w: int, h: int, seed: int, octaves: int = 5, cx: float = 4.0,
         stretch: float = 1.0) -> np.ndarray:
    top = cx * (2 ** (octaves - 1))
    need = int(min(w, max(64, top * 3.0)))
    ww = min(w, max(FBM_WORK, need)) if w > FBM_WORK else w
    hh = max(2, int(round(h * ww / w)))
    out = np.zeros((hh, ww), np.float32)
    amp = 1.0
    tot = 0.0
    c = cx
    for i in range(octaves):
        cy = max(2, int(round(c * hh / ww / stretch)))
        out += _noise(ww, hh, seed + i * 101, int(c), cy) * amp
        tot += amp
        amp *= 0.5
        c *= 2
    out /= tot
    if (hh, ww) != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return out
def _fbm1(w: int, seed: int, octaves: int = 4, cx: float = 4.0) -> np.ndarray:
    out = np.zeros(w)
    amp = 1.0
    tot = 0.0
    c = cx
    for i in range(octaves):
        cc = max(2, int(c))
        rng = np.random.default_rng(seed + i * 71)
        g = rng.random(cc + 1)
        xs = np.linspace(0, cc, w, endpoint=False)
        x0 = np.floor(xs).astype(int)
        fx = xs - x0
        fx = fx * fx * (3 - 2 * fx)
        out += (g[x0] * (1 - fx) + g[x0 + 1] * fx) * amp
        tot += amp
        amp *= 0.5
        c *= 2
    return out / tot
def _grain(img: np.ndarray, seed: int, amount: float) -> None:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    img += (rng.random((h, w, 1)) - 0.5) * amount
def _detail(img: np.ndarray, seed: int, amount: float = 17.0,
            cx: float = 9.0, tint: tuple = (1.0, 1.0, 1.0)) -> None:
    h, w = img.shape[:2]
    d = _fbm(w, h, seed, 6, cx) - 0.5
    img += d[:, :, None] * (np.array(tint, float) * amount)[None, None, :]
def _specks(img: np.ndarray, seed: int, count: int, lo: float, hi: float,
            tint: tuple, size: tuple = (1, 3)) -> None:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    n = int(count * (w * h) / (SIZE[0] * SIZE[1])) + 20
    xs = rng.integers(0, w, n)
    ys = rng.integers(0, h, n)
    mag = rng.uniform(lo, hi, n)
    rad = rng.integers(size[0], size[1] + 1, n)
    col = np.array(tint, float)
    for x, y, m, r in zip(xs, ys, mag, rad):
        if r <= 1:
            img[y, x] += col * m
        else:
            cv2.circle(img, (int(x), int(y)), int(r),
                       tuple(float(v) for v in col * m), -1, cv2.LINE_AA)
def dusk(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, h)[:, None]
    xx = np.linspace(0.0, 1.0, w)[None, :]
    horizon = 0.60
    t = np.clip(yy / horizon, 0, 1)[:, :, None] if False else \
        np.clip(yy / horizon, 0, 1)
    top = np.array([38, 44, 92], float)
    low = np.array([244, 158, 92], float)
    img = top[None, None, :] * (1 - t[:, :, None]) \
        + low[None, None, :] * t[:, :, None]
    img = np.repeat(img, w, axis=1) if img.shape[1] == 1 else img
    sx, sy = 0.68, 0.545
    d2 = ((xx - sx) * 1.9) ** 2 + ((yy - sy) * 2.9) ** 2
    img += np.exp(-d2 * 9.0)[:, :, None] * np.array([120, 74, 20])
    disc = _smooth(0.0032, 0.0016, d2)
    img += disc[:, :, None] * np.array([90, 70, 40])
    lit = np.exp(-(((xx - sx) * 1.5) ** 2 + ((yy - sy) * 2.0) ** 2) * 3.0)
    for i, (cs, st, thr, op) in enumerate(((5.0, 3.4, 0.50, 0.85),
                                           (11.0, 5.0, 0.55, 0.55),
                                           (21.0, 7.0, 0.58, 0.38))):
        cloud = _fbm(w, h, 21 + i * 37, 5, cs, stretch=st)
        band = _smooth(thr, thr + 0.24, cloud) * _smooth(0.70, 0.30, yy)
        ccol = (np.array([88, 64, 78]) + i * np.array([18, 10, 6])
                + lit[:, :, None] * np.array([150, 90, 30]))
        a = (band * op)[:, :, None]
        img = img * (1 - a) + ccol * a
    streak = _fbm(w, h, 900, 5, 40.0, stretch=9.0)
    img += ((streak - 0.5) * _smooth(0.72, 0.18, yy))[:, :, None] \
        * np.array([34, 24, 20])
    img += (_fbm(w, h, 910, 6, 12.0)[:, :, None] - 0.5) \
        * np.array([20, 16, 18])
    ridges = ((0.615, _fbm1(w, 5, 4, 3.0), 0.050, (96, 92, 124)),
              (0.685, _fbm1(w, 9, 4, 5.0), 0.070, (62, 58, 88)),
              (0.790, _fbm1(w, 13, 5, 7.0), 0.095, (32, 30, 50)))
    rock = _fbm(w, h, 44, 6, 14.0)
    for base, prof, amp, col in ridges:
        line = base + (prof - 0.5) * amp
        below = yy >= line[None, :]
        depth = np.clip((yy - line[None, :]) * 5.0, 0, 1)
        c = np.array(col, float)[None, None, :]
        shade = c * (1.0 - depth[:, :, None] * 0.35)
        shade = shade + (rock[:, :, None] - 0.5) * 46.0
        img[below] = shade[below]
    _detail(img, 88, 15.0, 10.0, (1.0, 0.92, 0.86))
    _specks(img, 66, 150, 0.30, 0.95, (150, 130, 160), (1, 2))
    _grain(img, 77, 11.0)
    return _bgr(img)
def water(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, h)[:, None]
    xx = np.linspace(0.0, 1.0, w)[None, :]
    top = np.array([16, 62, 84], float)
    bot = np.array([44, 128, 140], float)
    img = top[None, None, :] * (1 - yy[:, :, None]) \
        + bot[None, None, :] * yy[:, :, None]
    img = np.repeat(img, w, axis=1)
    warp = _fbm(w, h, 33, 4, 6.0) - 0.5
    phase = (xx * 30.0 + np.sin(yy * 14.0) * 2.4 + warp * 5.0
             + yy * 6.0) * np.pi
    ripple = np.sin(phase) * 0.5 + 0.5
    crest = _smooth(0.62, 0.99, ripple) * (0.35 + yy * 0.65)
    img += crest[:, :, None] * np.array([120, 150, 130])
    trough = _smooth(0.42, 0.02, ripple)
    img -= trough[:, :, None] * np.array([12, 34, 38])
    path = np.exp(-(((xx - 0.52) * 3.1) ** 2) * 2.2)
    glit = _fbm(w, h, 51, 6, 22.0)
    spark = _smooth(0.78, 0.94, glit) * path * (0.25 + yy * 0.75)
    img += spark[:, :, None] * np.array([180, 190, 150])
    _grain(img, 91, 9.0)
    return _bgr(img)
def leaves(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    base = _fbm(w, h, 7, 5, 3.0)
    img = np.array([26, 54, 28], float)[None, None, :] \
        + base[:, :, None] * np.array([34, 58, 26])
    img = np.repeat(np.repeat(img, h // img.shape[0] if img.shape[0] == 1
                              else 1, axis=0), 1, axis=1)
    canvas = np.ascontiguousarray(img)
    rng = np.random.default_rng(19)
    n = int(210 * (w * h) / (SIZE[0] * SIZE[1])) + 60
    for _ in range(n):
        cx = int(rng.uniform(-0.05, 1.05) * w)
        cy = int(rng.uniform(-0.05, 1.05) * h)
        ln = rng.uniform(0.045, 0.115) * w
        ax = int(max(3, ln))
        ay = int(max(2, ln * rng.uniform(0.34, 0.52)))
        ang = rng.uniform(0, 180)
        lift = rng.uniform(0.0, 1.0)
        col = (np.array([44, 96, 38], float)
               + lift * np.array([92, 96, 40], float)
               + rng.uniform(-14, 14))
        cv2.ellipse(canvas, (cx, cy), (ax, ay), ang, 0, 360,
                    tuple(float(v) for v in col), -1, cv2.LINE_AA)
        r = np.deg2rad(ang)
        dx, dy = np.cos(r) * ax, np.sin(r) * ay * 0.0 + np.sin(r) * ax
        vein = col * 0.78
        cv2.line(canvas, (int(cx - dx), int(cy - dy)),
                 (int(cx + dx), int(cy + dy)),
                 tuple(float(v) for v in vein), 1, cv2.LINE_AA)
    sun = np.exp(-((((np.linspace(0, 1, w)[None, :] - 0.32) * 2.0) ** 2)
                   + (((np.linspace(0, 1, h)[:, None] - 0.22) * 2.0) ** 2))
                 * 2.4)
    canvas += sun[:, :, None] * np.array([70, 74, 30])
    _detail(canvas, 27, 22.0, 14.0, (0.8, 1.0, 0.7))
    _specks(canvas, 31, 240, 0.30, 0.95, (90, 130, 70), (1, 2))
    _grain(canvas, 23, 12.0)
    return _bgr(canvas)
def sand(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, h)[:, None]
    xx = np.linspace(0.0, 1.0, w)[None, :]
    img = np.repeat(np.array([226, 196, 148], float)[None, None, :]
                    * (1 - yy[:, :, None] * 0.28), w, axis=1)
    dune = _fbm(w, h, 61, 4, 3.0, stretch=2.2)
    slope = np.gradient(dune, axis=0)
    img += np.clip(slope * 260.0, -60, 60)[:, :, None] \
        * np.array([1.0, 0.86, 0.62])
    ripple = np.sin((xx * 46.0 + dune * 9.0 + yy * 3.0) * np.pi)
    depth = _smooth(0.10, 0.95, yy)
    img += (ripple * depth * 13.0)[:, :, None] * np.array([1.0, 0.88, 0.70])
    shade = _smooth(0.45, 0.85, dune)
    img -= shade[:, :, None] * np.array([26, 30, 34])
    grit = _fbm(w, h, 84, 6, 30.0)
    img += (grit[:, :, None] - 0.5) * np.array([40, 34, 26])
    rng = np.random.default_rng(87)
    n = int(90 * (w * h) / (SIZE[0] * SIZE[1])) + 30
    for _ in range(n):
        cx = int(rng.uniform(0, 1) * w)
        cy = int(rng.uniform(0, 1) * h)
        r = int(rng.uniform(0.004, 0.016) * w)
        tone = rng.uniform(-46, 34)
        cv2.circle(img, (cx, cy), max(1, r),
                   tuple(float(v) for v in np.array([tone, tone * 0.9,
                                                     tone * 0.74])),
                   -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy + max(1, r // 2)),
                    (max(2, int(r * 1.5)), max(1, int(r * 0.5))),
                    rng.uniform(0, 180), 0, 360, (-18.0, -16.0, -12.0),
                    -1, cv2.LINE_AA)
    _detail(img, 85, 16.0, 13.0, (1.0, 0.90, 0.72))
    _specks(img, 86, 260, 0.35, 1.0, (60, 48, 34), (1, 2))
    _grain(img, 83, 20.0)
    return _bgr(img)
def stones(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    bed = _fbm(w, h, 29, 4, 6.0)
    img = np.ascontiguousarray(
        np.array([74, 70, 66], float)[None, None, :]
        + bed[:, :, None] * np.array([30, 28, 26]))
    img = np.ascontiguousarray(np.broadcast_to(img, (h, w, 3)).copy())
    rng = np.random.default_rng(43)
    spots = []
    tries = 0
    want = int(120 * (w * h) / (SIZE[0] * SIZE[1])) + 40
    while len(spots) < want and tries < want * 40:
        tries += 1
        r = rng.uniform(0.028, 0.072) * w
        cx = rng.uniform(-0.02, 1.02) * w
        cy = rng.uniform(-0.02, 1.02) * h
        clash = False
        for ox, oy, orr in spots:
            if (cx - ox) ** 2 + (cy - oy) ** 2 < (r + orr) ** 2 * 0.62:
                clash = True
                break
        if not clash:
            spots.append((cx, cy, r))
    for cx, cy, r in spots:
        ang = rng.uniform(0, 180)
        ax = int(max(3, r))
        ay = int(max(3, r * rng.uniform(0.68, 0.94)))
        tone = rng.uniform(58, 168)
        tint = np.array([tone, tone * rng.uniform(0.94, 1.03),
                         tone * rng.uniform(0.90, 1.02)])
        cv2.ellipse(img, (int(cx), int(cy)), (ax, ay), ang, 0, 360,
                    tuple(float(v) for v in tint), -1, cv2.LINE_AA)
        cv2.ellipse(img, (int(cx), int(cy)), (ax, ay), ang, 0, 360,
                    tuple(float(v) for v in tint * 0.55), 2, cv2.LINE_AA)
        hx = int(cx - ax * 0.30)
        hy = int(cy - ay * 0.34)
        cv2.ellipse(img, (hx, hy), (max(2, int(ax * 0.46)),
                                    max(2, int(ay * 0.40))), ang, 0, 360,
                    tuple(float(v) for v in np.clip(tint * 1.28, 0, 255)),
                    -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), max(0.6, w / 900.0))
    _grain(img, 37, 10.0)
    return _bgr(img)
def petals(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    base = _fbm(w, h, 15, 5, 4.0)
    img = np.ascontiguousarray(
        np.broadcast_to(np.array([96, 112, 86], float)[None, None, :]
                        + base[:, :, None] * np.array([48, 52, 40]),
                        (h, w, 3)).copy())
    rng = np.random.default_rng(67)
    n = int(26 * (w * h) / (SIZE[0] * SIZE[1])) + 12
    for _ in range(n):
        cx = rng.uniform(-0.04, 1.04) * w
        cy = rng.uniform(-0.04, 1.04) * h
        r = rng.uniform(0.045, 0.098) * w
        hue = rng.uniform(0, 1)
        col = (np.array([236, 172, 196], float) * (1 - hue)
               + np.array([196, 150, 232], float) * hue)
        petal_n = rng.integers(5, 8)
        off = rng.uniform(0, 360)
        for k in range(int(petal_n)):
            a = off + 360.0 * k / petal_n
            ra = np.deg2rad(a)
            px = cx + np.cos(ra) * r * 0.56
            py = cy + np.sin(ra) * r * 0.56
            shade = col * rng.uniform(0.82, 1.06)
            cv2.ellipse(img, (int(px), int(py)),
                        (max(2, int(r * 0.52)), max(2, int(r * 0.30))),
                        a, 0, 360, tuple(float(v) for v in shade),
                        -1, cv2.LINE_AA)
        cv2.circle(img, (int(cx), int(cy)), max(2, int(r * 0.24)),
                   (250.0, 206.0, 96.0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(cx), int(cy)), max(2, int(r * 0.24)),
                   (214.0, 162.0, 64.0), 1, cv2.LINE_AA)
    rng2 = np.random.default_rng(68)
    nb = int(70 * (w * h) / (SIZE[0] * SIZE[1])) + 24
    for _ in range(nb):
        cx = int(rng2.uniform(0, 1) * w)
        cy = int(rng2.uniform(0, 1) * h)
        ln = rng2.uniform(0.02, 0.07) * w
        a = np.deg2rad(rng2.uniform(0, 360))
        col = np.array([58, 92, 52], float) * rng2.uniform(0.7, 1.3)
        cv2.line(img, (cx, cy),
                 (int(cx + np.cos(a) * ln), int(cy + np.sin(a) * ln)),
                 tuple(float(v) for v in col), 1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (max(2, int(ln * 0.22)),
                                    max(1, int(ln * 0.10))),
                    np.rad2deg(a), 0, 360,
                    tuple(float(v) for v in col * 1.2), -1, cv2.LINE_AA)
    _detail(img, 62, 20.0, 12.0, (0.9, 1.0, 0.9))
    _specks(img, 64, 200, 0.30, 0.9, (120, 140, 110), (1, 2))
    _grain(img, 59, 12.0)
    return _bgr(img)
def aurora(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, h)[:, None]
    img = np.repeat(np.array([26, 28, 52], float)[None, None, :]
                    + yy[:, :, None] * np.array([26, 30, 34]), w, axis=1)
    rng = np.random.default_rng(101)
    n_stars = int(320 * (w * h) / (SIZE[0] * SIZE[1])) + 80
    sx = rng.integers(0, w, n_stars)
    sy = (rng.random(n_stars) ** 1.7 * h).astype(int)
    bright = rng.uniform(70, 230, n_stars)[:, None]
    img[sy, sx] += bright * np.array([1.0, 1.0, 0.95])
    for i, (cx, amp, wid, col) in enumerate((
            (0.30, 0.30, 0.16, (60, 224, 150)),
            (0.54, 0.24, 0.12, (110, 236, 180)),
            (0.74, 0.34, 0.20, (96, 150, 232)))):
        drift = (_fbm1(h, 200 + i * 17, 4, 3.0) - 0.5) * amp
        centre = cx + drift[:, None]
        xs = np.linspace(0.0, 1.0, w)[None, :]
        band = np.exp(-(((xs - centre) / wid) ** 2) * 2.6)
        fade = _smooth(0.86, 0.10, yy)
        streak = 0.55 + 0.45 * _fbm(w, h, 300 + i * 23, 4, 26.0, stretch=6.0)
        img += (band * fade * streak)[:, :, None] * np.array(col, float)
    img = cv2.GaussianBlur(img, (0, 0), max(0.5, w / 1400.0))
    dust = _fbm(w, h, 117, 6, 16.0)
    img += (dust[:, :, None] - 0.5) * np.array([34, 38, 50])
    cloud = _fbm(w, h, 121, 5, 7.0)
    veil = _smooth(0.52, 0.88, cloud) * float(0.62)
    img += veil[:, :, None] * np.array([34, 52, 70])
    band2 = _fbm(w, h, 123, 5, 30.0, stretch=8.0)
    img += ((band2 - 0.5) * _smooth(0.88, 0.16, yy))[:, :, None] \
        * np.array([26, 40, 30])
    milky = np.exp(-(((np.linspace(0, 1, w)[None, :] * 1.1
                       - np.linspace(0, 1, h)[:, None] * 0.7 - 0.18)
                      / 0.16) ** 2) * 2.0)
    grains = _fbm(w, h, 127, 6, 34.0)
    img += (milky * _smooth(0.80, 0.22, yy)
            * (0.45 + grains * 0.55))[:, :, None] * np.array([36, 34, 44])
    horizon = _smooth(0.72, 1.0, yy)
    img += horizon[:, :, None] * np.array([24, 30, 22])
    ground = _fbm1(w, 133, 4, 4.0)
    line = 0.90 + (ground - 0.5) * 0.05
    below = yy >= line[None, :]
    tree = _fbm(w, h, 137, 6, 40.0)
    dark = np.broadcast_to(np.array([24, 30, 34], float), img.shape).copy()
    dark += (tree[:, :, None] - 0.5) * 40.0
    img[below] = dark[below]
    _specks(img, 141, 200, 0.25, 0.8, (90, 120, 150), (1, 2))
    _grain(img, 113, 9.0)
    return _bgr(img)
def mosaic(w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    img = np.ascontiguousarray(
        np.broadcast_to(np.array([38, 36, 40], float)[None, None, :],
                        (h, w, 3)).copy())
    cols = 26
    rows = max(6, int(round(cols * h / w)))
    cw = w / cols
    ch = h / rows
    field = _fbm(w, h, 131, 4, 3.0)
    rng = np.random.default_rng(73)
    palette = (np.array([206, 92, 74], float), np.array([236, 176, 78], float),
               np.array([76, 148, 166], float), np.array([54, 96, 138], float),
               np.array([214, 206, 182], float),
               np.array([132, 84, 146], float))
    for r in range(rows):
        for c in range(cols):
            x0 = int(c * cw) + 1
            y0 = int(r * ch) + 1
            x1 = int((c + 1) * cw) - 1
            y1 = int((r + 1) * ch) - 1
            if x1 <= x0 or y1 <= y0:
                continue
            f = float(field[min(h - 1, int((r + 0.5) * ch)),
                            min(w - 1, int((c + 0.5) * cw))])
            k = int(np.clip(f * len(palette), 0, len(palette) - 1))
            col = palette[k] * rng.uniform(0.82, 1.14)
            cv2.rectangle(img, (x0, y0), (x1, y1),
                          tuple(float(v) for v in col), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1),
                          tuple(float(v) for v in col * 0.74), 1)
            if rng.random() < 0.30:
                cv2.circle(img, ((x0 + x1) // 2, (y0 + y1) // 2),
                           max(1, int(min(cw, ch) * 0.18)),
                           tuple(float(v) for v in col * 1.22), -1,
                           cv2.LINE_AA)
    _detail(img, 151, 12.0, 8.0)
    _grain(img, 149, 12.0)
    return _bgr(img)
MIN_TILE_STD = 7.0
def _ensure_texture(img: np.ndarray, seed: int, target: float = MIN_TILE_STD,
                    grid: int = 12, rounds: int = 6) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    scale = min(1.0, 420.0 / max(w, h))
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
    ks = (max(3, sw // grid), max(3, sh // grid))
    for k in range(rounds):
        g = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        if (sw, sh) != (w, h):
            g = cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA)
        mean = cv2.blur(g, ks)
        sq = cv2.blur(g * g, ks)
        local = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        need = np.clip((target * 1.9 - local) / (target * 1.9), 0.0, 1.0)
        if float(need.max()) <= 0.01:
            break
        if (sw, sh) != (w, h):
            need = cv2.resize(need, (w, h), interpolation=cv2.INTER_LINEAR)
        d = (_fbm(w, h, seed + k * 211, 6, 16.0 + k * 9.0) - 0.5)
        out += (d * need)[:, :, None] * (34.0 + k * 10.0)
    return np.clip(out, 0, 255).astype(np.uint8)
GALLERY = {
    "dusk": dusk,
    "water": water,
    "leaves": leaves,
    "sand": sand,
    "stones": stones,
    "petals": petals,
    "aurora": aurora,
    "mosaic": mosaic,
}
NAMES = list(GALLERY)
_SEEDS = {n: 1000 + i * 137 for i, n in enumerate(NAMES)}
ART_MAX = 760
def render(name: str, w: int = SIZE[0], h: int = SIZE[1]) -> np.ndarray:
    rw, rh = w, h
    if w > ART_MAX:
        rw = ART_MAX
        rh = max(2, int(round(h * ART_MAX / w)))
    img = GALLERY[name](rw, rh)
    if (rw, rh) != (w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return _ensure_texture(img, _SEEDS[name])
