import time
import cv2
import numpy as np
import art
W, H = 1200, 800
def tiles(img, n):
    h, w = img.shape[:2]
    th, tw = h // n, w // n
    out = []
    for r in range(n):
        for c in range(n):
            out.append(img[r * th:(r + 1) * th, c * tw:(c + 1) * tw])
    return out
def descriptor(t):
    small = cv2.resize(t, (8, 8), interpolation=cv2.INTER_AREA)
    v = small.astype(np.float32).ravel()
    return v - v.mean()
print(f"{'name':8} {'ms':>6} {'std':>6} {'edges%':>7} {'flat8':>6} "
      f"{'flat10':>7} {'nearest':>8} {'dark%':>6}")
print("-" * 62)
worst = []
for name in art.NAMES:
    t0 = time.perf_counter()
    img = art.render(name, W, H)
    ms = (time.perf_counter() - t0) * 1000
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = float(g.std())
    edges = cv2.Canny(g, 60, 150)
    edge_pct = float((edges > 0).mean() * 100)
    dark = float((g < 26).mean() * 100)
    flats = {}
    for n in (8, 10):
        ts = tiles(img, n)
        stds = np.array([float(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).std())
                         for t in ts])
        flats[n] = int((stds < 6.0).sum())
    ts = tiles(img, 8)
    d = np.stack([descriptor(t) for t in ts])
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
    sim = d @ d.T
    np.fill_diagonal(sim, -1)
    nearest = float(sim.max())
    print(f"{name:8} {ms:6.0f} {std:6.1f} {edge_pct:7.2f} "
          f"{flats[8]:6} {flats[10]:7} {nearest:8.3f} {dark:6.1f}")
    if flats[10] > 0 or std < 18 or nearest > 0.995:
        worst.append(name)
print()
print("pictures with flat or near-duplicate tiles:",
      worst if worst else "none")
print()
print("== render scales to the sizes the game uses ==")
for w, h in ((1200, 800), (800, 530), (400, 260)):
    t0 = time.perf_counter()
    for n in art.NAMES:
        art.render(n, w, h)
    ms = (time.perf_counter() - t0) * 1000 / len(art.NAMES)
    print(f"  {w}x{h}: {ms:.0f}ms per picture")
