import numpy as np
import cv2

import draw
import scenes
from session import Session
from ui import Zones

W, H = 900, 600
sc = scenes.render_all(W, H)
s = Session(seed=3)
z = Zones()
c = np.zeros((H, W, 3), np.uint8)

lum = []
for k in range(700):
    s.advance(0.25, clamp=False)
    draw.draw_session(c, s, sc, z)
    lum.append(float(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean()))

lum = np.array(lum)
jump = float(np.abs(np.diff(lum)).max())
print(f"luma  min {lum.min():.1f}  max {lum.max():.1f}  mean {lum.mean():.1f}")
print(f"biggest frame-to-frame jump  {jump:.3f}")

real = draw._drift
draw._drift = lambda *a: None
s2 = Session(seed=3)
c2 = np.zeros((H, W, 3), np.uint8)
base = []
for k in range(700):
    s2.advance(0.25, clamp=False)
    draw.draw_session(c2, s2, sc, z)
    base.append(float(cv2.cvtColor(c2, cv2.COLOR_BGR2GRAY).mean()))
draw._drift = real
base = np.array(base)
base_jump = float(np.abs(np.diff(base)).max())
print(f"glow adds  {lum.mean() - base.mean():+.1f} luma")
print(f"same scene without glow jumps  {base_jump:.3f}")
jump = max(0.0, jump - base_jump)
print(f"glow's own jump  {jump:.3f}")

fails = []
if lum.max() > 120.0:
    fails.append(f"too bright: {lum.max():.1f}")
if jump > 1.0:
    fails.append(f"flicker: {jump:.3f} per frame")
print("glow is calm" if not fails else "FAILED " + ", ".join(fails))
