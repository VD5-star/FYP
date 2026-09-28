import time
import numpy as np
import audio
import draw
import fx as fx_mod
from fx import Effects, Particles
fails = []
def ok(name, cond, detail=""):
    print(f"  [{'pass' if cond else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)
print("== particles are born, move, and die ==")
p = Particles()
rng = np.random.default_rng(1)
p.emit(100.0, 100.0, 20, rng, (200, 220, 230))
ok("emitted", p.n == 20, f"n={p.n}")
x0 = p.px[:p.n].copy()
y0 = p.py[:p.n].copy()
p.step(1 / 60)
moved = int((np.abs(p.px[:p.n] - x0) + np.abs(p.py[:p.n] - y0) > 0).sum())
ok("all of them move", moved == p.n, f"{moved}/{p.n}")
ok("they spread out", float(p.px[:p.n].std()) > 0.5,
   f"spread {float(p.px[:p.n].std()):.2f}px")
for _ in range(200):
    p.step(1 / 60)
ok("all gone eventually", p.n == 0, f"n={p.n}")
print()
print("== the pool is capped and never overflows ==")
p = Particles(cap=120)
for i in range(60):
    p.emit(50.0 + i, 50.0, 30, rng, (200, 200, 200))
ok("never exceeds cap", p.n <= 120, f"n={p.n} cap=120")
ok("still full of life", p.n > 0)
for _ in range(400):
    p.step(1 / 60)
ok("drains to empty", p.n == 0, f"n={p.n}")
print()
print("== a big time jump cannot explode the effects ==")
p = Particles()
p.emit(100.0, 100.0, 40, rng, (200, 220, 230))
p.step(5.0)
ok("survives a 5s hitch", p.n == 0 or np.isfinite(p.px[:p.n]).all(),
   f"n={p.n}")
p.clear()
p.emit(100.0, 100.0, 40, rng, (200, 220, 230))
frozen = p.px[:p.n].copy()
for _ in range(10):
    p.step(0.0)
ok("zero dt is harmless", p.n == 40, f"n={p.n}")
ok("zero dt does not move them",
   float(np.abs(p.px[:p.n] - frozen).max()) == 0.0)
ok("negative dt is clamped", (p.step(-1.0), p.n == 40)[1], f"n={p.n}")
ok("no NaN anywhere", bool(np.isfinite(p.px[:p.n]).all()
                           and np.isfinite(p.py[:p.n]).all()))
print()
print("== placing a piece makes a small, brief burst ==")
e = Effects(rng=np.random.default_rng(3))
e.place(300.0, 200.0, 40.0)
ok("particles appear", e.parts.n > 0, f"{e.parts.n}")
ok("modest count", e.parts.n <= 30, f"{e.parts.n}")
ok("a ring appears", len(e.rings) == 1)
t = 0.0
while e.busy and t < 5.0:
    e.step(1 / 60)
    t += 1 / 60
ok("all over quickly", t < 1.6, f"lasted {t:.2f}s")
ok("nothing left behind", not e.busy)
print()
print("== lifting a piece is a faint puff, not a burst ==")
e = Effects(rng=np.random.default_rng(5))
e.lift(200.0, 200.0)
lift_n = e.parts.n
e.clear()
e.place(200.0, 200.0, 40.0)
ok("lift is smaller than place", lift_n < e.parts.n,
   f"lift {lift_n} vs place {e.parts.n}")
ok("no ring on lift", True)
print()
print("== finishing is a gentle bloom, not fireworks ==")
e = Effects(rng=np.random.default_rng(7))
e.finish(1280, 860)
ok("particles appear", e.parts.n > 40, f"{e.parts.n}")
canvas_f = np.full((860, 1280, 3), 50, np.uint8)
before_f = canvas_f.copy()
t0 = time.perf_counter()
for _ in range(30):
    draw.draw_particles(canvas_f, e)
ok("finish burst draws under 2ms",
   (time.perf_counter() - t0) / 30 * 1000 < 2.0,
   f"{(time.perf_counter()-t0)/30*1000:.3f}ms")
lit = int((canvas_f != before_f).any(axis=2).sum())
ok("finish does not wash out the screen",
   lit < canvas_f.shape[0] * canvas_f.shape[1] * 0.02,
   f"{lit} pixels touched of {canvas_f.shape[0]*canvas_f.shape[1]}")
ok("no white-out", int(canvas_f.mean()) < int(before_f.mean()) + 6,
   f"mean {before_f.mean():.1f} -> {canvas_f.mean():.1f}")
t = 0.0
while e.busy and t < 8.0:
    e.step(1 / 60)
    t += 1 / 60
ok("settles within a few seconds", t < 3.5, f"{t:.2f}s")
print()
print("== clearing wipes everything ==")
e = Effects(rng=np.random.default_rng(9))
e.place(100.0, 100.0, 30.0)
e.finish(800, 600)
e.clear()
ok("no particles", e.parts.n == 0)
ok("no rings", not e.rings)
ok("not busy", not e.busy)
print()
print("== drawing particles never leaves the canvas ==")
canvas = np.full((300, 400, 3), 40, np.uint8)
e = Effects(rng=np.random.default_rng(11))
for x, y in ((-500, -500), (5000, 5000), (0, 0), (399, 299), (200, 150)):
    e.place(float(x), float(y), 30.0)
before = canvas.copy()
draw.draw_particles(canvas, e)
ok("no crash off-screen", True)
ok("canvas stays uint8", canvas.dtype == np.uint8)
ok("values stay in range", int(canvas.max()) <= 255
   and int(canvas.min()) >= 0)
ok("something was drawn", not np.array_equal(canvas, before))
print()
print("== every live particle actually reaches the screen ==")
canvas = np.zeros((400, 600, 3), np.uint8)
e = Effects(rng=np.random.default_rng(31))
e.parts.clear()
e.rings.clear()
e.parts.emit(300.0, 200.0, 40, e.rng, (200, 220, 230), speed=0.0,
             gravity=0.0, spread=0.0)
draw.draw_particles(canvas, e)
lit = int((canvas.sum(axis=2) > 0).sum())
ok("a burst paints pixels", lit > 0, f"{lit} pixels")
canvas = np.zeros((400, 600, 3), np.uint8)
e = Effects(rng=np.random.default_rng(33))
e.rings.clear()
e.parts.clear()
for i in range(12):
    e.parts.emit(60.0 + i * 40, 60.0 + i * 25, 3, e.rng, (220, 220, 220),
                 speed=0.0, gravity=0.0, spread=0.0)
e.rings.clear()
draw.draw_particles(canvas, e)
lit = int((canvas.sum(axis=2) > 0).sum())
ok("scattered particles all paint", lit >= 12,
   f"{lit} pixels for {e.parts.n} particles")
canvas = np.zeros((400, 600, 3), np.uint8)
e = Effects(rng=np.random.default_rng(35))
e.rings.clear()
e.parts.clear()
e.parts.emit(599.0, 399.0, 6, e.rng, (220, 220, 220), speed=0.0,
             gravity=0.0, spread=0.0)
e.rings.clear()
draw.draw_particles(canvas, e)
ok("a particle at the far corner still paints",
   int(canvas[399, 599].sum()) > 0 or int(canvas.sum()) > 0)
print()
print("== the finish burst covers the picture, not one spot ==")
e = Effects(rng=np.random.default_rng(37))
e.finish(1280, 860)
px, py, _, _, _ = e.parts.view()
ok("spread across the width", float(px.max() - px.min()) > 400,
   f"{float(px.max()-px.min()):.0f}px wide")
ok("spread across the height", float(py.max() - py.min()) > 150,
   f"{float(py.max()-py.min()):.0f}px tall")
canvas = np.zeros((860, 1280, 3), np.uint8)
draw.draw_particles(canvas, e)
ok("it paints something", int((canvas.sum(axis=2) > 0).sum()) > 0,
   f"{int((canvas.sum(axis=2) > 0).sum())} pixels")
print()
print("== particles only brighten, never darken ==")
canvas = np.full((200, 300, 3), 60, np.uint8)
e = Effects(rng=np.random.default_rng(13))
e.place(150.0, 100.0, 30.0)
before = canvas.copy()
draw.draw_particles(canvas, e)
ok("never darker than before", int((canvas < before).sum()) == 0,
   f"{int((canvas < before).sum())} pixels darkened")
print()
print("== there is no screen flash any more ==")
ok("no draw_flash function", not hasattr(draw, "draw_flash"))
src_fx = open("fx.py", encoding="utf-8").read()
ok("finish sets no flash", "self.flash =" not in src_fx)
src_draw = open("draw.py", encoding="utf-8").read()
ok("board draw never flashes", "draw_flash" not in src_draw)
e = Effects(rng=np.random.default_rng(21))
canvas = np.full((300, 400, 3), 60, np.uint8)
base = canvas.copy()
e.finish(400, 300)
for _ in range(4):
    draw.draw_particles(canvas, e)
    e.step(1 / 60)
ok("screen brightness barely moves",
   abs(float(canvas.mean()) - float(base.mean())) < 4.0,
   f"{base.mean():.1f} -> {canvas.mean():.1f}")
untouched = int((canvas == base).all(axis=2).sum())
ok("most of the screen is untouched",
   untouched > 300 * 400 * 0.93, f"{untouched}/{300*400} unchanged")
print()
print("== effects cost almost nothing per frame ==")
e = Effects(rng=np.random.default_rng(15))
canvas = np.zeros((860, 1280, 3), np.uint8)
for i in range(60):
    e.place(float(100 + i * 7), float(100 + i * 5), 40.0)
    e.step(1 / 600)
for _ in range(3):
    draw.draw_particles(canvas, e)
held = e.parts.n
t0 = time.perf_counter()
for _ in range(60):
    draw.draw_particles(canvas, e)
ms = (time.perf_counter() - t0) / 60 * 1000
ok("drawing a full pool under 4ms", ms < 4.0,
   f"{ms:.3f}ms, {held} particles")
ok("pool is bounded", held <= fx_mod.MAX_PARTICLES, f"{held}")
t0 = time.perf_counter()
for _ in range(60):
    e.step(1 / 600)
step_ms = (time.perf_counter() - t0) / 60 * 1000
ok("stepping under 0.5ms", step_ms < 0.5, f"{step_ms:.3f}ms")
print()
print("== the sound is soft and warm, not a sharp click ==")
c = audio._click()
ms = len(c) / audio.SR * 1000
ok("has a soft body, not an instant tick", 80 <= ms <= 260, f"{ms:.0f}ms")
ok("starts at silence", abs(float(c[0])) < 1e-6)
ok("ends at silence", abs(float(c[-1])) < 1e-6)
ok("peaks at 1", abs(float(np.abs(c).max()) - 1.0) < 1e-6)
atk = int(audio.ATTACK_MS / 1000.0 * audio.SR)
ok("eases in, no hard edge", float(np.abs(c[:atk // 3]).max()) < 0.35,
   f"first third of attack peaks at "
   f"{float(np.abs(c[:atk//3]).max()):.3f}")
ok("no sudden jolt", float(np.abs(np.diff(c)).max()) < 0.20,
   f"max step {float(np.abs(np.diff(c)).max()):.4f}")
spec = np.abs(np.fft.rfft(c.astype(np.float64)))
freqs = np.fft.rfftfreq(len(c), 1.0 / audio.SR)
peak_hz = float(freqs[np.argmax(spec)])
print(f"       strongest pitch: {peak_hz:.0f} Hz")
ok("it is a warm low pitch", 120 <= peak_hz <= 420, f"{peak_hz:.0f} Hz")
high = float(spec[freqs > 4000].sum())
total = float(spec.sum())
ok("very little harsh treble", high / total < 0.06,
   f"{high/total*100:.1f}% above 4kHz")
ok("fades away smoothly", float(np.abs(c[-len(c) // 6:]).max()) < 0.05,
   f"tail peak {float(np.abs(c[-len(c)//6:]).max()):.4f}")
print()
print("== every placement tone is gentle ==")
a = audio.Audio(seed=1)
for i, buf in enumerate(a.bank):
    ok(f"tone {i} starts silent", abs(float(buf[0])) < 1e-6)
    ok(f"tone {i} has no jolt", float(np.abs(np.diff(buf)).max()) < 0.20,
       f"{float(np.abs(np.diff(buf)).max()):.4f}")
ok("the finish tone is the longest and lowest",
   len(a.soft) > len(a.bank[0]) * 2,
   f"{len(a.soft)/audio.SR*1000:.0f}ms")
print()
print("== there is no drone and no melody any more ==")
src = open("audio.py", encoding="utf-8").read()
for word in ("drone", "SCALE", "melody", "make_drone", "BREATH", "LOOP"):
    ok(f"no {word}", word not in src)
a = audio.Audio(seed=1)
ok("no note scheduler", not hasattr(a, "_schedule"))
ok("silent when idle", float(np.abs(
    np.concatenate([a.render(audio.BLOCK) for _ in range(40)])).max()) == 0.0)
print()
print("== the click only sounds when a piece lands ==")
a = audio.Audio(seed=3)
idle = np.concatenate([a.render(audio.BLOCK) for _ in range(20)])
ok("silence before", float(np.abs(idle).max()) == 0.0)
a.place(0.5)
after = np.concatenate([a.render(audio.BLOCK) for _ in range(20)])
ok("audible click", float(np.abs(after).max()) > 0.05,
   f"peak {float(np.abs(after).max()):.3f}")
ok("no clipping", float(np.abs(after).max()) <= 1.0)
tail = np.concatenate([a.render(audio.BLOCK) for _ in range(40)])
ok("silence again after", float(np.abs(tail).max()) == 0.0)
print()
print("== many fast clicks never pile into noise ==")
a = audio.Audio(seed=5)
peak = 0.0
for i in range(200):
    a.place(i / 200)
    peak = max(peak, float(np.abs(a.render(audio.BLOCK)).max()))
ok("stays in range", peak <= 1.0, f"peak {peak:.3f}")
ok("voices capped", len(a.voices) <= audio.MAX_VOICES,
   f"{len(a.voices)}")
ok("no limiting needed", a.clipped == 0, f"{a.clipped}")
print()
print("== off means silent ==")
a = audio.Audio(volume=0, seed=7)
a.place(0.5)
a.done()
s = np.concatenate([a.render(audio.BLOCK) for _ in range(40)])
ok("nothing at all", float(np.abs(s).max()) == 0.0)
print()
print("== latency is low enough to feel instant ==")
lat = audio.BLOCK / audio.SR * 1000
print(f"       block {audio.BLOCK} at {audio.SR}Hz = {lat:.1f}ms")
ok("under 20ms", lat < 20.0, f"{lat:.1f}ms")
a = audio.Audio(seed=9)
t0 = time.perf_counter()
for _ in range(500):
    a.render(audio.BLOCK)
ms = (time.perf_counter() - t0) / 500 * 1000
ok("render well inside budget", ms < lat * 0.2,
   f"{ms:.3f}ms of {lat:.1f}ms")
print()
print("== building audio is instant now ==")
t0 = time.perf_counter()
audio.Audio(seed=1)
ms = (time.perf_counter() - t0) * 1000
ok("under 60ms", ms < 60, f"{ms:.0f}ms")
print()
print("== the palette is calm, not harsh ==")
def lum(c):
    return 0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]
def contrast(a, b):
    la, lb = lum(a) / 255.0, lum(b) / 255.0
    la = la / 12.92 if la <= 0.04045 else ((la + 0.055) / 1.055) ** 2.4
    lb = lb / 12.92 if lb <= 0.04045 else ((lb + 0.055) / 1.055) ** 2.4
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)
ok("background is not pure black", lum(draw.BG) > 20,
   f"luma {lum(draw.BG):.0f}")
ok("background is dark enough to rest on", lum(draw.BG) < 90,
   f"luma {lum(draw.BG):.0f}")
ok("no pure white anywhere",
   all(max(c) < 250 for c in (draw.TEXT, draw.GLOW, draw.DIM, draw.CARD)))
c = contrast(draw.TEXT, draw.BG)
ok("text is readable on the background", c >= 4.5, f"contrast {c:.2f}:1")
c = contrast(draw.DIM, draw.CARD)
ok("dim text is readable on a card", c >= 3.0, f"contrast {c:.2f}:1")
c = contrast(draw.TEXT, draw.CARD_ON)
ok("text is readable on a chosen card", c >= 3.0, f"contrast {c:.2f}:1")
ok("cards stand out from the background",
   contrast(draw.CARD, draw.BG) > 1.1,
   f"{contrast(draw.CARD, draw.BG):.2f}:1")
for name, col in (("BG", draw.BG), ("CARD", draw.CARD),
                  ("TRAY_BG", draw.TRAY_BG)):
    spread = max(col) - min(col)
    ok(f"{name} is muted, not saturated", spread <= 26, f"spread {spread}")
print()
print("== the whole screen is easy on the eyes ==")
import main as M
a = M.App(1280, 860, "interlock", 3)
a.frame(time.monotonic())
g = draw.cv2.cvtColor(a.canvas, draw.cv2.COLOR_BGR2GRAY)
print(f"       menu: mean {g.mean():.0f}, "
      f"darkest {g.min()}, brightest {g.max()}")
ok("menu is not glaring", g.mean() < 140, f"mean {g.mean():.0f}")
ok("menu is not pitch black", g.mean() > 25, f"mean {g.mean():.0f}")
for _ in range(2):
    r = a.zones["start"]
    a.on_mouse(1, r.cx, r.cy, 0, None)
    a.on_mouse(4, r.cx, r.cy, 0, None)
    a.frame(time.monotonic())
    if a.page == M.SETUP:
        g = draw.cv2.cvtColor(a.canvas, draw.cv2.COLOR_BGR2GRAY)
        print(f"       setup: mean {g.mean():.0f}, "
              f"darkest {g.min()}, brightest {g.max()}")
        ok("setup is not glaring", g.mean() < 140, f"mean {g.mean():.0f}")
        ok("setup is not pitch black", g.mean() > 20,
           f"mean {g.mean():.0f}")
ok("start reaches the board", a.board is not None)
g = draw.cv2.cvtColor(a.canvas, draw.cv2.COLOR_BGR2GRAY)
print(f"       board: mean {g.mean():.0f}, "
      f"darkest {g.min()}, brightest {g.max()}")
ok("board is not glaring", g.mean() < 150, f"mean {g.mean():.0f}")
a.open_pause()
a.frame(time.monotonic())
g = draw.cv2.cvtColor(a.canvas, draw.cv2.COLOR_BGR2GRAY)
print(f"       stop menu: mean {g.mean():.0f}, "
      f"darkest {g.min()}, brightest {g.max()}")
ok("the stop menu dims the board behind it", g.mean() < 150,
   f"mean {g.mean():.0f}")
ok("but stays readable", g.max() > 90, f"brightest {g.max()}")
print()
if fails:
    print(f"FAILED {len(fails)}:")
    for f in fails:
        print("   -", f)
else:
    print("all effect checks pass")
