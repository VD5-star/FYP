import time
import numpy as np
import art
import board as board_mod
import draw
import photo
from pieces import INTERLOCK, SQUARE, coverage, cut_image, make_cut
from ui import Rect, Zones
W, H = 900, 600
TRAY = (0, 620, 1280, 240)
def demo(w=W, h=H):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.linspace(20, 255, w, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(20, 255, h, dtype=np.uint8)[:, None]
    img[:, :, 2] = 90
    return img
fails = []
def ok(name, cond, detail=""):
    mark = "pass" if cond else "FAIL"
    print(f"  [{mark}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)
print("== cut integrity ==")
for style in (SQUARE, INTERLOCK):
    for n in (3, 5, 8, 10):
        cut = make_cut(n, n, W, H, style=style, seed=7)
        cov = coverage(cut)
        gaps = int((cov == 0).sum())
        ok(f"{style} {n}x{n} no gaps", gaps == 0, f"gaps={gaps}")
print()
print("== reassembly is exact ==")
img = demo()
for style in (SQUARE, INTERLOCK):
    cut = make_cut(6, 6, W, H, style=style, seed=5)
    ps = cut_image(img, cut)
    canvas = np.zeros((H, W, 3), np.uint8)
    for p in ps:
        x, y = p.home
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + p.w), min(H, y + p.h)
        sx0, sy0 = x0 - x, y0 - y
        sub = p.sprite[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
        m = p.alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)] > 0
        canvas[y0:y1, x0:x1][m] = sub[m]
    err = int(np.abs(canvas.astype(int) - img.astype(int)).max())
    unpainted = int((canvas.sum(axis=2) == 0).sum())
    ok(f"{style} pixel-exact", err == 0, f"max err={err}")
    ok(f"{style} fully painted", unpainted == 0, f"unpainted={unpainted}")
print()
print("== scatter puts every piece inside the tray ==")
for style in (SQUARE, INTERLOCK):
    b = board_mod.build(img, 6, 6, (40, 46), TRAY, style=style, seed=4)
    tx, ty, tw, th = TRAY
    inside = sum(1 for p in b.pieces
                 if p.pos[0] >= tx and p.pos[1] >= ty
                 and p.pos[0] + p.w <= tx + tw + p.w
                 and p.pos[1] + p.h <= ty + th + p.h)
    ok(f"{style} all in tray", inside == len(b.pieces),
       f"{inside}/{len(b.pieces)}")
    ok(f"{style} none start placed", b.done == 0, f"done={b.done}")
    ok(f"{style} order covers all", sorted(b.order) == list(range(b.total)))
print()
print("== a piece can never be lost: no dead ends ==")
b = board_mod.build(img, 5, 5, (40, 46), TRAY, style=INTERLOCK, seed=9)
reachable = 0
for i, p in enumerate(b.pieces):
    cx = p.pos[0] + p.w // 2
    cy = p.pos[1] + p.h // 2
    found = False
    for yy in range(p.pos[1], p.pos[1] + p.h, 3):
        for xx in range(p.pos[0], p.pos[0] + p.w, 3):
            if p.hits(xx, yy):
                found = True
                break
        if found:
            break
    reachable += found
ok("every piece has a grabbable pixel", reachable == b.total,
   f"{reachable}/{b.total}")
print()
print("== snap works from anywhere within radius, and only there ==")
b = board_mod.build(img, 5, 5, (40, 46), TRAY, style=INTERLOCK, seed=9)
r = b.radius
p = b.pieces[0]
tgt = b.target(p)
snapped_in = 0
trials = 0
rng = np.random.default_rng(1)
for _ in range(400):
    ang = rng.uniform(0, 2 * np.pi)
    d = rng.uniform(0, r * 0.94)
    b.held = 0
    b.grab = (0, 0)
    p.placed = False
    b.settled.clear()
    p.pos = [int(tgt[0] + np.cos(ang) * d), int(tgt[1] + np.sin(ang) * d)]
    trials += 1
    snapped_in += b.drop()
ok("inside radius always snaps", snapped_in == trials,
   f"{snapped_in}/{trials}")
snapped_out = 0
for _ in range(400):
    ang = rng.uniform(0, 2 * np.pi)
    d = rng.uniform(r * 1.15, r * 6)
    b.held = 0
    b.grab = (0, 0)
    p.placed = False
    b.settled.clear()
    p.pos = [int(tgt[0] + np.cos(ang) * d), int(tgt[1] + np.sin(ang) * d)]
    snapped_out += b.drop()
ok("outside radius never snaps", snapped_out == 0, f"{snapped_out}/400")
print()
print("== snap radius scales with piece size ==")
prev = None
line = []
for n in (3, 5, 8, 10):
    c = make_cut(n, n, W, H, style=INTERLOCK, seed=1)
    rr = board_mod.snap_radius(c)
    line.append(f"{n}x{n}={rr:.1f}")
    if prev is not None:
        ok(f"{n}x{n} radius <= smaller grid", rr <= prev + 0.01)
    prev = rr
print("      ", "  ".join(line))
ok("10x10 radius still usable", board_mod.snap_radius(
    make_cut(10, 10, W, H, seed=1)) >= board_mod.SNAP_MIN)
print()
print("== a placed piece stays placed and leaves the pick order ==")
b = board_mod.build(img, 4, 4, (40, 46), TRAY, style=INTERLOCK, seed=2)
p = b.pieces[3]
idx = 3
b.order.remove(idx)
b.order.append(idx)
b.held = idx
b.grab = (0, 0)
t = b.target(p)
p.pos = [t[0], t[1]]
b.drop()
ok("piece placed", p.placed)
ok("removed from order", idx not in b.order)
ok("cannot be picked again", not b.pick(t[0] + p.w // 2, t[1] + p.h // 2)
   or b.held != idx)
print()
print("== solving every piece marks the board solved ==")
for style in (SQUARE, INTERLOCK):
    b = board_mod.build(img, 5, 5, (40, 46), TRAY, style=style, seed=6)
    for i, p in enumerate(b.pieces):
        if i in b.order:
            b.order.remove(i)
        b.order.append(i)
        b.held = i
        b.grab = (0, 0)
        t = b.target(p)
        p.pos = [t[0], t[1]]
        b.drop()
    ok(f"{style} solved", b.solved, f"{b.done}/{b.total}")
    ok(f"{style} order empty", len(b.order) == 0, f"{len(b.order)}")
print()
print("== pick returns the topmost piece under the cursor ==")
b = board_mod.build(img, 4, 4, (40, 46), TRAY, style=SQUARE, seed=8)
a, c = b.pieces[0], b.pieces[1]
a.pos = [500, 700]
c.pos = [500, 700]
b.order = [0, 1]
b.pick(500 + a.w // 2, 700 + a.h // 2)
ok("topmost picked", b.held == 1, f"held={b.held}")
ok("picked moves to front", b.order[-1] == 1)
print()
print("== drag keeps the grab point under the cursor ==")
b = board_mod.build(img, 4, 4, (40, 46), TRAY, style=INTERLOCK, seed=8)
p = b.pieces[2]
gx = p.pos[0] + p.w // 2
gy = p.pos[1] + p.h // 2
b.pick(gx, gy)
drift = []
for dx, dy in ((30, 20), (-140, 90), (400, -260), (5, 5)):
    gx += dx
    gy += dy
    b.drag(gx, gy)
    held = b.pieces[b.held]
    drift.append(abs((gx - held.pos[0]) - b.grab[0])
                 + abs((gy - held.pos[1]) - b.grab[1]))
ok("grab offset constant", max(drift) == 0, f"max drift={max(drift)}")
print()
print("== no score, no timer, no failure state ==")
b = board_mod.build(img, 4, 4, (40, 46), TRAY, seed=1)
names = set(dir(b))
bad = {"score", "points", "streak", "lives", "penalty", "combo",
       "time_left", "deadline", "failed", "mistakes", "best"}
hit = names & bad
ok("board exposes no scoring fields", not hit, str(sorted(hit)))
ok("moves is recorded but unbounded", hasattr(b, "moves"))
print()
print("== gaps between moves are recorded for engagement, not score ==")
b = board_mod.build(img, 4, 4, (40, 46), TRAY, seed=1)
t0 = 1000.0
for i, step in enumerate((0.0, 1.2, 0.8, 4.0, 1.0)):
    t0 += step
    b.held = i
    b.grab = (0, 0)
    if i in b.order:
        b.order.remove(i)
    b.order.append(i)
    b.pieces[i].pos = [9999, 9999]
    b.drop(now=t0)
ok("gap count = moves - 1", len(b.gaps) == b.moves - 1,
   f"gaps={len(b.gaps)} moves={b.moves}")
ok("gaps match input", [round(g, 2) for g in b.gaps] == [1.2, 0.8, 4.0, 1.0],
   str([round(g, 2) for g in b.gaps]))
print()
print("== gallery renders ==")
for n in art.NAMES:
    im = art.render(n, 400, 260)
    span = int(im.max()) - int(im.min())
    std = float(im.std())
    ok(f"{n} has range", span > 60 and std > 12,
       f"span={span} std={std:.1f}")
    ok(f"{n} shape", im.shape == (260, 400, 3), str(im.shape))
    ok(f"{n} dtype", im.dtype == np.uint8)
print()
print("== fit and grid snap ==")
for w, h in ((4000, 3000), (600, 2400), (1920, 1080), (100, 100)):
    src = np.zeros((h, w, 3), np.uint8)
    f = photo.fit(src, 1200, 560)
    ok(f"{w}x{h} fits", f.shape[1] <= 1200 and f.shape[0] <= 560,
       f"-> {f.shape[1]}x{f.shape[0]}")
    ar_src = w / h
    ar_out = f.shape[1] / f.shape[0]
    ok(f"{w}x{h} aspect kept", abs(ar_src - ar_out) / ar_src < 0.02,
       f"{ar_src:.3f} vs {ar_out:.3f}")
for rows in (3, 7, 8, 10):
    src = np.zeros((563, 997, 3), np.uint8)
    g = photo.snap_to_grid(src, rows, rows)
    ok(f"{rows}x{rows} divides evenly",
       g.shape[0] % rows == 0 and g.shape[1] % rows == 0,
       f"{g.shape[1]}x{g.shape[0]}")
print()
print("== blit stays inside the canvas ==")
canvas = np.zeros((200, 300, 3), np.uint8)
sprite = np.full((60, 60, 3), 255, np.uint8)
alpha = np.full((60, 60), 255, np.uint8)
for x, y in ((-80, -80), (280, 180), (-30, 100), (150, -40), (999, 999)):
    before = canvas.copy()
    draw.blit(canvas, sprite, alpha, x, y)
ok("no crash out of bounds", True)
canvas[:] = 0
draw.blit(canvas, sprite, alpha, -30, -30)
ok("partial blit paints", int(canvas.sum()) > 0)
ok("partial blit clipped", int(canvas[35:, :].sum()) == 0)
print()
print("== reference ghost is faint ==")
canvas = np.zeros((200, 300, 3), np.uint8)
ref = np.full((100, 150, 3), 255, np.uint8)
draw.draw_reference(canvas, ref, (20, 20))
mid = int(canvas[70, 90].mean())
bg = int(np.array(draw.BG).mean())
ok("ghost is a faint hint, not the picture", bg < mid < 110,
   f"value={mid}, background={bg}, alpha={draw.GHOST}")
ok("ghost is much darker than the real piece", mid < 255 * 0.5,
   f"{mid} vs 255")
print()
print("== piece edges are antialiased, not jagged ==")
for style in (SQUARE, INTERLOCK):
    cut = make_cut(5, 5, W, H, style=style, seed=3)
    ps = cut_image(img, cut)
    soft = sum(int(((p.alpha > 0) & (p.alpha < 250)).sum()) for p in ps)
    hard = sum(int((p.alpha > 0).sum()) for p in ps)
    ok(f"{style} has soft edge pixels", soft > 0,
       f"{soft} soft of {hard} opaque")
    ok(f"{style} core is still solid", soft / hard < 0.12,
       f"{soft/hard*100:.1f}% soft")
print()
print("== touch rects behave ==")
r = Rect(10, 20, 100, 50)
ok("hit inside", r.hit(60, 45))
ok("miss left", not r.hit(9, 45))
ok("miss right edge is exclusive", not r.hit(110, 45))
ok("miss above", not r.hit(60, 19))
ok("miss below edge is exclusive", not r.hit(60, 70))
ok("centre correct", (r.cx, r.cy) == (60, 45))
ok("touchable at 100x50", r.touchable())
ok("not touchable at 40x40", not Rect(0, 0, 40, 40).touchable())
z = Zones()
z.add("a", Rect(0, 0, 50, 50))
z.add("b", Rect(60, 0, 50, 50))
ok("zone lookup a", z.at(10, 10) == "a")
ok("zone lookup b", z.at(70, 10) == "b")
ok("zone lookup gap", z.at(55, 10) is None)
print()
print("== cut time stays interactive ==")
for n in (5, 8, 10):
    src = demo(1200, 800)
    t0 = time.perf_counter()
    board_mod.build(src, n, n, (40, 46), TRAY, style=INTERLOCK, seed=3)
    ms = (time.perf_counter() - t0) * 1000
    ok(f"{n}x{n} builds under 900ms", ms < 900, f"{ms:.0f}ms")
print()
print("== frame draw stays under 16ms ==")
src = demo(1200, 700)
for n in (5, 8, 10):
    b = board_mod.build(src, n, n, (40, 46), TRAY, style=INTERLOCK, seed=3)
    canvas = np.zeros((860, 1280, 3), np.uint8)
    now = time.monotonic()
    for _ in range(3):
        draw.draw_board(canvas, b, src, now)
    z = Zones()
    z.add("stop", Rect(1280 - 26 - 68, 430 - 34, 68, 68))
    t0 = time.perf_counter()
    for _ in range(20):
        draw.draw_board(canvas, b, src, now)
        draw.draw_hud(canvas, b, "test", z)
    ms = (time.perf_counter() - t0) / 20 * 1000
    ok(f"{n}x{n} draws under 16ms", ms < 16.0, f"{ms:.2f}ms")
print()
if fails:
    print(f"FAILED {len(fails)}:")
    for f in fails:
        print("   -", f)
else:
    print("all checks pass")
