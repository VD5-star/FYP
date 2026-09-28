import threading
import time
import cv2
import numpy as np
import audio
import draw
import main as M
from main import SIZES, App
from pieces import INTERLOCK, SQUARE
from ui import TOUCH_MIN
DOWN = cv2.EVENT_LBUTTONDOWN
MOVE = cv2.EVENT_MOUSEMOVE
UP = cv2.EVENT_LBUTTONUP
fails = []
def ok(name, cond, detail=""):
    print(f"  [{'pass' if cond else 'FAIL'}] {name}"
          f"{('  ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)
def tap(app, x, y):
    app.on_mouse(DOWN, x, y, 0, None)
    app.on_mouse(UP, x, y, 0, None)
def tap_zone(app, key):
    r = app.zones[key]
    tap(app, r.cx, r.cy)
def go_setup(app):
    if app.page == M.PICK:
        tap_zone(app, "start")
def play(app):
    go_setup(app)
    tap_zone(app, "start")
def quiesce(timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        alive = [t for t in threading.enumerate()
                 if t is not threading.main_thread() and t.is_alive()]
        if not alive:
            return True
        for t in alive:
            t.join(max(0.0, end - time.time()))
    return False
def grab_point(p):
    for yy in range(p.pos[1], p.pos[1] + p.h):
        for xx in range(p.pos[0], p.pos[0] + p.w):
            if p.hits(xx, yy):
                return xx, yy
    return None, None
def solve(app, steps=None):
    b = app.board
    guard = 0
    while not b.solved and guard < b.total * 4:
        guard += 1
        idx = b.order[-1]
        p = b.pieces[idx]
        gx, gy = grab_point(p)
        app.on_mouse(DOWN, gx, gy, 0, None)
        t = b.target(p)
        dx, dy = t[0] + (gx - p.pos[0]), t[1] + (gy - p.pos[1])
        app.on_mouse(MOVE, dx, dy, 0, None)
        app.on_mouse(UP, dx, dy, 0, None)
    return b.solved
print("== the app has no keyboard handler at all ==")
a = App(1280, 860, INTERLOCK, 3)
ok("no menu_key", not hasattr(a, "menu_key"))
ok("no game_key", not hasattr(a, "game_key"))
src = open("main.py", encoding="utf-8").read()
ok("no waitKeyEx", "waitKeyEx" not in src)
ok("only esc-to-close remains", src.count("waitKey(") == 1)
ok("no ord( key compares", "ord(" not in src)
print()
print("== every control is reachable and big enough to touch ==")
a = App(1280, 860, INTERLOCK, 3)
small = [(k, r.w, r.h) for k, r in a.zones.items.items()
         if k != "lbl" and not r.touchable()]
ok("menu targets >= 44px", not small, str(small))
a.start()
small = [(k, r.w, r.h) for k, r in a.zones.items.items()
         if not r.touchable()]
ok("game targets >= 44px", not small, str(small))
a.board.solved and None
a.done_zones()
small = [(k, r.w, r.h) for k, r in a.zones.items.items()
         if not r.touchable()]
ok("done targets >= 44px", not small, str(small))
print()
print("== zones never overlap ==")
for label, mk in (("menu", lambda x: x.menu_zones()),
                  ("setup", lambda x: x.setup_zones()),
                  ("game", lambda x: x.game_zones()),
                  ("pause", lambda x: x.pause_zones()),
                  ("done", lambda x: x.done_zones())):
    a = App(1280, 860, INTERLOCK, 3)
    mk(a)
    items = [(k, r) for k, r in a.zones.items.items()
             if k not in ("lbl", "panel")]
    clash = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ka, ra = items[i]
            kb, rb = items[j]
            if (ra.x < rb.x1 and rb.x < ra.x1
                    and ra.y < rb.y1 and rb.y < ra.y1):
                clash.append((ka, kb))
    ok(f"{label} zones disjoint", not clash, str(clash))
print()
print("== zones stay inside the window ==")
for wh in ((1280, 860), (1024, 720), (1600, 1000)):
    a = App(wh[0], wh[1], INTERLOCK, 3)
    out = [k for k, r in a.zones.items.items()
           if r.x < 0 or r.y < 0 or r.x1 > wh[0] or r.y1 > wh[1]]
    ok(f"{wh[0]}x{wh[1]} menu inside", not out, str(out))
    a.start()
    out = [k for k, r in a.zones.items.items()
           if r.x < 0 or r.y < 0 or r.x1 > wh[0] or r.y1 > wh[1]]
    ok(f"{wh[0]}x{wh[1]} game inside", not out, str(out))
print()
print("== the picture page is a 3 by 3 grid ==")
a = App(1280, 860, INTERLOCK, 3)
cells = [k for k in a.zones.items if k.startswith("pic:")]
ok("nine cells in all", len(cells) + 1 == 9, f"{len(cells)} + your picture")
ok("your picture is the ninth", "own" in a.zones)
ok("only start and settings below", "start" in a.zones
   and "settings" in a.zones)
ok("no piece controls on the picture page", "pieces" not in a.zones
   and "cut" not in a.zones)
xs = sorted({a.zones[k].x for k in cells + ["own"]})
ys = sorted({a.zones[k].y for k in cells + ["own"]})
ok("three columns", len(xs) == 3, str(xs))
ok("three rows", len(ys) == 3, str(ys))
sx, st_ = a.zones["settings"], a.zones["start"]
ok("the two buttons sit side by side", sx.y == st_.y)
ok("and below the grid", sx.y > max(ys))
ok("and near the middle",
   abs(((sx.x + st_.x1) // 2) - a.w // 2) < 12,
   str((sx.x + st_.x1) // 2))
print()
print("== a picture leads to the setup page ==")
a = App(1280, 860, INTERLOCK, 3)
tap_zone(a, "pic:3")
ok("one tap goes to setup", a.page == M.SETUP, a.page)
ok("no board yet", a.board is None)
ok("it remembers which picture", a.pick == 3, str(a.pick))
ok("setup has the pieces slider", "pieces" in a.zones)
ok("setup has both cuts", "square" in a.zones and "interlock" in a.zones)
ok("setup has start and back", "start" in a.zones and "back" in a.zones)
tap_zone(a, "back")
ok("back returns to the pictures", a.page == M.PICK)
ok("and the grid is there again", "pic:0" in a.zones)
tap_zone(a, "pic:3")
tap_zone(a, "start")
ok("start builds the board", a.board is not None)
ok("with the right picture", a.title == a.names[3], a.title)
ok("and lands on the game page", a.page == M.GAME)
print()
print("== start button works from any selection ==")
bad_start = []
for i in range(len(App(1280, 860, INTERLOCK, 3).names)):
    a = App(1280, 860, INTERLOCK, 3)
    key = f"pic:{i}"
    if key not in a.zones:
        continue
    tap_zone(a, key)
    tap_zone(a, "start")
    if a.board is None or a.title != a.names[i]:
        bad_start.append(i)
ok("start works for all pictures", not bad_start, str(bad_start))
print()
print("== the pieces slider works by drag ==")
a = App(1280, 860, INTERLOCK, 3)
go_setup(a)
ok("there is a pieces slider", "pieces" in a.zones)
ok("it is touchable", a.zones["pieces"].touchable())
r = a.zones["pieces"]
opts = a.size_options()
got = set()
for px in range(r.x, r.x1, 2):
    a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.cx, r.cy, 0, None)
    a.on_mouse(cv2.EVENT_MOUSEMOVE, px, r.cy, 0, None)
    a.on_mouse(cv2.EVENT_LBUTTONUP, px, r.cy, 0, None)
    got.add(SIZES[a.size_pick])
ok("every offered size is reachable", got == set(opts),
   f"got {sorted(got)} of {opts}")
a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.x + 4, r.cy, 0, None)
a.on_mouse(cv2.EVENT_LBUTTONUP, r.x + 4, r.cy, 0, None)
ok("far left is the fewest pieces", SIZES[a.size_pick] == opts[0],
   str(SIZES[a.size_pick]))
a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.x1 - 4, r.cy, 0, None)
a.on_mouse(cv2.EVENT_LBUTTONUP, r.x1 - 4, r.cy, 0, None)
ok("far right is the most pieces", SIZES[a.size_pick] == opts[-1],
   str(SIZES[a.size_pick]))
play(a)
ok("board matches the slider", a.board.total == SIZES[a.size_pick] ** 2,
   f"{a.board.total}")
names = [draw.piece_shape(s * s)[0] for s in opts]
ok("each size is described", all(names))
ok("the words change with size", len(set(names)) >= 2, str(names))
ok("the word never goes backwards",
   all(names.index(names[i]) <= names.index(names[i + 1])
       for i in range(len(names) - 1)), str(names))
print()
print("== the cut is chosen by tap on the setup page ==")
a = App(1280, 860, INTERLOCK, 3)
go_setup(a)
tap_zone(a, "square")
ok("picked straight", a.style == SQUARE, a.style)
tap_zone(a, "interlock")
ok("picked interlocking", a.style == INTERLOCK, a.style)
tap_zone(a, "square")
ok("tapping the same one again keeps it", a.style == SQUARE, a.style)
tap_zone(a, "start")
ok("board uses chosen cut", a.board.cut.style == SQUARE, a.board.cut.style)
print()
print("== tiny pieces are never offered ==")
for wh in ((1280, 860), (900, 640), (700, 520)):
    a = App(wh[0], wh[1], INTERLOCK, 3)
    offered = a.size_options()
    _, aw, ah = a.layout()
    worst = [min(aw / s, ah / s) for s in offered]
    ok(f"{wh[0]}x{wh[1]} pieces >= {M.MIN_PIECE_PX:.0f}px",
       min(worst) >= M.MIN_PIECE_PX,
       f"offered {offered} smallest {min(worst):.0f}px")
print()
print("== a tap on the board does not disturb a piece ==")
a = App(1280, 860, INTERLOCK, 3)
play(a)
b = a.board
p = b.pieces[b.order[-1]]
gx, gy = grab_point(p)
before = list(p.pos)
tap(a, gx, gy)
ok("piece unmoved by tap", p.pos == before, f"{before} -> {p.pos}")
ok("no move recorded", b.moves == 1, f"moves={b.moves}")
print()
print("== drag then drop moves and can snap ==")
a = App(1280, 860, INTERLOCK, 3)
play(a)
b = a.board
p = b.pieces[b.order[-1]]
gx, gy = grab_point(p)
a.on_mouse(DOWN, gx, gy, 0, None)
a.on_mouse(MOVE, gx + 120, gy - 80, 0, None)
a.on_mouse(UP, gx + 120, gy - 80, 0, None)
ok("piece followed the pointer", p.pos != [gx, gy] and not p.placed)
t = b.target(p)
gx, gy = grab_point(p)
a.on_mouse(DOWN, gx, gy, 0, None)
a.on_mouse(MOVE, t[0] + (gx - p.pos[0]), t[1] + (gy - p.pos[1]), 0, None)
a.on_mouse(UP, t[0] + (gx - p.pos[0]), t[1] + (gy - p.pos[1]), 0, None)
ok("snapped home", p.placed)
print()
print("== a drag that starts on a button never grabs a piece ==")
a = App(1280, 860, INTERLOCK, 3)
play(a)
b = a.board
r = a.zones["stop"]
a.on_mouse(DOWN, r.cx, r.cy, 0, None)
ok("no piece held", b.held is None)
a.on_mouse(MOVE, r.cx + 200, r.cy - 200, 0, None)
a.on_mouse(UP, r.cx + 200, r.cy - 200, 0, None)
ok("drag off a button does nothing", b.moves == 0 and b.done == 0)
print()
print("== full solve by pointer only, both cuts ==")
for style in (SQUARE, INTERLOCK):
    a = App(1280, 860, style, 3)
    play(a)
    b = a.board
    ok(f"{style} solved", solve(a), f"{b.done}/{b.total}")
    ok(f"{style} moves == pieces", b.moves == b.total, f"moves={b.moves}")
print()
print("== the done screen is reachable and its buttons work ==")
a = App(1280, 860, INTERLOCK, 3)
play(a)
solve(a)
a.frame(time.monotonic())
ok("done timer set", a.done_at > 0)
ok("again button present", "again" in a.zones)
ok("back button present", "back" in a.zones)
tap_zone(a, "again")
ok("again gives a fresh board", a.board is not None and a.board.done == 0)
solve(a)
a.frame(time.monotonic())
tap_zone(a, "back")
ok("back returns to menu", a.board is None)
print()
print("== the stop button opens a menu with everything in it ==")
a = App(1280, 860, INTERLOCK, 3)
play(a)
ok("only the stop button is on the board", list(a.zones.items) == ["stop"],
   str(sorted(a.zones.items)))
s = a.zones["stop"]
ok("it is round", s.w == s.h, f"{s.w}x{s.h}")
ok("it is big enough to touch", s.touchable())
ok("it sits on the right", s.cx > a.w * 0.8, f"cx={s.cx} of {a.w}")
ok("it sits at the middle height", abs(s.cy - a.h // 2) <= 8,
   f"cy={s.cy} of {a.h}")
tap_zone(a, "stop")
ok("tapping it pauses", a.paused)
for k in ("volume", "resume", "again", "shuffle", "home"):
    ok(f"the menu has {k}", k in a.zones)
    if k in a.zones and not a.zones[k].touchable():
        ok(f"{k} is touchable", False)
ok("every menu control is touchable",
   not [f for f in fails if "is touchable" in f])
pnl = a.zones["panel"]
ok("the panel is a middling size",
   260 <= pnl.w <= 520 and 200 <= pnl.h <= 340, f"{pnl.w}x{pnl.h}")
ok("it is centred", abs(pnl.cx - a.w // 2) <= 2
   and abs(pnl.cy - a.h // 2) <= 2)
vol = a.zones["volume"]
ok("sound sits on top, across the middle",
   vol.y < a.zones["resume"].y and abs(vol.cx - pnl.cx) <= 2)
pairs = {}
for k in ("resume", "again", "shuffle", "home"):
    pairs.setdefault(a.zones[k].y, []).append(k)
ok("the four buttons form two rows of two",
   len(pairs) == 2 and all(len(v) == 2 for v in pairs.values()),
   str({y: sorted(v) for y, v in pairs.items()}))
ok("no separate sound button any more", "settings" not in a.zones)
tap_zone(a, "resume")
ok("keep going returns to the board", not a.paused
   and list(a.zones.items) == ["stop"])
b1 = a.board
b1.pieces[0].placed = True
tap_zone(a, "stop")
tap_zone(a, "again")
ok("start over gives a fresh board",
   a.board is not b1 and a.board.done == 0)
ok("and leaves the menu", not a.paused)
a = App(1280, 860, INTERLOCK, 3)
play(a)
b = a.board
p0 = b.pieces[0]
tgt = b.target(p0)
p0.pos = list(tgt)
p0.placed = True
loose = [p for p in b.pieces if not p.placed]
before = [tuple(p.pos) for p in loose]
tap_zone(a, "stop")
tap_zone(a, "shuffle")
after = [tuple(p.pos) for p in loose]
ok("shuffle keeps the same board", a.board is b)
ok("shuffle keeps what you already placed",
   p0.placed and b.done == 1 and tuple(p0.pos) == tuple(tgt),
   f"done={b.done}")
ok("shuffle really moves the loose pieces", before != after)
ok("and closes the menu", not a.paused)
tap_zone(a, "stop")
r = a.zones["volume"]
a.on_mouse(DOWN, r.x + draw.SLIDER_PAD, r.cy, 0, None)
a.on_mouse(UP, r.x + draw.SLIDER_PAD, r.cy, 0, None)
ok("dragging sound to the left mutes", a.audio.volume == audio.VOL_MIN,
   str(a.audio.volume))
a.on_mouse(DOWN, r.x1 - draw.SLIDER_PAD, r.cy, 0, None)
a.on_mouse(UP, r.x1 - draw.SLIDER_PAD, r.cy, 0, None)
ok("and to the right is loudest", a.audio.volume == audio.VOL_MAX,
   str(a.audio.volume))
ok("changing sound never leaves the menu", a.paused and not a.settings)
ok("and never disturbs the board", a.board is b and b.done == 1)
tap_zone(a, "home")
ok("home returns to the pictures", a.board is None and a.page == M.PICK)
ok("custom forgotten", a.custom is None)
ok("not paused any more", not a.paused)
print()
print("== your own picture: chosen, used, then forgotten ==")
a = App(1280, 860, INTERLOCK, 3)
a.custom = np.dstack([
    np.tile(np.linspace(10, 250, 700, dtype=np.uint8), (500, 1)),
    np.tile(np.linspace(250, 10, 700, dtype=np.uint8), (500, 1)),
    np.full((500, 700), 130, np.uint8)])
a.start()
ok("custom used", a.title == "yours", a.title)
ok("board built from it", a.board.total == SIZES[a.size_pick] ** 2)
tap_zone(a, "stop")
tap_zone(a, "home")
ok("dropped on exit", a.custom is None and a.image is None)
print()
print("== no score, no timer, no failure ==")
a = App(1280, 860, INTERLOCK, 3)
play(a)
bad = {"score", "points", "streak", "lives", "penalty", "combo",
       "time_left", "deadline", "failed", "mistakes", "best"}
ok("board clean", not (set(dir(a.board)) & bad))
ok("app clean", not (set(dir(a)) & bad))
print()
print("== frames render in budget ==")
for si, s in enumerate(SIZES):
    a = App(1280, 860, INTERLOCK, 3)
    go_setup(a)
    if s not in a.size_options():
        continue
    a.size_pick = si
    tap_zone(a, "start")
    quiesce()
    now = time.monotonic()
    for _ in range(3):
        a.frame(now)
    ms = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            a.frame(now)
        ms = min(ms, (time.perf_counter() - t0) / 20 * 1000)
    ok(f"{s}x{s} frame under 16ms", ms < 16.0, f"{ms:.2f}ms")
a = App(1280, 860, INTERLOCK, 3)
quiesce()
t0 = time.perf_counter()
for _ in range(20):
    a.frame(time.monotonic())
ms = (time.perf_counter() - t0) / 20 * 1000
ok("menu frame under 16ms", ms < 16.0, f"{ms:.2f}ms")
print()
print("== menu opens fast ==")
t0 = time.perf_counter()
App(1280, 860, INTERLOCK, 3)
first = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
App(1280, 860, INTERLOCK, 3)
later = (time.perf_counter() - t0) * 1000
print(f"       first {first:.0f}ms, later {later:.0f}ms")
ok("menu ready under 3s", first < 3000, f"{first:.0f}ms")
print()
print("== even a cold start is not a long stall ==")
for s in (5, 8, 10):
    best = 1e9
    for _ in range(3):
        a = App(1280, 860, INTERLOCK, 3)
        quiesce()
        go_setup(a)
        a._art.clear()
        if s in a.size_options():
            a.size_pick = SIZES.index(s)
            t0 = time.perf_counter()
            tap_zone(a, "start")
            best = min(best, (time.perf_counter() - t0) * 1000)
    ok(f"{s}x{s} cold start under 1500ms", best < 1500, f"{best:.0f}ms")
print()
print("== prewarm makes the start tap feel instant ==")
a = App(1280, 860, INTERLOCK, 3)
quiesce()
t0 = time.perf_counter()
play(a)
ms = (time.perf_counter() - t0) * 1000
ok("warm start under 120ms", ms < 120, f"{ms:.0f}ms")
ok("board really built", a.board is not None and a.board.total > 0)
best = 1e9
for _ in range(3):
    a = App(1280, 860, INTERLOCK, 3)
    tap_zone(a, "pic:5")
    quiesce()
    _, aw, ah = a.layout()
    a.art_at(a.names[5], aw, ah)
    t0 = time.perf_counter()
    tap_zone(a, "start")
    best = min(best, (time.perf_counter() - t0) * 1000)
ok("warm start after picking under 120ms", best < 120, f"{best:.0f}ms")
ok("right picture used", a.title == a.names[5], a.title)
print()
print("== a cold start still works, just slower ==")
a = App(1280, 860, INTERLOCK, 3)
a.pick = 2
a._art.clear()
play(a)
ok("cold start builds", a.board is not None and a.board.done == 0)
ok("image matches the choice", a.title == a.names[2], a.title)
print()
print("== prewarming never blocks or corrupts the menu ==")
a = App(1280, 860, INTERLOCK, 3)
t0 = time.perf_counter()
for i in range(len(a.names)):
    if f"pic:{i}" not in a.zones:
        continue
    tap_zone(a, f"pic:{i}")
    tap_zone(a, "back")
ms = (time.perf_counter() - t0) * 1000
ok("rapid picture switching is instant", ms < 400, f"{ms:.0f}ms")
ok("still in the picture page", a.board is None and a.page == M.PICK)
for t in threading.enumerate():
    pass
alive = [t for t in threading.enumerate() if t is not threading.main_thread()]
ok("warm threads are daemons", all(t.daemon for t in alive),
   f"{len(alive)} background")
for t in alive:
    t.join(30)
a.frame(time.monotonic())
ok("menu still renders after warms", a.canvas.sum() > 0)
print()
print("== sound is a touch control like everything else ==")
a = App(1280, 860, INTERLOCK, 3)
ok("settings button in menu", "settings" in a.zones)
ok("settings button is touchable", a.zones["settings"].touchable())
tap_zone(a, "settings")
ok("settings opens", a.settings)
ok("slider is there", "volume" in a.zones)
ok("slider is touchable", a.zones["volume"].touchable())
r = a.zones["volume"]
ok("the panel does not swallow the slider",
   a.zones.at(r.cx, r.cy) == "volume", str(a.zones.at(r.cx, r.cy)))
c = a.zones["close"]
ok("the panel does not swallow done",
   a.zones.at(c.cx, c.cy) == "close", str(a.zones.at(c.cx, c.cy)))
a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.x + 4, r.cy, 0, None)
a.on_mouse(cv2.EVENT_MOUSEMOVE, r.cx, r.cy, 0, None)
mid = a.audio.volume
a.on_mouse(cv2.EVENT_MOUSEMOVE, r.x1 - 4, r.cy, 0, None)
top = a.audio.volume
a.on_mouse(cv2.EVENT_LBUTTONUP, r.x1 - 4, r.cy, 0, None)
ok("dragging sets the volume", 0 < mid < top, f"{mid} then {top}")
ok("the far right is full", top == audio.VOL_MAX, str(top))
a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.x - 500, r.cy, 0, None)
a.on_mouse(cv2.EVENT_LBUTTONUP, r.x - 500, r.cy, 0, None)
ok("a press far outside is ignored", a.audio.volume == top)
steps = set()
for px in range(r.x, r.x1, 3):
    a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.cx, r.cy, 0, None)
    a.on_mouse(cv2.EVENT_MOUSEMOVE, px, r.cy, 0, None)
    a.on_mouse(cv2.EVENT_LBUTTONUP, px, r.cy, 0, None)
    steps.add(a.audio.volume)
ok("the slider is smooth, not four steps", len(steps) > 40, str(len(steps)))
tap_zone(a, "close")
ok("done closes settings", not a.settings)
ok("and lands back in the menu", a.board is None and "start" in a.zones)
a = App(1280, 860, INTERLOCK, 3)
play(a)
tap_zone(a, "stop")
ok("sound is right inside the stop menu", "volume" in a.zones)
r = a.zones["volume"]
steps = set()
for px in range(r.x, r.x1, 3):
    a.on_mouse(DOWN, r.cx, r.cy, 0, None)
    a.on_mouse(MOVE, px, r.cy, 0, None)
    a.on_mouse(UP, px, r.cy, 0, None)
    steps.add(a.audio.volume)
ok("it slides smoothly there too", len(steps) > 40, str(len(steps)))
ok("no extra panel opens", not a.settings and a.paused)
tap_zone(a, "resume")
ok("and back to the board", "stop" in a.zones and not a.paused)
ok("board untouched", a.board.done == 0 and a.board.moves == 0)
print()
print("== placing a piece makes a sound, a tap does not ==")
a = App(1280, 860, INTERLOCK, 3)
play(a)
b = a.board
p = b.pieces[b.order[-1]]
gx, gy = grab_point(p)
before = len(a.audio.voices)
tap(a, gx, gy)
ok("a tap is silent", len(a.audio.voices) == before)
t = b.target(p)
a.on_mouse(DOWN, gx, gy, 0, None)
a.on_mouse(MOVE, t[0] + (gx - p.pos[0]), t[1] + (gy - p.pos[1]), 0, None)
a.on_mouse(UP, t[0] + (gx - p.pos[0]), t[1] + (gy - p.pos[1]), 0, None)
ok("a snap makes a sound", len(a.audio.voices) > before,
   f"{before} -> {len(a.audio.voices)}")
a2 = App(1280, 860, INTERLOCK, 3)
play(a2)
b2 = a2.board
p2 = b2.pieces[b2.order[-1]]
gx, gy = grab_point(p2)
before = len(a2.audio.voices)
a2.on_mouse(DOWN, gx, gy, 0, None)
a2.on_mouse(MOVE, gx + 200, gy - 60, 0, None)
a2.on_mouse(UP, gx + 200, gy - 60, 0, None)
ok("a miss is silent", len(a2.audio.voices) == before)
print()
print("== finishing plays the closing chord once ==")
a = App(1280, 860, INTERLOCK, 3)
play(a)
solve(a)
a.audio.voices.clear()
a.audio.fade = 1.0
quiet = np.concatenate([a.audio.render(audio.BLOCK) for _ in range(4)])
a.frame(time.monotonic())
ok("chord queued on finish", len(a.audio.voices) > 0,
   f"{len(a.audio.voices)} voices")
loud = np.concatenate([a.audio.render(audio.BLOCK) for _ in range(120)])
ok("chord is audible", float(np.abs(loud).max())
   > float(np.abs(quiet).max()) * 1.15,
   f"{float(np.abs(quiet).max()):.3f} -> {float(np.abs(loud).max()):.3f}")
ok("chord does not clip", float(np.abs(loud).max()) <= 1.0)
a.audio.voices.clear()
for _ in range(5):
    a.frame(time.monotonic())
ok("not replayed every frame", len(a.audio.voices) == 0,
   f"{len(a.audio.voices)}")
print()
print("== audio never slows the frame ==")
a = App(1280, 860, INTERLOCK, 3)
play(a)
b = App(1280, 860, INTERLOCK, 3, audio_on=False)
play(b)
quiesce()
now = time.monotonic()
def best_frame(app):
    for _ in range(3):
        app.frame(now)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            app.frame(now)
        best = min(best, (time.perf_counter() - t0) / 20 * 1000)
    return best
with_audio = best_frame(a)
without = best_frame(b)
ok("frame cost unchanged", abs(with_audio - without) < 3.0,
   f"{with_audio:.2f}ms with, {without:.2f}ms without")
print()
print("== the game runs fine with no audio at all ==")
a = App(1280, 860, INTERLOCK, 3, audio_on=False)
ok("no audio object", a.audio is None)
ok("label says so", a.sound_name() == "no sound", a.sound_name())
tap_zone(a, "settings")
ok("settings still opens", a.settings)
a.frame(time.monotonic())
ok("and draws without audio", a.canvas.sum() > 0)
r = a.zones["volume"]
a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.cx, r.cy, 0, None)
a.on_mouse(cv2.EVENT_LBUTTONUP, r.cx, r.cy, 0, None)
ok("dragging with no audio is harmless", a.settings)
tap_zone(a, "close")
ok("closes fine", not a.settings and a.board is None)
play(a)
ok("solves without audio", solve(a))
a.frame(time.monotonic())
ok("done screen fine", "again" in a.zones)
print()
print("== audio never writes files either ==")
import os
before_files = set(os.listdir("."))
a = App(1280, 860, INTERLOCK, 3)
play(a)
solve(a)
a.frame(time.monotonic())
ok("no files from audio", set(os.listdir(".")) == before_files)
print()
if fails:
    print(f"FAILED {len(fails)}:")
    for f in fails:
        print("   -", f)
else:
    print("all smoke checks pass")
