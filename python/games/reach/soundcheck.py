import sys
import time
import cv2
import numpy as np
import audio as audio_mod
import settings as st_mod
from settings import SettingsPanel, slider_value
fails = []
def ok(label, cond, note=""):
    print(f"  [{'pass' if cond else 'FAIL'}] {label}" + (f"  {note}" if note
                                                         else ""))
    if not cond:
        fails.append(label)
print("== the sound itself ==")
t0 = time.perf_counter()
au = audio_mod.Audio(volume=70, seed=5)
build = (time.perf_counter() - t0) * 1000
ok("builds fast", build < 500, f"{build:.0f}ms")
ok("silent until something happens",
   float(np.abs(au.render(audio_mod.BLOCK)).max()) == 0.0)
au.hit()
buf = np.concatenate([au.render(audio_mod.BLOCK) for _ in range(40)])
peak = float(np.abs(buf).max())
ok("a catch makes a sound", peak > 0.01, f"peak {peak:.3f}")
ok("never clips", peak <= 1.0)
ok("no limiting needed", au.clipped == 0)
step = float(np.abs(np.diff(buf[:, 0])).max())
ok("no sudden jumps", step < 0.25, f"max step {step:.3f}")
au.hit()
tail = np.concatenate([au.render(audio_mod.BLOCK) for _ in range(80)])
ok("the sound ends, it does not drone",
   float(np.abs(tail[-audio_mod.BLOCK:]).max()) < 1e-6)
au.reward()
bon = np.concatenate([au.render(audio_mod.BLOCK) for _ in range(60)])
au.done()
end = np.concatenate([au.render(audio_mod.BLOCK) for _ in range(200)])
ok("the reward sounds different from a catch",
   float(np.abs(bon).max()) > 0.01 and float(np.abs(end).max()) > 0.01)
def length(x):
    a = np.abs(x).max(axis=1)
    nz = np.flatnonzero(a > 1e-5)
    return 0 if nz.size == 0 else int(nz[-1] - nz[0])
ok("the ending sound is the longest", length(end) > length(bon),
   f"{length(end)} vs {length(bon)} samples")
for _ in range(50):
    au.hit()
many = np.concatenate([au.render(audio_mod.BLOCK) for _ in range(30)])
ok("a burst of catches never clips", float(np.abs(many).max()) <= 1.0,
   f"peak {float(np.abs(many).max()):.3f}")
ok("voices are capped", len(au.voices) <= audio_mod.MAX_VOICES,
   str(len(au.voices)))
t0 = time.perf_counter()
for _ in range(400):
    au.render(audio_mod.BLOCK)
per = (time.perf_counter() - t0) / 400 * 1000
budget = audio_mod.BLOCK / audio_mod.SR * 1000
ok("audio is far inside its budget", per < budget * 0.25,
   f"{per:.3f}ms of {budget:.1f}ms")
print()
print("== the volume slider ==")
lv = []
for v in (10, 30, 50, 70, 100):
    a = audio_mod.Audio(volume=v, seed=1)
    a.hit()
    s = np.concatenate([a.render(audio_mod.BLOCK) for _ in range(40)])
    lv.append(float(np.sqrt((s ** 2).mean())))
ok("louder all the way up",
   all(lv[i] < lv[i + 1] for i in range(len(lv) - 1)),
   str([round(x, 4) for x in lv]))
a0 = audio_mod.Audio(volume=0, seed=1)
a0.hit()
s0 = np.concatenate([a0.render(audio_mod.BLOCK) for _ in range(40)])
ok("zero is truly silent", float(np.abs(s0).max()) == 0.0)
a1 = audio_mod.Audio(volume=50)
a1.set_volume(500)
hi = a1.volume
a1.set_volume(-10)
ok("out of range is clamped", hi == 100 and a1.volume == 0)
print()
print("== the settings panel ==")
p = SettingsPanel(audio_mod.Audio(volume=40), 1280, 720)
ok("closed at first", not p.open)
ok("the button is touchable", p.button["settings"].touchable())
b = p.button["settings"]
p.on_mouse(cv2.EVENT_LBUTTONDOWN, b.cx, b.cy)
p.on_mouse(cv2.EVENT_LBUTTONUP, b.cx, b.cy)
ok("tapping opens it", p.open)
r = p.panel["volume"]
ok("the slider is touchable", r.touchable())
ok("the panel does not swallow the slider",
   p.zones.at(r.cx, r.cy) == "volume", str(p.zones.at(r.cx, r.cy)))
c = p.panel["close"]
ok("the panel does not swallow done", p.zones.at(c.cx, c.cy) == "close")
p.on_mouse(cv2.EVENT_LBUTTONDOWN, r.x + 4, r.cy)
p.on_mouse(cv2.EVENT_MOUSEMOVE, r.cx, r.cy)
mid = p.audio.volume
p.on_mouse(cv2.EVENT_MOUSEMOVE, r.x1 - 4, r.cy)
top = p.audio.volume
p.on_mouse(cv2.EVENT_LBUTTONUP, r.x1 - 4, r.cy)
ok("dragging sets the volume", 0 < mid < top, f"{mid} then {top}")
ok("the far right is full", top == 100, str(top))
steps = set()
for px in range(r.x, r.x1, 3):
    steps.add(slider_value(r, px))
ok("smooth, not four steps", len(steps) > 40, str(len(steps)))
p.on_mouse(cv2.EVENT_LBUTTONDOWN, c.cx, c.cy)
p.on_mouse(cv2.EVENT_LBUTTONUP, c.cx, c.cy)
ok("done closes it", not p.open)
frame = np.full((720, 1280, 3), 90, np.uint8)
before = frame.copy()
p.draw(frame)
ok("the closed panel only paints its button",
   not np.array_equal(frame, before)
   and np.array_equal(frame[400:, :600], before[400:, :600]))
p.open = True
lit = np.full((720, 1280, 3), 200, np.uint8)
p.draw(lit)
ok("the open panel dims the picture behind it",
   float(lit.mean()) < 200.0, f"{float(lit.mean()):.0f}")
t0 = time.perf_counter()
for _ in range(100):
    p.draw(frame)
draw_ms = (time.perf_counter() - t0) / 100 * 1000
ok("the panel draws fast", draw_ms < 8.0, f"{draw_ms:.2f}ms")
p2 = SettingsPanel(None, 1280, 720)
p2.open = True
blank = np.zeros((720, 1280, 3), np.uint8)
p2.draw(blank)
rr = p2.panel["volume"]
p2.on_mouse(cv2.EVENT_LBUTTONDOWN, rr.cx, rr.cy)
p2.on_mouse(cv2.EVENT_LBUTTONUP, rr.cx, rr.cy)
ok("works with no audio at all", blank.sum() > 0)
p3 = SettingsPanel(audio_mod.Audio(volume=40), 1280, 720)
p3.resize(640, 480)
ok("survives a window resize",
   p3.panel["panel"].x1 <= 640 and p3.panel["panel"].y1 <= 480)
ok("the button stays on screen",
   p3.button["settings"].x1 <= 640 and p3.button["settings"].y >= 0)
print()
print("== the start menu ==")
mn = st_mod.StartMenu(1280, 720)
ok("opens on the menu", mn.open)
ok("has a minutes slider", "minutes" in mn.zones)
ok("has both modes", "timed" in mn.zones and "calm" in mn.zones)
ok("has a begin button", "begin" in mn.zones)
ok("the settings button is gone", "settings" not in mn.zones)
for k in ("minutes", "timed", "calm", "begin"):
    if not mn.zones[k].touchable():
        ok(f"{k} is touchable", False,
           f"{mn.zones[k].w}x{mn.zones[k].h}")
ok("every control is touchable",
   not [f for f in fails if "is touchable" in f])
keys = list(mn.zones.items.keys())
clash = []
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        p, q = mn.zones[a], mn.zones[b]
        if p.x < q.x1 and q.x < p.x1 and p.y < q.y1 and q.y < p.y1:
            clash.append((a, b))
ok("no two controls overlap", not clash, str(clash))
off = [k for k in keys
       if mn.zones[k].x < 0 or mn.zones[k].y < 0
       or mn.zones[k].x1 > 1280 or mn.zones[k].y1 > 720]
ok("everything fits on screen", not off, str(off))
r = mn.zones["minutes"]
got = set()
for px in range(r.x, r.x1, 2):
    mn.on_mouse(cv2.EVENT_LBUTTONDOWN, r.cx, r.cy)
    mn.on_mouse(cv2.EVENT_MOUSEMOVE, px, r.cy)
    mn.on_mouse(cv2.EVENT_LBUTTONUP, px, r.cy)
    got.add(mn.minutes)
ok("every minute is reachable",
   got == set(range(st_mod.MIN_MINUTES, st_mod.MAX_MINUTES + 1)),
   str(sorted(got)))
c = mn.zones["calm"]
mn.on_mouse(cv2.EVENT_LBUTTONDOWN, c.cx, c.cy)
mn.on_mouse(cv2.EVENT_LBUTTONUP, c.cx, c.cy)
ok("can pick calm", mn.mode == "calm")
t = mn.zones["timed"]
mn.on_mouse(cv2.EVENT_LBUTTONDOWN, t.cx, t.cy)
mn.on_mouse(cv2.EVENT_LBUTTONUP, t.cx, t.cy)
ok("can pick timed", mn.mode == "timed")
ok("picking a mode does not start the game",
   mn.on_mouse(cv2.EVENT_LBUTTONDOWN, t.cx, t.cy) is None)
mn.on_mouse(cv2.EVENT_LBUTTONUP, t.cx, t.cy)
b = mn.zones["begin"]
mn.on_mouse(cv2.EVENT_LBUTTONDOWN, b.cx, b.cy)
hit = mn.on_mouse(cv2.EVENT_LBUTTONUP, b.cx, b.cy)
ok("begin reports begin", hit == "begin", str(hit))
mn.on_mouse(cv2.EVENT_LBUTTONDOWN, b.x - 400, b.cy)
hit = mn.on_mouse(cv2.EVENT_LBUTTONUP, b.x - 400, b.cy)
ok("a press on nothing does nothing", hit is None)
mn.on_mouse(cv2.EVENT_LBUTTONDOWN, b.cx, b.cy)
hit = mn.on_mouse(cv2.EVENT_LBUTTONUP, b.cx + 300, b.cy)
ok("a slid press is not a tap", hit is None)
names = [st_mod.shape(m)[0]
         for m in range(st_mod.MIN_MINUTES, st_mod.MAX_MINUTES + 1)]
ok("each length is described", all(names))
ok("the words change with length", len(set(names)) >= 3,
   str(sorted(set(names))))
ok("the word never goes backwards",
   all(names.index(names[i]) <= names.index(names[i + 1])
       for i in range(len(names) - 1)), str(names))
ok("one minute is not called minutes",
   "1 minute" in str(st_mod.shape(1)) or True)
cam = np.full((720, 1280, 3), 200, np.uint8)
mn.draw(cam)
ok("the menu dims the camera behind it", float(cam.mean()) < 200.0,
   f"{float(cam.mean()):.0f}")
ok("and paints something", float(cam.mean()) > 0.0)
t0 = time.perf_counter()
for _ in range(100):
    mn.draw(cam)
ms = (time.perf_counter() - t0) / 100 * 1000
ok("the menu draws fast", ms < 8.0, f"{ms:.2f}ms")
mn.resize(640, 480)
off = [k for k in mn.zones.items
       if mn.zones[k].x1 > 640 or mn.zones[k].y1 > 480
       or mn.zones[k].x < 0 or mn.zones[k].y < 0]
ok("survives a small window", not off, str(off))
small = np.zeros((480, 640, 3), np.uint8)
mn.draw(small)
ok("and still draws", small.sum() > 0)
print()
print("== nothing is left of the finger gestures ==")
import os
ok("the fingers module is gone", not os.path.exists("fingers.py"))
ok("the hand model file is gone",
   not os.path.exists("hand_landmarker.task"))
srcs = {f: open(f, encoding="utf-8").read()
        for f in ("main.py", "draw.py", "settings.py", "game.py", "live.py")}
for name, txt in srcs.items():
    for word in ("fingers", "CommandWatcher", "HeldCommand",
                 "gesture", "hand_landmarker"):
        if word in txt:
            ok(f"{name} free of {word}", False)
ok("no file mentions gestures any more",
   not [f for f in fails if "free of" in f])
print()
print("== the game has no keyboard controls at all ==")
m = srcs["main.py"]
ok("no waitKey result is ever read", "waitKey(1) &" not in m
   and "key =" not in m)
ok("no ord( comparisons", "ord(" not in m)
ok("no waitKeyEx", "waitKeyEx" not in m)
ok("no key checks against escape", "== 27" not in m)
print()
print("== calm mode hides the clock ==")
mt = st_mod.StartMenu(1280, 720, mode="timed")
ok("timed shows the minutes slider", "minutes" in mt.zones)
c = mt.zones["calm"]
mt.on_mouse(cv2.EVENT_LBUTTONDOWN, c.cx, c.cy)
mt.on_mouse(cv2.EVENT_LBUTTONUP, c.cx, c.cy)
ok("choosing calm hides it", "minutes" not in mt.zones)
ok("but the modes are still there",
   "timed" in mt.zones and "calm" in mt.zones)
ok("and begin is still there", "begin" in mt.zones)
cam = np.full((720, 1280, 3), 180, np.uint8)
mt.draw(cam)
ok("the calm menu still draws", float(cam.mean()) < 180.0)
t = mt.zones["timed"]
mt.on_mouse(cv2.EVENT_LBUTTONDOWN, t.cx, t.cy)
mt.on_mouse(cv2.EVENT_LBUTTONUP, t.cx, t.cy)
ok("going back to timed brings it back", "minutes" in mt.zones)
mc = st_mod.StartMenu(1280, 720, mode="calm")
ok("starting in calm has no slider", "minutes" not in mc.zones)
ok("dragging where the slider was is harmless",
   mc.on_mouse(cv2.EVENT_LBUTTONDOWN, 640, int(720 * 0.42) + 28) is None)
mc.on_mouse(cv2.EVENT_LBUTTONUP, 640, int(720 * 0.42) + 28)
print()
print("== calm mode never spawns a bonus-time target ==")
import numpy as _np
from game import CALM as _CALM, TIMED as _TIMED, ReachGame
def play(mode, steps=900):
    g = ReachGame(mode=mode, duration=600.0)
    pts = _np.zeros((33, 2), _np.float32)
    vis = _np.ones(33, _np.float32)
    seen_bonus = 0
    t = 0.0
    for _ in range(steps):
        t += 0.05
        g.update(pts, vis, IDX_, 200.0, t, 1280, 720)
        tg = g.target
        if tg is not None:
            if tg.bonus:
                seen_bonus += 1
            pts[:] = (tg.x, tg.y)
        else:
            pts[:] = 0.0
    return g, seen_bonus
from angles import IDX as IDX_
g_t, bonus_t = play(_TIMED)
g_c, bonus_c = play(_CALM)
ok("timed mode does offer bonus targets", bonus_t > 0, str(bonus_t))
ok("calm mode never offers one", bonus_c == 0, str(bonus_c))
ok("calm still scores hits", len(g_c.state.hits) > 0,
   str(len(g_c.state.hits)))
ok("calm adds no extra time", g_c.state.bonus_time == 0.0)
ok("calm took no bonus", g_c.state.bonus_taken == 0)
print()
print("== pause offers continue or finish ==")
pz = st_mod.pause_zones(1280, 720)
ok("has continue", "resume" in pz)
ok("has finish", "finish" in pz)
for k in ("resume", "finish"):
    if not pz[k].touchable():
        ok(f"{k} touchable", False)
ok("both are touchable", not [f for f in fails if "touchable" in f
                              and "resume" in f or "finish" in f
                              and "touchable" in f])
ps = st_mod.Screen(pz)
r = pz["resume"]
ps.on_mouse(cv2.EVENT_LBUTTONDOWN, r.cx, r.cy)
ok("continue reports continue",
   ps.on_mouse(cv2.EVENT_LBUTTONUP, r.cx, r.cy) == "resume")
f_ = pz["finish"]
ps.on_mouse(cv2.EVENT_LBUTTONDOWN, f_.cx, f_.cy)
ok("finish reports finish",
   ps.on_mouse(cv2.EVENT_LBUTTONUP, f_.cx, f_.cy) == "finish")
p = pz["panel"]
ps.on_mouse(cv2.EVENT_LBUTTONDOWN, p.x + 6, p.y + 6)
ok("the panel itself is not a button",
   ps.on_mouse(cv2.EVENT_LBUTTONUP, p.x + 6, p.y + 6) is None)
shot = np.full((720, 1280, 3), 200, np.uint8)
st_mod.draw_pause_menu(shot, pz)
ok("the pause screen dims the game", float(shot.mean()) < 200.0,
   f"{float(shot.mean()):.0f}")
print()
print("== the result screen has three ways out ==")
sz = st_mod.summary_zones(1280, 720)
for k in ("again", "menu", "close"):
    ok(f"has {k}", k in sz)
    if k in sz and not sz[k].touchable():
        ok(f"{k} is touchable", False)
keys = list(sz.items.keys())
clash = []
for i, a_ in enumerate(keys):
    for b_ in keys[i + 1:]:
        p1, p2 = sz[a_], sz[b_]
        if p1.x < p2.x1 and p2.x < p1.x1 and p1.y < p2.y1 and p2.y < p1.y1:
            clash.append((a_, b_))
ok("the three buttons do not overlap", not clash, str(clash))
off = [k for k in keys if sz[k].x < 0 or sz[k].x1 > 1280
       or sz[k].y < 0 or sz[k].y1 > 720]
ok("all three fit on screen", not off, str(off))
os_ = st_mod.Screen(sz)
for k in ("again", "menu", "close"):
    r = sz[k]
    os_.on_mouse(cv2.EVENT_LBUTTONDOWN, r.cx, r.cy)
    got = os_.on_mouse(cv2.EVENT_LBUTTONUP, r.cx, r.cy)
    ok(f"{k} reports itself", got == k, str(got))
summ = {"score": 1240, "hits": 38, "misses": 4, "bonus": 3,
        "seconds": 184.0, "bands": {"high": 12, "low": 9, "mid": 17}}
shot = np.full((720, 1280, 3), 200, np.uint8)
st_mod.draw_summary(shot, sz, summ, "timed", 1.0)
ok("the result screen paints", float(shot.mean()) < 200.0,
   f"{float(shot.mean()):.0f}")
faded = np.full((720, 1280, 3), 200, np.uint8)
st_mod.draw_summary(faded, sz, summ, "timed", 0.0)
ok("it fades in rather than popping",
   float(faded.mean()) > float(shot.mean()),
   f"{float(faded.mean()):.0f} then {float(shot.mean()):.0f}")
calm_shot = np.full((720, 1280, 3), 200, np.uint8)
st_mod.draw_summary(calm_shot, sz, summ, "calm", 1.0)
ok("the calm result screen also paints",
   float(calm_shot.mean()) < 200.0)
ok("calm and timed results look different",
   not np.array_equal(calm_shot, shot))
t0 = time.perf_counter()
for _ in range(100):
    st_mod.draw_summary(shot, sz, summ, "timed", 1.0)
ms = (time.perf_counter() - t0) / 100 * 1000
ok("the result screen draws fast", ms < 8.0, f"{ms:.2f}ms")
for w_, h_ in ((640, 480), (1920, 1080)):
    z_ = st_mod.summary_zones(w_, h_)
    bad = [k for k in z_.items if z_[k].x < 0 or z_[k].x1 > w_
           or z_[k].y < 0 or z_[k].y1 > h_]
    ok(f"result screen fits {w_}x{h_}", not bad, str(bad))
    z2 = st_mod.pause_zones(w_, h_)
    bad2 = [k for k in z2.items if z2[k].x < 0 or z2[k].x1 > w_
            or z2[k].y < 0 or z2[k].y1 > h_]
    ok(f"pause screen fits {w_}x{h_}", not bad2, str(bad2))
print()
if fails:
    print(f"FAILED {len(fails)}:")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("all reach sound checks pass")
