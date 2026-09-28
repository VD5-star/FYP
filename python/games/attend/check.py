import threading
import time
import cv2
import numpy as np
import audio as audio_mod
import draw
import scenes as scenes_mod
import session as S
import sounds
from main import MENU, OVER, RUN, App
from session import Session
from ui import TOUCH_MIN, Rect, Zones
DOWN = cv2.EVENT_LBUTTONDOWN
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
MINUTES = list(range(S.MIN_MINUTES, S.MAX_MINUTES + 1))
print("== the session follows the exercise, in order ==")
for m in (S.MIN_MINUTES, S.DEFAULT_MINUTES, S.MAX_MINUTES):
    s = Session(minutes=m, seed=3)
    stages = [st.stage for st in s.steps]
    ok(f"{m}min starts with single focus", stages[0] == S.FOCUS)
    ok(f"{m}min has one focus per sound",
       stages.count(S.FOCUS) == len(sounds.NAMES),
       f"{stages.count(S.FOCUS)}")
    ok(f"{m}min then switches", stages.count(S.SWITCH) >= S.MIN_SWITCHES,
       f"{stages.count(S.SWITCH)}")
    ok(f"{m}min ends with divide then settle",
       stages[-2] == S.DIVIDE and stages[-1] == S.SETTLE,
       f"{stages[-2]}, {stages[-1]}")
    first_switch = stages.index(S.SWITCH)
    ok(f"{m}min all focus comes before all switching",
       all(x == S.FOCUS for x in stages[:first_switch]))
    ok(f"{m}min divide comes last",
       stages.index(S.DIVIDE) == len(stages) - 2)
print()
print("== focus visits every sound once ==")
for m in MINUTES:
    for seed in (1, 2, 3, 7, 11):
        s = Session(minutes=m, seed=seed)
        seen = [st.target for st in s.steps if st.stage == S.FOCUS]
        if sorted(seen) != sorted(sounds.NAMES):
            ok(f"{m}min seed {seed} covers all", False, str(seen))
ok("every sound is introduced once", not [f for f in fails
                                          if "covers all" in f])
print()
print("== switching never asks for the sound you are already on ==")
bad = 0
for seed in range(40):
    s = Session(minutes=S.MAX_MINUTES, seed=seed)
    prev = None
    for st in s.steps:
        if st.target is not None:
            if st.target == prev:
                bad += 1
            prev = st.target
ok("never repeats back to back", bad == 0, f"{bad} repeats in 40 sessions")
print()
print("== switching is unpredictable ==")
counts = {n: 0 for n in sounds.NAMES}
for seed in range(60):
    s = Session(minutes=S.MAX_MINUTES, seed=seed)
    for st in s.steps:
        if st.stage == S.SWITCH:
            counts[st.target] += 1
lo, hi = min(counts.values()), max(counts.values())
ok("all sounds get used", lo > 0, str(counts))
ok("roughly even", hi / max(1, lo) < 1.6, f"{lo}-{hi}")
print()
print("== the session lasts exactly what it promises ==")
worst = 0.0
for m in MINUTES:
    for seed in (0, 1, 5, 9):
        s = Session(minutes=m, seed=seed)
        worst = max(worst, abs(s.total - m * 60))
ok("every length is exact to the second", worst < 0.5,
   f"worst drift {worst:.3f}s over {len(MINUTES) * 4} sessions")
for m in (S.MIN_MINUTES, S.DEFAULT_MINUTES, S.MAX_MINUTES):
    s = Session(minutes=m, seed=1)
    ok(f"{m} minutes reads {S.clock(m*60)}",
       S.clock(s.total) == S.clock(m * 60), S.clock(s.total))
ok("asking for less than the minimum is clamped",
   Session(minutes=1).minutes == S.MIN_MINUTES)
ok("asking for more than the maximum is clamped",
   Session(minutes=99).minutes == S.MAX_MINUTES)
print()
print("== every length is a real exercise, not a squashed one ==")
for m in MINUTES:
    s = Session(minutes=m, seed=1)
    holds = {st.stage: st.hold for st in s.steps}
    focus = [st.hold for st in s.steps if st.stage == S.FOCUS][0]
    switch = [st.hold for st in s.steps if st.stage == S.SWITCH][0]
    if focus < 10.0:
        ok(f"{m}min focus long enough", False, f"{focus:.1f}s")
    if switch < 4.0 or switch > 9.0:
        ok(f"{m}min switch sensible", False, f"{switch:.1f}s")
    if holds[S.DIVIDE] < 30.0:
        ok(f"{m}min divide long enough", False, f"{holds[S.DIVIDE]:.1f}s")
    if holds[S.SETTLE] < S.SETTLE_MIN - 0.01:
        ok(f"{m}min settle long enough", False, f"{holds[S.SETTLE]:.1f}s")
ok("no stage is ever squashed",
   not [f for f in fails if "long enough" in f or "sensible" in f])
lens = [Session(minutes=m, seed=1).total for m in MINUTES]
ok("longer setting is always a longer session",
   all(lens[i] < lens[i + 1] for i in range(len(lens) - 1)),
   str([int(v) for v in lens]))
print()
print("== each length is described to the user ==")
names = []
for m in MINUTES:
    name, detail = S.shape(m)
    names.append(name)
    if not name or not detail:
        ok(f"{m}min has words", False)
ok("every length has a word and a line", len(names) == len(MINUTES))
ok("the words change as it gets longer", len(set(names)) >= 3,
   str(sorted(set(names))))
ok("the shortest and longest read differently",
   S.shape(S.MIN_MINUTES)[0] != S.shape(S.MAX_MINUTES)[0],
   f"{S.shape(S.MIN_MINUTES)[0]} vs {S.shape(S.MAX_MINUTES)[0]}")
order = [S.shape(m)[0] for m in MINUTES]
ok("the word never goes backwards",
   all(order.index(order[i]) <= order.index(order[i + 1])
       for i in range(len(order) - 1)), str(order))
print()
print("== time moves forward and ends exactly once ==")
s = Session(minutes=S.MIN_MINUTES, seed=5)
total = s.total
t = 0.0
ends = 0
guard = 0
while not s.finished and guard < 100000:
    guard += 1
    s.advance(1 / 60)
    t += 1 / 60
    if s.finished:
        ends += 1
ok("it finishes", s.finished)
ok("after about the right time", abs(t - total) < 1.0,
   f"{t:.1f}s vs {total:.1f}s")
ok("progress reaches the end", abs(s.progress - 1.0) < 1e-6)
ok("nothing left", s.remaining < 1e-6)
ok("advancing again does nothing", not s.advance(1.0))
print()
print("== a long stall cannot skip the whole session ==")
s = Session(minutes=S.MIN_MINUTES, seed=5)
s.advance(10.0)
ok("one huge jump is capped", not s.finished, f"index {s.index}")
ok("progress is still small", s.progress < 0.25,
   f"{s.progress*100:.0f}%")
print()
print("== pause really stops time ==")
s = Session(minutes=S.MIN_MINUTES, seed=5)
s.advance(3.0)
p = s.position
s.toggle_pause()
for _ in range(100):
    s.advance(1 / 60)
ok("paused means frozen", abs(s.position - p) < 1e-9,
   f"{p:.3f} -> {s.position:.3f}")
s.toggle_pause()
s.advance(1.0)
ok("resuming moves again", s.position > p)
print()
print("== no sound is ever turned down ==")
s = Session(minutes=S.MAX_MINUTES, seed=2)
worst = 1.0
while not s.finished:
    g = s.gains()
    worst = min(worst, min(g.values()))
    s.advance(0.2)
ok("every sound stays at full level the whole time", worst == 1.0,
   f"quietest was {worst}")
src = open("session.py", encoding="utf-8").read()
ok("nothing ducks the other sounds", "duck" not in src.lower())
print()
print("== the picture follows the sound you are asked to hear ==")
s = Session(minutes=S.MIN_MINUTES, seed=4)
s.advance(S.CROSSFADE + 0.1)
w = s.scene_weights()
ok("weights sum to one", abs(sum(w.values()) - 1.0) < 1e-5,
   f"{sum(w.values()):.4f}")
ok("the asked sound is the picture", w[s.target] > 0.99,
   f"{w[s.target]:.2f}")
ok("no negative weight", min(w.values()) >= 0.0)
s2 = Session(minutes=S.MIN_MINUTES, seed=4)
s2.advance(s2.steps[0].hold + 0.05, clamp=False)
w2 = s2.scene_weights()
mixed = sum(1 for v in w2.values() if v > 0.01)
ok("pictures cross-fade, never cut", mixed == 2, f"{mixed} pictures showing")
ok("still sums to one", abs(sum(w2.values()) - 1.0) < 1e-5)
s2b = Session(minutes=S.MIN_MINUTES, seed=4)
blend_frames = 0
while not s2b.finished:
    s2b.advance(1 / 60)
    if sum(1 for v in s2b.scene_weights().values() if v > 0.01) > 1:
        blend_frames += 1
ok("blending happens in real play", blend_frames > 0,
   f"{blend_frames} frames blended")
ok("but most frames are a single picture",
   blend_frames < s2b.total * 60 * 0.5,
   f"{blend_frames} of {int(s2b.total*60)}")
print()
print("== at the end all pictures share the screen ==")
s = Session(minutes=S.MIN_MINUTES, seed=4)
while s.stage != S.DIVIDE and not s.finished:
    s.advance(0.2)
w = s.scene_weights()
ok("every picture is present", all(v > 0 for v in w.values()), str(w))
ok("evenly", max(w.values()) - min(w.values()) < 1e-6)
print()
print("== the scenes are calm and dark enough for a dim room ==")
sc = scenes_mod.render_all(560, 360)
for n in sounds.NAMES:
    img = sc[n]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"       {n:6}: brightness {g.mean():5.1f}  "
          f"darkest {g.min():3}  brightest {g.max():3}")
    ok(f"{n} is not glaring", g.mean() <= scenes_mod.MAX_LUMA + 1,
       f"{g.mean():.1f}")
    ok(f"{n} is not pure black", g.mean() > 12, f"{g.mean():.1f}")
    ok(f"{n} has some texture", float(g.std()) > 6,
       f"std {float(g.std()):.1f}")
    ok(f"{n} no blown highlights",
       float((g > 245).mean()) < 0.01,
       f"{float((g>245).mean())*100:.2f}% near white")
print()
print("== the five pictures look different from each other ==")
small = {n: cv2.resize(sc[n], (12, 8)).astype(np.float32).ravel()
         for n in sounds.NAMES}
worst = (1e9, "", "")
for i, a in enumerate(sounds.NAMES):
    for b in sounds.NAMES[i + 1:]:
        d = float(np.linalg.norm(small[a] - small[b])
                  / np.sqrt(small[a].size))
        if d < worst[0]:
            worst = (d, a, b)
ok("even the closest pair differs", worst[0] > 6.0,
   f"{worst[1]} vs {worst[2]} = {worst[0]:.1f}")
print()
print("== the screen text stays readable over every picture ==")
canvas = np.zeros((360, 560, 3), np.uint8)
for n in sounds.NAMES:
    canvas[:] = sc[n]
    draw.darken(canvas, draw.VEIL)
    g = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    bg = float(g.mean())
    fg = 0.114 * draw.TEXT[0] + 0.587 * draw.TEXT[1] + 0.299 * draw.TEXT[2]
    ratio = (fg + 12) / (bg + 12)
    ok(f"text stands out on {n}", ratio > 2.6,
       f"text {fg:.0f} vs picture {bg:.0f} = {ratio:.1f}x")
print()
print("== every control is big enough to touch, and none overlap ==")
a = App(560, 420, audio_on=False)
for label, mk in (("menu", a.menu_zones), ("running", a.run_zones),
                  ("stop menu", a.pause_zones), ("done", a.over_zones)):
    mk()
    small_z = [(k, r.w, r.h) for k, r in a.zones.items.items()
               if not r.touchable()]
    ok(f"{label} targets >= {TOUCH_MIN}px", not small_z, str(small_z))
    items = [(k, r) for k, r in a.zones.items.items() if k != "panel"]
    clash = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ka, ra = items[i]
            kb, rb = items[j]
            if (ra.x < rb.x1 and rb.x < ra.x1
                    and ra.y < rb.y1 and rb.y < ra.y1):
                clash.append((ka, kb))
    ok(f"{label} zones disjoint", not clash, str(clash))
    out = [k for k, r in a.zones.items.items()
           if r.x < 0 or r.y < 0 or r.x1 > 560 or r.y1 > 420]
    ok(f"{label} zones on screen", not out, str(out))
print()
print("== it is touch only ==")
src = open("main.py", encoding="utf-8").read()
ok("no key handler", "def menu_key" not in src and "def game_key" not in src)
ok("no waitKeyEx", "waitKeyEx" not in src)
ok("only the close key remains", src.count("waitKey(") == 1)
ok("no ord( comparisons", "ord(" not in src)
print()
print("== the whole thing runs by touch ==")
a = App(700, 520, audio_on=False)
ok("starts in the menu", a.mode == MENU)
ok("waits before it is ready", not a.ready)
a.wait_ready(90)
ok("becomes ready", a.ready)
r = a.zones["minutes"]
ok("the minutes slider is there", "minutes" in a.zones)
ok("it is touchable", r.touchable())
a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.x + 4, r.cy, 0, None)
a.on_mouse(cv2.EVENT_MOUSEMOVE, r.x1 - 4, r.cy, 0, None)
a.on_mouse(cv2.EVENT_LBUTTONUP, r.x1 - 4, r.cy, 0, None)
ok("dragging right gives the longest", a.minutes == S.MAX_MINUTES,
   str(a.minutes))
a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.cx, r.cy, 0, None)
a.on_mouse(cv2.EVENT_MOUSEMOVE, r.x + 4, r.cy, 0, None)
a.on_mouse(cv2.EVENT_LBUTTONUP, r.x + 4, r.cy, 0, None)
ok("dragging left gives the shortest", a.minutes == S.MIN_MINUTES,
   str(a.minutes))
reach = set()
for px in range(r.x, r.x1, 2):
    a.on_mouse(cv2.EVENT_LBUTTONDOWN, r.cx, r.cy, 0, None)
    a.on_mouse(cv2.EVENT_MOUSEMOVE, px, r.cy, 0, None)
    a.on_mouse(cv2.EVENT_LBUTTONUP, px, r.cy, 0, None)
    reach.add(a.minutes)
ok("every minute from 3 to 10 is reachable",
   reach == set(range(S.MIN_MINUTES, S.MAX_MINUTES + 1)),
   str(sorted(reach)))
a.minutes = 6
tap_zone(a, "begin")
ok("the session matches the slider", a.sess.total == 6 * 60,
   f"{a.sess.total:.0f}s")
tap_zone(a, "open")
tap_zone(a, "home")
tap_zone(a, "begin")
ok("begins", a.mode == RUN and a.sess is not None)
ok("starts at the first step", a.sess.index == 0)
ok("one round button and nothing else",
   sorted(a.zones.items) == ["open"], str(sorted(a.zones.items)))
o = a.zones["open"]
ok("it is a circle", o.w == o.h, f"{o.w}x{o.h}")
ok("on the right, halfway down",
   o.cx > a.w * 0.8 and abs(o.cy - a.h // 2) <= 8, f"{o.cx},{o.cy}")
tap_zone(a, "open")
ok("it pauses the session", a.sess.paused)
for k in ("volume", "resume", "again", "home", "quit"):
    ok(f"the stop menu holds {k}", k in a.zones)
tap_zone(a, "resume")
ok("keep going resumes", not a.sess.paused)
ok("and puts the button back", sorted(a.zones.items) == ["open"])
tap_zone(a, "open")
tap_zone(a, "home")
ok("back returns to the menu", a.mode == MENU and a.sess is None)
print()
print("== it reaches the end and offers to go again ==")
a = App(700, 520, minutes=S.MIN_MINUTES, seed=5, audio_on=False)
a.wait_ready(90)
tap_zone(a, "begin")
t = a.last_t
guard = 0
while a.mode != OVER and guard < 40000:
    guard += 1
    t += 0.2
    a.frame(t)
ok("it finishes", a.mode == OVER, f"mode={a.mode}")
ok("again is offered", "again" in a.zones)
t += 2.0
a.frame(t)
tap_zone(a, "again")
ok("again starts a fresh session", a.mode == RUN and a.sess.index == 0)
print()
print("== every frame renders, in every state ==")
a = App(700, 520, minutes=S.MIN_MINUTES, seed=2, audio_on=False)
a.wait_ready(90)
a.frame()
ok("menu draws", int(a.canvas.sum()) > 0)
tap_zone(a, "begin")
seen = set()
t = a.last_t
blank = 0
steps_needed = int(a.sess.total / 0.2) + 60
for _ in range(steps_needed):
    t += 0.2
    a.frame(t)
    if a.sess is not None:
        seen.add(a.sess.stage)
    if int(a.canvas.sum()) == 0:
        blank += 1
    if a.mode == OVER:
        break
ok("no blank frames", blank == 0, f"{blank} blank")
ok("all four stages were shown",
   {S.FOCUS, S.SWITCH, S.DIVIDE, S.SETTLE} <= seen, str(sorted(seen)))
ok("it reached the end", a.mode == OVER, f"mode={a.mode}")
print()
print("== speed ==")
a = App(1100, 700, minutes=S.MAX_MINUTES, seed=1, audio_on=False)
a.wait_ready(90)
t = a.last_t
tap_zone(a, "begin")
for _ in range(5):
    t += 1 / 60
    a.frame(t)
best = 1e9
for _ in range(4):
    t0 = time.perf_counter()
    for _ in range(30):
        t += 1 / 60
        a.frame(t)
    best = min(best, (time.perf_counter() - t0) / 30 * 1000)
ok("a frame draws under 16ms", best < 16.0, f"{best:.2f}ms")
a2 = App(1100, 700, audio_on=False)
a2.wait_ready(90)
t0 = time.perf_counter()
for _ in range(30):
    a2.frame()
ok("menu draws under 16ms", (time.perf_counter() - t0) / 30 * 1000 < 16.0,
   f"{(time.perf_counter()-t0)/30*1000:.2f}ms")
t0 = time.perf_counter()
a3 = App(1100, 700, audio_on=False)
ctor = (time.perf_counter() - t0) * 1000
ok("opening the app is instant", ctor < 120, f"{ctor:.0f}ms")
t0 = time.perf_counter()
a3.wait_ready(90)
ok("ready within 8s", (time.perf_counter() - t0) * 1000 < 8000,
   f"{(time.perf_counter()-t0)*1000:.0f}ms")
print()
print("== audio ==")
au = audio_mod.Audio(volume=70)
first = au.render(audio_mod.BLOCK)
ok("silent before it is built", float(np.abs(first).max()) == 0.0)
ok("builds", au.wait(90))
au.fade = 1.0
au.gain = au.target
buf = np.concatenate([au.render(audio_mod.BLOCK) for _ in range(300)])
ok("makes sound", float(np.abs(buf).max()) > 0.01,
   f"peak {float(np.abs(buf).max()):.3f}")
ok("never clips", float(np.abs(buf).max()) <= 1.0)
ok("no limiting needed", au.clipped == 0, f"{au.clipped}")
ok("no sudden jumps", float(np.abs(np.diff(buf[:, 0])).max()) < 0.25,
   f"max step {float(np.abs(np.diff(buf[:,0])).max()):.3f}")
lr = abs(float(np.sqrt((buf[:, 0] ** 2).mean()))
         - float(np.sqrt((buf[:, 1] ** 2).mean())))
ok("left and right are balanced", lr < 0.02, f"{lr:.4f}")
ok("really stereo", not np.allclose(buf[:, 0], buf[:, 1]))
au2 = audio_mod.Audio(volume=100)
au2.wait(90)
first = au2.render(audio_mod.BLOCK)
ok("fades in from silence", float(np.abs(first).max()) < 0.02,
   f"{float(np.abs(first).max()):.4f}")
rise = [float(np.abs(au2.render(audio_mod.BLOCK)).max())
        for _ in range(80)]
ok("and rises", rise[-1] > rise[0], f"{rise[0]:.3f} -> {rise[-1]:.3f}")
au3 = audio_mod.Audio(volume=0)
au3.wait(90)
au3.fade = 1.0
s0 = np.concatenate([au3.render(audio_mod.BLOCK) for _ in range(60)])
ok("zero is silent", float(np.abs(s0).max()) == 0.0)
lvls = []
for lv in (10, 30, 50, 70, 100):
    a_ = audio_mod.Audio(volume=lv)
    a_.wait(90)
    a_.fade = 1.0
    a_.gain = a_.target
    s_ = np.concatenate([a_.render(audio_mod.BLOCK) for _ in range(120)])
    lvls.append(float(np.sqrt((s_ ** 2).mean())))
ok("the slider gets louder all the way up",
   all(lvls[i] < lvls[i + 1] for i in range(len(lvls) - 1)),
   str([round(v, 4) for v in lvls]))
au5 = audio_mod.Audio(volume=80)
au5.wait(90)
for v in range(0, 101):
    au5.set_volume(v)
ok("every step 0..100 is accepted", au5.volume == 100)
au5.set_volume(250)
au5.set_volume(-40)
ok("out of range is clamped", au5.volume == audio_mod.VOL_MIN)
au6 = audio_mod.Audio(volume=60)
au6.wait(90)
au6.fade = 1.0
au6.gain = au6.target
au6.render(audio_mod.BLOCK)
au6.stop()
after = np.concatenate([au6.render(audio_mod.BLOCK) for _ in range(4)])
ok("stop rewinds and re-fades", au6.pos <= audio_mod.BLOCK * 4
   and float(np.abs(after[:8]).max()) < 0.02,
   f"pos {au6.pos}, head {float(np.abs(after[:8]).max()):.4f}")
ok("stop is safe to repeat", (au6.stop(), au6.stop(), True)[-1])
t0 = time.perf_counter()
for _ in range(400):
    au.render(audio_mod.BLOCK)
per = (time.perf_counter() - t0) / 400 * 1000
budget = audio_mod.BLOCK / audio_mod.SR * 1000
ok("audio is far inside its budget", per < budget * 0.25,
   f"{per:.3f}ms of {budget:.1f}ms")
import sys
real = sys.modules.get("sounddevice")
sys.modules["sounddevice"] = None
bad_au = audio_mod.Audio(volume=55)
ok("a missing sound card is handled", bad_au.start() is False)
ok("and marked", bad_au.failed)
if real is not None:
    sys.modules["sounddevice"] = real
else:
    del sys.modules["sounddevice"]
a4 = App(560, 420, audio_on=False)
ok("runs with no audio at all", a4.audio is None)
ok("and says so", a4.sound_name() == "no sound")
a4.wait_ready(90)
tap_zone(a4, "begin")
ok("still begins", a4.mode == RUN)
print()
print("== nothing is scored, nothing is kept ==")
a = App(560, 420, audio_on=False)
a.wait_ready(90)
tap_zone(a, "begin")
bad_fields = {"score", "points", "streak", "best", "record", "history",
              "level_reached", "failed", "mistakes", "stats"}
ok("session has no score", not (set(dir(a.sess)) & bad_fields))
ok("app has no score", not (set(dir(a)) & bad_fields))
import os
before = set(os.listdir("."))
t = a.last_t
for _ in range(600):
    t += 1 / 30
    a.frame(t)
if a.mode == RUN:
    tap_zone(a, "open")
    tap_zone(a, "home")
ok("no files written", set(os.listdir(".")) == before,
   str(sorted(set(os.listdir(".")) - before)))
for mod, name in ((sounds, "sounds"), (scenes_mod, "scenes"),
                  (S, "session"), (audio_mod, "audio")):
    bad_fn = [x for x in dir(mod)
              if "save" in x.lower() or "write" in x.lower()]
    ok(f"{name} cannot save anything", not bad_fn, str(bad_fn))
print()
if fails:
    print(f"FAILED {len(fails)}:")
    for f in fails:
        print("   -", f)
else:
    print("all checks pass")
