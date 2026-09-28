import time
import numpy as np
from angles import IDX
from body import BodyTracker
from camera import CameraStream, shape_frame
from draw import draw_hud, draw_targets
from game import TIMED, ReachGame
from smooth import PointSmoother
W, H = 1280, 720
stream = CameraStream(0)
if not stream.opened:
    print("camera failed")
    raise SystemExit(1)
t = BodyTracker()
sm = PointSmoother()
g = ReachGame(mode=TIMED, duration=60)
shape_ms, pose_ms, game_ms, draw_ms = [], [], [], []
start = time.perf_counter()
n = 0
seen = 0
last_seq = -1
torsos = []
stalls = 0
gaps = []
seen_targets = set()
while n < 150:
    raw, seq = stream.read()
    if raw is None or seq == last_seq:
        stalls += 1
        time.sleep(0.001)
        continue
    last_seq = seq
    a = time.perf_counter()
    f = shape_frame(raw, W, H, "fill", mirror=True)
    b = time.perf_counter()
    now = time.perf_counter() - start
    t.update(f, now)
    c = time.perf_counter()
    drawn = sm(t.points, t.visibility, now) if t.points is not None else None
    g.update(drawn, t.visibility, IDX, t.torso, now, W, H)
    e = time.perf_counter()
    draw_targets(f, g.targets, now)
    draw_hud(f, g, now)
    fin = time.perf_counter()
    shape_ms.append((b - a) * 1000)
    pose_ms.append((c - b) * 1000)
    game_ms.append((e - c) * 1000)
    draw_ms.append((fin - e) * 1000)
    if t.points is not None:
        seen += 1
    if t.torso:
        torsos.append(t.torso)
    tg = g.target
    if tg is not None and id(tg) not in seen_targets and drawn is not None:
        seen_targets.add(id(tg))
        pts = drawn[(t.visibility >= 0.15)]
        if len(pts):
            gaps.append(float(np.min(np.hypot(pts[:, 0] - tg.x,
                                              pts[:, 1] - tg.y))) / t.torso)
    n += 1
el = time.perf_counter() - start
def show(name, xs):
    xs = sorted(xs)
    print(f"{name:6} avg {sum(xs)/len(xs):6.1f}  p90 {xs[int(len(xs)*0.9)]:6.1f}"
          f"  max {xs[-1]:6.1f}")
print("frames", n, "fps", round(n / el, 1), "pose_seen", seen, "waits", stalls)
show("shape", shape_ms)
show("pose", pose_ms)
show("game", game_ms)
show("draw", draw_ms)
if torsos:
    print("torso px", round(min(torsos), 1), "-", round(max(torsos), 1))
print("score", g.state.score, "hits", len(g.state.hits),
      "misses", g.state.misses, "bonus", g.state.bonus_taken)
print("spawn rejected", g.spawner.rejected, "relaxed", g.spawner.relaxed)
if gaps:
    gs = sorted(gaps)
    print(f"live gap to body: min {gs[0]:.2f} med {gs[len(gs)//2]:.2f} "
          f"max {gs[-1]:.2f} torso units")
stream.release()
t.close()
