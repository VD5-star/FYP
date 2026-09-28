import cv2
import numpy as np
from angles import IDX
from game import CALM, TIMED, ReachGame
from targets import (ARM_LENGTH, BODY_GAP, BONUS_EVERY, LIFETIME,
                     TargetSpawner, edge_points)
W, H = 1280, 720
def body(cx=640.0, cy=360.0, torso=180.0, cut_top=False):
    pts = np.full((33, 2), np.nan)
    vis = np.zeros(33)
    top = cy - 150 if not cut_top else 2.0
    pts[IDX["nose"]] = [cx, top]
    vis[IDX["nose"]] = 1.0
    for name, dx, dy in (("leftShoulder", -50, -90), ("rightShoulder", 50, -90),
                         ("leftElbow", -80, 0), ("rightElbow", 80, 0),
                         ("leftWrist", -90, 70), ("rightWrist", 90, 70),
                         ("leftHip", -40, 90), ("rightHip", 40, 90),
                         ("leftKnee", -45, 220), ("rightKnee", 45, 220),
                         ("leftAnkle", -48, 340), ("rightAnkle", 48, 340)):
        pts[IDX[name]] = [cx + dx, cy + dy]
        vis[IDX[name]] = 1.0
    return pts, vis, torso
USED = ["nose", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
        "leftWrist", "rightWrist", "leftHip", "rightHip", "leftKnee",
        "rightKnee", "leftAnkle", "rightAnkle"]
def body_points(pts, vis):
    return np.array([pts[IDX[n]] for n in USED if vis[IDX[n]] >= 0.15])
print("== gap measured to nearest body part, not to old target ==")
sp = TargetSpawner(seed=3)
p, v, torso = body()
bp = body_points(p, v)
anchor = np.array([640.0, 360.0])
gaps = []
fails = 0
for _ in range(500):
    t = sp.spawn(anchor, torso, 0.0, W, H, body=bp)
    if t is None:
        fails += 1
        continue
    d = np.min(np.hypot(bp[:, 0] - t.x, bp[:, 1] - t.y)) / torso
    gaps.append(d)
gaps.sort()
print(f"gap to nearest body part (torso units): min {gaps[0]:.2f} "
      f"p10 {gaps[int(len(gaps)*0.1)]:.2f} med {gaps[len(gaps)//2]:.2f} "
      f"max {gaps[-1]:.2f}")
print(f"target >= {BODY_GAP} (arm = {ARM_LENGTH})")
print("below arm length:", sum(1 for g in gaps if g < ARM_LENGTH * 0.95),
      "/", len(gaps), " spawn failures:", fails,
      " relaxed:", sp.relaxed, " rejected:", sp.rejected)
print()
print("== consecutive targets can be close to each other ==")
sp2 = TargetSpawner(seed=9)
prev = None
close = 0
n = 0
for _ in range(300):
    t = sp2.spawn(anchor, torso, 0.0, W, H, body=bp)
    if t is None:
        continue
    if prev is not None:
        d = np.hypot(t.x - prev.x, t.y - prev.y) / torso
        if d < 1.0:
            close += 1
        n += 1
    prev = t
print(f"consecutive pairs closer than 1 torso: {close}/{n} "
      "(previous rule forbade this entirely)")
print()
print("== body cut at top: nothing spawns past that edge ==")
pc, vc, _ = body(cut_top=True)
bpc = body_points(pc, vc)
fence = edge_points(bpc, torso, W, H)
print("body top y:", round(float(bpc[:, 1].min()), 1),
      " fence points generated:", len(fence))
sp3 = TargetSpawner(seed=4)
above = 0
made = 0
gaps3 = []
for _ in range(500):
    t = sp3.spawn(anchor, torso, 0.0, W, H, body=bpc)
    if t is None:
        continue
    made += 1
    x0, x1 = bpc[:, 0].min() - torso * 0.5, bpc[:, 0].max() + torso * 0.5
    if t.y < bpc[:, 1].min() + torso * 0.5 and x0 <= t.x <= x1:
        above += 1
    gaps3.append(float(np.min(np.hypot(bpc[:, 0] - t.x, bpc[:, 1] - t.y))) / torso)
print(f"spawned in the cut-off zone above the body: {above}/{made}")
print(f"min gap still held: {min(gaps3):.2f}")
print()
print("== normal body: no fence, full circle available ==")
fence_n = edge_points(bp, torso, W, H)
print("fence points when body is fully inside:", len(fence_n))
print()
print("== points scale with distance from body ==")
sp4 = TargetSpawner(seed=6)
by_band = {}
for _ in range(600):
    t = sp4.spawn(anchor, torso, 0.0, W, H, body=bp)
    if t is None:
        continue
    d = np.min(np.hypot(bp[:, 0] - t.x, bp[:, 1] - t.y)) / torso
    by_band.setdefault(t.band, []).append(d)
for band in ("near", "far", "very_far"):
    ds = sorted(by_band.get(band, []))
    if ds:
        print(f"  {band:9} n={len(ds):3} gap {ds[0]:.2f}-{ds[-1]:.2f} "
              f"med {ds[len(ds)//2]:.2f}")
print()
print("== game still plays ==")
g = ReachGame(mode=TIMED, duration=30, seed=7)
p2, v2, _ = body()
g.update(p2, v2, IDX, torso, 0.0, W, H)
print("target exists:", g.target is not None)
t = g.target
d = np.min(np.hypot(bp[:, 0] - t.x, bp[:, 1] - t.y)) / torso
print("gap from body:", round(float(d), 2), "torso units")
p2[IDX["rightWrist"]] = [t.x, t.y]
hits = g.update(p2, v2, IDX, torso, 0.1, W, H)
print("hit:", [(h.band, h.points) for h in hits], "score", g.state.score,
      "new target:", g.target is not t)
print()
print("== bonus every 5 still works ==")
g2 = ReachGame(mode=TIMED, duration=30, seed=11)
p3, v3, _ = body()
now = 0.0
seen = []
for step in range(13):
    now += 0.05
    g2.update(p3, v3, IDX, torso, now, W, H)
    tg = g2.target
    if tg is None:
        continue
    if tg.bonus:
        seen.append(g2.state.streak)
    p3[IDX["rightWrist"]] = [tg.x, tg.y]
    g2.update(p3, v3, IDX, torso, now + 0.01, W, H)
    p3[IDX["rightWrist"]] = [-999, -999]
print("bonus at streaks:", seen, "bonus time:", g2.state.bonus_time,
      "total:", g2.total_time())
print()
print("== shrink over 1.5s ==")
sp5 = TargetSpawner(seed=1)
t5 = sp5.spawn(anchor, torso, 0.0, W, H, body=bp)
for now in (0.0, 0.75, 1.49, 1.5):
    print(f"  t={now:4.2f} scale {t5.scale(now):.2f} "
          f"r {t5.current_radius(now):5.1f} expired {t5.expired(now)}")
print()
print("== touch: any part of the body, not just extremities ==")
from game import LIMBS, LIMB_STEPS
g6 = ReachGame(mode=TIMED, duration=30, seed=2)
p6, v6, _ = body()
g6.update(p6, v6, IDX, torso, 0.0, W, H)
pts6 = g6._touch_points(p6, v6, IDX)
print("landmarks used:", int(v6.sum()), " touch points total:", len(pts6))
print("limb segments:", len(LIMBS), "x", LIMB_STEPS - 1, "interpolated")
mid_upper_arm = (p6[IDX["leftShoulder"]] + p6[IDX["leftElbow"]]) / 2
mid_thigh = (p6[IDX["leftHip"]] + p6[IDX["leftKnee"]]) / 2
mid_chest = (p6[IDX["leftShoulder"]] + p6[IDX["rightHip"]]) / 2
for name, pt in (("mid upper arm", mid_upper_arm), ("mid thigh", mid_thigh)):
    d = np.min(np.hypot(pts6[:, 0] - pt[0], pts6[:, 1] - pt[1]))
    print(f"  {name:14} nearest touch point: {d:.1f}px")
tg6 = g6.target
for name, pt in (("upper arm", mid_upper_arm), ("thigh", mid_thigh)):
    moved = p6.copy()
    shift = np.array([tg6.x, tg6.y]) - pt
    for i in range(33):
        if v6[i] >= 0.15:
            moved[i] = p6[i] + shift
    gg = ReachGame(mode=TIMED, duration=30, seed=2)
    gg.update(p6, v6, IDX, torso, 0.0, W, H)
    t_before = gg.target
    shift = np.array([t_before.x, t_before.y]) - pt
    moved = p6.copy()
    for i in range(33):
        if v6[i] >= 0.15:
            moved[i] = p6[i] + shift
    hits = gg.update(moved, v6, IDX, torso, 0.05, W, H)
    print(f"  touching with {name}: {'HIT' if hits else 'miss'}")
print()
print("== smoothing ==")
from smooth import PointSmoother
sm = PointSmoother()
noisy = []
base, bv, _ = body()
rng = np.random.default_rng(4)
for k in range(40):
    jit = base.copy()
    for i in range(33):
        if bv[i] >= 0.15:
            jit[i] = base[i] + rng.normal(0, 2.0, 2)
    noisy.append(jit)
out = [sm(p, bv, k / 30.0) for k, p in enumerate(noisy)]
i = IDX["leftWrist"]
raw_j = np.median(np.hypot(*np.diff([p[i] for p in noisy], axis=0).T))
sm_j = np.median(np.hypot(*np.diff([p[i] for p in out], axis=0).T))
print(f"wrist jitter {raw_j:.2f}px -> {sm_j:.2f}px "
      f"({(1-sm_j/raw_j)*100:.0f}% less)")
