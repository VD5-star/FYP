import builtins
import io
import os
import numpy as np
import main
import photo
writes = []
real_open = builtins.open
def watched_open(file, mode="r", *a, **k):
    if any(c in str(mode) for c in ("w", "a", "x", "+")):
        writes.append((str(file), str(mode)))
    return real_open(file, mode, *a, **k)
builtins.open = watched_open
real_save = np.save
def watched_save(file, *a, **k):
    writes.append((str(file), "np.save"))
    return real_save(file, *a, **k)
np.save = watched_save
import cv2
imwrite = cv2.imwrite
def watched_imwrite(path, *a, **k):
    writes.append((str(path), "cv2.imwrite"))
    return imwrite(path, *a, **k)
cv2.imwrite = watched_imwrite
before = {}
for root in (".", os.path.expanduser("~")):
    try:
        before[root] = set(os.listdir(root))
    except OSError:
        pass
app = main.App(1280, 860, "interlock", 3)
app.custom = np.dstack([
    np.tile(np.linspace(10, 250, 700, dtype=np.uint8), (500, 1)),
    np.tile(np.linspace(250, 10, 700, dtype=np.uint8), (500, 1)),
    np.full((500, 700), 130, np.uint8)])
app.start()
b = app.board
guard = 0
while not b.solved and guard < b.total * 4:
    guard += 1
    idx = b.order[-1]
    p = b.pieces[idx]
    gx = gy = None
    for yy in range(p.pos[1], p.pos[1] + p.h):
        for xx in range(p.pos[0], p.pos[0] + p.w):
            if p.hits(xx, yy):
                gx, gy = xx, yy
                break
        if gx is not None:
            break
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, gx, gy, 0, None)
    t = b.target(p)
    dx, dy = t[0] + (gx - p.pos[0]), t[1] + (gy - p.pos[1])
    app.on_mouse(cv2.EVENT_MOUSEMOVE, dx, dy, 0, None)
    app.on_mouse(cv2.EVENT_LBUTTONUP, dx, dy, 0, None)
app.frame(__import__("time").monotonic())
r = app.zones["back"]
app.on_mouse(cv2.EVENT_LBUTTONDOWN, r.cx, r.cy, 0, None)
app.on_mouse(cv2.EVENT_LBUTTONUP, r.cx, r.cy, 0, None)
print("solved:", b.solved, f"{b.done}/{b.total}")
print("custom image still in memory after esc:", app.custom is not None)
print("board dropped:", app.board is None)
print()
print("write calls during a whole custom-image session:", len(writes))
for w in writes:
    print("   ", w)
for root, names in before.items():
    now = set(os.listdir(root))
    new = now - names
    print(f"new entries in {root}: {len(new)} {sorted(new) if new else ''}")
print()
print("photo module has no save/write function:",
      not [n for n in dir(photo) if "save" in n or "write" in n])
print("main module has no save/write function:",
      not [n for n in dir(main) if "save" in n or "write" in n])
