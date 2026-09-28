from __future__ import annotations
import numpy as np
import cv2
from angles import IDX
from body import BONES, DRAW_THRESHOLD, HEAD
from targets import BONUS, BONUS_SECONDS, FAR, NEAR, VERY_FAR
COLOR_BODY = (235, 235, 235)
COLOR_FAINT = (140, 140, 140)
COLOR_TEXT = (255, 255, 255)
COLOR_SHADOW = (0, 0, 0)
COLOR_ENDING = (120, 120, 255)
BAND_COLORS = {
    NEAR: (120, 220, 120),
    FAR: (240, 200, 90),
    VERY_FAR: (120, 160, 255),
    BONUS: (255, 150, 255),
}
FLASH_SECONDS = 0.35
def clip_to_frame(a, b, width, height):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    direction = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-direction[0], a[0]), (direction[0], width - a[0]),
                 (-direction[1], a[1]), (direction[1], height - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return a + direction * t0, a + direction * t1
def draw_body(image, points, visibility, discarded, unit):
    if points is None or visibility is None:
        return
    h, w = image.shape[:2]
    if discarded is None:
        discarded = np.zeros(len(points), dtype=bool)
    thickness = max(2, int(unit * 1.1))
    for a, b in BONES:
        ia, ib = IDX[a], IDX[b]
        if discarded[ia] or discarded[ib]:
            continue
        if visibility[ia] < DRAW_THRESHOLD or visibility[ib] < DRAW_THRESHOLD:
            continue
        clipped = clip_to_frame(points[ia], points[ib], w, h)
        if clipped is None:
            continue
        pa, pb = clipped
        cv2.line(image, tuple(pa.astype(int)), tuple(pb.astype(int)),
                 COLOR_SHADOW, thickness + 4, cv2.LINE_AA)
        cv2.line(image, tuple(pa.astype(int)), tuple(pb.astype(int)),
                 COLOR_BODY, thickness, cv2.LINE_AA)
    ls, rs = IDX["leftShoulder"], IDX["rightShoulder"]
    lh, rh = IDX["leftHip"], IDX["rightHip"]
    if not any(discarded[i] for i in (ls, rs, lh, rh)) and \
            all(visibility[i] >= DRAW_THRESHOLD for i in (ls, rs, lh, rh)):
        mid_s = (points[ls] + points[rs]) / 2
        mid_h = (points[lh] + points[rh]) / 2
        for pair in ((mid_s, mid_h), (points[ls], points[rs]),
                     (points[lh], points[rh])):
            clipped = clip_to_frame(pair[0], pair[1], w, h)
            if clipped is None:
                continue
            pa, pb = clipped
            cv2.line(image, tuple(pa.astype(int)), tuple(pb.astype(int)),
                     COLOR_SHADOW, thickness + 4, cv2.LINE_AA)
            cv2.line(image, tuple(pa.astype(int)), tuple(pb.astype(int)),
                     COLOR_BODY, thickness, cv2.LINE_AA)
    head = [IDX[n] for n in HEAD
            if not discarded[IDX[n]] and visibility[IDX[n]] >= DRAW_THRESHOLD]
    if head:
        xs = points[head, 0]
        ys = points[head, 1]
        cx = float(xs.mean())
        cy = float(ys.mean())
        r = max(unit * 2.2, float(max(xs.max() - xs.min(),
                                      ys.max() - ys.min())) * 0.9)
        cv2.circle(image, (int(cx), int(cy)), int(r), COLOR_SHADOW,
                   thickness + 4, cv2.LINE_AA)
        cv2.circle(image, (int(cx), int(cy)), int(r), COLOR_BODY,
                   thickness, cv2.LINE_AA)
    for name in ("leftWrist", "rightWrist", "leftAnkle", "rightAnkle",
                 "leftElbow", "rightElbow", "leftKnee", "rightKnee"):
        i = IDX[name]
        if discarded[i] or visibility[i] < DRAW_THRESHOLD:
            continue
        cv2.circle(image, tuple(points[i].astype(int)), max(3, int(unit * 0.8)),
                   COLOR_BODY, -1, cv2.LINE_AA)
def draw_targets(image, targets, now, fade=1.0, colour_override=None):
    for target in targets:
        colour = colour_override or BAND_COLORS.get(target.band, COLOR_BODY)
        cx, cy = int(target.x), int(target.y)
        life = target.life(now)
        r = int(target.current_radius(now) * fade)
        if r < 2:
            continue
        p0 = (cx - r, cy - r)
        p1 = (cx + r, cy + r)
        alpha = 0.30 * fade * (0.30 + 0.70 * life)
        overlay = image.copy()
        cv2.rectangle(overlay, p0, p1, colour, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.rectangle(image, p0, p1, COLOR_SHADOW, 6, cv2.LINE_AA)
        cv2.rectangle(image, p0, p1, colour, 3, cv2.LINE_AA)
        if target.bonus:
            label = f"+{int(BONUS_SECONDS)}s"
        else:
            label = str(target.points)
        scale = max(0.45, r / 40.0)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        origin = (cx - size[0] // 2, cy + size[1] // 2)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    COLOR_SHADOW, 4, cv2.LINE_AA)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    colour, 2, cv2.LINE_AA)
def draw_flash(image, state, now):
    if state.last_hit is None:
        return
    age = now - state.last_hit
    if age > FLASH_SECONDS:
        return
    strength = 1.0 - age / FLASH_SECONDS
    colour = BAND_COLORS.get(state.last_band, COLOR_BODY)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), colour, -1)
    cv2.addWeighted(overlay, 0.18 * strength, image, 1 - 0.18 * strength, 0,
                    image)
def _text(image, label, origin, scale, colour, weight=2):
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                COLOR_SHADOW, weight + 3, cv2.LINE_AA)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, weight, cv2.LINE_AA)
def draw_hud(image, game, now):
    h, w = image.shape[:2]
    _text(image, str(game.state.score), (24, 64), 1.6, COLOR_TEXT, 3)
    streak = game.state.streak % 5
    for i in range(5):
        cx = 30 + i * 18
        colour = BAND_COLORS[BONUS] if i < streak else (70, 70, 70)
        cv2.circle(image, (cx, 96), 5, colour, -1, cv2.LINE_AA)
    if game.mode == "timed":
        left = game.remaining(now)
        label = f"{left:04.1f}"
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        colour = (110, 110, 255) if left <= 10.0 else COLOR_TEXT
        _text(image, label, (w - size[0] - 24, 60), 1.2, colour, 3)
        bar = int((w - 48) * min(1.0, left / max(game.total_time(), 1e-6)))
        cv2.rectangle(image, (24, 78), (w - 24, 86), (60, 60, 60), -1)
        cv2.rectangle(image, (24, 78), (24 + bar, 86), colour, -1)
def draw_waiting(image):
    h, w = image.shape[:2]
    label = "step back so I can see you"
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    _text(image, label, ((w - size[0]) // 2, h - 60), 0.9, COLOR_FAINT)
