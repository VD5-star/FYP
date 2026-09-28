from __future__ import annotations
from collections import OrderedDict
import cv2
import numpy as np
import session as S
import sounds
from ui import Rect, Zones
BG = (30, 28, 26)
TEXT = (226, 222, 216)
DIM = (168, 163, 156)
FAINT = (112, 108, 103)
CARD = (58, 54, 50)
CARD_ON = (86, 96, 100)
EDGE = (104, 99, 94)
ACCENT = (198, 212, 216)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL = (44, 41, 38)
VEIL = 0.42
CACHE_KEEP = 8
SLIDER_PAD = 22
ORB = (13, 12, 10)
DRIFT_ORBS = 3
DRIFT_SPEED = 0.10
DRIFT_SHRINK = 8
_mix_cache: OrderedDict = OrderedDict()
def blend(scenes: dict, weights: dict, w: int, h: int) -> np.ndarray:
    live = [(n, v) for n, v in weights.items() if v > 0.002]
    if not live:
        return np.full((h, w, 3), BG, np.uint8)
    if len(live) == 1:
        return scenes[live[0][0]]
    key = tuple(sorted((n, round(v, 2)) for n, v in live)) + (w, h)
    hit = _mix_cache.get(key)
    if hit is not None:
        _mix_cache.move_to_end(key)
        return hit
    tot = sum(v for _, v in live)
    acc = np.zeros((h, w, 3), np.float32)
    for n, v in live:
        acc += scenes[n].astype(np.float32) * (v / tot)
    out = np.clip(acc, 0, 255).astype(np.uint8)
    _mix_cache[key] = out
    while len(_mix_cache) > CACHE_KEEP:
        _mix_cache.popitem(last=False)
    return out
def darken(canvas: np.ndarray, amount: float) -> None:
    if amount <= 0.001:
        return
    cv2.convertScaleAbs(canvas, dst=canvas, alpha=1.0 - amount, beta=0.0)
def _text_mid(canvas, txt, rect: Rect, scale=0.5, col=TEXT, thick=1):
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, thick)
    cv2.putText(canvas, txt, (rect.cx - tw // 2, rect.cy + th // 2),
                FONT, scale, col, thick, cv2.LINE_AA)
def _centre(canvas, txt, y, scale, col, thick=1):
    h, w = canvas.shape[:2]
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, thick)
    cv2.putText(canvas, txt, ((w - tw) // 2, y), FONT, scale, col, thick,
                cv2.LINE_AA)
    return th
def _round_rect(canvas, r: Rect, col, rad=12, edge=None):
    x, y, x1, y1 = r.x, r.y, r.x1 - 1, r.y1 - 1
    rad = max(0, min(rad, r.w // 2, r.h // 2))
    cv2.rectangle(canvas, (x + rad, y), (x1 - rad, y1), col, -1)
    cv2.rectangle(canvas, (x, y + rad), (x1, y1 - rad), col, -1)
    for cx, cy in ((x + rad, y + rad), (x1 - rad, y + rad),
                   (x + rad, y1 - rad), (x1 - rad, y1 - rad)):
        cv2.circle(canvas, (cx, cy), rad, col, -1, cv2.LINE_AA)
    if edge is None:
        return
    cv2.line(canvas, (x + rad, y), (x1 - rad, y), edge, 1, cv2.LINE_AA)
    cv2.line(canvas, (x + rad, y1), (x1 - rad, y1), edge, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y + rad), (x, y1 - rad), edge, 1, cv2.LINE_AA)
    cv2.line(canvas, (x1, y + rad), (x1, y1 - rad), edge, 1, cv2.LINE_AA)
    for cx, cy, a0, a1 in ((x + rad, y + rad, 180, 270),
                           (x1 - rad, y + rad, 270, 360),
                           (x + rad, y1 - rad, 90, 180),
                           (x1 - rad, y1 - rad, 0, 90)):
        cv2.ellipse(canvas, (cx, cy), (rad, rad), 0, a0, a1, edge, 1,
                    cv2.LINE_AA)
PROMPTS = {
    S.FOCUS: "follow",
    S.SWITCH: "now",
    S.DIVIDE: "hold them all at once",
    S.SETTLE: "let it go",
}
def draw_session(canvas: np.ndarray, sess, scenes: dict,
                 zones: Zones) -> None:
    h, w = canvas.shape[:2]
    canvas[:] = blend(scenes, sess.scene_weights(), w, h)
    darken(canvas, VEIL)
    _drift(canvas, sess, w, h)
    stage = sess.stage
    t = sess.target
    if stage in (S.FOCUS, S.SWITCH):
        label = sounds.LABELS.get(t, t or "")
        _centre(canvas, PROMPTS[stage], h // 2 - 34, 0.62, DIM)
        _centre(canvas, label, h // 2 + 26, 1.25, TEXT)
    elif stage == S.DIVIDE:
        _centre(canvas, PROMPTS[stage], h // 2 + 8, 0.92, TEXT)
    elif stage == S.SETTLE:
        _centre(canvas, PROMPTS[stage], h // 2 + 8, 0.92, DIM)
    _breath(canvas, sess, w, h)
    _bar(canvas, sess, w, h)
    if "panel" in zones:
        return
    _buttons(canvas, zones, sess)
    if sess.paused:
        _paused(canvas, w, h)
def _drift(canvas, sess, w, h) -> None:
    t = sess.elapsed
    sw = max(24, w // DRIFT_SHRINK)
    sh = max(24, h // DRIFT_SHRINK)
    small = np.zeros((sh, sw, 3), np.uint8)
    for i in range(DRIFT_ORBS):
        p = t * DRIFT_SPEED + i * (2 * np.pi / DRIFT_ORBS)
        rx = sw * (0.20 + 0.05 * np.sin(p * 0.37 + i))
        ry = sh * (0.13 + 0.04 * np.cos(p * 0.29 + i * 1.7))
        x = int(sw * 0.5 + rx * np.sin(p))
        y = int(sh * 0.5 + ry * np.sin(p * 0.61 + i * 0.9))
        rad = int(min(sw, sh) * (0.085 + 0.022 * np.sin(p * 0.53 + i)))
        cv2.circle(small, (x, y), max(1, rad), ORB, -1, cv2.LINE_AA)
    cv2.GaussianBlur(small, (0, 0), min(sw, sh) * 0.10, dst=small)
    glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.add(canvas, glow, dst=canvas)
def _breath(canvas, sess, w, h) -> None:
    if sess.stage not in (S.FOCUS, S.SWITCH, S.DIVIDE, S.SETTLE):
        return
    r0 = 16
    grow = 10 if sess.stage in (S.FOCUS, S.SWITCH) else 6
    r = int(r0 + grow * (1.0 - abs(sess.step_progress * 2 - 1)))
    cv2.circle(canvas, (w // 2, h - 108), r, FAINT, 1, cv2.LINE_AA)
    cv2.circle(canvas, (w // 2, h - 108), 3, DIM, -1, cv2.LINE_AA)
def _bar(canvas, sess, w, h) -> None:
    bw = int(w * 0.52)
    x0 = (w - bw) // 2
    y = h - 66
    cv2.line(canvas, (x0, y), (x0 + bw, y), (72, 68, 64), 3, cv2.LINE_AA)
    p = sess.progress
    if p > 0:
        cv2.line(canvas, (x0, y), (x0 + int(bw * p), y), ACCENT, 3,
                 cv2.LINE_AA)
    left = int(sess.remaining)
    txt = f"{left // 60}:{left % 60:02d}"
    (tw, _), _ = cv2.getTextSize(txt, FONT, 0.42, 1)
    cv2.putText(canvas, txt, (x0 + bw + 14, y + 5), FONT, 0.42, FAINT, 1,
                cv2.LINE_AA)
def _buttons(canvas, zones: Zones, sess) -> None:
    if "open" not in zones:
        return
    r = zones["open"]
    rad = r.w // 2
    cv2.circle(canvas, (r.cx, r.cy), rad, (22, 21, 19), -1, cv2.LINE_AA)
    cv2.circle(canvas, (r.cx, r.cy), rad - 2, CARD, -1, cv2.LINE_AA)
    cv2.circle(canvas, (r.cx, r.cy), rad - 2, EDGE, 1, cv2.LINE_AA)
    s = max(8, rad // 3)
    cv2.rectangle(canvas, (r.cx - s, r.cy - s), (r.cx + s, r.cy + s),
                  DIM, -1, cv2.LINE_AA)
def draw_pause(canvas: np.ndarray, zones: Zones, volume: int = 0,
               mute: bool = False) -> None:
    darken(canvas, 0.42)
    if "panel" in zones:
        _round_rect(canvas, zones["panel"], PANEL, rad=18, edge=EDGE)
    if "volume" in zones:
        r = zones["volume"]
        y = r.cy + 6
        x0, x1 = r.x + SLIDER_PAD, r.x1 - SLIDER_PAD
        label = "sound off" if mute else f"sound {volume}"
        (lw, _), _ = cv2.getTextSize(label, FONT, 0.44, 1)
        cv2.putText(canvas, label, (r.cx - lw // 2, r.y + 8), FONT, 0.44,
                    FAINT if mute else DIM, 1, cv2.LINE_AA)
        cv2.line(canvas, (x0, y), (x1, y), (70, 66, 62), 5, cv2.LINE_AA)
        kx = slider_knob(r, volume)
        if kx > x0 and not mute:
            cv2.line(canvas, (x0, y), (kx, y), ACCENT, 5, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 13, PANEL, -1, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 11, EDGE if mute else ACCENT, -1,
                   cv2.LINE_AA)
    for key, label, lead in (("resume", "keep going", True),
                             ("again", "start over", False),
                             ("home", "back", False),
                             ("quit", "close", False)):
        if key not in zones:
            continue
        r = zones[key]
        _round_rect(canvas, r, CARD_ON if lead else CARD,
                    edge=ACCENT if lead else EDGE)
        _text_mid(canvas, label, r, 0.46, TEXT if lead else DIM)
def _paused(canvas, w, h) -> None:
    darken(canvas, 0.30)
    _centre(canvas, "paused", h // 2 - 110, 0.56, DIM)
def draw_menu(canvas: np.ndarray, scenes: dict, minutes: int,
              zones: Zones, sound: str, ready: bool) -> None:
    h, w = canvas.shape[:2]
    first = scenes.get(sounds.NAMES[0]) if scenes else None
    if first is not None:
        canvas[:] = first
        darken(canvas, 0.58)
    else:
        canvas[:] = BG
    _centre(canvas, "attend", 96, 1.05, TEXT)
    _centre(canvas, "move your attention between sounds", 138, 0.48, DIM)
    _centre(canvas, "eyes open or closed, whichever is easier", 168, 0.44,
            FAINT)
    if "minutes" in zones:
        r = zones["minutes"]
        y = r.cy
        x0, x1 = r.x + SLIDER_PAD, r.x1 - SLIDER_PAD
        cv2.line(canvas, (x0, y), (x1, y), (72, 68, 64), 5, cv2.LINE_AA)
        kx = slider_knob(r, minutes, S.MIN_MINUTES, S.MAX_MINUTES)
        if kx > x0:
            cv2.line(canvas, (x0, y), (kx, y), ACCENT, 5, cv2.LINE_AA)
        for m in range(S.MIN_MINUTES, S.MAX_MINUTES + 1):
            tx = slider_knob(r, m, S.MIN_MINUTES, S.MAX_MINUTES)
            cv2.circle(canvas, (tx, y + 20), 2, FAINT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 14, BG, -1, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 12, ACCENT, -1, cv2.LINE_AA)
        name, detail = S.shape(minutes)
        big = f"{minutes} minutes"
        (tw, _), _ = cv2.getTextSize(big, FONT, 0.66, 1)
        cv2.putText(canvas, big, (r.cx - tw // 2, y - 40), FONT, 0.66,
                    TEXT, 1, cv2.LINE_AA)
        (tw2, _), _ = cv2.getTextSize(name, FONT, 0.50, 1)
        cv2.putText(canvas, name, (r.cx - tw2 // 2, y + 52), FONT, 0.50,
                    ACCENT, 1, cv2.LINE_AA)
        (tw3, _), _ = cv2.getTextSize(detail, FONT, 0.40, 1)
        cv2.putText(canvas, detail, (r.cx - tw3 // 2, y + 78), FONT, 0.40,
                    FAINT, 1, cv2.LINE_AA)
    if "settings" in zones:
        r = zones["settings"]
        _round_rect(canvas, r, CARD, edge=EDGE)
        _text_mid(canvas, "settings", r, 0.42, DIM)
    if "begin" in zones:
        r = zones["begin"]
        _round_rect(canvas, r, CARD_ON if ready else CARD,
                    edge=ACCENT if ready else EDGE)
        _text_mid(canvas, "begin" if ready else "getting ready", r,
                  0.54 if ready else 0.44, TEXT if ready else FAINT)
    _centre(canvas, "headphones help, but are not required", h - 46, 0.40,
            FAINT)
def slider_value(rect: Rect, x: int, lo: int = 0, hi: int = 100) -> int:
    inner = max(1, rect.w - 2 * SLIDER_PAD)
    f = (float(x) - (rect.x + SLIDER_PAD)) / inner
    return int(round(lo + (hi - lo) * float(np.clip(f, 0.0, 1.0))))
def slider_knob(rect: Rect, value: int, lo: int = 0, hi: int = 100) -> int:
    inner = max(1, rect.w - 2 * SLIDER_PAD)
    f = (float(value) - lo) / max(1, hi - lo)
    return int(rect.x + SLIDER_PAD + inner * float(np.clip(f, 0.0, 1.0)))
def draw_settings(canvas: np.ndarray, zones: Zones, volume: int,
                  muted: bool) -> None:
    h, w = canvas.shape[:2]
    darken(canvas, 0.55)
    if "panel" in zones:
        _round_rect(canvas, zones["panel"], PANEL, rad=18, edge=EDGE)
    _centre(canvas, "settings", int(h * 0.30), 0.68, TEXT)
    if "volume" in zones:
        r = zones["volume"]
        y = r.cy
        x0 = r.x + SLIDER_PAD
        x1 = r.x1 - SLIDER_PAD
        cv2.line(canvas, (x0, y), (x1, y), (74, 70, 66), 5, cv2.LINE_AA)
        kx = slider_knob(r, volume)
        if kx > x0:
            cv2.line(canvas, (x0, y), (kx, y),
                     FAINT if muted else ACCENT, 5, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 13, BG, -1, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 11,
                   FAINT if muted else ACCENT, -1, cv2.LINE_AA)
        cv2.putText(canvas, "volume", (x0, y - 30), FONT, 0.44, DIM, 1,
                    cv2.LINE_AA)
        txt = "off" if muted else str(volume)
        (tw, _), _ = cv2.getTextSize(txt, FONT, 0.46, 1)
        cv2.putText(canvas, txt, (x1 - tw, y - 30), FONT, 0.46,
                    DIM if muted else TEXT, 1, cv2.LINE_AA)
    if "close" in zones:
        r = zones["close"]
        _round_rect(canvas, r, CARD_ON, edge=ACCENT)
        _text_mid(canvas, "done", r, 0.5, TEXT)
def draw_over(canvas: np.ndarray, scenes: dict, zones: Zones,
              alpha: float) -> None:
    h, w = canvas.shape[:2]
    darken(canvas, 0.30 * alpha)
    c = int(150 * alpha) + 80
    _centre(canvas, "done", h // 2 - 40, 0.9, (c, c, c))
    _centre(canvas, "nothing to score, nothing to keep", h // 2 - 4, 0.42,
            FAINT)
    if alpha > 0.5:
        for key, label in (("again", "again"), ("stop", "close")):
            if key not in zones:
                continue
            r = zones[key]
            _round_rect(canvas, r, CARD, edge=EDGE)
            _text_mid(canvas, label, r, 0.48, TEXT)
