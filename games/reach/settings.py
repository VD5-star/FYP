from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
import audio as audio_mod
TOUCH_MIN = 44
SLIDER_PAD = 22
PANEL = (52, 47, 43)
CARD_ON = (84, 96, 100)
EDGE = (96, 90, 84)
GLOW = (198, 214, 218)
TEXT = (226, 222, 216)
DIM = (150, 145, 138)
FAINT = (104, 99, 94)
BG = (34, 31, 29)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BUTTON_W = 122
BUTTON_H = 46
BUTTON_PAD = 20
CARD = (58, 54, 50)
ACCENT = (198, 214, 218)
MIN_MINUTES = 1
MAX_MINUTES = 10
DEFAULT_MINUTES = 1
SHAPES = (
    (2, "quick", "a short warm up"),
    (4, "steady", "long enough to find a rhythm"),
    (7, "full", "a proper session"),
    (10, "long", "settle in and keep moving"),
)
def clamp_minutes(m: int) -> int:
    return int(min(MAX_MINUTES, max(MIN_MINUTES, int(m))))
def shape(minutes: int) -> tuple[str, str]:
    m = clamp_minutes(minutes)
    for top, name, detail in SHAPES:
        if m <= top:
            return name, detail
    return SHAPES[-1][1], SHAPES[-1][2]
@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int
    def hit(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w \
            and self.y <= py < self.y + self.h
    @property
    def cx(self) -> int:
        return self.x + self.w // 2
    @property
    def cy(self) -> int:
        return self.y + self.h // 2
    @property
    def x1(self) -> int:
        return self.x + self.w
    @property
    def y1(self) -> int:
        return self.y + self.h
    def touchable(self) -> bool:
        return self.w >= TOUCH_MIN and self.h >= TOUCH_MIN
@dataclass
class Zones:
    items: dict = field(default_factory=dict)
    def add(self, key: str, r: Rect) -> Rect:
        self.items[key] = r
        return r
    def at(self, px: int, py: int) -> str | None:
        for k, r in reversed(list(self.items.items())):
            if r.hit(px, py):
                return k
        return None
    def clear(self) -> None:
        self.items.clear()
    def __contains__(self, key: str) -> bool:
        return key in self.items
    def __getitem__(self, key: str) -> Rect:
        return self.items[key]
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
def _text_mid(canvas, txt, r: Rect, scale=0.5, col=TEXT, thick=1):
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, thick)
    cv2.putText(canvas, txt, (r.cx - tw // 2, r.cy + th // 2), FONT, scale,
                col, thick, cv2.LINE_AA)
def slider_value(rect: Rect, x: int, lo: int = 0, hi: int = 100) -> int:
    inner = max(1, rect.w - 2 * SLIDER_PAD)
    f = (float(x) - (rect.x + SLIDER_PAD)) / inner
    return int(round(lo + (hi - lo) * float(np.clip(f, 0.0, 1.0))))
def slider_knob(rect: Rect, value: int, lo: int = 0, hi: int = 100) -> int:
    inner = max(1, rect.w - 2 * SLIDER_PAD)
    f = (float(value) - lo) / max(1, hi - lo)
    return int(rect.x + SLIDER_PAD + inner * float(np.clip(f, 0.0, 1.0)))
def button_zone(w: int, h: int) -> Zones:
    z = Zones()
    z.add("settings", Rect(w - BUTTON_W - BUTTON_PAD, BUTTON_PAD,
                           BUTTON_W, BUTTON_H))
    return z
def panel_zones(w: int, h: int) -> Zones:
    z = Zones()
    pw = min(520, w - 80)
    ph = 260
    px = (w - pw) // 2
    py = (h - ph) // 2
    z.add("panel", Rect(px, py, pw, ph))
    z.add("volume", Rect(px + 30, py + 112, pw - 60, 56))
    z.add("close", Rect(w // 2 - 85, py + ph - 76, 170, 54))
    return z
def draw_button(canvas: np.ndarray, zones: Zones) -> None:
    if "settings" not in zones:
        return
    r = zones["settings"]
    _round_rect(canvas, r, PANEL, edge=EDGE)
    _text_mid(canvas, "settings", r, 0.44, TEXT)
def draw_panel(canvas: np.ndarray, zones: Zones, volume: int,
               muted: bool) -> None:
    h, w = canvas.shape[:2]
    cv2.convertScaleAbs(canvas, dst=canvas, alpha=0.45, beta=0.0)
    if "panel" in zones:
        _round_rect(canvas, zones["panel"], PANEL, rad=18, edge=EDGE)
    (tw, _), _ = cv2.getTextSize("settings", FONT, 0.68, 1)
    cv2.putText(canvas, "settings", ((w - tw) // 2, int(h * 0.30)), FONT,
                0.68, TEXT, 1, cv2.LINE_AA)
    if "volume" in zones:
        r = zones["volume"]
        y = r.cy
        x0, x1 = r.x + SLIDER_PAD, r.x1 - SLIDER_PAD
        cv2.line(canvas, (x0, y), (x1, y), (78, 72, 66), 5, cv2.LINE_AA)
        kx = slider_knob(r, volume)
        if kx > x0:
            cv2.line(canvas, (x0, y), (kx, y), FAINT if muted else GLOW, 5,
                     cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 13, BG, -1, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 11, FAINT if muted else GLOW, -1,
                   cv2.LINE_AA)
        cv2.putText(canvas, "volume", (x0, y - 30), FONT, 0.44, DIM, 1,
                    cv2.LINE_AA)
        txt = "off" if muted else str(volume)
        (tw2, _), _ = cv2.getTextSize(txt, FONT, 0.46, 1)
        cv2.putText(canvas, txt, (x1 - tw2, y - 30), FONT, 0.46,
                    DIM if muted else TEXT, 1, cv2.LINE_AA)
    if "close" in zones:
        r = zones["close"]
        _round_rect(canvas, r, CARD_ON, edge=GLOW)
        _text_mid(canvas, "done", r, 0.5, TEXT)
def menu_zones(w: int, h: int, mode: str = "timed") -> Zones:
    z = Zones()
    sw = min(460, w - 120)
    y = int(h * 0.42)
    if mode != "calm":
        z.add("minutes", Rect((w - sw) // 2, y, sw, 56))
    bw, bh, gap = 150, 54, 20
    x0 = (w - (bw * 2 + gap)) // 2
    z.add("timed", Rect(x0, y + 112, bw, bh))
    z.add("calm", Rect(x0 + bw + gap, y + 112, bw, bh))
    z.add("begin", Rect((w - 220) // 2, y + 190, 220, 62))
    return z
def _slider(canvas, r: Rect, value: int, lo: int, hi: int) -> None:
    y = r.cy
    x0, x1 = r.x + SLIDER_PAD, r.x1 - SLIDER_PAD
    cv2.line(canvas, (x0, y), (x1, y), (72, 68, 64), 5, cv2.LINE_AA)
    kx = slider_knob(r, value, lo, hi)
    if kx > x0:
        cv2.line(canvas, (x0, y), (kx, y), ACCENT, 5, cv2.LINE_AA)
    for m in range(lo, hi + 1):
        tx = slider_knob(r, m, lo, hi)
        cv2.circle(canvas, (tx, y + 20), 2, FAINT, -1, cv2.LINE_AA)
    cv2.circle(canvas, (kx, y), 14, BG, -1, cv2.LINE_AA)
    cv2.circle(canvas, (kx, y), 12, ACCENT, -1, cv2.LINE_AA)
def _centre(canvas, txt, y, scale, col, thick=1):
    w = canvas.shape[1]
    (tw, _), _ = cv2.getTextSize(txt, FONT, scale, thick)
    cv2.putText(canvas, txt, ((w - tw) // 2, y), FONT, scale, col, thick,
                cv2.LINE_AA)
def draw_menu(canvas: np.ndarray, zones: Zones, minutes: int, mode: str,
              ready: bool = True) -> None:
    h, w = canvas.shape[:2]
    cv2.convertScaleAbs(canvas, dst=canvas, alpha=0.38, beta=0.0)
    _centre(canvas, "reach", 96, 1.05, TEXT)
    _centre(canvas, "move your body to meet the targets", 138, 0.48, DIM)
    _centre(canvas, "stand back so your whole body fits", 168, 0.44, FAINT)
    if "minutes" in zones:
        r = zones["minutes"]
        _slider(canvas, r, minutes, MIN_MINUTES, MAX_MINUTES)
        word = "minute" if minutes == 1 else "minutes"
        name, detail = shape(minutes)
        _centre(canvas, f"{minutes} {word}", r.cy - 40, 0.66, TEXT)
        _centre(canvas, name, r.cy + 52, 0.50, ACCENT)
        _centre(canvas, detail, r.cy + 78, 0.40, FAINT)
    elif "timed" in zones:
        y = zones["timed"].y - 58
        _centre(canvas, "no clock", y, 0.62, TEXT)
        _centre(canvas, "stop whenever you want", y + 28, 0.40, FAINT)
    for key, label, sub in (("timed", "timed", "the clock runs"),
                            ("calm", "calm", "no clock at all")):
        if key not in zones:
            continue
        r = zones[key]
        on = mode == key
        _round_rect(canvas, r, CARD_ON if on else CARD,
                    edge=ACCENT if on else EDGE)
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.52, 1)
        cv2.putText(canvas, label, (r.cx - tw // 2, r.cy - 2), FONT, 0.52,
                    TEXT if on else DIM, 1, cv2.LINE_AA)
        (tw2, _), _ = cv2.getTextSize(sub, FONT, 0.36, 1)
        cv2.putText(canvas, sub, (r.cx - tw2 // 2, r.cy + 20), FONT, 0.36,
                    DIM if on else FAINT, 1, cv2.LINE_AA)
    if "begin" in zones:
        r = zones["begin"]
        _round_rect(canvas, r, CARD_ON if ready else CARD,
                    edge=ACCENT if ready else EDGE)
        _text_mid(canvas, "begin" if ready else "finding the camera", r,
                  0.54 if ready else 0.42, TEXT if ready else FAINT)
    _centre(canvas, "stand where the camera can see all of you", h - 46,
            0.40, FAINT)
class StartMenu:
    def __init__(self, width: int, height: int,
                 minutes: int = DEFAULT_MINUTES,
                 mode: str = "timed") -> None:
        self.minutes = clamp_minutes(minutes)
        self.mode = mode
        self.open = True
        self.sliding = False
        self.press: tuple[int, int] | None = None
        self.press_key: str | None = None
        self.resize(width, height)
    def resize(self, width: int, height: int) -> None:
        self.w = width
        self.h = height
        self.zones = menu_zones(width, height, self.mode)
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.zones = menu_zones(self.w, self.h, mode)
    def set_minutes(self, x: int) -> None:
        self.minutes = slider_value(self.zones["minutes"], x,
                                    MIN_MINUTES, MAX_MINUTES)
    def on_mouse(self, event, x, y, flags=0, param=None) -> str | None:
        if not self.open:
            return None
        if event == cv2.EVENT_LBUTTONDOWN:
            self.press = (x, y)
            self.press_key = self.zones.at(x, y)
            if self.press_key == "minutes":
                self.sliding = True
                self.set_minutes(x)
            return None
        if event == cv2.EVENT_MOUSEMOVE:
            if self.sliding:
                self.set_minutes(x)
            return None
        if event == cv2.EVENT_LBUTTONUP:
            hit = None
            if self.sliding:
                self.set_minutes(x)
                self.sliding = False
            elif self.press is not None:
                moved = abs(x - self.press[0]) + abs(y - self.press[1])
                key = self.zones.at(x, y)
                if moved <= 8 and key is not None and key == self.press_key:
                    if key in ("timed", "calm"):
                        self.set_mode(key)
                    else:
                        hit = key
            self.press = None
            self.press_key = None
            return hit
        return None
    def draw(self, canvas: np.ndarray, ready: bool = True) -> None:
        draw_menu(canvas, self.zones, self.minutes, self.mode, ready)
def pause_zones(w: int, h: int) -> Zones:
    z = Zones()
    pw = min(520, w - 80)
    ph = 250
    px = (w - pw) // 2
    py = (h - ph) // 2
    z.add("panel", Rect(px, py, pw, ph))
    bw, bh, gap = 190, 58, 22
    x0 = (w - (bw * 2 + gap)) // 2
    z.add("resume", Rect(x0, py + ph - 96, bw, bh))
    z.add("finish", Rect(x0 + bw + gap, py + ph - 96, bw, bh))
    return z
def draw_pause_menu(canvas: np.ndarray, zones: Zones) -> None:
    h, w = canvas.shape[:2]
    cv2.convertScaleAbs(canvas, dst=canvas, alpha=0.40, beta=0.0)
    if "panel" in zones:
        _round_rect(canvas, zones["panel"], PANEL, rad=18, edge=EDGE)
    p = zones["panel"] if "panel" in zones else None
    top = (p.y + 76) if p is not None else h // 2
    _centre(canvas, "paused", top, 0.82, TEXT)
    _centre(canvas, "take your time", top + 34, 0.44, FAINT)
    for key, label, lead in (("resume", "continue", True),
                             ("finish", "finish", False)):
        if key not in zones:
            continue
        r = zones[key]
        _round_rect(canvas, r, CARD_ON if lead else CARD,
                    edge=ACCENT if lead else EDGE)
        _text_mid(canvas, label, r, 0.54, TEXT if lead else DIM)
def summary_zones(w: int, h: int) -> Zones:
    z = Zones()
    bw, bh, gap = 200, 58, 20
    total = bw * 3 + gap * 2
    x0 = (w - total) // 2
    y = int(h * 0.72)
    z.add("again", Rect(x0, y, bw, bh))
    z.add("menu", Rect(x0 + bw + gap, y, bw, bh))
    z.add("close", Rect(x0 + (bw + gap) * 2, y, bw, bh))
    return z
def _stat(canvas, label: str, value: str, cx: int, y: int) -> None:
    (vw, _), _ = cv2.getTextSize(value, FONT, 0.92, 1)
    cv2.putText(canvas, value, (cx - vw // 2, y), FONT, 0.92, TEXT, 1,
                cv2.LINE_AA)
    (lw, _), _ = cv2.getTextSize(label, FONT, 0.40, 1)
    cv2.putText(canvas, label, (cx - lw // 2, y + 26), FONT, 0.40, FAINT, 1,
                cv2.LINE_AA)
def draw_summary(canvas: np.ndarray, zones: Zones, summary: dict,
                 mode: str = "timed", alpha: float = 1.0) -> None:
    h, w = canvas.shape[:2]
    cv2.convertScaleAbs(canvas, dst=canvas,
                        alpha=1.0 - 0.62 * float(np.clip(alpha, 0, 1)),
                        beta=0.0)
    if alpha < 0.35:
        return
    top = int(h * 0.26)
    _centre(canvas, "well done", top, 0.92, TEXT)
    _centre(canvas, "nothing is saved, nothing is ranked", top + 32, 0.40,
            FAINT)
    secs = float(summary.get("seconds", 0.0))
    mins = int(secs) // 60
    rest = int(secs) % 60
    stats = [("touched", str(summary.get("hits", 0))),
             ("time", f"{mins}:{rest:02d}")]
    if mode == "timed":
        stats.insert(0, ("score", str(summary.get("score", 0))))
        stats.append(("bonus", str(summary.get("bonus", 0))))
    n = len(stats)
    span = min(w - 120, 190 * n)
    step = span // max(1, n)
    x0 = (w - span) // 2 + step // 2
    ys = int(h * 0.46)
    for i, (label, value) in enumerate(stats):
        _stat(canvas, label, value, x0 + i * step, ys)
    bands = summary.get("bands") or {}
    if bands:
        parts = [f"{k} {v}" for k, v in sorted(bands.items())]
        _centre(canvas, "   ".join(parts), ys + 70, 0.42, DIM)
    for key, label, lead in (("again", "again", True),
                             ("menu", "main menu", False),
                             ("close", "close", False)):
        if key not in zones:
            continue
        r = zones[key]
        _round_rect(canvas, r, CARD_ON if lead else CARD,
                    edge=ACCENT if lead else EDGE)
        _text_mid(canvas, label, r, 0.50, TEXT if lead else DIM)
class Screen:
    def __init__(self, zones: Zones) -> None:
        self.zones = zones
        self.press: tuple[int, int] | None = None
        self.press_key: str | None = None
    def on_mouse(self, event, x, y, flags=0, param=None) -> str | None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.press = (x, y)
            self.press_key = self.zones.at(x, y)
            return None
        if event == cv2.EVENT_LBUTTONUP:
            hit = None
            if self.press is not None:
                moved = abs(x - self.press[0]) + abs(y - self.press[1])
                key = self.zones.at(x, y)
                if moved <= 8 and key is not None and key == self.press_key \
                        and key != "panel":
                    hit = key
            self.press = None
            self.press_key = None
            return hit
        return None
class SettingsPanel:
    def __init__(self, audio, width: int, height: int) -> None:
        self.audio = audio
        self.open = False
        self.sliding = False
        self.press: tuple[int, int] | None = None
        self.press_key: str | None = None
        self.resize(width, height)
    def resize(self, width: int, height: int) -> None:
        self.w = width
        self.h = height
        self.button = button_zone(width, height)
        self.panel = panel_zones(width, height)
    @property
    def zones(self) -> Zones:
        return self.panel if self.open else self.button
    def set_volume(self, x: int) -> None:
        if self.audio is None:
            return
        v = slider_value(self.panel["volume"], x, audio_mod.VOL_MIN,
                         audio_mod.VOL_MAX)
        self.audio.set_volume(v)
        if v <= audio_mod.VOL_MIN:
            self.audio.stop()
        elif not self.audio.on:
            self.audio.start()
    def on_mouse(self, event, x, y, flags=0, param=None) -> bool:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.press = (x, y)
            self.press_key = self.zones.at(x, y)
            if self.open and self.press_key == "volume":
                self.sliding = True
                self.set_volume(x)
                return True
            return self.press_key is not None or self.open
        if event == cv2.EVENT_MOUSEMOVE:
            if self.sliding:
                self.set_volume(x)
                return True
            return self.open
        if event == cv2.EVENT_LBUTTONUP:
            taken = self.open or self.press_key is not None
            if self.sliding:
                self.set_volume(x)
                self.sliding = False
            elif self.press is not None:
                moved = abs(x - self.press[0]) + abs(y - self.press[1])
                key = self.zones.at(x, y)
                if moved <= 8 and key is not None and key == self.press_key:
                    if key == "settings":
                        self.open = True
                    elif key == "close":
                        self.open = False
            self.press = None
            self.press_key = None
            return taken
        return self.open
    def draw(self, canvas: np.ndarray) -> None:
        if not self.open:
            draw_button(canvas, self.button)
            return
        vol = self.audio.volume if self.audio is not None else 0
        muted = self.audio is None or self.audio.failed or vol <= 0
        draw_panel(canvas, self.panel, vol, muted)
