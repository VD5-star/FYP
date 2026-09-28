from __future__ import annotations
from collections import OrderedDict
import cv2
import numpy as np
from board import SETTLE_SECONDS, Board
from ui import Rect, Zones
BG = (42, 38, 34)
TRAY_BG = (52, 47, 43)
FRAME = (78, 72, 66)
GHOST = 0.17
TEXT = (214, 209, 202)
DIM = (150, 145, 138)
FAINT = (104, 99, 94)
GLOW = (198, 214, 218)
SPARK = (200, 220, 230)
CARD = (60, 55, 50)
CARD_ON = (84, 96, 100)
EDGE = (96, 90, 84)
FONT = cv2.FONT_HERSHEY_SIMPLEX
def blit(canvas: np.ndarray, sprite: np.ndarray, alpha: np.ndarray,
         x: int, y: int, scale: float = 1.0,
         af: np.ndarray | None = None) -> None:
    ch, cw = canvas.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + w), min(ch, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    sy1, sx1 = sy0 + (y1 - y0), sx0 + (x1 - x0)
    sub = sprite[sy0:sy1, sx0:sx1]
    dst = canvas[y0:y1, x0:x1]
    if af is None:
        a = (alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[:, :, None]
    else:
        a = af[sy0:sy1, sx0:sx1]
    if scale >= 1.0:
        opaque = a[:, :, 0] > 0.996
        if opaque.all():
            dst[:] = sub
            return
        np.copyto(dst, sub, where=opaque[:, :, None])
        edge = ~opaque & (a[:, :, 0] > 0.0)
        if not edge.any():
            return
        ae = a[edge]
        dst[edge] = (dst[edge] * (1 - ae) + sub[edge] * ae).astype(np.uint8)
        return
    a = a * scale
    canvas[y0:y1, x0:x1] = (dst * (1 - a) + sub * a).astype(np.uint8)
CACHE_KEEP = 4
_ghost_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
def ghost_of(image: np.ndarray) -> np.ndarray:
    key = (id(image), image.shape)
    hit = _ghost_cache.get(key)
    if hit is not None:
        _ghost_cache.move_to_end(key)
        return hit
    g = (image.astype(np.float32) * GHOST
         + np.array(BG, np.float32) * (1 - GHOST)).astype(np.uint8)
    _ghost_cache[key] = g
    while len(_ghost_cache) > CACHE_KEEP:
        _ghost_cache.popitem(last=False)
    return g
def draw_reference(canvas: np.ndarray, image: np.ndarray,
                   origin: tuple[int, int]) -> None:
    h, w = image.shape[:2]
    x, y = origin
    ch, cw = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + w), min(ch, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    g = ghost_of(image)
    canvas[y0:y1, x0:x1] = g[y0 - y:y1 - y, x0 - x:x1 - x]
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x1, y1), FRAME, 1, cv2.LINE_AA)
def fast_blit(canvas: np.ndarray, p, x: int, y: int) -> bool:
    ch, cw = canvas.shape[:2]
    cx0, cy0, cx1, cy1 = p.crop
    ox, oy = x + cx0, y + cy0
    if ox < 0 or oy < 0 or ox + (cx1 - cx0) > cw or oy + (cy1 - cy0) > ch:
        return False
    dst = canvas[oy:oy + (cy1 - cy0), ox:ox + (cx1 - cx0)]
    np.copyto(dst, p.csprite, where=p.ccore3)
    ys, xs = p.ceidx
    if ys.size:
        cur = dst[ys, xs, :].astype(np.float32)
        dst[ys, xs, :] = (cur + (p.ergb - cur) * p.ea).astype(np.uint8)
    return True
def base_of(canvas: np.ndarray, board: Board, image: np.ndarray) -> np.ndarray:
    key = (id(image), image.shape, canvas.shape, board.origin, board.tray)
    hit = _base_cache.get(key)
    if hit is not None:
        _base_cache.move_to_end(key)
        return hit
    base = np.empty_like(canvas)
    base[:] = BG
    tx, ty, tw, th = board.tray
    cv2.rectangle(base, (tx, ty), (tx + tw, ty + th), TRAY_BG, -1)
    _paint_reference(base, image, board.origin)
    _base_cache[key] = base
    while len(_base_cache) > CACHE_KEEP:
        _base_cache.popitem(last=False)
    return base
_base_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
def _paint_reference(canvas, image, origin) -> None:
    h, w = image.shape[:2]
    x, y = origin
    ch, cw = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + w), min(ch, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    g = ghost_of(image)
    canvas[y0:y1, x0:x1] = g[y0 - y:y1 - y, x0 - x:x1 - x]
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x1, y1), FRAME, 1, cv2.LINE_AA)
def draw_board(canvas: np.ndarray, board: Board, image: np.ndarray,
               now: float, fx=None) -> None:
    canvas[:] = base_of(canvas, board, image)
    for p in board.pieces:
        if not p.placed:
            continue
        age = board.settle_age(p, now)
        if not fast_blit(canvas, p, p.pos[0], p.pos[1]):
            blit(canvas, p.sprite, p.alpha, p.pos[0], p.pos[1], af=p.af)
        if age < SETTLE_SECONDS:
            _halo(canvas, p, 1.0 - age / SETTLE_SECONDS)
    for idx in board.order:
        p = board.pieces[idx]
        if p.placed:
            continue
        lift = 3 if idx == board.held else 0
        if lift:
            _shadow(canvas, p)
        if not fast_blit(canvas, p, p.pos[0] - lift, p.pos[1] - lift):
            blit(canvas, p.sprite, p.alpha, p.pos[0] - lift, p.pos[1] - lift,
                 af=p.af)
    if fx is not None:
        draw_particles(canvas, fx)
def _shadow(canvas: np.ndarray, p) -> None:
    sh = np.zeros_like(p.sprite)
    blit(canvas, sh, p.alpha, p.pos[0] + 5, p.pos[1] + 6, scale=0.26, af=p.af)
def _halo(canvas: np.ndarray, p, t: float) -> None:
    glow = np.full_like(p.sprite, GLOW, dtype=np.uint8)
    blit(canvas, glow, p.alpha, p.pos[0], p.pos[1], scale=0.30 * t, af=p.af)
def draw_particles(canvas: np.ndarray, fx) -> None:
    for r in fx.rings:
        t = r.life / max(1e-6, r.full)
        rad = int(r.r0 * (1.0 + (1.0 - t) * 2.1))
        a = t * t * 0.5
        if rad < 1 or a <= 0.01:
            continue
        col = tuple(float(c * a) for c in r.colour)
        _add_circle(canvas, int(r.x), int(r.y), rad, col, 2)
    px, py, sz, col, t = fx.parts.view()
    if len(px) == 0:
        return
    h, w = canvas.shape[:2]
    xi = px.astype(np.int32)
    yi = py.astype(np.int32)
    keep = (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
    if not keep.any():
        return
    xi, yi = xi[keep], yi[keep]
    a = (t[keep] ** 1.4)[:, None]
    c = col[keep] * a
    big = sz[keep] > 1.9
    if big.any():
        bx, by, bc = xi[big], yi[big], c[big] * 0.55
        ax = np.concatenate((xi, np.clip(bx + 1, 0, w - 1), bx))
        ay = np.concatenate((yi, by, np.clip(by + 1, 0, h - 1)))
        ac = np.concatenate((c, bc, bc))
    else:
        ax, ay, ac = xi, yi, c
    flat = canvas.reshape(-1, 3)
    idx = ay.astype(np.intp) * w + ax
    cur = flat[idx].astype(np.int16)
    cur += ac.astype(np.int16)
    np.clip(cur, 0, 255, out=cur)
    flat[idx] = cur.astype(np.uint8)
def _add_circle(canvas, cx, cy, rad, colour, thick) -> None:
    x0, y0 = max(0, cx - rad - thick), max(0, cy - rad - thick)
    x1 = min(canvas.shape[1], cx + rad + thick + 1)
    y1 = min(canvas.shape[0], cy + rad + thick + 1)
    if x1 <= x0 or y1 <= y0:
        return
    patch = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    cv2.circle(patch, (cx - x0, cy - y0), rad, colour, thick, cv2.LINE_AA)
    dst = canvas[y0:y1, x0:x1]
    np.clip(dst.astype(np.int16) + patch, 0, 255, out=dst,
            casting="unsafe")
def _text_mid(canvas, txt, rect: Rect, scale=0.5, col=TEXT, thick=1):
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, thick)
    cv2.putText(canvas, txt, (rect.cx - tw // 2, rect.cy + th // 2),
                FONT, scale, col, thick, cv2.LINE_AA)
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
def draw_hud(canvas: np.ndarray, board: Board, name: str,
             zones: Zones, sound: str = "") -> None:
    h, w = canvas.shape[:2]
    if "stop" in zones:
        r = zones["stop"]
        rad = r.w // 2
        cv2.circle(canvas, (r.cx, r.cy), rad, (26, 24, 22), -1, cv2.LINE_AA)
        cv2.circle(canvas, (r.cx, r.cy), rad - 2, CARD, -1, cv2.LINE_AA)
        cv2.circle(canvas, (r.cx, r.cy), rad - 2, EDGE, 1, cv2.LINE_AA)
        s = max(8, rad // 3)
        cv2.rectangle(canvas, (r.cx - s, r.cy - s), (r.cx + s, r.cy + s),
                      DIM, -1, cv2.LINE_AA)
    cv2.putText(canvas, name, (26, 36), FONT, 0.48, FAINT, 1, cv2.LINE_AA)
    _progress(canvas, board, w, h)
def draw_pause(canvas: np.ndarray, zones: Zones, volume: int = 0,
               mute: bool = False) -> None:
    cv2.convertScaleAbs(canvas, dst=canvas, alpha=0.42, beta=0.0)
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
        cv2.line(canvas, (x0, y), (x1, y), (72, 68, 64), 5, cv2.LINE_AA)
        kx = slider_knob(r, volume, 0, 100)
        if kx > x0 and not mute:
            cv2.line(canvas, (x0, y), (kx, y), GLOW, 5, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 13, PANEL, -1, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 11, EDGE if mute else GLOW, -1,
                   cv2.LINE_AA)
    for key, label, lead in (("resume", "keep going", True),
                             ("again", "start over", False),
                             ("shuffle", "shuffle", False),
                             ("home", "pictures", False)):
        if key not in zones:
            continue
        r = zones[key]
        _round_rect(canvas, r, CARD_ON if lead else CARD,
                    edge=GLOW if lead else EDGE)
        _text_mid(canvas, label, r, 0.46, TEXT if lead else DIM)
def draw_setup(canvas: np.ndarray, zones: Zones, sizes: list[int],
               size_pick: int, style: str, options: list[int],
               thumb: np.ndarray | None = None) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], canvas.shape[0]), BG, -1)
    h, w = canvas.shape[:2]
    (tw2, _), _ = cv2.getTextSize("how do you want it", FONT, 0.62, 1)
    cv2.putText(canvas, "how do you want it", ((w - tw2) // 2, 64), FONT,
                0.62, TEXT, 1, cv2.LINE_AA)
    if thumb is not None:
        top = 96
        th, tw = thumb.shape[:2]
        room = max(0, int(h * 0.40) - top)
        if th > room:
            sc = room / float(th)
            thumb = cv2.resize(thumb, (max(1, int(tw * sc)), max(1, room)),
                               interpolation=cv2.INTER_AREA)
            th, tw = thumb.shape[:2]
        x = (w - tw) // 2
        cut = canvas[top:top + th, x:x + tw]
        cv2.addWeighted(thumb, 0.5, cut, 0.5, 0.0, dst=cut)
    if "pieces" in zones:
        r = zones["pieces"]
        side = sizes[size_pick]
        idx = options.index(side) if side in options else 0
        top = max(1, len(options) - 1)
        y = r.cy
        x0, x1 = r.x + SLIDER_PAD, r.x1 - SLIDER_PAD
        cv2.line(canvas, (x0, y), (x1, y), (72, 68, 64), 5, cv2.LINE_AA)
        kx = slider_knob(r, idx, 0, top)
        if kx > x0:
            cv2.line(canvas, (x0, y), (kx, y), GLOW, 5, cv2.LINE_AA)
        for i in range(len(options)):
            cv2.circle(canvas, (slider_knob(r, i, 0, top), y + 20), 2,
                       FAINT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 14, BG, -1, cv2.LINE_AA)
        cv2.circle(canvas, (kx, y), 12, GLOW, -1, cv2.LINE_AA)
        count = side * side
        name, detail = piece_shape(count)
        big = f"{count} pieces"
        (bw, _), _ = cv2.getTextSize(big, FONT, 0.66, 1)
        cv2.putText(canvas, big, (r.cx - bw // 2, y - 40), FONT, 0.66, TEXT,
                    1, cv2.LINE_AA)
        (nw, _), _ = cv2.getTextSize(name, FONT, 0.48, 1)
        cv2.putText(canvas, name, (r.cx - nw // 2, y + 52), FONT, 0.48,
                    GLOW, 1, cv2.LINE_AA)
        (dw, _), _ = cv2.getTextSize(detail, FONT, 0.40, 1)
        cv2.putText(canvas, detail, (r.cx - dw // 2, y + 78), FONT, 0.40,
                    FAINT, 1, cv2.LINE_AA)
    for key, label, sub in (("square", "straight", "plain square edges"),
                            ("interlock", "interlocking", "tabs that hook")):
        if key not in zones:
            continue
        r = zones[key]
        on = style == key
        _round_rect(canvas, r, CARD_ON if on else CARD,
                    edge=GLOW if on else EDGE)
        (lw, _), _ = cv2.getTextSize(label, FONT, 0.50, 1)
        cv2.putText(canvas, label, (r.cx - lw // 2, r.cy - 2), FONT, 0.50,
                    TEXT if on else DIM, 1, cv2.LINE_AA)
        (sw2, _), _ = cv2.getTextSize(sub, FONT, 0.34, 1)
        cv2.putText(canvas, sub, (r.cx - sw2 // 2, r.cy + 20), FONT, 0.34,
                    DIM if on else FAINT, 1, cv2.LINE_AA)
    if "back" in zones:
        r = zones["back"]
        _round_rect(canvas, r, CARD, edge=EDGE)
        _text_mid(canvas, "back", r, 0.44, DIM)
    if "start" in zones:
        r = zones["start"]
        _round_rect(canvas, r, CARD_ON, edge=GLOW)
        _text_mid(canvas, "start", r, 0.54, (232, 240, 242))
def _progress(canvas: np.ndarray, board: Board, w: int, h: int) -> None:
    bw = 210
    x0 = w - bw - 26
    y = 30
    frac = board.done / max(1, board.total)
    cv2.line(canvas, (x0, y), (x0 + bw, y), (64, 59, 54), 4, cv2.LINE_AA)
    if frac > 0:
        cv2.line(canvas, (x0, y), (x0 + int(bw * frac), y),
                 (150, 176, 184), 4, cv2.LINE_AA)
def draw_done(canvas: np.ndarray, alpha: float, zones: Zones) -> None:
    h, w = canvas.shape[:2]
    canvas[:] = (canvas.astype(np.float32) * (1 - alpha * 0.42)).astype(
        np.uint8)
    c = int(140 * alpha) + 90
    (tw, th), _ = cv2.getTextSize("done", FONT, 0.9, 1)
    cv2.putText(canvas, "done", ((w - tw) // 2, h // 2 - 34),
                FONT, 0.9, (c, c, c), 1, cv2.LINE_AA)
    if alpha > 0.55:
        for key, label in (("again", "again"), ("back", "pictures")):
            if key not in zones:
                continue
            r = zones[key]
            _round_rect(canvas, r, CARD, edge=EDGE)
            _text_mid(canvas, label, r, 0.48, TEXT)
PIECE_WORDS = (
    (9, "gentle", "a few big pieces"),
    (25, "easy", "room to see the picture"),
    (49, "steady", "a real puzzle"),
    (64, "busy", "small pieces, long sit"),
    (100, "deep", "the longest one"),
)
def piece_shape(count: int) -> tuple[str, str]:
    for top, name, detail in PIECE_WORDS:
        if count <= top:
            return name, detail
    return PIECE_WORDS[-1][1], PIECE_WORDS[-1][2]
def draw_menu(canvas: np.ndarray, thumbs: dict, names: list[str],
              pick: int, sizes: list[int], size_pick: int,
              style: str, zones: Zones, sound: str = "",
              options: list[int] | None = None) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], canvas.shape[0]), BG, -1)
    h, w = canvas.shape[:2]
    (tw0, _), _ = cv2.getTextSize("jigsaw", FONT, 0.86, 1)
    cv2.putText(canvas, "jigsaw", ((w - tw0) // 2, 52), FONT, 0.86, TEXT, 1,
                cv2.LINE_AA)
    (tw1, _), _ = cv2.getTextSize("pick a picture", FONT, 0.44, 1)
    cv2.putText(canvas, "pick a picture", ((w - tw1) // 2, 78), FONT, 0.44,
                FAINT, 1, cv2.LINE_AA)
    for i, n in enumerate(names):
        key = f"pic:{i}"
        if key not in zones:
            continue
        r = zones[key]
        t = thumbs.get(n)
        if t is not None:
            th_, tw_ = t.shape[:2]
            canvas[r.y:r.y + th_, r.x:r.x + tw_] = t
        if i == pick:
            cv2.rectangle(canvas, (r.x - 2, r.y - 2), (r.x1 + 1, r.y1 + 1),
                          GLOW, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (r.x - 1, r.y - 1), (r.x1, r.y1),
                          FRAME, 1, cv2.LINE_AA)
    if "own" in zones:
        r = zones["own"]
        _round_rect(canvas, r, CARD, edge=EDGE)
        _text_mid(canvas, "your picture", r, 0.44, TEXT)
    if "settings" in zones:
        r = zones["settings"]
        _round_rect(canvas, r, CARD, edge=EDGE)
        _text_mid(canvas, "settings", r, 0.42, DIM)
    if "start" in zones:
        r = zones["start"]
        _round_rect(canvas, r, CARD_ON, edge=GLOW)
        _text_mid(canvas, "start", r, 0.52, (232, 240, 242))
SLIDER_PAD = 22
PANEL = (56, 51, 47)
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
def make_thumb(image: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = image.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(iw * s + 0.5)), max(h, int(ih * s + 0.5))
    r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (nw - w) // 2
    y = (nh - h) // 2
    return np.ascontiguousarray(r[y:y + h, x:x + w])
