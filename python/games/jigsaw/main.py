from __future__ import annotations
import argparse
import threading
import time
import cv2
import numpy as np
import art
import audio as audio_mod
import board as board_mod
import draw
import fx as fx_mod
import photo
from pieces import INTERLOCK, SQUARE
from ui import Rect, Zones
WINDOW = "jigsaw"
SIZES = [3, 4, 5, 6, 7, 8, 10]
TRAY_FRAC = 0.30
DONE_FADE = 1.6
TAP_SLOP = 8
MIN_PIECE_PX = 52.0
PICK = "pick"
SETUP = "setup"
GAME = "game"
STOP_R = 34
STOP_PAD = 26
class App:
    def __init__(self, width: int, height: int, style: str,
                 seed: int | None, sound: int = 2,
                 audio_on: bool = True) -> None:
        self.w = width
        self.h = height
        self.style = style
        self.seed = seed
        self.canvas = np.zeros((height, width, 3), np.uint8)
        self.names = list(art.NAMES)
        self.pick = 0
        self.size_pick = 2
        self.board: board_mod.Board | None = None
        self.image: np.ndarray | None = None
        self.title = ""
        self.custom: np.ndarray | None = None
        self.done_at = 0.0
        self.zones = Zones()
        self.press: tuple[int, int] | None = None
        self.press_key: str | None = None
        self.dragging = False
        self.thumbs: dict[str, np.ndarray] = {}
        self._art: dict = {}
        self._lock = threading.Lock()
        self._warming = None
        self._warm_thread: threading.Thread | None = None
        self.audio = audio_mod.Audio(volume=sound, seed=seed) \
            if audio_on else None
        self.settings = False
        self.sliding = False
        self.page = PICK
        self.paused = False
        self.fx = fx_mod.Effects(rng=np.random.default_rng(seed))
        self.last_t = time.monotonic()
        self.menu_zones()
        self.warm(self.names[self.pick])
    def sizes(self) -> list[int]:
        _, aw, ah = self.layout()
        out = []
        for s in SIZES:
            if min(aw / s, ah / s) >= MIN_PIECE_PX:
                out.append(s)
        return out or [SIZES[0]]
    def layout(self):
        tray_h = int(self.h * TRAY_FRAC)
        area_h = self.h - tray_h - 76
        area_w = self.w - 80
        tray = (0, self.h - tray_h, self.w, tray_h)
        return tray, area_w, area_h
    def menu_zones(self) -> None:
        z = self.zones
        z.clear()
        cols = 3
        rows = 3
        pad = 20
        bh = 58
        top = 96
        foot = bh + 34
        room_h = self.h - top - foot - 24
        room_w = self.w - 60
        ch = (room_h - pad * (rows - 1)) // rows
        cw = int(ch / 0.62)
        if cw * cols + pad * (cols - 1) > room_w:
            cw = (room_w - pad * (cols - 1)) // cols
            ch = int(cw * 0.62)
        gw = cw * cols + pad * (cols - 1)
        gh = ch * rows + pad * (rows - 1)
        x0 = (self.w - gw) // 2
        n = min(len(self.names), cols * rows - 1)
        for i in range(n):
            r, c = divmod(i, cols)
            z.add(f"pic:{i}", Rect(x0 + c * (cw + pad),
                                   top + r * (ch + pad), cw, ch))
        r, c = divmod(n, cols)
        z.add("own", Rect(x0 + c * (cw + pad), top + r * (ch + pad), cw, ch))
        y = top + gh + 24
        bw, gap = 190, 20
        sx = (self.w - (bw * 2 + gap)) // 2
        z.add("settings", Rect(sx, y, bw, bh))
        z.add("start", Rect(sx + bw + gap, y, bw, bh))
        self.build_thumbs()
    def setup_zones(self) -> None:
        z = self.zones
        z.clear()
        avail = self.sizes()
        if self.size_pick >= len(SIZES) or SIZES[self.size_pick] not in avail:
            self.size_pick = SIZES.index(avail[-1])
        sw = min(460, self.w - 140)
        y = int(self.h * 0.44)
        z.add("pieces", Rect((self.w - sw) // 2, y, sw, 56))
        bw, bh, gap = 180, 56, 20
        x0 = (self.w - (bw * 2 + gap)) // 2
        z.add("square", Rect(x0, y + 128, bw, bh))
        z.add("interlock", Rect(x0 + bw + gap, y + 128, bw, bh))
        z.add("start", Rect((self.w - 220) // 2, y + 206, 220, 62))
        z.add("back", Rect(26, 26, 140, 50))
    def size_options(self) -> list[int]:
        return self.sizes()
    def set_size(self, x: int) -> None:
        if "pieces" not in self.zones:
            return
        opts = self.size_options()
        i = draw.slider_value(self.zones["pieces"], x, 0, len(opts) - 1)
        self.size_pick = SIZES.index(opts[i])
    def build_thumbs(self) -> None:
        if self.thumbs:
            return
        for i, n in enumerate(self.names):
            key = f"pic:{i}"
            if key not in self.zones:
                continue
            r = self.zones[key]
            tw = max(r.w, 200)
            th = max(r.h, int(tw * art.SIZE[1] / art.SIZE[0]))
            self.thumbs[n] = draw.make_thumb(art.render(n, tw, th), r.w, r.h)
    def art_at(self, name: str, w: int, h: int) -> np.ndarray:
        key = (name, w, h)
        with self._lock:
            hit = self._art.get(key)
        if hit is not None:
            return hit
        ratio = art.SIZE[0] / art.SIZE[1]
        if w / h > ratio:
            rw, rh = w, int(round(w / ratio))
        else:
            rw, rh = int(round(h * ratio)), h
        img = art.render(name, rw, rh)
        with self._lock:
            self._art = {key: img}
        return img
    def warm(self, name: str) -> None:
        _, aw, ah = self.layout()
        key = (name, aw, ah)
        with self._lock:
            if key in self._art or self._warming == key:
                return
            self._warming = key
        def job():
            try:
                self.art_at(name, aw, ah)
            finally:
                with self._lock:
                    if self._warming == key:
                        self._warming = None
        t = threading.Thread(target=job, daemon=True)
        t.start()
        self._warm_thread = t
    def game_zones(self) -> None:
        z = self.zones
        z.clear()
        cy = int(self.h * 0.5)
        z.add("stop", Rect(self.w - STOP_PAD - STOP_R * 2, cy - STOP_R,
                           STOP_R * 2, STOP_R * 2))
    def pause_zones(self) -> None:
        z = self.zones
        z.clear()
        pw = min(440, self.w - 80)
        ph = 286
        px = (self.w - pw) // 2
        py = (self.h - ph) // 2
        z.add("panel", Rect(px, py, pw, ph))
        pad = 28
        gap = 14
        bw = (pw - pad * 2 - gap) // 2
        bh = 52
        x0 = px + pad
        x1 = x0 + bw + gap
        z.add("volume", Rect(x0, py + 56, pw - pad * 2, 52))
        y = py + 138
        z.add("resume", Rect(x0, y, bw, bh))
        z.add("again", Rect(x1, y, bw, bh))
        z.add("shuffle", Rect(x0, y + bh + gap, bw, bh))
        z.add("home", Rect(x1, y + bh + gap, bw, bh))
    def done_zones(self) -> None:
        z = self.zones
        z.clear()
        z.add("again", Rect(self.w // 2 - 200, self.h // 2 + 16, 180, 54))
        z.add("back", Rect(self.w // 2 + 20, self.h // 2 + 16, 180, 54))
    def settings_zones(self) -> None:
        z = self.zones
        z.clear()
        pw = min(520, self.w - 80)
        ph = 260
        px = (self.w - pw) // 2
        py = (self.h - ph) // 2
        z.add("panel", Rect(px, py, pw, ph))
        z.add("volume", Rect(px + 30, py + 112, pw - 60, 56))
        z.add("close", Rect(self.w // 2 - 85, py + ph - 76, 170, 54))
    def start(self) -> None:
        rows = SIZES[self.size_pick]
        tray, area_w, area_h = self.layout()
        if self.custom is not None:
            src = self.custom
            self.title = "yours"
        else:
            self.title = self.names[self.pick]
            src = self.art_at(self.title, area_w, area_h)
        img = photo.fit(src, area_w, area_h)
        img = photo.snap_to_grid(img, rows, rows)
        self.image = img
        ox = (self.w - img.shape[1]) // 2
        oy = 52
        self.board = board_mod.build(img, rows, rows, (ox, oy), tray,
                                     style=self.style, seed=self.seed)
        self.done_at = 0.0
        self.paused = False
        self.sliding = False
        self.fx.clear()
        self.page = GAME
        self.game_zones()
    def to_menu(self) -> None:
        self.board = None
        self.image = None
        self.custom = None
        self.done_at = 0.0
        self.paused = False
        self.sliding = False
        self.page = PICK
        self.fx.clear()
        self.menu_zones()
        self.warm(self.names[self.pick])
    def to_setup(self) -> None:
        self.board = None
        self.image = None
        self.done_at = 0.0
        self.paused = False
        self.sliding = False
        self.page = SETUP
        self.fx.clear()
        self.setup_zones()
        if self.custom is None:
            self.warm(self.names[self.pick])
    def open_pause(self) -> None:
        if self.paused or self.board is None or self.board.solved:
            return
        self.paused = True
        self.sliding = False
        self.pause_zones()
    def close_pause(self) -> None:
        self.paused = False
        self.sliding = False
        self.game_zones()
    def open_own(self) -> None:
        p = photo.ask_file()
        if not p:
            return
        img = photo.load(p)
        if img is None:
            return
        self.custom = img
        self.to_setup()
    def tap_menu(self, key: str) -> None:
        if key.startswith("pic:"):
            i = int(key.split(":")[1])
            self.pick = i
            self.custom = None
            self.to_setup()
        elif key == "settings":
            self.open_settings()
        elif key == "start":
            self.custom = None
            self.to_setup()
        elif key == "own":
            self.open_own()
    def tap_setup(self, key: str) -> None:
        if key == "square":
            self.style = SQUARE
        elif key == "interlock":
            self.style = INTERLOCK
        elif key == "start":
            self.start()
        elif key == "back":
            self.to_menu()
    def tap_pause(self, key: str) -> None:
        if key == "resume":
            self.close_pause()
        elif key == "again":
            self.start()
        elif key == "shuffle":
            if self.board is not None:
                self.board.reshuffle()
            self.close_pause()
        elif key == "home":
            self.to_menu()
    def open_settings(self) -> None:
        if self.settings:
            return
        self.settings = True
        self.settings_zones()
    def close_settings(self) -> None:
        self.sliding = False
        self.settings = False
        if self.paused:
            self.pause_zones()
        elif self.page == PICK:
            self.menu_zones()
        elif self.page == SETUP:
            self.setup_zones()
        elif self.solved_view():
            self.done_zones()
        else:
            self.game_zones()
    def set_volume(self, x: int) -> None:
        if self.audio is None or "volume" not in self.zones:
            return
        v = draw.slider_value(self.zones["volume"], x,
                              audio_mod.VOL_MIN, audio_mod.VOL_MAX)
        self.audio.set_volume(v)
        if v <= audio_mod.VOL_MIN:
            self.audio.stop()
        elif not self.audio.on:
            self.audio.start()
    def sound_name(self) -> str:
        if self.audio is None:
            return "no sound"
        if self.audio.failed:
            return "no sound"
        return f"sound {self.audio.level_name}"
    def tap_game(self, key: str) -> None:
        if key == "stop":
            self.open_pause()
    def tap_done(self, key: str) -> None:
        if key == "again":
            self.start()
        elif key == "back":
            self.to_menu()
    def solved_view(self) -> bool:
        return self.board is not None and self.board.solved \
            and self.done_at > 0.0
    def on_mouse(self, event, x, y, flags, param) -> None:
        if self.settings:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.press = (x, y)
                self.press_key = self.zones.at(x, y)
                if self.press_key == "volume":
                    self.sliding = True
                    self.set_volume(x)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.sliding:
                    self.set_volume(x)
            elif event == cv2.EVENT_LBUTTONUP:
                if self.sliding:
                    self.set_volume(x)
                    self.sliding = False
                elif self.press is not None:
                    moved = abs(x - self.press[0]) + abs(y - self.press[1])
                    key = self.zones.at(x, y)
                    if moved <= TAP_SLOP and key == self.press_key \
                            and key == "close":
                        self.close_settings()
                self.press = None
                self.press_key = None
            return
        if self.page == SETUP:
            if event == cv2.EVENT_LBUTTONDOWN and \
                    self.zones.at(x, y) == "pieces":
                self.press = (x, y)
                self.press_key = "pieces"
                self.sliding = True
                self.set_size(x)
                return
            if self.sliding:
                if event == cv2.EVENT_MOUSEMOVE:
                    self.set_size(x)
                    return
                if event == cv2.EVENT_LBUTTONUP:
                    self.set_size(x)
                    self.sliding = False
                    self.press = None
                    self.press_key = None
                    return
        if self.paused:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.press = (x, y)
                self.press_key = self.zones.at(x, y)
                if self.press_key == "volume":
                    self.sliding = True
                    self.set_volume(x)
                    return
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.sliding:
                    self.set_volume(x)
                    return
            elif self.sliding and event == cv2.EVENT_LBUTTONUP:
                self.set_volume(x)
                self.sliding = False
                self.press = None
                self.press_key = None
                return
            if event == cv2.EVENT_LBUTTONUP:
                if self.press is not None:
                    moved = abs(x - self.press[0]) + abs(y - self.press[1])
                    key = self.zones.at(x, y)
                    if moved <= TAP_SLOP and key is not None \
                            and key == self.press_key and key != "panel":
                        self.tap_pause(key)
                self.press = None
                self.press_key = None
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.press = (x, y)
            self.press_key = self.zones.at(x, y)
            self.dragging = False
            if self.board is not None and not self.board.solved \
                    and self.press_key is None:
                self.dragging = self.board.pick(x, y)
                if self.dragging:
                    self.fx.lift(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.board is not None:
                self.board.drag(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging and self.board is not None:
                b = self.board
                held = b.pieces[b.held] if b.held is not None else None
                if b.drop():
                    frac = b.done / max(1, b.total)
                    if held is not None:
                        cx = held.pos[0] + held.w * 0.5
                        cy = held.pos[1] + held.h * 0.5
                        rad = min(b.cut.cell_w, b.cut.cell_h) * 0.5
                        self.fx.place(cx, cy, rad, draw.SPARK,
                                      0.8 + frac * 0.5)
                        pan = float(np.clip(
                            (cx / max(1, self.w)) * 2.0 - 1.0, -1, 1))
                    else:
                        pan = 0.0
                    if self.audio is not None:
                        self.audio.place(frac, pan * 0.6)
                self.dragging = False
                self.press = None
                self.press_key = None
                return
            if self.press is None:
                return
            moved = abs(x - self.press[0]) + abs(y - self.press[1])
            key = self.zones.at(x, y)
            if moved <= TAP_SLOP and key is not None \
                    and key == self.press_key:
                if self.page == PICK:
                    self.tap_menu(key)
                elif self.page == SETUP:
                    self.tap_setup(key)
                elif self.solved_view():
                    self.tap_done(key)
                else:
                    self.tap_game(key)
            self.press = None
            self.press_key = None
    def frame(self, now: float) -> None:
        dt = now - self.last_t
        self.last_t = now
        self.fx.step(max(0.0, min(0.05, dt)))
        if self.settings:
            vol = self.audio.volume if self.audio is not None else 0
            mute = self.audio is None or self.audio.failed or vol <= 0
            draw.draw_settings(self.canvas, self.zones, vol, mute)
            return
        if self.page == PICK:
            draw.draw_menu(self.canvas, self.thumbs, self.names, self.pick,
                           SIZES, self.size_pick, self.style, self.zones,
                           self.sound_name(), self.size_options())
            return
        if self.page == SETUP:
            thumb = None
            if self.custom is None:
                thumb = self.thumbs.get(self.names[self.pick])
            draw.draw_setup(self.canvas, self.zones, SIZES, self.size_pick,
                            self.style, self.size_options(), thumb)
            return
        draw.draw_board(self.canvas, self.board, self.image, now, self.fx)
        if not self.board.solved:
            if self.paused:
                vol = self.audio.volume if self.audio is not None else 0
                mute = self.audio is None or self.audio.failed or vol <= 0
                draw.draw_pause(self.canvas, self.zones, vol, mute)
            else:
                draw.draw_hud(self.canvas, self.board, self.title,
                              self.zones, self.sound_name())
            return
        if not self.done_at:
            self.done_at = now
            self.done_zones()
            self.fx.finish(self.w, self.h)
            if self.audio is not None:
                self.audio.done()
        a = min(1.0, (now - self.done_at) / DONE_FADE)
        draw.draw_done(self.canvas, a, self.zones)
    def run(self) -> None:
        if self.audio is not None and self.audio.volume > 0:
            self.audio.start()
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                self.frame(time.monotonic())
                cv2.imshow(WINDOW, self.canvas)
                if cv2.waitKey(16) == 27:
                    if self.settings:
                        self.close_settings()
                        continue
                    if self.paused:
                        self.close_pause()
                        continue
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if self.audio is not None:
                self.audio.stop()
            cv2.destroyAllWindows()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=860)
    ap.add_argument("--cut", choices=[INTERLOCK, SQUARE], default=INTERLOCK)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sound", type=int, default=audio_mod.VOL_DEFAULT)
    ap.add_argument("--no-sound", action="store_true")
    a = ap.parse_args()
    App(a.width, a.height, a.cut, a.seed, sound=a.sound,
        audio_on=not a.no_sound).run()
if __name__ == "__main__":
    main()
