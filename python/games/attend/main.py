from __future__ import annotations
import argparse
import threading
import time
import cv2
import numpy as np
import audio as audio_mod
import draw
import scenes as scenes_mod
import session as S
import sounds
from session import Session
from ui import Rect, Zones
WINDOW = "attend"
TAP_SLOP = 8
CORNER_PAD = 22
OVER_FADE = 1.4
STOP_R = 34
STOP_PAD = 26
MENU = "menu"
RUN = "run"
OVER = "over"
SETTINGS = "settings"
class App:
    def __init__(self, width: int = 1100, height: int = 700,
                 minutes: int = S.DEFAULT_MINUTES, seed: int | None = None,
                 sound: int = audio_mod.VOL_DEFAULT,
                 audio_on: bool = True) -> None:
        self.w = width
        self.h = height
        self.minutes = S.clamp_minutes(minutes)
        self.seed = seed
        self.canvas = np.zeros((height, width, 3), np.uint8)
        self.zones = Zones()
        self.mode = MENU
        self.sess: Session | None = None
        self.scenes: dict = {}
        self.over_at = 0.0
        self.press: tuple[int, int] | None = None
        self.press_key: str | None = None
        self.last_t = time.monotonic()
        self.audio = audio_mod.Audio(volume=sound) if audio_on else None
        self.prev_mode = MENU
        self.dragging = False
        self.quitting = False
        self._lock = threading.Lock()
        self._scene_thread = threading.Thread(target=self._build_scenes,
                                              daemon=True)
        self._scene_thread.start()
        self.menu_zones()
    def _build_scenes(self) -> None:
        built = scenes_mod.render_all(self.w, self.h)
        with self._lock:
            self.scenes = built
    @property
    def ready(self) -> bool:
        with self._lock:
            have = len(self.scenes) == len(sounds.NAMES)
        if self.audio is None:
            return have
        return have and self.audio.ready.is_set()
    def wait_ready(self, timeout: float = 60.0) -> bool:
        self._scene_thread.join(timeout)
        if self.audio is not None:
            self.audio.wait(timeout)
        return self.ready
    def menu_zones(self) -> None:
        z = self.zones
        z.clear()
        sw = min(460, self.w - 120)
        y = int(self.h * 0.42)
        z.add("minutes", Rect((self.w - sw) // 2, y, sw, 56))
        z.add("begin", Rect((self.w - 220) // 2, y + 128, 220, 62))
        z.add("settings", Rect(self.w - 158, 28, 130, 46))
    def run_zones(self) -> None:
        z = self.zones
        z.clear()
        z.add("open", Rect(self.w - STOP_PAD - STOP_R * 2,
                           self.h // 2 - STOP_R, STOP_R * 2, STOP_R * 2))
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
        z.add("home", Rect(x0, y + bh + gap, bw, bh))
        z.add("quit", Rect(x1, y + bh + gap, bw, bh))
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
    def over_zones(self) -> None:
        z = self.zones
        z.clear()
        z.add("again", Rect(self.w // 2 - 190, self.h // 2 + 30, 170, 56))
        z.add("stop", Rect(self.w // 2 + 20, self.h // 2 + 30, 170, 56))
    def sound_name(self) -> str:
        if self.audio is None or self.audio.failed:
            return "no sound"
        return f"sound {self.audio.level_name}"
    def begin(self) -> None:
        if not self.ready:
            return
        self.silence()
        self.sess = Session(minutes=self.minutes, seed=self.seed)
        self.mode = RUN
        self.over_at = 0.0
        self.run_zones()
        if self.audio is not None and self.audio.volume > 0:
            self.audio.start()
    def to_menu(self) -> None:
        self.mode = MENU
        self.sess = None
        self.over_at = 0.0
        self.silence()
        self.menu_zones()
    def silence(self) -> None:
        if self.audio is not None:
            self.audio.stop()
    def open_settings(self) -> None:
        if self.mode == SETTINGS:
            return
        self.prev_mode = self.mode
        self.mode = SETTINGS
        self.settings_zones()
    def close_settings(self) -> None:
        self.dragging = False
        self.mode = self.prev_mode
        if self.mode == OVER:
            self.over_zones()
        else:
            self.menu_zones()
    def open_pause(self) -> None:
        if self.mode != RUN or self.sess is None:
            return
        if not self.sess.paused:
            self.sess.toggle_pause()
        self.silence()
        self.dragging = False
        self.pause_zones()
    def close_pause(self) -> None:
        self.dragging = False
        if self.sess is not None and self.sess.paused:
            self.sess.toggle_pause()
        self.run_zones()
        if self.audio is not None and self.audio.volume > 0:
            self.audio.start()
    @property
    def paused(self) -> bool:
        return self.mode == RUN and self.sess is not None \
            and self.sess.paused
    def set_minutes(self, x: int) -> None:
        if "minutes" not in self.zones:
            return
        self.minutes = draw.slider_value(self.zones["minutes"], x,
                                         S.MIN_MINUTES, S.MAX_MINUTES)
    def set_volume(self, x: int) -> None:
        if self.audio is None or "volume" not in self.zones:
            return
        v = draw.slider_value(self.zones["volume"], x,
                              audio_mod.VOL_MIN, audio_mod.VOL_MAX)
        self.audio.set_volume(v)
        if v <= audio_mod.VOL_MIN:
            self.audio.stop()
        elif self.mode == RUN and not self.paused and not self.audio.on:
            self.audio.start()
    def tap(self, key: str) -> None:
        if self.mode == SETTINGS:
            if key == "close":
                self.close_settings()
            return
        if self.mode == MENU:
            if key == "begin":
                self.begin()
            elif key == "settings":
                self.open_settings()
            return
        if self.mode == RUN:
            if key == "open":
                self.open_pause()
            elif key == "resume":
                self.close_pause()
            elif key == "again":
                self.begin()
            elif key == "home":
                self.to_menu()
            elif key == "quit":
                self.quitting = True
            return
        if key == "again":
            self.begin()
        elif key == "stop":
            self.to_menu()
    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.press = (x, y)
            self.press_key = self.zones.at(x, y)
            if self.press_key == "volume":
                self.dragging = "volume"
                self.set_volume(x)
            elif self.mode == MENU and self.press_key == "minutes":
                self.dragging = "minutes"
                self.set_minutes(x)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging == "volume":
                self.set_volume(x)
            elif self.dragging == "minutes":
                self.set_minutes(x)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                if self.dragging == "volume":
                    self.set_volume(x)
                else:
                    self.set_minutes(x)
                self.dragging = False
                self.press = None
                self.press_key = None
                return
            if self.press is None:
                return
            moved = abs(x - self.press[0]) + abs(y - self.press[1])
            key = self.zones.at(x, y)
            if moved <= TAP_SLOP and key is not None \
                    and key == self.press_key and key != "panel":
                self.tap(key)
            self.press = None
            self.press_key = None
    def frame(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        dt = max(0.0, min(0.25, now - self.last_t))
        self.last_t = now
        with self._lock:
            sc = self.scenes
        if self.mode == SETTINGS:
            vol = self.audio.volume if self.audio is not None else 0
            mute = self.audio is None or self.audio.failed or vol <= 0
            draw.draw_settings(self.canvas, self.zones, vol, mute)
            return
        if self.mode == MENU:
            draw.draw_menu(self.canvas, sc, self.minutes, self.zones,
                           self.sound_name(), self.ready)
            return
        if self.sess is None:
            self.to_menu()
            return
        self.sess.advance(dt)
        draw.draw_session(self.canvas, self.sess, sc, self.zones)
        if self.mode == RUN and "panel" in self.zones:
            vol = self.audio.volume if self.audio is not None else 0
            mute = self.audio is None or self.audio.failed or vol <= 0
            draw.draw_pause(self.canvas, self.zones, vol, mute)
            return
        if self.sess.finished:
            if self.mode != OVER:
                self.mode = OVER
                self.over_at = now
                self.silence()
                self.over_zones()
            a = min(1.0, (now - self.over_at) / OVER_FADE)
            draw.draw_over(self.canvas, sc, self.zones, a)
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                self.frame()
                if self.quitting:
                    break
                cv2.imshow(WINDOW, self.canvas)
                if cv2.waitKey(16) == 27:
                    if self.mode == SETTINGS:
                        self.close_settings()
                        continue
                    if "panel" in self.zones:
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
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument("--minutes", type=int, default=S.DEFAULT_MINUTES)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sound", type=int, default=audio_mod.VOL_DEFAULT)
    ap.add_argument("--no-sound", action="store_true")
    a = ap.parse_args()
    App(a.width, a.height, a.minutes, a.seed, sound=a.sound,
        audio_on=not a.no_sound).run()
if __name__ == "__main__":
    main()
