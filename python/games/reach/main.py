from __future__ import annotations
import argparse
import sys
import time
import cv2
from angles import IDX
from body import BodyTracker
from camera import CameraStream, shape_frame
from draw import (COLOR_ENDING, draw_body, draw_flash, draw_hud, draw_targets,
                  draw_waiting)
from smooth import PointSmoother
from game import CALM, ENDING, FINISHED, PAUSED, RUNNING, TIMED, ReachGame
from settings import (Screen, StartMenu, draw_pause_menu, draw_summary,
                      pause_zones, summary_zones)
from targets import LIFETIME
import audio as audio_mod
SUMMARY_FADE = 0.9
STOP_W = 118
STOP_H = 46
STOP_PAD = 20
def stop_button(w: int, h: int):
    from settings import Rect, Zones
    z = Zones()
    z.add("stop", Rect(w - STOP_W - STOP_PAD, STOP_PAD, STOP_W, STOP_H))
    return z
def draw_stop(canvas, zones) -> None:
    from settings import CARD, DIM, EDGE, _round_rect, _text_mid
    if "stop" not in zones:
        return
    r = zones["stop"]
    _round_rect(canvas, r, CARD, edge=EDGE)
    cv2.rectangle(canvas, (r.x + 22, r.cy - 8), (r.x + 34, r.cy + 4),
                  DIM, -1, cv2.LINE_AA)
    _text_mid(canvas, "stop", r, 0.44, DIM)
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", choices=(TIMED, CALM), default=TIMED)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--lifetime", type=float, default=LIFETIME)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fit", choices=("fit", "fill"), default="fill")
    parser.add_argument("--body", action="store_true")
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--sound", type=int, default=audio_mod.VOL_DEFAULT)
    parser.add_argument("--no-sound", action="store_true")
    args = parser.parse_args()
    stream = CameraStream(args.camera)
    if not stream.opened:
        return 1
    tracker = BodyTracker()
    smoother = None if args.no_smoothing else PointSmoother()
    if not tracker.available:
        stream.release()
        return 1
    game = ReachGame(mode=args.mode, duration=args.duration,
                     lifetime=args.lifetime)
    menu = StartMenu(args.width, args.height,
                     minutes=max(1, int(round(args.duration / 60.0))),
                     mode=args.mode)
    sound = None if args.no_sound else audio_mod.Audio(volume=args.sound)
    if sound is not None and sound.volume > 0:
        sound.start()
    view = {"w": args.width, "h": args.height}
    stop = stop_button(args.width, args.height)
    stop_screen = Screen(stop)
    pause_screen = Screen(pause_zones(args.width, args.height))
    over_screen = Screen(summary_zones(args.width, args.height))
    window = "reach"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.width, args.height)
    start = time.perf_counter()
    last_seq = -1
    ended = False
    finished_at = 0.0
    quitting = {"now": False}
    def relayout(w: int, h: int) -> None:
        view["w"], view["h"] = w, h
        menu.resize(w, h)
        stop_screen.zones = stop_button(w, h)
        pause_screen.zones = pause_zones(w, h)
        over_screen.zones = summary_zones(w, h)
    def start_session() -> None:
        nonlocal start, ended
        game.mode = menu.mode
        game.duration = menu.minutes * 60.0
        game.reset()
        start = time.perf_counter()
        ended = False
        menu.open = False
    def to_menu() -> None:
        menu.open = True
    def on_mouse(event, x, y, flags, param):
        if menu.open:
            hit = menu.on_mouse(event, x, y, flags, param)
            if hit == "begin":
                start_session()
            return
        phase = game.state.phase
        if phase == FINISHED:
            hit = over_screen.on_mouse(event, x, y, flags, param)
            if hit == "again":
                start_session()
            elif hit == "menu":
                to_menu()
            elif hit == "close":
                quitting["now"] = True
            return
        if phase == PAUSED:
            hit = pause_screen.on_mouse(event, x, y, flags, param)
            if hit == "resume":
                game.resume(time.perf_counter() - start)
            elif hit == "finish":
                game.begin_ending(time.perf_counter() - start)
            return
        if phase == RUNNING:
            hit = stop_screen.on_mouse(event, x, y, flags, param)
            if hit == "stop":
                game.pause(time.perf_counter() - start)
    cv2.setMouseCallback(window, on_mouse)
    while not quitting["now"]:
        raw, seq = stream.read()
        if raw is None:
            if not stream.opened:
                break
            time.sleep(0.002)
            continue
        if seq == last_seq:
            time.sleep(0.001)
            continue
        last_seq = seq
        frame = shape_frame(raw, args.width, args.height, args.fit,
                            mirror=not args.no_mirror)
        h, w = frame.shape[:2]
        if w != view["w"] or h != view["h"]:
            relayout(w, h)
        if menu.open:
            menu.draw(frame, ready=True)
            cv2.imshow(window, frame)
            cv2.waitKey(1)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        now = time.perf_counter() - start
        tracker.update(frame, now)
        drawn = tracker.points
        if smoother is not None and drawn is not None:
            drawn = smoother(drawn, tracker.visibility, now)
        scored = game.update(drawn, tracker.visibility, IDX, tracker.torso,
                             now, w, h)
        if sound is not None:
            for hit in scored:
                if hit.bonus:
                    sound.reward()
                else:
                    sound.hit()
        if args.body:
            unit = (tracker.torso / 22.0) if tracker.torso else 4.0
            draw_body(frame, drawn, tracker.visibility, tracker.discarded,
                      unit)
        phase = game.state.phase
        if phase == FINISHED:
            if not ended:
                if sound is not None:
                    sound.done()
                finished_at = now
                ended = True
            fade = min(1.0, (now - finished_at) / SUMMARY_FADE)
            draw_summary(frame, over_screen.zones, game.summary(now),
                         game.mode, fade)
        elif phase == ENDING:
            fade = 1.0 - game.ending_progress(now)
            draw_targets(frame, game.targets, now, fade=fade,
                         colour_override=COLOR_ENDING)
            draw_hud(frame, game, now)
        elif phase == PAUSED:
            draw_pause_menu(frame, pause_screen.zones)
        else:
            draw_targets(frame, game.targets, now)
            draw_flash(frame, game.state, now)
            draw_hud(frame, game, now)
            if tracker.torso is None:
                draw_waiting(frame)
            draw_stop(frame, stop_screen.zones)
        cv2.imshow(window, frame)
        cv2.waitKey(1)
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    if sound is not None:
        sound.stop()
    tracker.close()
    stream.release()
    cv2.destroyAllWindows()
    return 0
if __name__ == "__main__":
    sys.exit(main())
