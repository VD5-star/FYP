from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.config import CONFIG, EMOTIONS_AR
from engine.core import MoodEngine
from engine.core.trainer import FaceTrainer
from engine.db import MoodDatabase


_EMOTION_COLOURS = {
    "happy": (80, 220, 100), "sad": (200, 130, 60),
    "anger": (60, 60, 235), "fear": (170, 90, 200),
    "surprise": (60, 210, 240), "disgust": (90, 160, 110),
    "neutral": (190, 190, 190),
}


def _draw_overlay(frame: np.ndarray, r) -> np.ndarray:
    out = frame
    if not r.ok or r.bbox is None:
        cv2.putText(out, "No face", (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (60, 60, 235), 2)
        return out

    x1, y1, x2, y2 = r.bbox
    emo = r.emotion.label if r.emotion else "?"
    colour = _EMOTION_COLOURS.get(emo, (200, 200, 200))
    if r.spoof and r.spoof.is_spoof:
        colour = (0, 0, 255)

    cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

    name = r.person_name or "?"
    if r.spoof and r.spoof.is_spoof:
        name += "  [SPOOF?]"
    cv2.rectangle(out, (x1, max(0, y1 - 28)), (x1 + 260, y1), colour, -1)
    cv2.putText(out, f"{name}  {r.match_score:.2f}",
                (x1 + 6, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (20, 20, 20), 2)

    if r.emotion:
        cv2.putText(out, f"{emo} {r.emotion.confidence:.2f}",
                    (x1, y2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    h = out.shape[0]
    lines = []
    if r.emotion:
        lines.append(f"V {r.emotion.valence:+.2f}  A {r.emotion.arousal:.2f}")
    if r.pose:
        lines.append(f"yaw {r.pose.yaw:+.0f}  pitch {r.pose.pitch:+.0f}")
        lines.append(f"attention {r.pose.attention:.2f}")
    if r.spoof:
        lines.append(f"liveness {r.spoof.liveness:.2f}")
    if r.age is not None:
        lines.append(f"age {r.age:.0f}  {r.gender or '?'}")
    lines.append(f"quality {r.quality:.2f}  {r.elapsed_ms:.0f} ms")

    for i, text in enumerate(lines):
        cv2.putText(out, text, (12, h - 12 - (len(lines) - 1 - i) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1,
                    cv2.LINE_AA)
    return out


def cmd_live(args: argparse.Namespace) -> int:
    engine = MoodEngine(auto_enrol_unknown=not args.no_enrol)
    print("Loading models ...")
    engine.load()
    print(f"Emotion backend : {engine.emotion.backend}")
    print(f"Known people    : {engine.db.stats()['persons']}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    engine.start_session("cli-live")
    print("\n[q] quit   [n] name current person   [r] reset tracking\n")

    every = max(1, args.every)
    last = None
    frames = 0
    t_start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            if frames % every == 0:
                last = engine.analyse_frame(frame, persist=not args.no_save)
            if last is not None:
                frame = _draw_overlay(frame, last)
            fps = frames / max(1e-6, time.time() - t_start)
            cv2.putText(frame, f"{fps:.1f} fps", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)

            if not args.headless:
                cv2.imshow("Mood Engine", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    engine.reset()
                    print("tracking reset")
                if key == ord("n") and last and last.person_id:
                    cv2.destroyWindow("Mood Engine")
                    new = input(f"New name for '{last.person_name}': ").strip()
                    if new:
                        engine.rename_person(last.person_id, new)
                        print(f"renamed -> {new}")
            elif args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.end_session()
        trend = engine.trend()
        print("\n--- session ---")
        print(f"frames    : {frames}")
        print(f"dominant  : {trend.get('dominant')} "
              f"({trend.get('stability', 0):.0%} stable)")
        print(f"avg V/A   : {trend.get('avg_valence', 0):+.2f} / "
              f"{trend.get('avg_arousal', 0):.2f}")
        engine.close()
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    engine = MoodEngine()
    engine.load()
    trainer = FaceTrainer(engine, min_quality=args.min_quality,
                          max_per_person=args.max_per_person)

    root = Path(args.folder)
    groups = trainer.discover(root)
    if not groups:
        print(f"No images found under {root}", file=sys.stderr)
        engine.close()
        return 2

    print(f"Found {len(groups)} people, "
          f"{sum(len(v) for v in groups.values())} images\n")

    def progress(label: str, done: int, total: int) -> None:
        pct = done / max(1, total) * 100
        print(f"\r[{done}/{total}] {pct:5.1f}%  {label[:52]:<52}",
              end="", flush=True)

    report = trainer.import_folder(root, progress=progress,
                                   dry_run=args.dry_run)
    print("\n\n--- import ---")
    print(json.dumps(report.summary(), indent=2, ensure_ascii=False))

    if args.verbose:
        for it in report.items:
            if not it.ok:
                print(f"  {it.reason:18s} {Path(it.path).name}")
    engine.close()
    return 0


def cmd_persons(args: argparse.Namespace) -> int:
    db = MoodDatabase()
    people = db.list_persons()
    if not people:
        print("No people enrolled yet.")
    else:
        print(f"{'id':>4}  {'name':<24} {'vectors':>7} {'seen':>7}  status")
        print("-" * 62)
        for p in people:
            status = "provisional" if p["is_provisional"] else "named"
            print(f"{p['id']:>4}  {p['name']:<24} "
                  f"{p['embedding_count']:>7} {p['observation_count']:>7}  "
                  f"{status}")
    db.close()
    return 0


def cmd_rename(args: argparse.Namespace) -> int:
    engine = MoodEngine()
    person = engine.db.get_person(args.person_id)
    if not person:
        print(f"No person with id {args.person_id}", file=sys.stderr)
        engine.close()
        return 2
    engine.rename_person(args.person_id, args.name)
    print(f"{person['name']} -> {args.name}")
    engine.close()
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    engine = MoodEngine(auto_enrol_unknown=False)
    engine.load()
    img = FaceTrainer._read_image(Path(args.image))
    if img is None:
        print(f"Could not read {args.image}", file=sys.stderr)
        engine.close()
        return 2
    r = engine.analyse_frame(img, persist=not args.no_save)
    print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    engine.close()
    return 0 if r.ok else 1


def cmd_history(args: argparse.Namespace) -> int:
    db = MoodDatabase()
    rows = db.recent_observations(person_id=args.person_id, limit=args.limit)
    if not rows:
        print("No observations recorded yet.")
    else:
        print(f"{'time':<20} {'person':<18} {'emotion':<10} {'conf':>5} "
              f"{'V':>6} {'A':>5}")
        print("-" * 70)
        for o in rows:
            print(f"{str(o['ts'])[:19]:<20} {str(o['person_name'] or '-'):<18} "
                  f"{o['emotion']:<10} {o['emotion_conf']:>5.2f} "
                  f"{o['valence']:>+6.2f} {o['arousal']:>5.2f}")
        print("\n--- distribution ---")
        for s in db.emotion_summary(person_id=args.person_id):
            ar = EMOTIONS_AR.get(s["emotion"], "")
            print(f"  {s['emotion']:<10} {ar:<8} n={s['n']:<5} "
                  f"conf={s['avg_conf']:.2f}")
    db.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    engine = MoodEngine()
    if args.full:
        engine.load()
    print(json.dumps(engine.stats(), indent=2, ensure_ascii=False))
    engine.close()
    return 0


def cmd_wipe(args: argparse.Namespace) -> int:
    db = MoodDatabase()
    before = db.stats()
    print("This permanently erases all stored faces and history:")
    print(json.dumps(before, indent=2))
    if not args.yes:
        if input("\nType DELETE to confirm: ").strip() != "DELETE":
            print("Cancelled.")
            db.close()
            return 1
    removed = db.wipe_all()
    print("Removed:", json.dumps(removed, indent=2))
    db.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mood-engine",
        description="Live face, identity and emotion analysis engine.")
    sub = p.add_subparsers(dest="command", required=True)

    lv = sub.add_parser("live", help="analyse the camera feed live")
    lv.add_argument("--camera", type=int, default=CONFIG.camera_index)
    lv.add_argument("--width", type=int, default=640)
    lv.add_argument("--height", type=int, default=480)
    lv.add_argument("--every", type=int, default=CONFIG.analyse_every_n_frames,
                    help="analyse every Nth frame")
    lv.add_argument("--no-save", action="store_true",
                    help="do not write observations to the database")
    lv.add_argument("--no-enrol", action="store_true",
                    help="do not auto-create Unknown-N people")
    lv.add_argument("--headless", action="store_true")
    lv.add_argument("--max-frames", type=int, default=0)
    lv.set_defaults(func=cmd_live)

    im = sub.add_parser("import", help="learn people from a photo folder")
    im.add_argument("folder")
    im.add_argument("--min-quality", type=float, default=0.35)
    im.add_argument("--max-per-person", type=int, default=None)
    im.add_argument("--dry-run", action="store_true")
    im.add_argument("--verbose", "-v", action="store_true")
    im.set_defaults(func=cmd_import)

    ps = sub.add_parser("persons", help="list known people")
    ps.set_defaults(func=cmd_persons)

    rn = sub.add_parser("rename", help="rename a person")
    rn.add_argument("person_id", type=int)
    rn.add_argument("name")
    rn.set_defaults(func=cmd_rename)

    an = sub.add_parser("analyse", help="analyse a single image")
    an.add_argument("image")
    an.add_argument("--no-save", action="store_true")
    an.set_defaults(func=cmd_analyse)

    hs = sub.add_parser("history", help="show recent observations")
    hs.add_argument("--person-id", type=int, default=None)
    hs.add_argument("--limit", type=int, default=20)
    hs.set_defaults(func=cmd_history)

    st = sub.add_parser("stats", help="database and model status")
    st.add_argument("--full", action="store_true", help="also load models")
    st.set_defaults(func=cmd_stats)

    wp = sub.add_parser("wipe", help="erase all stored data")
    wp.add_argument("--yes", action="store_true")
    wp.set_defaults(func=cmd_wipe)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
