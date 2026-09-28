from __future__ import annotations

import argparse
import collections
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from engine.core import MoodEngine
from engine.db import MoodDatabase

CANDIDATES = [
    ROOT / "data" / "emo" / "sad1.jpg",
    ROOT / "data" / "emo" / "neutral1.jpg",
    ROOT / "data" / "emo" / "angry1.jpg",
    ROOT / "data" / "emo" / "surprise1.jpg",
    ROOT / "data" / "emo" / "happy1.jpg",
    ROOT / "data" / "ds_raw" / "biden.jpg",
]


def synthesise(image: np.ndarray, frames: int) -> list[np.ndarray]:
    h, w = image.shape[:2]
    rng = np.random.default_rng(7)
    sequence = []
    for i in range(frames):
        phase = i / max(1, frames - 1) * 2 * np.pi
        angle = 7.0 * np.sin(phase)
        scale = 1.0 + 0.06 * np.sin(phase * 1.7)
        shift_x = 14.0 * np.sin(phase * 0.9)
        shift_y = 8.0 * np.cos(phase * 1.3)

        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y
        frame = cv2.warpAffine(image, matrix, (w, h),
                               borderMode=cv2.BORDER_REPLICATE)

        frame = cv2.convertScaleAbs(
            frame, alpha=1.0 + 0.10 * np.sin(phase * 2.1),
            beta=10.0 * np.sin(phase * 1.1))
        noise = rng.normal(0, 3.0, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        sequence.append(frame)
    return sequence


def run(engine: MoodEngine, frames: list[np.ndarray], label: str) -> dict:
    engine.reset()
    labels: list[str] = []
    valence: list[float] = []
    identities: list[str] = []
    detected = 0

    for frame in frames:
        result = engine.analyse_frame(frame, persist=False, save_snapshot=False)
        if not result.ok or result.emotion is None:
            continue
        detected += 1
        labels.append(result.emotion.label)
        valence.append(result.emotion.valence)
        identities.append(result.person_name or "?")

    if not labels:
        return {"label": label, "detected": 0}

    switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    dominant = collections.Counter(labels).most_common(1)[0]
    id_switches = sum(1 for a, b in zip(identities, identities[1:]) if a != b)

    return {
        "label": label,
        "detected": detected,
        "of": len(frames),
        "switches": switches,
        "switch_rate": switches / max(1, len(labels) - 1),
        "dominant": dominant[0],
        "dominant_share": dominant[1] / len(labels),
        "valence_std": float(np.std(valence)),
        "distinct": len(set(labels)),
        "id_switches": id_switches,
    }


def aggregate(rows: list[dict], label: str) -> dict:
    scored = [r for r in rows if r.get("detected")]
    if not scored:
        return {"label": label, "detected": 0}
    return {
        "label": label,
        "detected": sum(r["detected"] for r in scored),
        "of": sum(r["of"] for r in scored),
        "switches": sum(r["switches"] for r in scored),
        "switch_rate": float(np.mean([r["switch_rate"] for r in scored])),
        "dominant_share": float(np.mean([r["dominant_share"] for r in scored])),
        "valence_std": float(np.mean([r["valence_std"] for r in scored])),
        "id_switches": sum(r["id_switches"] for r in scored),
        "faces": len(scored),
    }


def report(row: dict) -> None:
    if not row.get("detected"):
        print(f"  {row['label']:24s} no detections")
        return
    print(f"  {row['label']:24s} "
          f"detected {row['detected']:4d}/{row['of']:<4d} "
          f"switches {row['switches']:4d} ({row['switch_rate']:5.1%})  "
          f"dominant share {row['dominant_share']:5.1%}  "
          f"valence sd {row['valence_std']:.3f}  "
          f"identity flips {row['id_switches']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()

    sources = [p for p in CANDIDATES if p.exists()]
    if not sources:
        raise SystemExit("no sample images available")

    sequences = []
    for path in sources:
        image = cv2.imread(str(path))
        if image is None:
            continue
        sequences.append((path.name,
                          synthesise(cv2.resize(image, (640, 480)),
                                     args.frames)))
    print(f"faces  : {len(sequences)}  ({', '.join(n for n, _ in sequences)})")
    print(f"frames : {args.frames} per face\n")

    db_path = ROOT / "data" / "_stability.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)

    engine = MoodEngine(db=MoodDatabase(db_path), auto_enrol_unknown=True)
    engine.load()

    print("Lower switch rate is better. No face changes expression, so every")
    print("switch is the engine disagreeing with itself.\n")

    from dataclasses import replace
    original = engine.emotion.config
    configs = [
        ("current pipeline", original),
        ("without anti-flicker",
         replace(original, switch_margin=0.0, switch_frames=1)),
        ("raw, no smoothing",
         replace(original, smoothing_alpha=1.0, smoothing_alpha_min=1.0,
                 switch_margin=0.0, switch_frames=1)),
    ]

    try:
        for label, config in configs:
            engine.emotion.config = config
            rows = [run(engine, frames, name) for name, frames in sequences]
            report(aggregate(rows, label))
        engine.emotion.config = original
    finally:
        engine.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
