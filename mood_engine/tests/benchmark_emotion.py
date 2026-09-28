from __future__ import annotations

import argparse
import collections
import csv
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from engine.config import EMOTIONS
from engine.core.detector import FaceDetector
from engine.core.emotion import EmotionAnalyzer

MANIFEST = ROOT / "data" / "fer_eval" / "manifest.csv"


def load_manifest() -> list[tuple[Path, str]]:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing dataset manifest: {MANIFEST}")
    with MANIFEST.open(encoding="utf-8") as f:
        return [(Path(r["path"]), r["label"]) for r in csv.DictReader(f)]


def evaluate(detector: FaceDetector, analyzer: EmotionAnalyzer,
             samples: list[tuple[Path, str]], *, aligned: bool,
             label: str) -> dict:
    correct = 0
    total = 0
    no_face = 0
    per_class = collections.defaultdict(lambda: [0, 0])
    confusion = collections.Counter()
    margins: list[float] = []
    times: list[float] = []

    for index, (path, truth) in enumerate(samples):
        if index and index % 50 == 0:
            print(f"    ...{index}/{len(samples)}", flush=True)
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            continue
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue

        faces = detector.detect(img)
        if not faces:
            no_face += 1
            continue
        face = max(faces, key=lambda f: f.area)

        crop = (detector.align(face, img) if aligned
                else face.crop(img, margin=0.15))

        t0 = time.perf_counter()
        result = analyzer.analyse(crop, face.landmarks_2d, smooth=False)
        times.append((time.perf_counter() - t0) * 1000)

        total += 1
        per_class[truth][1] += 1
        if result.label == truth:
            correct += 1
            per_class[truth][0] += 1
        else:
            confusion[(truth, result.label)] += 1

        ranked = sorted(result.probs.values(), reverse=True)
        margins.append(ranked[0] - ranked[1])

    accuracy = correct / total if total else 0.0
    balanced = float(np.mean([c / t for c, t in per_class.values() if t]))

    print(f"\n=== {label} ===")
    print(f"  images scored : {total}  (no face: {no_face})")
    print(f"  accuracy      : {accuracy:.1%}")
    print(f"  balanced acc  : {balanced:.1%}")
    print(f"  mean margin   : {np.mean(margins):.3f}  "
          f"(higher = more stable label)")
    print(f"  inference     : {np.mean(times):.1f} ms")
    print("  per class:")
    for emotion in EMOTIONS:
        c, t = per_class.get(emotion, (0, 0))
        if t:
            print(f"    {emotion:9s} {c:3d}/{t:3d}  {c / t:5.1%}")
    top = confusion.most_common(5)
    if top:
        print("  most common confusions:")
        for (truth, got), n in top:
            print(f"    {truth:9s} -> {got:9s}  {n}")

    return {
        "accuracy": accuracy,
        "balanced": balanced,
        "margin": float(np.mean(margins)),
        "ms": float(np.mean(times)),
        "total": total,
    }


def balanced_sample(samples: list[tuple[Path, str]],
                    limit: int) -> list[tuple[Path, str]]:
    if limit <= 0 or limit >= len(samples):
        return samples
    by_class: dict[str, list[tuple[Path, str]]] = collections.defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)
    per_class = max(1, limit // max(1, len(by_class)))
    rng = random.Random(42)
    chosen: list[tuple[Path, str]] = []
    for group in by_class.values():
        rng.shuffle(group)
        chosen.extend(group[:per_class])
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=210,
                        help="images to score (0 = all). Keeps runs bounded.")
    args = parser.parse_args()

    samples = balanced_sample(load_manifest(), args.limit)
    print(f"Dataset: {len(samples)} labelled images", flush=True)

    detector = FaceDetector()
    detector.load()
    analyzer = EmotionAnalyzer()
    print(f"Emotion backend: {analyzer.backend}", flush=True)
    print(f"Face pack      : {detector.config.model_pack}", flush=True)

    has_align = hasattr(detector, "align")
    before = evaluate(detector, analyzer, samples, aligned=False,
                      label="RAW CROP (current)")
    if has_align:
        after = evaluate(detector, analyzer, samples, aligned=True,
                         label="ALIGNED CROP (new)")
        print("\n=== DELTA ===")
        print(f"  accuracy : {before['accuracy']:.1%} -> {after['accuracy']:.1%}"
              f"  ({after['accuracy'] - before['accuracy']:+.1%})")
        print(f"  balanced : {before['balanced']:.1%} -> {after['balanced']:.1%}"
              f"  ({after['balanced'] - before['balanced']:+.1%})")
        print(f"  margin   : {before['margin']:.3f} -> {after['margin']:.3f}"
              f"  ({after['margin'] - before['margin']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
