from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import pose_sim as sim

HERE = Path(__file__).parent
MODELS = {
    "lite": HERE / "models" / "pose_landmarker_lite.task",
    "full": HERE / "models" / "pose_landmarker_full.task",
    "heavy": HERE / "models" / "pose_landmarker_heavy.task",
}
RESULTS = HERE / "results"

DEMPSTER = [
    (0.430, ["leftShoulder", "rightShoulder", "leftHip", "rightHip"]),  # trunk+head
    (0.100, ["leftHip", "leftKnee"]),      # left thigh
    (0.100, ["rightHip", "rightKnee"]),    # right thigh
    (0.0465, ["leftKnee", "leftAnkle"]),   # left shank
    (0.0465, ["rightKnee", "rightAnkle"]),
    (0.015, ["leftAnkle", "leftFootIndex"]),  # left foot
    (0.015, ["rightAnkle", "rightFootIndex"]),
    (0.028, ["leftShoulder", "leftElbow"]),   # left upper arm
    (0.028, ["rightShoulder", "rightElbow"]),
    (0.022, ["leftElbow", "leftWrist"]),      # left forearm+hand
    (0.022, ["rightElbow", "rightWrist"]),
]

TRACKED_ANGLES = {
    "leftElbow": ("leftElbow", "leftShoulder", "leftWrist"),
    "rightElbow": ("rightElbow", "rightShoulder", "rightWrist"),
    "leftShoulder": ("leftShoulder", "leftHip", "leftElbow"),
    "rightShoulder": ("rightShoulder", "rightHip", "rightElbow"),
    "leftKnee": ("leftKnee", "leftHip", "leftAnkle"),
    "rightKnee": ("rightKnee", "rightHip", "rightAnkle"),
    "leftHip": ("leftHip", "leftShoulder", "leftKnee"),
    "rightHip": ("rightHip", "rightShoulder", "rightKnee"),
}


def centre_of_mass(points: np.ndarray, visibility: np.ndarray) -> np.ndarray | None:
    """Mass-weighted centre of mass from the Dempster segment model.

    Returns None when too much of the body is unreliable to be worth averaging:
    a CoM computed from three visible landmarks is not the quantity the
    literature validated.
    """
    total_mass = 0.0
    acc = np.zeros(2)
    for mass, names in DEMPSTER:
        idxs = [sim.IDX[n] for n in names]
        if any(visibility[i] < 0.5 for i in idxs):
            continue
        acc += mass * points[idxs].mean(axis=0)
        total_mass += mass
    if total_mass < 0.6:
        return None
    return acc / total_mass


@dataclass
class Sample:
    """One frame's worth of measurements."""
    t: float
    latency_ms: float
    detected: bool
    angles: dict
    com: list | None
    torso_scale: float | None
    landmarks: list | None
    visibility: list | None


def circular_safe_std(values: list[float]) -> float:
    """Standard deviation of an angle series.

    Plain std is correct here because these are interior angles constrained to
    0..180 - there is no wraparound at 0/360 to worry about. Named explicitly so
    nobody later "fixes" it into a circular statistic that would be wrong.
    """
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def run_benchmark(model_key: str, seconds: float, camera: int,
                  label: str, warmup: float = 2.0,
                  video: str | None = None) -> dict:
    """Measure one backend, from the camera or from a recorded clip.

    The video path exists for two reasons. It is the only way to work when the
    camera is unavailable - but more importantly it is the only way to compare
    backends on *identical* input. Two camera runs never see the same frames, so
    part of any difference between them is the subject having moved. Replaying
    one clip removes that variable entirely, which is what makes a difference
    between models attributable to the models.
    """
    path = MODELS[model_key]
    if not path.exists():
        raise SystemExit(f"Model missing: {path}")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    from_video = video is not None
    if from_video:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open clip: {video}")
        warmup = 0.0
    else:
        cap = sim.open_camera(camera)

    samples: list[Sample] = []
    started = time.time()
    frame_index = 0

    print(f"\n[{model_key}] {label}")
    if from_video:
        print(f"  replaying {Path(video).name} ...")
    else:
        print(f"  warming up {warmup:.0f}s, then measuring {seconds:.0f}s ...")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if from_video:
                h, w = frame.shape[:2]
                elapsed = frame_index / 30.0
            else:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                elapsed = time.time() - started
                if elapsed > warmup + seconds:
                    break
            frame_index += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(image, int(elapsed * 1000))
            latency_ms = (time.perf_counter() - t0) * 1000

            if (from_video and frame_index <= 10) or (not from_video and elapsed < warmup):
                continue

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                norm = np.array([[p.x, p.y] for p in lm], dtype=np.float64)
                vis = np.array([p.visibility for p in lm], dtype=np.float64)

                angles = {}
                for name, (v, a, b) in TRACKED_ANGLES.items():
                    ang = sim.joint_angle(norm, vis, sim.IDX[v], sim.IDX[a], sim.IDX[b])
                    if ang is not None:
                        angles[name] = ang

                px = norm * np.array([w, h])
                com = centre_of_mass(px, vis)
                tf = sim.torso_frame(norm, vis)

                samples.append(Sample(
                    t=elapsed, latency_ms=latency_ms, detected=True,
                    angles=angles,
                    com=None if com is None else com.tolist(),
                    torso_scale=None if tf is None else tf[1],
                    landmarks=px.tolist(),
                    visibility=vis.tolist(),
                ))
            else:
                samples.append(Sample(
                    t=elapsed, latency_ms=latency_ms, detected=False,
                    angles={}, com=None, torso_scale=None,
                    landmarks=None, visibility=None,
                ))

    cap.release()
    duration = (time.time() - started) if from_video else seconds
    result = analyse(model_key, label, samples, duration)
    result["source"] = Path(video).name if from_video else f"camera{camera}"
    return result


def analyse(model_key: str, label: str, samples: list[Sample],
            seconds: float) -> dict:
    """Reduce the raw samples to the numbers that decide the choice."""
    detected = [s for s in samples if s.detected]
    latencies = [s.latency_ms for s in samples]

    out: dict = {
        "model": model_key,
        "label": label,
        "frames": len(samples),
        "detected_frames": len(detected),
        "detection_rate": len(detected) / len(samples) if samples else 0.0,
        "fps": len(samples) / seconds if seconds else 0.0,
    }

    if latencies:
        out["latency_ms"] = {
            "best": round(min(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
        }

    angle_stats = {}
    for name in TRACKED_ANGLES:
        series = [s.angles[name] for s in detected if name in s.angles]
        if len(series) < 10:
            continue
        angle_stats[name] = {
            "n": len(series),
            "mean": round(statistics.mean(series), 2),
            "std": round(circular_safe_std(series), 3),
            "range": round(max(series) - min(series), 2),
            "p95_step": round(
                sorted(abs(b - a) for a, b in zip(series, series[1:]))[
                    max(0, int((len(series) - 1) * 0.95) - 1)
                ], 3
            ) if len(series) > 2 else 0.0,
        }
    out["angles"] = angle_stats

    if angle_stats:
        stds = [v["std"] for v in angle_stats.values()]
        out["angle_noise_summary"] = {
            "worst_joint": max(angle_stats, key=lambda k: angle_stats[k]["std"]),
            "worst_std": round(max(stds), 3),
            "median_std": round(statistics.median(stds), 3),
            "best_std": round(min(stds), 3),
        }

    coms = [s.com for s in detected if s.com is not None]
    scales = [s.torso_scale for s in detected if s.torso_scale]
    if len(coms) > 10 and scales:
        arr = np.array(coms)
        com_std_px = float(np.mean([arr[:, 0].std(), arr[:, 1].std()]))
        out["com_stability_px"] = round(com_std_px, 3)

        for joint in ("leftWrist", "leftAnkle", "leftKnee"):
            idx = sim.IDX[joint]
            pts = np.array([s.landmarks[idx] for s in detected
                            if s.landmarks is not None])
            if len(pts) > 10:
                out[f"{joint}_stability_px"] = round(
                    float(np.mean([pts[:, 0].std(), pts[:, 1].std()])), 3
                )

    with_lms = [s for s in detected if s.landmarks is not None]
    if len(with_lms) > 10:
        arr = np.array([s.landmarks for s in with_lms])
        vis = np.array([s.visibility for s in with_lms])
        jitter = {}
        for i, name in enumerate(sim.LANDMARK_NAMES):
            if float(vis[:, i].mean()) < 0.5:
                continue
            jitter[name] = round(
                float(np.mean([arr[:, i, 0].std(), arr[:, i, 1].std()])), 3
            )
        out["landmark_jitter_px"] = dict(
            sorted(jitter.items(), key=lambda kv: -kv[1])
        )

    return out


def print_report(r: dict) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {r['model'].upper()}  -  {r['label']}")
    print(f"{'=' * 62}")
    print(f"  fps                {r['fps']:.1f}")
    if "latency_ms" in r:
        l = r["latency_ms"]
        print(f"  inference ms       best {l['best']:.1f}   median {l['median']:.1f}   p95 {l['p95']:.1f}")
    print(f"  detection rate     {r['detection_rate'] * 100:.1f}%  "
          f"({r['detected_frames']}/{r['frames']})")

    if "angle_noise_summary" in r:
        s = r["angle_noise_summary"]
        print(f"\n  JOINT ANGLE NOISE FLOOR (still subject: all of this is error)")
        print(f"    best  {s['best_std']:.2f} deg   median {s['median_std']:.2f} deg"
              f"   worst {s['worst_std']:.2f} deg ({s['worst_joint']})")
        print(f"\n    {'joint':<16} {'std':>7} {'range':>8} {'p95 step':>9}")
        for name, v in sorted(r["angles"].items(), key=lambda kv: -kv[1]["std"]):
            print(f"    {name:<16} {v['std']:>7.2f} {v['range']:>8.1f} {v['p95_step']:>9.2f}")

    if "com_stability_px" in r:
        print(f"\n  STABILITY (pixel std, lower is steadier)")
        print(f"    centre of mass   {r['com_stability_px']:.2f}")
        for j in ("leftKnee", "leftAnkle", "leftWrist"):
            k = f"{j}_stability_px"
            if k in r:
                ratio = r[k] / r["com_stability_px"] if r["com_stability_px"] else 0
                print(f"    {j:<16} {r[k]:.2f}   ({ratio:.1f}x the CoM)")

    if "landmark_jitter_px" in r:
        items = list(r["landmark_jitter_px"].items())
        print(f"\n  NOISIEST LANDMARKS")
        for name, v in items[:6]:
            print(f"    {name:<18} {v:.2f} px")
        print(f"  STEADIEST")
        for name, v in items[-4:]:
            print(f"    {name:<18} {v:.2f} px")


def main() -> int:
    p = argparse.ArgumentParser(description="Measure pose backends on this machine")
    p.add_argument("--model", choices=list(MODELS), default="full")
    p.add_argument("--all", action="store_true", help="measure every model in turn")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--label", type=str, default="unlabelled")
    p.add_argument("--video", type=str, default=None,
                   help="measure against a recorded clip instead of the camera")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        for k, v in MODELS.items():
            size = f"{v.stat().st_size / 1e6:.1f} MB" if v.exists() else "MISSING"
            print(f"  {k:<8} {size:>10}  {v.name}")
        return 0

    RESULTS.mkdir(exist_ok=True)
    keys = list(MODELS) if args.all else [args.model]

    for key in keys:
        if args.all and not args.video:
            print(f"\n--- next: {key}. Hold the pose still. Starting in 3s ---")
            time.sleep(3)
        r = run_benchmark(key, args.seconds, args.camera, args.label,
                          video=args.video)
        print_report(r)
        stamp = time.strftime("%H%M%S")
        out = RESULTS / f"{key}_{args.label.replace(' ', '_')}_{stamp}.json"
        out.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\n  saved {out.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())