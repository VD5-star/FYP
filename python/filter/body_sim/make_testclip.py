from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "testdata"

W, H = 720, 1280  # portrait, as a propped-up phone would be


def draw_person(canvas: np.ndarray, knee_bend: float = 0.0,
                arm_raise: float = 0.0) -> dict:
    """Render a simple human figure and return its true joint positions.

    Deliberately crude but anatomically proportioned: MediaPipe was trained on
    photographs, so a rendered figure is out-of-distribution for it. That is
    acceptable and even useful - it stresses the detector rather than flattering
    it, and the question being asked is about *stability*, not accuracy.

    knee_bend  0 = standing, 1 = deep squat
    arm_raise  0 = arms down, 1 = arms horizontal
    """
    cx = W // 2
    ground = int(H * 0.92)

    head_r = int(H * 0.045)
    shoulder_y = int(H * 0.26)
    hip_y = int(H * 0.50 + knee_bend * H * 0.10)
    knee_y = int(H * 0.70 + knee_bend * H * 0.02)
    shoulder_w = int(W * 0.22)
    hip_w = int(W * 0.15)

    knee_forward = int(knee_bend * W * 0.06)

    joints = {
        "head": (cx, shoulder_y - head_r - int(H * 0.02)),
        "leftShoulder": (cx + shoulder_w // 2, shoulder_y),
        "rightShoulder": (cx - shoulder_w // 2, shoulder_y),
        "leftHip": (cx + hip_w // 2, hip_y),
        "rightHip": (cx - hip_w // 2, hip_y),
        "leftKnee": (cx + hip_w // 2 + knee_forward, knee_y),
        "rightKnee": (cx - hip_w // 2 - knee_forward, knee_y),
        "leftAnkle": (cx + hip_w // 2, ground),
        "rightAnkle": (cx - hip_w // 2, ground),
    }

    arm_len = int(H * 0.14)
    angle = math.radians(90 - arm_raise * 90)
    for side, sign in (("left", 1), ("right", -1)):
        sx, sy = joints[f"{side}Shoulder"]
        ex = sx + sign * int(arm_len * math.cos(angle) * 0.5)
        ey = sy + int(arm_len * math.sin(angle))
        joints[f"{side}Elbow"] = (ex, ey)
        joints[f"{side}Wrist"] = (
            ex + sign * int(arm_len * math.cos(angle) * 0.7),
            ey + int(arm_len * math.sin(angle) * 0.9),
        )

    skin = (150, 170, 195)
    shirt = (140, 95, 70)
    trouser = (70, 60, 55)

    def limb(a, b, colour, thickness):
        cv2.line(canvas, joints[a], joints[b], colour, thickness, cv2.LINE_AA)

    trunk = np.array([
        joints["leftShoulder"], joints["rightShoulder"],
        joints["rightHip"], joints["leftHip"],
    ], np.int32)
    cv2.fillPoly(canvas, [trunk], shirt, cv2.LINE_AA)

    limb("leftShoulder", "leftElbow", shirt, 26)
    limb("leftElbow", "leftWrist", skin, 22)
    limb("rightShoulder", "rightElbow", shirt, 26)
    limb("rightElbow", "rightWrist", skin, 22)
    limb("leftHip", "leftKnee", trouser, 34)
    limb("leftKnee", "leftAnkle", trouser, 28)
    limb("rightHip", "rightKnee", trouser, 34)
    limb("rightKnee", "rightAnkle", trouser, 28)

    hx, hy = joints["head"]
    cv2.circle(canvas, (hx, hy), head_r, skin, -1, cv2.LINE_AA)
    cv2.circle(canvas, (hx - head_r // 3, hy - head_r // 5), 4, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.circle(canvas, (hx + head_r // 3, hy - head_r // 5), 4, (40, 40, 40), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (hx, hy + head_r // 3), (head_r // 3, head_r // 6),
                0, 0, 180, (60, 60, 90), 3, cv2.LINE_AA)
    cv2.line(canvas, (hx, hy + head_r),
             (cx, shoulder_y), skin, 18, cv2.LINE_AA)

    for side in ("left", "right"):
        cv2.circle(canvas, joints[f"{side}Wrist"], 13, skin, -1, cv2.LINE_AA)
        ax, ay = joints[f"{side}Ankle"]
        sign = 1 if side == "left" else -1
        cv2.ellipse(canvas, (ax + sign * 12, ay), (26, 12), 0, 0, 360,
                    (40, 40, 45), -1, cv2.LINE_AA)

    return joints


def background(seed: int = 0) -> np.ndarray:
    """A plain but non-uniform background.

    A flat colour would be unrealistically easy: real rooms have gradients and
    texture, and a detector's stability depends partly on having consistent
    context to lock onto.
    """
    rng = np.random.default_rng(seed)
    canvas = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        shade = 150 - int(40 * y / H)
        canvas[y, :] = (shade + 12, shade + 6, shade)
    cv2.line(canvas, (0, int(H * 0.92)), (W, int(H * 0.92)), (95, 100, 108), 6)
    cv2.line(canvas, (int(W * 0.78), 0), (int(W * 0.78), int(H * 0.92)),
             (120, 126, 134), 4)
    return canvas


def add_sensor_noise(frame: np.ndarray, rng: np.random.Generator,
                     sigma: float = 3.5) -> np.ndarray:
    """Add Gaussian noise at a level typical of a webcam in room light.

    Without this the clip would be unrealistically clean and would understate
    the noise floor. 3.5 grey levels is a reasonable figure for a consumer
    sensor at moderate ISO; it is a modelling choice, not a measurement, and is
    recorded as such.
    """
    noise = rng.normal(0, sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def write_clip(path: Path, frames: int, fps: int,
               pose_at, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    truth = []
    for i in range(frames):
        knee, arm = pose_at(i, frames)
        canvas = background(seed)
        joints = draw_person(canvas, knee, arm)
        writer.write(add_sensor_noise(canvas, rng))
        truth.append({"frame": i, "knee_bend": knee, "arm_raise": arm,
                      "joints": {k: list(v) for k, v in joints.items()}})
    writer.release()
    (path.with_suffix(".truth.json")).write_text(
        __import__("json").dumps(truth), encoding="utf-8")
    print(f"  {path.name}: {frames} frames @ {fps} fps")


def main() -> None:
    OUT.mkdir(exist_ok=True)

    write_clip(OUT / "still.mp4", 300, 30, lambda i, n: (0.0, 0.0))

    def squat(i: int, n: int):
        phase = (math.sin(2 * math.pi * i / (n / 3)) + 1) / 2
        return phase * 0.9, 0.0
    write_clip(OUT / "squat.mp4", 300, 30, squat, seed=1)

    def arms(i: int, n: int):
        phase = (math.sin(2 * math.pi * i / (n / 3)) + 1) / 2
        return 0.0, phase
    write_clip(OUT / "arms.mp4", 300, 30, arms, seed=2)


if __name__ == "__main__":
    main()