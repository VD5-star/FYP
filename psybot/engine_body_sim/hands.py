from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

WRIST = 0
FINGER_JOINTS: dict[str, tuple[int, int, int]] = {
    "thumb": (4, 3, 2),
    "index": (8, 6, 5),
    "middle": (12, 10, 9),
    "ring": (16, 14, 13),
    "pinky": (20, 18, 17),
}

FINGER_NUMBERS: dict[str, int] = {
    "pinky": 1, "ring": 2, "middle": 3, "index": 4, "thumb": 5,
}

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"


def finger_extended(points: np.ndarray, finger: str) -> bool:
    """Whether one finger is extended.

    Tested by comparing the tip's distance from the wrist against the middle
    joint's distance from the wrist. A folded finger curls its tip back towards
    the palm, so the tip ends up *closer* to the wrist than its own knuckle.

    Distance from the wrist, rather than comparing y coordinates: the y test
    only works for an upright hand and fails the moment the hand is rotated,
    which it will be. Distance is rotation-invariant.
    """
    tip, middle, _ = FINGER_JOINTS[finger]
    wrist = points[WRIST]
    return bool(np.linalg.norm(points[tip] - wrist)
                > np.linalg.norm(points[middle] - wrist))


def read_fingers(points: np.ndarray) -> dict[str, bool]:
    """Extended state of all five fingers."""
    return {name: finger_extended(points, name) for name in FINGER_JOINTS}


def finger_numbers(states: dict[str, bool]) -> list[int]:
    """Extended fingers, as the user's numbers, sorted."""
    return sorted(FINGER_NUMBERS[n] for n, up in states.items() if up)


def is_stop_gesture(states: dict[str, bool]) -> bool:
    """Fingers 1 and 5 up, the rest down."""
    return (states.get("pinky", False) and states.get("thumb", False)
            and not states.get("index", True)
            and not states.get("middle", True)
            and not states.get("ring", True))


@dataclass
class HandReading:
    """What was seen of one hand on one frame."""

    points: np.ndarray
    handedness: str
    states: dict[str, bool]

    @property
    def numbers(self) -> list[int]:
        return finger_numbers(self.states)

    @property
    def stop(self) -> bool:
        return is_stop_gesture(self.states)


@dataclass
class GestureState:
    """Progress of a held gesture."""

    progress: float = 0.0
    fired: bool = False
    hands: list[HandReading] = field(default_factory=list)


class HandReader:
    """Finds hands near the pose's wrists and reads their fingers.

    Holds its own landmarker, and is safe to construct when the model file is
    missing: `available` is then False and the simulator runs without the
    feature rather than refusing to start.
    """

    def __init__(self, every_n_frames: int = 4, num_hands: int = 2,
                 crop_size: int = 256) -> None:
        self.every_n_frames = max(1, every_n_frames)
        self.crop_size = crop_size
        self._frame = 0
        self._landmarker = None
        self.available = MODEL_PATH.exists()
        if self.available:
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=num_hands,
                # region, so it is softer than a native close-up.
                min_hand_detection_confidence=0.3,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)

    def tick(self) -> bool:
        """Advance the frame counter, returning whether to sample this frame.

        Counting lives here rather than in the caller so that skipping a frame
        cannot silently stop advancing the counter - which would freeze the
        sampler either permanently on or permanently off.
        """
        due = self._frame % self.every_n_frames == 0
        self._frame += 1
        return due

    def read(self, image: np.ndarray, wrists: list[np.ndarray],
             torso: float) -> list[HandReading]:
        """Read every hand whose wrist the pose model located.

        `wrists` are pixel positions and `torso` sets the crop size, so the box
        scales with the subject's distance instead of being a fixed number of
        pixels that is right at one distance only.
        """
        if not self.available or self._landmarker is None or not wrists:
            return []

        h, w = image.shape[:2]
        half = max(24.0, torso * 0.55)
        out: list[HandReading] = []

        for wrist in wrists:
            if not np.all(np.isfinite(wrist)):
                continue
            x0 = int(max(0, wrist[0] - half))
            y0 = int(max(0, wrist[1] - half))
            x1 = int(min(w, wrist[0] + half))
            y1 = int(min(h, wrist[1] + half))
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue

            crop = image[y0:y1, x0:x1]
            scale = self.crop_size / max(crop.shape[0], crop.shape[1])
            if scale > 1:
                crop = cv2.resize(
                    crop, (int(crop.shape[1] * scale),
                           int(crop.shape[0] * scale)),
                    interpolation=cv2.INTER_CUBIC)

            result = self._landmarker.detect(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            if not result.hand_landmarks:
                continue

            ch, cw = crop.shape[:2]
            for i, hand in enumerate(result.hand_landmarks):
                pts = np.array([
                    [x0 + p.x * cw / (scale if scale > 1 else 1.0),
                     y0 + p.y * ch / (scale if scale > 1 else 1.0)]
                    for p in hand])
                side = "unknown"
                if result.handedness and i < len(result.handedness):
                    side = result.handedness[i][0].category_name
                out.append(HandReading(points=pts, handedness=side,
                                       states=read_fingers(pts)))
        return out

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


class HeldGesture:
    """Requires a gesture to persist before it counts.

    Time-based rather than frame-based, because the reader samples
    intermittently and the frame rate varies. A user holding a shape for 1.5
    seconds means 1.5 seconds of wall clock, not a frame count that changes
    with the machine.
    """

    def __init__(self, seconds: float = 1.5, grace_seconds: float = 0.4) -> None:
        self.seconds = seconds
        self.grace_seconds = grace_seconds
        """How long the gesture may be lost before progress resets.

        The reader samples every few frames and a hand can be missed for one of
        them. Without grace, a single dropped sample would restart the hold and
        the gesture would be nearly impossible to complete.
        """
        self._started: float | None = None
        self._last_seen: float | None = None
        self._fired = False

    def update(self, present: bool, timestamp: float) -> GestureState:
        if present:
            self._last_seen = timestamp
            if self._started is None:
                self._started = timestamp
        elif self._started is not None and self._last_seen is not None:
            if timestamp - self._last_seen > self.grace_seconds:
                self._started = None
                self._fired = False
                return GestureState()

        if self._started is None:
            return GestureState()

        held = timestamp - self._started
        progress = min(1.0, held / self.seconds)
        fired = progress >= 1.0 and not self._fired
        if fired:
            self._fired = True
        return GestureState(progress=progress, fired=fired)

    def reset(self) -> None:
        self._started = None
        self._last_seen = None
        self._fired = False