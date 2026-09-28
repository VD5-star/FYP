from __future__ import annotations

import numpy as np
import pytest

from engine.config import CONFIG, EMOTIONS
from engine.core.baseline import BaselineTracker, PersonBaseline


def _ready_baseline(resting: list[float]) -> BaselineTracker:
    tracker = BaselineTracker()
    baseline = tracker.baseline_for(1)
    probs = np.array(resting, dtype=float)
    probs = probs / probs.sum()
    rng = np.random.default_rng(7)
    for _ in range(PersonBaseline.REQUIRED + 5):
        noisy = np.clip(probs + rng.normal(0, 0.015, len(EMOTIONS)), 1e-4, None)
        baseline.observe(noisy / noisy.sum(), {})
    assert baseline.ready
    return tracker


class TestBaselineDoesNotInventEmotions:


    RESTING = [0.55, 0.12, 0.13, 0.05, 0.04, 0.06, 0.05]

    def test_neutral_face_stays_neutral(self) -> None:
        tracker = _ready_baseline(self.RESTING)
        probs = np.array([0.42, 0.08, 0.15, 0.06, 0.07, 0.12, 0.10])
        probs = probs / probs.sum()

        out, _ = tracker.adjust(probs, {}, 1, learn=False)

        assert EMOTIONS[int(np.argmax(out))] == "neutral"

    def test_neutral_mass_is_preserved_exactly(self) -> None:
        tracker = _ready_baseline(self.RESTING)
        probs = np.array([0.42, 0.08, 0.15, 0.06, 0.07, 0.12, 0.10])
        probs = probs / probs.sum()

        out, _ = tracker.adjust(probs, {}, 1, learn=False)

        neutral = EMOTIONS.index("neutral")
        assert out[neutral] == pytest.approx(probs[neutral], abs=1e-9)

    @pytest.mark.parametrize("name,reading", [
        ("happy", [0.15, 0.62, 0.05, 0.07, 0.03, 0.04, 0.04]),
        ("anger", [0.12, 0.03, 0.10, 0.05, 0.08, 0.12, 0.50]),
        ("sad", [0.30, 0.05, 0.38, 0.05, 0.07, 0.08, 0.07]),
        ("surprise", [0.10, 0.06, 0.05, 0.55, 0.10, 0.07, 0.07]),
    ])
    def test_real_expressions_survive(self, name: str,
                                      reading: list[float]) -> None:
        tracker = _ready_baseline(self.RESTING)
        probs = np.array(reading) / sum(reading)

        out, _ = tracker.adjust(probs, {}, 1, learn=False)

        assert EMOTIONS[int(np.argmax(out))] == name

    def test_output_is_a_distribution(self) -> None:
        tracker = _ready_baseline(self.RESTING)
        rng = np.random.default_rng(1)
        for _ in range(50):
            probs = rng.random(len(EMOTIONS))
            probs = probs / probs.sum()
            out, _ = tracker.adjust(probs, {}, 1, learn=False)
            assert out.sum() == pytest.approx(1.0)
            assert np.all(out >= 0)
            assert np.all(np.isfinite(out))

    def test_amplification_is_capped(self) -> None:
        tracker = _ready_baseline([0.70, 0.24, 0.02, 0.01, 0.01, 0.01, 0.01])
        probs = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS))

        out, _ = tracker.adjust(probs, {}, 1, learn=False)

        assert out.max() / out.min() < 8.0


class TestDetectorScaling:


    def test_config_limits_detector_input(self) -> None:
        assert CONFIG.detection.detect_max_width > 0
        assert CONFIG.detection.detect_max_width >= 480

    def test_rescale_maps_boxes_back(self) -> None:
        from engine.core.detector import DetectedFace, FaceDetector

        face = DetectedFace(
            bbox=np.array([10.0, 20.0, 60.0, 90.0], dtype=np.float32),
            det_score=0.9,
            embedding=None,
            keypoints=np.array([[10.0, 20.0]] * 5, dtype=np.float32),
            landmarks_2d=None,
            landmarks_3d=None,
            age=None,
            gender=None,
            pose=None,
            raw=None,
        )

        FaceDetector._rescale_faces([face], 2.0, 1280, 720)

        assert list(face.bbox) == pytest.approx([20.0, 40.0, 120.0, 180.0])
        assert face.keypoints[0][0] == pytest.approx(20.0)

    def test_rescale_clamps_to_frame(self) -> None:
        from engine.core.detector import DetectedFace, FaceDetector

        face = DetectedFace(
            bbox=np.array([600.0, 300.0, 700.0, 400.0], dtype=np.float32),
            det_score=0.9,
            embedding=None,
            keypoints=np.zeros((5, 2), dtype=np.float32),
            landmarks_2d=None,
            landmarks_3d=None,
            age=None,
            gender=None,
            pose=None,
            raw=None,
        )

        FaceDetector._rescale_faces([face], 2.0, 640, 480)

        assert face.bbox[2] <= 639.0
        assert face.bbox[3] <= 479.0
