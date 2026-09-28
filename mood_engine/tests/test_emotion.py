from __future__ import annotations

import numpy as np
import pytest

from conftest import load_image
from engine.config import EMOTIONS, EMOTIONS_AR

REF_IDX = {0: "anger", 1: "contempt", 2: "disgust", 3: "fear",
           4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}
IMAGE_KEYS = ["obama_a", "obama_b", "biden", "stranger"]


def _reference_predict(model_path: str, crop, size: int) -> str:
    import cv2
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(model_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    x = cv2.resize(crop, (size, size)) / 255
    x[..., 0] = (x[..., 0] - 0.485) / 0.229
    x[..., 1] = (x[..., 1] - 0.456) / 0.224
    x[..., 2] = (x[..., 2] - 0.406) / 0.225
    x = x.transpose(2, 0, 1).astype("float32")[np.newaxis, ...]
    scores = sess.run(None, {"input": x})[0][0]
    label = REF_IDX[int(np.argmax(scores))]
    return "disgust" if label == "contempt" else label


def test_onnx_model_is_active(analyzer):
    assert analyzer.backend == "onnx", (
        "trained emotion model not loaded - accuracy would be much lower")
    assert analyzer.model_path is not None


def test_matches_reference_implementation(detector, analyzer):
    if analyzer.backend != "onnx":
        pytest.skip("no ONNX model available")

    size = analyzer._input_size[0]
    mismatches = []
    for key in IMAGE_KEYS:
        img = load_image(key)
        faces = detector.detect(img)
        if not faces:
            continue
        face = max(faces, key=lambda f: f.area)
        crop = face.crop(img, margin=0.15)

        ours = analyzer.analyse(crop, face.landmarks_2d, smooth=False).label
        theirs = _reference_predict(analyzer.model_path, crop, size)
        if ours != theirs:
            mismatches.append(f"{key}: ours={ours} reference={theirs}")

    assert not mismatches, "diverged from reference: " + "; ".join(mismatches)


def test_probabilities_form_valid_distribution(detector, analyzer):
    img = load_image("obama_a")
    face = max(detector.detect(img), key=lambda f: f.area)
    result = analyzer.analyse(face.crop(img, 0.15), face.landmarks_2d,
                              smooth=False)

    assert set(result.probs) == set(EMOTIONS)
    assert abs(sum(result.probs.values()) - 1.0) < 1e-5
    assert all(0.0 <= p <= 1.0 for p in result.probs.values())
    assert result.label == max(result.probs, key=result.probs.get)
    assert result.confidence == pytest.approx(result.probs[result.label])


def test_valence_arousal_within_bounds(detector, analyzer):
    for key in IMAGE_KEYS:
        img = load_image(key)
        faces = detector.detect(img)
        if not faces:
            continue
        face = max(faces, key=lambda f: f.area)
        r = analyzer.analyse(face.crop(img, 0.15), face.landmarks_2d,
                             smooth=False)
        assert -1.0 <= r.valence <= 1.0, f"{key}: valence {r.valence}"
        assert 0.0 <= r.arousal <= 1.0, f"{key}: arousal {r.arousal}"


def test_happy_face_has_positive_valence(detector, analyzer):
    img = load_image("stranger")
    faces = detector.detect(img)
    if not faces:
        pytest.skip("no face in sample")
    face = max(faces, key=lambda f: f.area)
    r = analyzer.analyse(face.crop(img, 0.15), face.landmarks_2d, smooth=False)

    if r.label == "happy" and r.confidence > 0.7:
        assert r.valence > 0.3, f"happy face reported valence {r.valence}"


def test_arabic_label_present(detector, analyzer):
    img = load_image("obama_a")
    face = max(detector.detect(img), key=lambda f: f.area)
    r = analyzer.analyse(face.crop(img, 0.15), face.landmarks_2d, smooth=False)
    assert r.label_ar == EMOTIONS_AR[r.label]
    assert r.label_ar.strip()


def test_smoothing_reduces_jitter(detector):
    from engine.core.emotion import EmotionAnalyzer

    a = EmotionAnalyzer()
    img_happy = load_image("stranger")
    img_other = load_image("biden")

    faces_h = detector.detect(img_happy)
    faces_o = detector.detect(img_other)
    if not faces_h or not faces_o:
        pytest.skip("samples unavailable")

    fh = max(faces_h, key=lambda f: f.area)
    fo = max(faces_o, key=lambda f: f.area)

    for _ in range(6):
        a.analyse(fh.crop(img_happy, 0.15), fh.landmarks_2d, smooth=True)
    settled = a.analyse(fh.crop(img_happy, 0.15), fh.landmarks_2d, smooth=True)

    perturbed = a.analyse(fo.crop(img_other, 0.15), fo.landmarks_2d,
                          smooth=True)
    unsmoothed = EmotionAnalyzer().analyse(fo.crop(img_other, 0.15),
                                           fo.landmarks_2d, smooth=False)

    drift = abs(perturbed.probs[settled.label] - settled.probs[settled.label])
    raw_drift = abs(unsmoothed.probs[settled.label]
                    - settled.probs[settled.label])
    assert drift <= raw_drift + 1e-9


def test_trend_reports_dominant_emotion(detector, analyzer):
    from engine.core.emotion import EmotionAnalyzer

    a = EmotionAnalyzer()
    img = load_image("stranger")
    faces = detector.detect(img)
    if not faces:
        pytest.skip("no face")
    face = max(faces, key=lambda f: f.area)

    for _ in range(8):
        a.analyse(face.crop(img, 0.15), face.landmarks_2d)

    trend = a.trend()
    assert trend["samples"] == 8
    assert trend["dominant"] in EMOTIONS
    assert 0.0 <= trend["stability"] <= 1.0
