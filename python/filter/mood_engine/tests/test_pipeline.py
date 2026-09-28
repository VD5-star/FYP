from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from conftest import load_image


def test_padding_recovers_tightly_cropped_faces(detector):
    img = load_image("obama_a")
    faces = detector.detect(img)
    assert faces, "baseline detection failed"

    face = max(faces, key=lambda f: f.area)
    tight = face.crop(img, margin=0.0)
    assert tight.size > 0

    recovered = detector.detect(tight)
    assert recovered, "tight crop was not recovered by the padding retry"

    box = recovered[0].bbox
    h, w = tight.shape[:2]
    assert 0 <= box[0] <= w and 0 <= box[2] <= w
    assert 0 <= box[1] <= h and 0 <= box[3] <= h
    assert box[2] > box[0] and box[3] > box[1]


def test_padding_does_not_invent_faces(detector):
    for frame in (np.zeros((240, 320, 3), np.uint8),
                  np.full((240, 320, 3), 127, np.uint8)):
        assert detector.detect(frame) == []


def test_align_returns_square_crop_of_configured_size(detector):
    img = load_image("obama_a")
    face = max(detector.detect(img), key=lambda f: f.area)
    aligned = detector.align(face, img)

    size = detector.config.align_size
    assert aligned.shape == (size, size, 3)
    assert aligned.dtype == np.uint8


def test_align_levels_the_eyes(detector):
    img = load_image("obama_a")
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 20.0, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    faces = detector.detect(rotated)
    if not faces:
        pytest.skip("detector could not find the rotated face")
    face = max(faces, key=lambda f: f.area)

    aligned = detector.align(face, rotated)
    assert aligned.shape[0] == aligned.shape[1]

    redetected = detector.detect(aligned)
    if not redetected:
        pytest.skip("no face found in the aligned crop")
    kps = redetected[0].keypoints
    if kps is None or len(kps) < 2:
        pytest.skip("no keypoints available")

    dy = abs(float(kps[1][1] - kps[0][1]))
    dx = abs(float(kps[1][0] - kps[0][0])) or 1.0
    tilt = np.degrees(np.arctan2(dy, dx))
    assert tilt < 12.0, f"eyes still tilted by {tilt:.1f} degrees after alignment"


def test_align_survives_missing_keypoints(detector):
    img = load_image("obama_a")
    face = max(detector.detect(img), key=lambda f: f.area)
    face.keypoints = np.zeros((0, 2), dtype=np.float32)

    aligned = detector.align(face, img)
    size = detector.config.align_size
    assert aligned.shape == (size, size, 3)


def _probs(**values: float) -> np.ndarray:
    from engine.config import EMOTIONS

    vector = np.zeros(len(EMOTIONS))
    for name, value in values.items():
        vector[EMOTIONS.index(name)] = value
    total = vector.sum()
    return vector / total if total else vector


def test_label_does_not_flip_on_a_marginal_lead():
    from engine.core.emotion import EmotionAnalyzer

    analyzer = EmotionAnalyzer()
    analyzer.reset()

    assert analyzer._stable_label(_probs(happy=0.50, sad=0.30)) == 'happy'
    assert analyzer._stable_label(_probs(sad=0.41, happy=0.39)) == 'happy'
    assert analyzer._stable_label(_probs(sad=0.42, happy=0.40)) == 'happy'


def test_label_switches_on_a_sustained_clear_lead():
    from engine.core.emotion import EmotionAnalyzer

    analyzer = EmotionAnalyzer()
    analyzer.reset()
    analyzer._stable_label(_probs(happy=0.60, sad=0.20))

    strong = _probs(sad=0.70, happy=0.10)
    for _ in range(analyzer.config.switch_frames):
        analyzer._stable_label(strong)
    assert analyzer._stable_label(strong) == 'sad'


def test_reset_clears_sticky_label():
    from engine.core.emotion import EmotionAnalyzer

    analyzer = EmotionAnalyzer()
    analyzer._stable_label(_probs(anger=0.9))
    analyzer.reset()
    assert analyzer._sticky_label is None
    assert analyzer._stable_label(_probs(happy=0.9)) == 'happy'


def test_low_quality_frames_move_the_estimate_less():
    from engine.core.emotion import EmotionAnalyzer

    start = _probs(neutral=1.0)
    shock = _probs(anger=1.0)

    good = EmotionAnalyzer()
    good._smoothed = start.copy()
    good._smooth(shock, quality=1.0)

    poor = EmotionAnalyzer()
    poor._smoothed = start.copy()
    poor._smooth(shock, quality=0.0)

    from engine.config import EMOTIONS
    index = EMOTIONS.index('anger')
    assert good._smoothed[index] > poor._smoothed[index], (
        "frame quality is not affecting the smoothing rate")


def test_smoothed_output_stays_a_distribution():
    from engine.core.emotion import EmotionAnalyzer

    analyzer = EmotionAnalyzer()
    for quality in (0.0, 0.5, 1.0):
        out = analyzer._smooth(_probs(happy=0.6, sad=0.4), quality=quality)
        assert abs(out.sum() - 1.0) < 1e-6
        assert (out >= 0).all()


def test_writer_hook_keeps_persistence_off_the_frame_path(engine):
    jobs: list = []
    engine.set_writer(lambda job: (jobs.append(job), True)[1])
    engine.start_session("writer-test")

    result = engine.analyse_frame(load_image("obama_a"), persist=True)
    assert result.ok
    assert engine.db.stats()["observations"] == 0, "write happened inline"
    assert jobs, "no write was queued"

    for job in jobs:
        job()
    assert engine.db.stats()["observations"] == 1

    engine.set_writer(None)


def test_inline_writes_still_work_without_a_writer(engine):
    engine.start_session("inline-test")
    result = engine.analyse_frame(load_image("obama_a"), persist=True)
    assert result.observation_id is not None
    assert result.observation_id > 0
    assert engine.db.stats()["observations"] == 1


def test_snapshots_are_rate_limited(engine):
    from engine.config import FACES_DIR

    before = {p.name for p in FACES_DIR.glob("*.jpg")}
    img = load_image("obama_a")
    for _ in range(8):
        engine.analyse_frame(img, persist=False, save_snapshot=True)
    after = {p.name for p in FACES_DIR.glob("*.jpg")}

    written = after - before
    for name in written:
        (FACES_DIR / name).unlink(missing_ok=True)
    assert len(written) <= 1, f"{len(written)} snapshots for 8 frames"


def test_identity_cache_is_off_by_default(engine):
    assert engine.detector._identity_every == 1

    def embedding_of(name):
        faces = engine.detector.detect(load_image(name))
        assert faces, f"no face found in {name}"
        vector = max(faces, key=lambda f: f.area).embedding
        return vector / np.linalg.norm(vector)

    similarity = float(np.dot(embedding_of("obama_a"), embedding_of("biden")))
    assert similarity < 0.42, (
        "different people share an embedding; identity was cached when it "
        f"should not have been (similarity {similarity:.3f})")


def test_identity_cache_holds_identity_across_frames(engine):
    engine.detector.enable_identity_cache(5)
    try:
        image = load_image("obama_a")
        embeddings = []
        for _ in range(6):
            faces = engine.detector.detect(image)
            if faces:
                embeddings.append(max(faces, key=lambda f: f.area).embedding)

        assert len(embeddings) >= 5
        first = embeddings[0] / np.linalg.norm(embeddings[0])
        for other in embeddings[1:]:
            other = other / np.linalg.norm(other)
            assert float(np.dot(first, other)) > 0.9
    finally:
        engine.detector.enable_identity_cache(1)


def test_enabling_the_cache_clears_stale_identity(engine):
    engine.analyse_frame(load_image("obama_a"), persist=False)
    engine.detector.enable_identity_cache(5)
    assert engine.detector._cached_identity is None
    engine.detector.enable_identity_cache(1)


def test_reset_tracking_clears_the_identity_cache(engine):
    engine.detector.enable_identity_cache(5)
    try:
        engine.analyse_frame(load_image("obama_a"), persist=False)
        engine.detector.reset_tracking()
        assert engine.detector._cached_identity is None
    finally:
        engine.detector.enable_identity_cache(1)


