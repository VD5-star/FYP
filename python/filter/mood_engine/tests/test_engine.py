from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import load_image


def test_full_frame_analysis(engine):
    r = engine.analyse_frame(load_image("obama_a"))

    assert r.ok
    assert r.face_count >= 1
    assert r.bbox is not None and len(r.bbox) == 4
    assert r.emotion is not None and r.pose is not None and r.spoof is not None
    assert 0.0 <= r.quality <= 1.0
    assert r.observation_id is not None, "observation was not persisted"

    x1, y1, x2, y2 = r.bbox
    assert x2 > x1 and y2 > y1, "bounding box is degenerate"


def test_no_face_is_handled_cleanly(engine):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    r = engine.analyse_frame(blank)

    assert not r.ok
    assert r.message == "no_face"
    assert r.face_count == 0
    assert r.person_id is None


def test_result_is_json_serialisable(engine):
    import json

    r = engine.analyse_frame(load_image("obama_a"))
    payload = json.dumps(r.to_dict(), ensure_ascii=False)
    restored = json.loads(payload)

    assert restored["ok"] is True
    assert restored["emotion"]["label"] in restored["emotion"]["probs"]


def test_closest_face_is_selected(engine, detector):
    import cv2

    a = cv2.resize(load_image("obama_a"), (320, 400))
    b = cv2.resize(load_image("biden"), (160, 200))
    canvas = np.zeros((400, 640, 3), dtype=np.uint8)
    canvas[0:400, 0:320] = a
    canvas[0:200, 400:560] = b

    faces = detector.detect(canvas)
    if len(faces) < 2:
        pytest.skip("test composite did not yield two faces")

    subject = detector.select_subject(faces, canvas.shape)
    assert subject.area == max(f.area for f in faces)


def test_observations_accumulate(engine):
    img = load_image("obama_a")
    engine.start_session("test")
    for _ in range(3):
        engine.analyse_frame(img)
    engine.end_session()

    assert engine.db.stats()["observations"] == 3
    rows = engine.db.recent_observations(limit=10)
    assert len(rows) == 3
    assert all(r["session_id"] is not None for r in rows)


def test_auto_enrol_creates_provisional_person(tmp_path):
    from engine.core import MoodEngine
    from engine.db import MoodDatabase

    eng = MoodEngine(db=MoodDatabase(tmp_path / "auto.db"),
                     auto_enrol_unknown=True)
    eng.load()
    try:
        r = eng.analyse_frame(load_image("obama_a"))
        assert r.person_name == "Unknown-1"
        assert r.is_new_person and r.is_provisional

        r2 = eng.analyse_frame(load_image("obama_c"))
        assert r2.person_id == r.person_id
        assert not r2.is_new_person
        assert eng.db.stats()["persons"] == 1
    finally:
        eng.close()


def test_auto_enrol_can_be_disabled(engine):
    r = engine.analyse_frame(load_image("obama_a"))
    assert r.person_id is None
    assert engine.db.stats()["persons"] == 0


def test_spoof_detector_accepts_real_photo(engine):
    r = engine.analyse_frame(load_image("obama_a"))
    assert r.spoof is not None
    assert 0.0 <= r.spoof.liveness <= 1.0
    assert set(r.spoof.signals) == {"depth", "texture", "colour", "moire",
                                    "motion"}


def test_pose_and_attention_ranges(engine):
    r = engine.analyse_frame(load_image("obama_a"))
    p = r.pose
    assert -180.0 <= p.yaw <= 180.0
    assert -180.0 <= p.pitch <= 180.0
    assert 0.0 <= p.attention <= 1.0
    assert -1.0 <= p.gaze_x <= 1.0
    assert -1.0 <= p.gaze_y <= 1.0


def test_age_and_gender_are_plausible(engine):
    r = engine.analyse_frame(load_image("obama_a"))
    assert r.age is not None and 1 <= r.age <= 100
    assert r.gender in ("M", "F")


def _build_dataset(tmp_path):
    import shutil

    from conftest import FACE_IMAGES

    root = tmp_path / "dataset"
    for person, keys in (("Obama", ["obama_a", "obama_c"]), ("Biden", ["biden"])):
        folder = root / person
        folder.mkdir(parents=True)
        for i, key in enumerate(keys):
            src = FACE_IMAGES[key]
            if not src.exists():
                pytest.skip("sample images missing")
            shutil.copy(src, folder / f"{i}.jpg")
    return root


def test_import_folder_learns_people(engine, tmp_path):
    from engine.core.trainer import FaceTrainer

    root = _build_dataset(tmp_path)
    report = FaceTrainer(engine).import_folder(root)

    assert report.added == 3
    assert report.failed == 0
    assert report.persons == {"Obama": 2, "Biden": 1}

    names = {p["name"] for p in engine.list_persons()}
    assert names == {"Obama", "Biden"}


def test_import_skips_duplicates(engine, tmp_path):
    from engine.core.trainer import FaceTrainer

    root = _build_dataset(tmp_path)
    trainer = FaceTrainer(engine)
    trainer.import_folder(root)
    second = trainer.import_folder(root)

    assert second.added == 0
    assert second.skipped == 3
    assert engine.db.count_embeddings() == 3


def test_dry_run_changes_nothing(engine, tmp_path):
    from engine.core.trainer import FaceTrainer

    root = _build_dataset(tmp_path)
    report = FaceTrainer(engine).import_folder(root, dry_run=True)

    assert report.added == 3
    assert engine.db.count_embeddings() == 0
    assert engine.db.stats()["persons"] == 0


def test_import_then_identify(engine, tmp_path, detector):
    from engine.core.trainer import FaceTrainer

    FaceTrainer(engine).import_folder(_build_dataset(tmp_path))

    img = load_image("obama_b")
    face = max(detector.detect(img), key=lambda f: f.area)
    match = engine.recognizer.match(face.embedding)

    assert match.name == "Obama"
    assert match.score > 0.5


def test_import_rejects_images_without_faces(engine, tmp_path):
    import cv2

    from engine.core.trainer import FaceTrainer

    folder = tmp_path / "ds" / "Ghost"
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "blank.jpg"),
                np.zeros((200, 200, 3), dtype=np.uint8))

    report = FaceTrainer(engine).import_folder(tmp_path / "ds")
    assert report.added == 0
    assert report.failed == 1
    assert report.items[0].reason == "no_face_detected"


def test_analysis_latency_budget(engine):
    img = load_image("obama_a")
    engine.analyse_frame(img, persist=False)

    timings = []
    for _ in range(7):
        t0 = time.perf_counter()
        engine.analyse_frame(img, persist=False)
        timings.append((time.perf_counter() - t0) * 1000.0)

    best = float(np.min(timings))
    assert best < 400.0, (
        f"frame analysis too slow: best {best:.0f}ms of "
        f"{[round(t) for t in timings]}")


def test_empty_frame_is_fast(engine):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    engine.analyse_frame(blank, persist=False)

    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        engine.analyse_frame(blank, persist=False)
        timings.append((time.perf_counter() - t0) * 1000.0)

    assert float(np.median(timings)) < 120.0
