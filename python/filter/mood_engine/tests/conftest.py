from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
FACE_IMAGES = {
    "obama_a": DATA / "ds_raw" / "obama.jpg",
    "obama_b": DATA / "ds_raw" / "obama2.jpg",
    "obama_c": DATA / "ds_raw" / "obama_partial_face.jpg",
    "obama_d": DATA / "ds_raw" / "obama_partial_face2.jpg",
    "biden": DATA / "ds_raw" / "biden.jpg",
    "stranger": DATA / "expressions" / "happy" / "ffhq_0.png",
}


@pytest.fixture(scope="session")
def detector():
    from engine.core.detector import FaceDetector
    d = FaceDetector()
    d.load()
    return d


@pytest.fixture(scope="session")
def analyzer():
    from engine.core.emotion import EmotionAnalyzer
    return EmotionAnalyzer()


@pytest.fixture
def temp_db(tmp_path):
    from engine.db import MoodDatabase
    db = MoodDatabase(tmp_path / "test.db")
    yield db
    db.close()


@pytest.fixture
def engine(tmp_path):
    from engine.core import MoodEngine
    from engine.db import MoodDatabase
    eng = MoodEngine(db=MoodDatabase(tmp_path / "engine.db"),
                     auto_enrol_unknown=False)
    eng.load()
    yield eng
    eng.close()


def load_image(key: str):
    import cv2
    path = FACE_IMAGES[key]
    if not path.exists():
        pytest.skip(f"sample image missing: {path}")
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"unreadable image: {path}")
    return img
