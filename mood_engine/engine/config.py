from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
DATA_DIR = PROJECT_ROOT / "data"
FACES_DIR = DATA_DIR / "faces"
DB_PATH = DATA_DIR / "mood_engine.db"
CA_BUNDLE = PROJECT_ROOT / "certs" / "win_ca_bundle.pem"

for _d in (ASSETS_DIR, MODELS_DIR, DATA_DIR, FACES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

if CA_BUNDLE.exists():
    os.environ.setdefault("SSL_CERT_FILE", str(CA_BUNDLE))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(CA_BUNDLE))


EMOTIONS = (
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
)

EMOTIONS_AR = {
    "neutral": "محايد",
    "happy": "سعيد",
    "sad": "حزين",
    "surprise": "مندهش",
    "fear": "خائف",
    "disgust": "مشمئز",
    "anger": "غاضب",
}

EMOTION_VA = {
    "neutral":  (0.00, 0.20),
    "happy":    (0.80, 0.65),
    "sad":      (-0.70, 0.25),
    "surprise": (0.25, 0.90),
    "fear":     (-0.65, 0.85),
    "disgust":  (-0.60, 0.55),
    "anger":    (-0.75, 0.85),
}


@dataclass(frozen=True)
class DetectionConfig:
    det_size: tuple[int, int] = (320, 320)
    min_det_score: float = 0.45
    min_face_ratio: float = 0.05
    detect_max_width: int = 640
    model_pack: str = "buffalo_s"
    intra_op_threads: int = 6
    modules: tuple[str, ...] = ("detection", "recognition", "genderage",
                                "landmark_2d_106")
    identity_every_n_frames: int = 1
    live_identity_every_n_frames: int = 5
    retry_with_padding: bool = True
    padding_ratio: float = 0.25
    align_size: int = 224
    align_eye_ratio: float = 0.34


@dataclass(frozen=True)
class RecognitionConfig:
    match_threshold: float = 0.42
    new_threshold: float = 0.32
    max_embeddings_per_person: int = 40
    auto_enrol_low: float = 0.45
    auto_enrol_high: float = 0.92


@dataclass(frozen=True)
class EmotionConfig:
    smoothing_alpha: float = 0.35
    smoothing_alpha_min: float = 0.12
    history_len: int = 30
    min_confidence: float = 0.25
    switch_margin: float = 0.06
    switch_frames: int = 2


@dataclass(frozen=True)
class SpoofConfig:
    enabled: bool = True
    threshold: float = 0.55
    history_len: int = 12


@dataclass(frozen=True)
class TrackingConfig:
    max_centre_drift: float = 0.18
    max_missing_frames: int = 15


@dataclass(frozen=True)
class EngineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    spoof: SpoofConfig = field(default_factory=SpoofConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    camera_index: int = 0
    camera_sizes: tuple[tuple[int, int], ...] = (
        (1280, 720), (1024, 768), (960, 540), (800, 600), (640, 480),
    )
    camera_fourcc: str = "MJPG"
    analyse_every_n_frames: int = 1
    guided_calibration: bool = True
    session_period_s: float = 300.0
    save_snapshots: bool = True
    snapshot_min_confidence: float = 0.80
    snapshot_min_interval_s: float = 20.0


CONFIG = EngineConfig()
