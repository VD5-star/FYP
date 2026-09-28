from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..config import CONFIG, FACES_DIR, EngineConfig
from ..db import MoodDatabase
from .affect import AffectAnalyzer, AffectReading
from .baseline import BaselineTracker, MoodSummary
from .attributes import PoseEstimator, PoseResult, SpoofDetector, SpoofResult
from .detector import DetectedFace, FaceDetector
from .emotion import EmotionAnalyzer, EmotionResult
from .mood import MoodAnalyzer, MoodReading
from .calibration import CalibrationManager, rest_frames_only
from .recognizer import FaceRecognizer, MatchResult
from .session import SessionAnalyser, SessionReport


@dataclass
class FrameResult:
    ok: bool
    frame_index: int
    timestamp: float
    elapsed_ms: float
    face_count: int = 0
    bbox: list[int] | None = None
    det_score: float = 0.0
    person_id: int | None = None
    person_name: str | None = None
    is_new_person: bool = False
    is_provisional: bool = False
    match_score: float = 0.0
    match_uncertain: bool = False
    emotion: EmotionResult | None = None
    pose: PoseResult | None = None
    spoof: SpoofResult | None = None
    affect: AffectReading | None = None
    mood: MoodSummary | None = None
    age: float | None = None
    gender: str | None = None
    quality: float = 0.0
    observation_id: int | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "face_count": self.face_count,
            "bbox": self.bbox,
            "det_score": round(self.det_score, 4),
            "person": {
                "id": self.person_id,
                "name": self.person_name,
                "is_new": self.is_new_person,
                "is_provisional": self.is_provisional,
                "match_score": round(self.match_score, 4),
                "uncertain": self.match_uncertain,
            },
            "emotion": self.emotion.to_dict() if self.emotion else None,
            "pose": self.pose.to_dict() if self.pose else None,
            "spoof": self.spoof.to_dict() if self.spoof else None,
            "affect": self.affect.to_dict() if self.affect else None,
            "mood": self.mood.to_dict() if self.mood else None,
            "age": self.age,
            "gender": self.gender,
            "quality": round(self.quality, 4),
            "observation_id": self.observation_id,
            "message": self.message,
        }


class MoodEngine:


    def __init__(self, db: MoodDatabase | None = None,
                 config: EngineConfig | None = None,
                 auto_enrol_unknown: bool = True) -> None:
        self.config = config or CONFIG
        self.db = db if db is not None else MoodDatabase()
        self.detector = FaceDetector(self.config.detection, self.config.tracking)
        self.recognizer = FaceRecognizer(self.db, self.config.recognition)
        self.emotion = EmotionAnalyzer(self.config.emotion)
        self.pose = PoseEstimator()
        self.spoof = SpoofDetector(self.config.spoof)
        self.affect = AffectAnalyzer()
        self.baseline = BaselineTracker()
        self.calibration = CalibrationManager(
            enabled=self.config.guided_calibration)
        self.session = SessionAnalyser(
            period_s=self.config.session_period_s)
        self.recent_reports: list[SessionReport] = []
        self.auto_enrol_unknown = auto_enrol_unknown

        self.frame_index = 0
        self.session_id: int | None = None
        self._current_person: int | None = None
        self._last_result: FrameResult | None = None
        self._loaded = False
        self._writer: Callable[[Callable[[], None]], bool] | None = None
        self._last_snapshot_at = 0.0

    def load(self) -> None:
        if self._loaded:
            return
        self.detector.load()
        try:
            for person_id, payload in self.db.all_baselines().items():
                self.baseline.load(person_id, payload)
        except Exception:
            pass
        self._loaded = True

    def start_session(self, note: str | None = None) -> int:
        self.session_id = self.db.start_session(note)
        self.frame_index = 0
        return self.session_id

    def end_session(self) -> None:
        if self.session_id is not None:
            self.db.end_session(self.session_id, self.frame_index)
            self.session_id = None

    def reset(self) -> None:
        self.detector.reset_tracking()
        self.emotion.reset()
        self.spoof.reset()
        self.affect.reset()
        self.baseline.reset()
        self._current_person = None
        self._last_result = None

    def set_writer(self,
                   writer: Callable[[Callable[[], None]], bool] | None) -> None:
        self._writer = writer

    def _submit(self, job: Callable[[], None]) -> bool:
        if self._writer is not None:
            return self._writer(job)
        job()
        return True

    def analyse_frame(self, frame: np.ndarray, *, persist: bool = True,
                      save_snapshot: bool | None = None) -> FrameResult:
        self.load()
        t0 = time.perf_counter()
        self.frame_index += 1

        subject, faces = self.detector.detect_subject(frame)
        if subject is None:
            self._current_person = None
            res = FrameResult(
                ok=False, frame_index=self.frame_index, timestamp=time.time(),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                face_count=len(faces), message="no_face")
            self._last_result = res
            return res

        quality = self.detector.quality_score(subject, frame)
        spoof = self.spoof.check(subject, frame)
        pose = self.pose.estimate(subject, frame)

        match = self.recognizer.match(subject.embedding)
        person_id, person_name = match.person_id, match.name
        is_new = False
        is_provisional = False

        if not spoof.is_spoof:
            if match.is_new:
                if self.auto_enrol_unknown and quality >= 0.45:
                    snap = self._save_face(subject, frame, "auto")
                    person_id, person_name = self.recognizer.enrol_provisional(
                        subject.embedding, quality=quality, image_path=snap)
                    is_new = True
                    is_provisional = True
            elif person_id is not None:
                self.recognizer.maybe_auto_enrol(
                    person_id, subject.embedding, match.score, quality)
                rec = self.db.get_person(person_id)
                is_provisional = bool(rec and rec["is_provisional"])

        if person_id is not None and person_id != self._current_person:
            if self._current_person is not None:
                self.emotion.reset()
            self._current_person = person_id
            self.db.touch_person(person_id)

        crop = self.detector.align(subject, frame)

        cal_session = self.calibration.ensure(
            person_id, person_name, self.baseline.baseline_for(person_id))
        if cal_session is not None and not cal_session.complete:
            cal_session.offer(
                has_face=True, quality=quality,
                yaw=pose.yaw if pose else 0.0,
                pitch=pose.pitch if pose else 0.0)

        emotion = self.emotion.analyse(
            crop, subject.landmarks_2d, quality=quality,
            rebalance=lambda p, aus: self.baseline.adjust(
                p, aus, person_id,
                learn=rest_frames_only(cal_session, p))[0],
        )
        affect = self.affect.update(emotion, pose, emotion.action_units)
        mood = self.baseline.update_mood(
            emotion.valence, emotion.arousal,
            affect.tension, affect.volatility, person_id)

        session_report = self.session.observe(
            person_id=person_id,
            person_name=person_name,
            valence=emotion.valence,
            arousal=emotion.arousal,
            mood_state=mood.state if mood else None,
            emotion=emotion.label,
            confidence=emotion.confidence,
            units={
                "engagement": affect.engagement,
                "fatigue": affect.fatigue,
                "tension": affect.tension,
            },
            now=time.time(),
        )
        if session_report is not None:
            self._store_session(session_report)

        snapshot_path = None
        want_snap = (self.config.save_snapshots if save_snapshot is None
                     else save_snapshot)
        now = time.time()
        if (want_snap and person_id is not None
                and emotion.confidence >= self.config.snapshot_min_confidence
                and now - self._last_snapshot_at
                >= self.config.snapshot_min_interval_s):
            self._last_snapshot_at = now
            crop_copy = subject.crop(frame, margin=0.25).copy()
            snapshot_path = str(FACES_DIR / f"{int(now * 1000)}_"
                                            f"{emotion.label}.jpg")
            target = snapshot_path
            self._submit(lambda: self._write_jpeg(target, crop_copy))

        result = FrameResult(
            ok=True,
            frame_index=self.frame_index,
            timestamp=time.time(),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            face_count=len(faces),
            bbox=[int(v) for v in subject.bbox],
            det_score=subject.det_score,
            person_id=person_id,
            person_name=person_name,
            is_new_person=is_new,
            is_provisional=is_provisional,
            match_score=match.score,
            match_uncertain=match.is_uncertain,
            emotion=emotion,
            pose=pose,
            spoof=spoof,
            affect=affect,
            mood=mood,
            age=subject.age,
            gender=subject.gender,
            quality=quality,
            message="spoof_suspected" if spoof.is_spoof else "ok",
        )

        if persist:
            result.observation_id = self._persist(result, snapshot_path)

        self._last_result = result
        return result

    @staticmethod
    def _write_jpeg(path: str, image: np.ndarray) -> None:
        try:
            cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        except Exception:
            pass

    def _persist(self, r: FrameResult, snapshot: str | None) -> int:
        assert r.emotion is not None
        if self._writer is not None:
            payload = self._observation_payload(r, snapshot)
            self._submit(lambda: self.db.add_observation(**payload))
            return -1
        return self._write_observation(r, snapshot)

    def _observation_payload(self, r: FrameResult,
                             snapshot: str | None) -> dict[str, Any]:
        assert r.emotion is not None
        return dict(
            person_id=r.person_id,
            session_id=self.session_id,
            emotion=r.emotion.label,
            emotion_conf=r.emotion.confidence,
            valence=r.emotion.valence,
            arousal=r.emotion.arousal,
            probs=r.emotion.probs,
            age=r.age,
            gender=r.gender,
            yaw=r.pose.yaw if r.pose else None,
            pitch=r.pose.pitch if r.pose else None,
            roll=r.pose.roll if r.pose else None,
            gaze_x=r.pose.gaze_x if r.pose else None,
            gaze_y=r.pose.gaze_y if r.pose else None,
            attention=r.pose.attention if r.pose else None,
            liveness=r.spoof.liveness if r.spoof else None,
            is_spoof=int(bool(r.spoof and r.spoof.is_spoof)),
            match_score=r.match_score,
            det_score=r.det_score,
            snapshot_path=snapshot,
            duchenne=r.affect.duchenne if r.affect else None,
            smile_type=r.affect.smile_type if r.affect else None,
            compound=r.affect.compound if r.affect else None,
            engagement=r.affect.engagement if r.affect else None,
            fatigue=r.affect.fatigue if r.affect else None,
            tension=r.affect.tension if r.affect else None,
            volatility=r.affect.volatility if r.affect else None,
            blink_rate=r.affect.blink_rate if r.affect else None,
            expressiveness=r.affect.expressiveness if r.affect else None,
            mood_state=r.mood.state if r.mood else None,
        )

    def _write_observation(self, r: FrameResult, snapshot: str | None) -> int:
        return self.db.add_observation(
            **self._observation_payload(r, snapshot))

    @staticmethod
    def _save_face(face: DetectedFace, frame: np.ndarray,
                   tag: str) -> str | None:
        try:
            crop = face.crop(frame, margin=0.25)
            if crop.size == 0:
                return None
            name = f"{int(time.time() * 1000)}_{tag}.jpg"
            path = FACES_DIR / name
            cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            return str(path)
        except Exception:
            return None

    def enrol_image(self, name: str, image: np.ndarray, *,
                    source: str = "import",
                    save_crop: bool = True) -> dict[str, Any]:
        self.load()
        faces = self.detector.detect(image)
        if not faces:
            return {"ok": False, "reason": "no_face_detected"}
        face = max(faces, key=lambda f: f.area)
        quality = self.detector.quality_score(face, image)
        path = self._save_face(face, image, name) if save_crop else None
        person_id = self.recognizer.enrol(
            name, face.embedding, source=source, quality=quality,
            image_path=path)
        return {"ok": True, "person_id": person_id, "name": name,
                "quality": round(quality, 4),
                "det_score": round(face.det_score, 4),
                "faces_in_image": len(faces)}

    def rename_person(self, person_id: int, new_name: str) -> None:
        existing = self.db.get_person_by_name(new_name)
        if existing and int(existing["id"]) != person_id:
            self.db.merge_persons(person_id, int(existing["id"]))
            return
        self.db.rename_person(person_id, new_name)

    def list_persons(self) -> list[dict[str, Any]]:
        return self.db.list_persons()

    def trend(self) -> dict[str, Any]:
        return self.emotion.trend()

    def stats(self) -> dict[str, Any]:
        return {
            **self.db.stats(),
            "emotion_backend": self.emotion.backend,
            "emotion_model": self.emotion.model_path,
            "detector_models": self.detector.model_names if self._loaded else [],
        }

    def _store_session(self, report: SessionReport) -> None:
        self.recent_reports.insert(0, report)
        del self.recent_reports[12:]
        session_id = self.session_id
        self._submit(lambda: self.db.save_session_report(
            report.to_dict(), session_id=session_id))

    def end_period(self) -> SessionReport | None:
        report = self.session.flush()
        if report is not None:
            self._store_session(report)
        return report

    def save_baselines(self) -> int:
        saved = 0
        for person_id in list(self.baseline._baselines):
            payload = self.baseline.export(person_id)
            if payload and payload.get("samples", 0) >= 20:
                try:
                    self.db.save_baseline(person_id, payload)
                    saved += 1
                except Exception:
                    continue
        return saved

    def close(self) -> None:
        try:
            self.save_baselines()
        except Exception:
            pass
        self.end_session()
        self.db.close()

    def __enter__(self) -> "MoodEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
