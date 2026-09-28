from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ImportItem:
    path: str
    person: str
    ok: bool
    reason: str = ""
    quality: float = 0.0
    det_score: float = 0.0
    faces_found: int = 0


@dataclass
class ImportReport:
    added: int = 0
    skipped: int = 0
    failed: int = 0
    persons: dict[str, int] = field(default_factory=dict)
    items: list[ImportItem] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.added + self.skipped + self.failed

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "added": self.added,
            "skipped": self.skipped,
            "failed": self.failed,
            "persons": dict(sorted(self.persons.items())),
        }


class FaceTrainer:


    def __init__(self, engine: Any, *, min_quality: float = 0.35,
                 duplicate_threshold: float = 0.97,
                 max_per_person: int | None = None) -> None:
        self.engine = engine
        self.min_quality = min_quality
        self.duplicate_threshold = duplicate_threshold
        self.max_per_person = max_per_person

    @staticmethod
    def discover(root: Path | str) -> dict[str, list[Path]]:
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        groups: dict[str, list[Path]] = {}
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            images = sorted(
                p for p in child.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if images:
                groups[child.name] = images

        if not groups:
            loose = sorted(p for p in root.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            for p in loose:
                groups.setdefault(p.stem, []).append(p)
        return groups

    def import_folder(self, root: Path | str, *,
                      progress: Callable[[str, int, int], None] | None = None,
                      dry_run: bool = False) -> ImportReport:
        groups = self.discover(root)
        report = ImportReport()
        total = sum(len(v) for v in groups.values())
        done = 0

        for person, images in groups.items():
            if self.max_per_person:
                images = images[:self.max_per_person]
            existing = self._existing_vectors(person)
            for img_path in images:
                done += 1
                if progress:
                    progress(f"{person}/{img_path.name}", done, total)
                item = self._import_one(person, img_path, existing,
                                        dry_run=dry_run)
                report.items.append(item)
                if item.ok:
                    report.added += 1
                    report.persons[person] = report.persons.get(person, 0) + 1
                elif item.reason in ("duplicate", "low_quality"):
                    report.skipped += 1
                else:
                    report.failed += 1
        return report

    def import_person(self, name: str, images: Sequence[Path | str],
                      *, dry_run: bool = False) -> ImportReport:
        report = ImportReport()
        existing = self._existing_vectors(name)
        for p in images:
            item = self._import_one(name, Path(p), existing, dry_run=dry_run)
            report.items.append(item)
            if item.ok:
                report.added += 1
                report.persons[name] = report.persons.get(name, 0) + 1
            elif item.reason in ("duplicate", "low_quality"):
                report.skipped += 1
            else:
                report.failed += 1
        return report

    def _import_one(self, person: str, path: Path,
                    existing: list[np.ndarray],
                    *, dry_run: bool) -> ImportItem:
        image = self._read_image(path)
        if image is None:
            return ImportItem(str(path), person, False, "unreadable")

        self.engine.load()
        faces = self.engine.detector.detect(image)
        if not faces:
            return ImportItem(str(path), person, False, "no_face_detected")

        face = max(faces, key=lambda f: f.area)
        quality = self.engine.detector.quality_score(face, image)
        if quality < self.min_quality:
            return ImportItem(str(path), person, False, "low_quality",
                              quality, face.det_score, len(faces))

        emb = np.asarray(face.embedding, dtype=np.float32).ravel()
        n = float(np.linalg.norm(emb))
        emb = emb / n if n > 0 else emb

        for prev in existing:
            if float(np.dot(prev, emb)) >= self.duplicate_threshold:
                return ImportItem(str(path), person, False, "duplicate",
                                  quality, face.det_score, len(faces))

        if not dry_run:
            crop_path = self.engine._save_face(face, image, person)
            self.engine.recognizer.enrol(
                person, emb, source="import", quality=quality,
                image_path=crop_path)
            existing.append(emb)

        return ImportItem(str(path), person, True, "added", quality,
                          face.det_score, len(faces))

    def _existing_vectors(self, person: str) -> list[np.ndarray]:
        rec = self.engine.db.get_person_by_name(person)
        if not rec:
            return []
        pid = int(rec["id"])
        matrix = self.engine.db.embedding_matrix
        if matrix is None:
            return []
        ids = np.asarray(self.engine.db.embedding_person_ids)
        return [row for row in matrix[ids == pid]]

    @staticmethod
    def _read_image(path: Path) -> np.ndarray | None:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            longest = max(h, w)
            if longest > 1600:
                scale = 1600.0 / longest
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            return img
        except Exception:
            return None
