from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import CONFIG, RecognitionConfig
from ..db import MoodDatabase


@dataclass
class MatchResult:
    person_id: int | None
    name: str | None
    score: float
    is_new: bool
    is_uncertain: bool
    runner_up: tuple[str, float] | None = None

    @property
    def matched(self) -> bool:
        return self.person_id is not None and not self.is_new


class FaceRecognizer:


    def __init__(self, db: MoodDatabase,
                 config: RecognitionConfig | None = None) -> None:
        self.db = db
        self.config = config or CONFIG.recognition

    def match(self, embedding: np.ndarray, *, top_k: int = 3) -> MatchResult:
        matrix = self.db.embedding_matrix
        if matrix is None or matrix.shape[0] == 0:
            return MatchResult(None, None, 0.0, is_new=True, is_uncertain=False)

        query = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return MatchResult(None, None, 0.0, is_new=True, is_uncertain=False)
        query = query / norm

        if query.size != matrix.shape[1]:
            return MatchResult(None, None, 0.0, is_new=True, is_uncertain=False)

        sims = matrix @ query
        person_ids = np.asarray(self.db.embedding_person_ids)

        best: dict[int, float] = {}
        for pid in np.unique(person_ids):
            s = sims[person_ids == pid]
            k = min(top_k, s.size)
            best[int(pid)] = float(np.mean(np.sort(s)[-k:]))

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_score = ranked[0]

        runner_up = None
        if len(ranked) > 1:
            second = self.db.get_person(ranked[1][0])
            if second:
                runner_up = (str(second["name"]), float(ranked[1][1]))

        is_new = top_score < self.config.new_threshold
        is_uncertain = (not is_new) and top_score < self.config.match_threshold

        if is_new:
            return MatchResult(None, None, top_score, True, False, runner_up)

        person = self.db.get_person(top_id)
        return MatchResult(
            person_id=top_id,
            name=str(person["name"]) if person else None,
            score=top_score,
            is_new=False,
            is_uncertain=is_uncertain,
            runner_up=runner_up,
        )

    def enrol(self, name: str, embedding: np.ndarray, *,
              source: str = "camera", quality: float = 0.0,
              image_path: str | None = None,
              provisional: bool = False) -> int:
        person_id = self.db.get_or_create_person(name, provisional=provisional)
        self.db.add_embedding(person_id, embedding, source=source,
                              quality=quality, image_path=image_path)
        self._enforce_limit(person_id)
        return person_id

    def enrol_provisional(self, embedding: np.ndarray, *, quality: float = 0.0,
                          image_path: str | None = None) -> tuple[int, str]:
        name = self.db.next_provisional_name()
        person_id = self.db.create_person(name, provisional=True)
        self.db.add_embedding(person_id, embedding, source="auto",
                              quality=quality, image_path=image_path)
        return person_id, name

    def maybe_auto_enrol(self, person_id: int, embedding: np.ndarray,
                         score: float, quality: float) -> bool:
        cfg = self.config
        if not (cfg.auto_enrol_low <= score <= cfg.auto_enrol_high):
            return False
        if quality < 0.55:
            return False
        if self.db.count_embeddings(person_id) >= cfg.max_embeddings_per_person:
            return False
        self.db.add_embedding(person_id, embedding, source="auto",
                              quality=quality)
        return True

    def _enforce_limit(self, person_id: int) -> None:
        limit = self.config.max_embeddings_per_person
        if self.db.count_embeddings(person_id) > limit:
            self.db.prune_embeddings(person_id, limit)

    def identify_all(self, embeddings: np.ndarray) -> list[MatchResult]:
        return [self.match(e) for e in np.atleast_2d(embeddings)]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a / na, b / nb))
