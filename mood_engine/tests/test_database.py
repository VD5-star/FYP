from __future__ import annotations

import numpy as np
import pytest

from engine.db import MoodDatabase


def test_database_file_is_encrypted(tmp_path):
    path = tmp_path / "enc.db"
    db = MoodDatabase(path)
    db.create_person("Mohammed_Secret")
    db.add_observation(person_id=1, emotion="happy", emotion_conf=0.9,
                       valence=0.8, arousal=0.6, probs={"happy": 0.9})
    db.close()

    raw = path.read_bytes()
    assert b"Mohammed_Secret" not in raw, "person name found in plaintext"
    assert b"happy" not in raw, "emotion label found in plaintext"
    assert not raw.startswith(b"SQLite format 3"), "database is not encrypted"


def test_wrong_key_is_rejected(tmp_path):
    path = tmp_path / "key.db"
    db = MoodDatabase(path, key="a" * 64)
    db.create_person("Ali")
    db.close()

    with pytest.raises(Exception):
        MoodDatabase(path, key="b" * 64)

    reopened = MoodDatabase(path, key="a" * 64)
    assert reopened.get_person_by_name("Ali") is not None
    reopened.close()


def test_person_crud(temp_db):
    pid = temp_db.create_person("Ahmed")
    assert temp_db.get_person(pid)["name"] == "Ahmed"
    assert temp_db.get_or_create_person("Ahmed") == pid

    temp_db.rename_person(pid, "Ahmed Ali")
    person = temp_db.get_person(pid)
    assert person["name"] == "Ahmed Ali"
    assert person["is_provisional"] == 0

    temp_db.delete_person(pid)
    assert temp_db.get_person(pid) is None


def test_embedding_cache_is_normalised(temp_db):
    pid = temp_db.create_person("Sara")
    rng = np.random.default_rng(0)
    for _ in range(5):
        temp_db.add_embedding(pid, rng.normal(size=512) * 7.0)

    matrix = temp_db.embedding_matrix
    assert matrix.shape == (5, 512)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "cache vectors must be unit norm"
    assert temp_db.embedding_person_ids == [pid] * 5


def test_provisional_naming_fills_gaps(temp_db):
    assert temp_db.next_provisional_name() == "Unknown-1"
    temp_db.create_person("Unknown-1", provisional=True)
    temp_db.create_person("Unknown-3", provisional=True)
    assert temp_db.next_provisional_name() == "Unknown-2"


def test_prune_keeps_highest_quality(temp_db):
    pid = temp_db.create_person("Omar")
    rng = np.random.default_rng(1)
    for q in (0.1, 0.9, 0.5, 0.7, 0.3):
        temp_db.add_embedding(pid, rng.normal(size=512), quality=q)

    temp_db.prune_embeddings(pid, keep=2)
    rows = temp_db.conn.execute(
        "SELECT quality FROM embeddings WHERE person_id = ? "
        "ORDER BY quality DESC", (pid,)).fetchall()
    assert [round(r["quality"], 1) for r in rows] == [0.9, 0.7]


def test_merge_persons_moves_all_data(temp_db):
    a = temp_db.create_person("Unknown-1", provisional=True)
    b = temp_db.create_person("Khalid")
    rng = np.random.default_rng(2)
    temp_db.add_embedding(a, rng.normal(size=512))
    temp_db.add_observation(person_id=a, emotion="sad", emotion_conf=0.5,
                            valence=-0.5, arousal=0.3, probs={"sad": 0.5})

    temp_db.merge_persons(a, b)
    assert temp_db.get_person(a) is None
    assert temp_db.count_embeddings(b) == 1
    assert len(temp_db.recent_observations(person_id=b)) == 1


def test_cascade_delete_removes_children(temp_db):
    pid = temp_db.create_person("Nora")
    temp_db.add_embedding(pid, np.ones(512))
    temp_db.add_observation(person_id=pid, emotion="happy", emotion_conf=0.8,
                            valence=0.7, arousal=0.5, probs={"happy": 0.8})

    temp_db.delete_person_data(pid)
    assert temp_db.count_embeddings() == 0
    assert temp_db.stats()["observations"] == 0


def test_wipe_all_clears_everything(temp_db):
    pid = temp_db.create_person("Layla")
    temp_db.add_embedding(pid, np.ones(512))
    sid = temp_db.start_session()
    temp_db.add_observation(person_id=pid, session_id=sid, emotion="fear",
                            emotion_conf=0.6, valence=-0.6, arousal=0.8,
                            probs={"fear": 0.6})

    removed = temp_db.wipe_all()
    assert removed["persons"] == 1
    assert temp_db.stats() == {"persons": 0, "provisional": 0, "embeddings": 0,
                               "observations": 0, "sessions": 0}
    assert temp_db.embedding_matrix is None


def test_observation_summary(temp_db):
    pid = temp_db.create_person("Yousef")
    for emotion, n in (("happy", 3), ("sad", 1)):
        for _ in range(n):
            temp_db.add_observation(person_id=pid, emotion=emotion,
                                    emotion_conf=0.8, valence=0.0,
                                    arousal=0.5, probs={emotion: 0.8})

    summary = {s["emotion"]: s["n"] for s in temp_db.emotion_summary()}
    assert summary == {"happy": 3, "sad": 1}


def test_session_reports_round_trip(temp_db):
    person_id = temp_db.create_person("Reported")
    temp_db.save_session_report({
        "person_id": person_id,
        "person_name": "Reported",
        "started_at": 1_700_000_000.0,
        "ended_at": 1_700_000_300.0,
        "duration_s": 300.0,
        "samples": 480,
        "state": "tense",
        "confidence": 0.72,
        "dominance": 0.61,
        "valence": -0.18,
        "arousal": 0.66,
        "stability": 0.44,
        "engagement": 0.55,
        "fatigue": 0.21,
        "tension": 0.71,
        "emotion_mix": {"anger": 0.4, "neutral": 0.6},
        "notes": ["sustained tension"],
    })
    rows = temp_db.session_reports()
    assert len(rows) == 1
    row = rows[0]
    assert row["state"] == "tense"
    assert row["samples"] == 480
    assert row["confidence"] == pytest.approx(0.72)
    assert row["emotion_mix"] == {"anger": 0.4, "neutral": 0.6}
    assert row["notes"] == ["sustained tension"]


def test_session_reports_filter_by_person(temp_db):
    a = temp_db.create_person("A")
    b = temp_db.create_person("B")
    for pid, state in ((a, "low"), (b, "positive"), (a, "calm")):
        temp_db.save_session_report({
            "person_id": pid, "person_name": "x",
            "started_at": 1.0, "ended_at": 2.0, "duration_s": 1.0,
            "samples": 90, "state": state, "confidence": 0.5,
        })
    assert len(temp_db.session_reports(person_id=a)) == 2
    assert len(temp_db.session_reports(person_id=b)) == 1
    assert len(temp_db.session_reports()) == 3


def test_deleting_a_person_removes_their_reports(temp_db):
    person_id = temp_db.create_person("Temporary")
    temp_db.save_session_report({
        "person_id": person_id, "person_name": "Temporary",
        "started_at": 1.0, "ended_at": 2.0, "duration_s": 1.0,
        "samples": 90, "state": "low", "confidence": 0.5,
    })
    assert temp_db.session_reports(person_id=person_id)
    temp_db.delete_person(person_id)
    assert temp_db.session_reports(person_id=person_id) == []
