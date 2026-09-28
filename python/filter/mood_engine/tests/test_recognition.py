from __future__ import annotations

import itertools

import numpy as np
import pytest

from conftest import load_image


def _embed(detector, key):
    img = load_image(key)
    faces = detector.detect(img)
    assert faces, f"no face detected in {key}"
    face = max(faces, key=lambda f: f.area)
    return face.embedding, face, img


def test_same_person_scores_far_above_threshold(detector):
    from engine.config import CONFIG

    keys = ["obama_a", "obama_b", "obama_c", "obama_d"]
    embs = {k: _embed(detector, k)[0] for k in keys}

    scores = [float(np.dot(embs[a], embs[b]))
              for a, b in itertools.combinations(keys, 2)]
    assert min(scores) > CONFIG.recognition.match_threshold, (
        f"same-person similarity {min(scores):.3f} below match threshold")


def test_different_people_score_below_new_threshold(detector):
    from engine.config import CONFIG

    same_person = {k: _embed(detector, k)[0]
                   for k in ("obama_a", "obama_b", "obama_c", "obama_d")}
    others = {k: _embed(detector, k)[0] for k in ("biden", "stranger")}

    cross = [float(np.dot(a, b))
             for a in same_person.values() for b in others.values()]
    cross += [float(np.dot(others["biden"], others["stranger"]))]

    assert max(cross) < CONFIG.recognition.new_threshold, (
        f"different-person similarity {max(cross):.3f} above new threshold")


def test_separation_margin_is_wide(detector):
    keys = ["obama_a", "obama_b", "obama_c", "obama_d"]
    embs = {k: _embed(detector, k)[0] for k in keys}
    embs["biden"] = _embed(detector, "biden")[0]
    embs["stranger"] = _embed(detector, "stranger")[0]

    same = [float(np.dot(embs[a], embs[b]))
            for a, b in itertools.combinations(keys, 2)]
    diff = [float(np.dot(embs[a], embs[b]))
            for a, b in itertools.combinations(embs, 2)
            if not (a.startswith("obama") and b.startswith("obama"))]

    margin = min(same) - max(diff)
    assert margin > 0.30, f"identity margin too small: {margin:.3f}"


def test_engine_recognises_held_out_photo(engine, detector):
    for key in ("obama_a", "obama_c"):
        assert engine.enrol_image("Obama", load_image(key))["ok"]
    assert engine.enrol_image("Biden", load_image("biden"))["ok"]

    emb, _, _ = _embed(detector, "obama_b")
    match = engine.recognizer.match(emb)

    assert match.name == "Obama", f"expected Obama, got {match.name}"
    assert match.matched
    assert match.score > 0.5


def test_engine_rejects_unknown_person(engine, detector):
    assert engine.enrol_image("Obama", load_image("obama_a"))["ok"]

    emb, _, _ = _embed(detector, "stranger")
    match = engine.recognizer.match(emb)

    assert match.is_new, f"stranger wrongly matched to {match.name}"
    assert match.person_id is None


def test_provisional_person_lifecycle(engine, detector):
    emb, _, _ = _embed(detector, "obama_a")
    pid, name = engine.recognizer.enrol_provisional(emb, quality=0.8)

    assert name == "Unknown-1"
    assert engine.db.get_person(pid)["is_provisional"] == 1

    engine.rename_person(pid, "Obama")
    person = engine.db.get_person(pid)
    assert person["name"] == "Obama"
    assert person["is_provisional"] == 0


def test_renaming_into_existing_name_merges(engine, detector):
    assert engine.enrol_image("Obama", load_image("obama_a"))["ok"]
    existing = engine.db.get_person_by_name("Obama")

    emb, _, _ = _embed(detector, "obama_c")
    pid, _ = engine.recognizer.enrol_provisional(emb, quality=0.8)

    engine.rename_person(pid, "Obama")

    assert engine.db.get_person(pid) is None, "provisional record should be gone"
    assert engine.db.count_embeddings(int(existing["id"])) == 2


def test_embedding_is_unit_length(detector):
    emb, _, _ = _embed(detector, "obama_a")
    assert emb.shape == (512,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4
