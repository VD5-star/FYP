from __future__ import annotations

import base64
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

try:
    import sqlcipher3 as _sqlcipher
except ImportError:
    _sqlcipher = None

from ..config import DB_PATH, DATA_DIR

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
KEY_PATH = DATA_DIR / "db.key"
SCHEMA_VERSION = "3"

OBSERVATION_COLUMNS = (
    "person_id", "session_id", "ts", "emotion", "emotion_conf", "valence",
    "arousal", "probs", "age", "gender", "yaw", "pitch", "roll", "gaze_x",
    "gaze_y", "attention", "liveness", "is_spoof", "match_score", "det_score",
    "snapshot_path", "duchenne", "smile_type", "compound", "engagement",
    "fatigue", "tension", "volatility", "blink_rate", "expressiveness",
    "mood_state",
)


def _dpapi_protect(raw: bytes) -> bytes | None:
    try:
        import ctypes
        from ctypes import wintypes

        class BLOB(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD),
                        ("pbData", ctypes.POINTER(ctypes.c_char))]

        buf = ctypes.create_string_buffer(raw, len(raw))
        blob_in = BLOB(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        blob_out = BLOB()
        ok = ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(blob_in), None, None, None, None, 0,
            ctypes.byref(blob_out))
        if not ok:
            return None
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    except Exception:
        return None


def _dpapi_unprotect(blob: bytes) -> bytes | None:
    try:
        import ctypes
        from ctypes import wintypes

        class BLOB(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD),
                        ("pbData", ctypes.POINTER(ctypes.c_char))]

        buf = ctypes.create_string_buffer(blob, len(blob))
        blob_in = BLOB(len(blob), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        blob_out = BLOB()
        ok = ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(blob_in), None, None, None, None, 0,
            ctypes.byref(blob_out))
        if not ok:
            return None
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    except Exception:
        return None


def load_or_create_key(key_path: Path = KEY_PATH) -> str:
    if key_path.exists():
        blob = key_path.read_bytes()
        if blob.startswith(b"DPAPI:"):
            raw = _dpapi_unprotect(blob[6:])
            if raw is None:
                raise RuntimeError(
                    "Database key could not be decrypted. It is bound to the "
                    "Windows user that created it.")
            return raw.decode("ascii")
        return blob.decode("ascii").strip()

    key = secrets.token_hex(32)
    protected = _dpapi_protect(key.encode("ascii"))
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if protected is not None:
        key_path.write_bytes(b"DPAPI:" + protected)
    else:
        key_path.write_bytes(key.encode("ascii"))
    try:
        os.chmod(key_path, 0o600)
    except Exception:
        pass
    return key


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class MoodDatabase:


    def __init__(self, path: Path | str = DB_PATH, key: str | None = None,
                 encrypted: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.encrypted = encrypted and _sqlcipher is not None
        if encrypted and _sqlcipher is None:
            raise RuntimeError(
                "sqlcipher3 is not installed; encrypted storage unavailable.")
        self._key = key if key is not None else (
            load_or_create_key() if self.encrypted else "")
        self._conn: sqlite3.Connection | None = None
        self._matrix: np.ndarray | None = None
        self._emb_ids: list[int] = []
        self._emb_person_ids: list[int] = []
        self.connect()

    def connect(self) -> None:
        driver = _sqlcipher if self.encrypted else sqlite3
        conn = driver.connect(str(self.path), check_same_thread=False)
        conn.row_factory = driver.Row
        if self.encrypted:
            conn.execute(f"PRAGMA key = \"x'{self._key}'\"")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        self._conn = conn
        self._migrate()
        self._load_cache()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database is closed.")
        return self._conn

    _ADDED_COLUMNS = (
        ("observations", "duchenne", "REAL"),
        ("observations", "smile_type", "TEXT"),
        ("observations", "compound", "TEXT"),
        ("observations", "engagement", "REAL"),
        ("observations", "fatigue", "REAL"),
        ("observations", "tension", "REAL"),
        ("observations", "volatility", "REAL"),
        ("observations", "blink_rate", "REAL"),
        ("observations", "expressiveness", "REAL"),
        ("observations", "mood_state", "TEXT"),
        ("persons", "baseline", "TEXT"),
    )

    def _migrate(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        for table, column, kind in self._ADDED_COLUMNS:
            existing = {row["name"] for row in
                        self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (SCHEMA_VERSION,))
        self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
            except Exception:
                pass
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "MoodDatabase":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def create_person(self, name: str, *, provisional: bool = False,
                      notes: str | None = None) -> int:
        now = utcnow()
        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO persons(name, is_provisional, notes, created_at, "
                "updated_at) VALUES(?,?,?,?,?)",
                (name, int(provisional), notes, now, now))
        return int(cur.lastrowid)

    def get_person(self, person_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
        return dict(row) if row else None

    def get_person_by_name(self, name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM persons WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def get_or_create_person(self, name: str, *, provisional: bool = False) -> int:
        found = self.get_person_by_name(name)
        if found:
            return int(found["id"])
        return self.create_person(name, provisional=provisional)

    def list_persons(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT p.*, "
            "  (SELECT COUNT(*) FROM embeddings e WHERE e.person_id = p.id) "
            "     AS embedding_count, "
            "  (SELECT COUNT(*) FROM observations o WHERE o.person_id = p.id) "
            "     AS observation_count "
            "FROM persons p ORDER BY p.is_provisional, p.name"
        ).fetchall()
        return [dict(r) for r in rows]

    def rename_person(self, person_id: int, new_name: str) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE persons SET name = ?, is_provisional = 0, "
                "updated_at = ? WHERE id = ?",
                (new_name, utcnow(), person_id))

    def merge_persons(self, source_id: int, target_id: int) -> None:
        if source_id == target_id:
            return
        with self.transaction() as c:
            c.execute("UPDATE embeddings SET person_id = ? WHERE person_id = ?",
                      (target_id, source_id))
            c.execute("UPDATE observations SET person_id = ? WHERE person_id = ?",
                      (target_id, source_id))
            c.execute("DELETE FROM persons WHERE id = ?", (source_id,))
        self._load_cache()

    def touch_person(self, person_id: int) -> None:
        with self.transaction() as c:
            c.execute("UPDATE persons SET last_seen_at = ? WHERE id = ?",
                      (utcnow(), person_id))

    def save_baseline(self, person_id: int, payload: dict[str, Any]) -> None:
        import json
        with self.transaction() as c:
            c.execute("UPDATE persons SET baseline = ? WHERE id = ?",
                      (json.dumps(payload), person_id))

    def load_baseline(self, person_id: int) -> dict[str, Any] | None:
        import json
        row = self.conn.execute(
            "SELECT baseline FROM persons WHERE id = ?",
            (person_id,)).fetchone()
        if not row or not row["baseline"]:
            return None
        try:
            return json.loads(row["baseline"])
        except Exception:
            return None

    def all_baselines(self) -> dict[int, dict[str, Any]]:
        import json
        out: dict[int, dict[str, Any]] = {}
        for row in self.conn.execute(
                "SELECT id, baseline FROM persons "
                "WHERE baseline IS NOT NULL"):
            try:
                out[int(row["id"])] = json.loads(row["baseline"])
            except Exception:
                continue
        return out

    def save_session_report(self, report: dict[str, Any],
                            session_id: int | None = None) -> int:
        import json
        from datetime import datetime, timezone

        def stamp(value: Any) -> str:
            try:
                return datetime.fromtimestamp(
                    float(value), tz=timezone.utc).isoformat()
            except Exception:
                return datetime.now(tz=timezone.utc).isoformat()

        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO session_reports("
                " session_id, person_id, person_name, started_at, ended_at,"
                " duration_s, samples, state, confidence, dominance, valence,"
                " arousal, stability, engagement, fatigue, tension,"
                " emotion_mix, notes)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (session_id, report.get("person_id"),
                 report.get("person_name"),
                 stamp(report.get("started_at")), stamp(report.get("ended_at")),
                 report.get("duration_s", 0.0), report.get("samples", 0),
                 report.get("state", "neutral"),
                 report.get("confidence", 0.0), report.get("dominance"),
                 report.get("valence"), report.get("arousal"),
                 report.get("stability"), report.get("engagement"),
                 report.get("fatigue"), report.get("tension"),
                 json.dumps(report.get("emotion_mix") or {}),
                 json.dumps(report.get("notes") or [])))
            return int(cur.lastrowid or 0)

    def session_reports(self, person_id: int | None = None,
                        limit: int = 50) -> list[dict[str, Any]]:
        import json
        sql = ("SELECT * FROM session_reports"
               + (" WHERE person_id = ?" if person_id is not None else "")
               + " ORDER BY ended_at DESC LIMIT ?")
        args: tuple[Any, ...] = ((person_id, limit) if person_id is not None
                                 else (limit,))
        out: list[dict[str, Any]] = []
        for row in self.conn.execute(sql, args):
            item = dict(row)
            for key in ("emotion_mix", "notes"):
                try:
                    item[key] = json.loads(item[key]) if item[key] else None
                except Exception:
                    item[key] = None
            out.append(item)
        return out

    def next_provisional_name(self) -> str:
        rows = self.conn.execute(
            "SELECT name FROM persons WHERE name LIKE 'Unknown-%'").fetchall()
        used = set()
        for r in rows:
            tail = str(r["name"]).split("-", 1)[-1]
            if tail.isdigit():
                used.add(int(tail))
        n = 1
        while n in used:
            n += 1
        return f"Unknown-{n}"

    def delete_person(self, person_id: int) -> None:
        with self.transaction() as c:
            c.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        self._load_cache()

    def add_embedding(self, person_id: int, vector: np.ndarray, *,
                      source: str = "camera", quality: float = 0.0,
                      image_path: str | None = None,
                      refresh_cache: bool = True) -> int:
        vec = np.asarray(vector, dtype=np.float32).ravel()
        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO embeddings(person_id, vector, dim, source, "
                "quality, image_path, created_at) VALUES(?,?,?,?,?,?,?)",
                (person_id, vec_to_blob(vec), int(vec.size), source,
                 float(quality), image_path, utcnow()))
        if refresh_cache:
            self._load_cache()
        return int(cur.lastrowid)

    def prune_embeddings(self, person_id: int, keep: int) -> int:
        with self.transaction() as c:
            cur = c.execute(
                "DELETE FROM embeddings WHERE id IN ("
                "  SELECT id FROM embeddings WHERE person_id = ? "
                "  ORDER BY quality DESC, id DESC LIMIT -1 OFFSET ?)",
                (person_id, keep))
        removed = cur.rowcount or 0
        if removed:
            self._load_cache()
        return removed

    def count_embeddings(self, person_id: int | None = None) -> int:
        if person_id is None:
            row = self.conn.execute("SELECT COUNT(*) n FROM embeddings").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) n FROM embeddings WHERE person_id = ?",
                (person_id,)).fetchone()
        return int(row["n"])

    def _load_cache(self) -> None:
        rows = self.conn.execute(
            "SELECT id, person_id, vector FROM embeddings ORDER BY id"
        ).fetchall()
        if not rows:
            self._matrix = None
            self._emb_ids = []
            self._emb_person_ids = []
            return
        decoded = [(r, blob_to_vec(r["vector"])) for r in rows]
        dim = max(v.size for _, v in decoded)
        decoded = [(r, v) for r, v in decoded if v.size == dim]
        keep = [r for r, _ in decoded]
        mat = np.vstack([v for _, v in decoded]).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = mat / norms
        self._emb_ids = [int(r["id"]) for r in keep]
        self._emb_person_ids = [int(r["person_id"]) for r in keep]

    @property
    def embedding_matrix(self) -> np.ndarray | None:
        return self._matrix

    @property
    def embedding_person_ids(self) -> list[int]:
        return self._emb_person_ids

    def refresh_cache(self) -> None:
        self._load_cache()

    def start_session(self, note: str | None = None) -> int:
        with self.transaction() as c:
            cur = c.execute(
                "INSERT INTO sessions(started_at, note) VALUES(?,?)",
                (utcnow(), note))
        return int(cur.lastrowid)

    def end_session(self, session_id: int, frame_count: int = 0) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE sessions SET ended_at = ?, frame_count = ? WHERE id = ?",
                (utcnow(), frame_count, session_id))

    def add_observation(self, **kw: Any) -> int:
        probs = kw.get("probs")
        if isinstance(probs, dict):
            kw["probs"] = json.dumps(probs, ensure_ascii=False)
        kw.setdefault("ts", utcnow())
        kw.setdefault("is_spoof", 0)
        cols = OBSERVATION_COLUMNS
        values = [kw.get(c) for c in cols]
        placeholders = ",".join("?" * len(cols))
        with self.transaction() as c:
            cur = c.execute(
                f"INSERT INTO observations({','.join(cols)}) "
                f"VALUES({placeholders})", values)
        return int(cur.lastrowid)

    def add_observations_bulk(self, rows: Iterable[dict[str, Any]]) -> int:
        cols = OBSERVATION_COLUMNS
        payload = []
        for kw in rows:
            probs = kw.get("probs")
            if isinstance(probs, dict):
                kw = {**kw, "probs": json.dumps(probs, ensure_ascii=False)}
            kw.setdefault("ts", utcnow())
            kw.setdefault("is_spoof", 0)
            payload.append([kw.get(c) for c in cols])
        if not payload:
            return 0
        placeholders = ",".join("?" * len(cols))
        with self.transaction() as c:
            c.executemany(
                f"INSERT INTO observations({','.join(cols)}) "
                f"VALUES({placeholders})", payload)
        return len(payload)

    def recent_observations(self, person_id: int | None = None,
                            limit: int = 100) -> list[dict[str, Any]]:
        if person_id is None:
            rows = self.conn.execute(
                "SELECT o.*, p.name AS person_name FROM observations o "
                "LEFT JOIN persons p ON p.id = o.person_id "
                "ORDER BY o.id DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT o.*, p.name AS person_name FROM observations o "
                "LEFT JOIN persons p ON p.id = o.person_id "
                "WHERE o.person_id = ? ORDER BY o.id DESC LIMIT ?",
                (person_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def emotion_summary(self, person_id: int | None = None,
                        since: str | None = None) -> list[dict[str, Any]]:
        sql = ("SELECT emotion, COUNT(*) AS n, AVG(emotion_conf) AS avg_conf, "
               "AVG(valence) AS avg_valence, AVG(arousal) AS avg_arousal "
               "FROM observations WHERE 1=1")
        args: list[Any] = []
        if person_id is not None:
            sql += " AND person_id = ?"
            args.append(person_id)
        if since is not None:
            sql += " AND ts >= ?"
            args.append(since)
        sql += " GROUP BY emotion ORDER BY n DESC"
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def wipe_all(self, *, delete_files: bool = True) -> dict[str, int]:
        counts = {
            "persons": self.conn.execute(
                "SELECT COUNT(*) n FROM persons").fetchone()["n"],
            "embeddings": self.conn.execute(
                "SELECT COUNT(*) n FROM embeddings").fetchone()["n"],
            "observations": self.conn.execute(
                "SELECT COUNT(*) n FROM observations").fetchone()["n"],
            "sessions": self.conn.execute(
                "SELECT COUNT(*) n FROM sessions").fetchone()["n"],
        }
        with self.transaction() as c:
            c.execute("DELETE FROM observations")
            c.execute("DELETE FROM embeddings")
            c.execute("DELETE FROM sessions")
            c.execute("DELETE FROM persons")
            c.execute("DELETE FROM sqlite_sequence")
        self.conn.execute("VACUUM")
        self._load_cache()
        if delete_files:
            from ..config import FACES_DIR
            for f in FACES_DIR.glob("**/*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except OSError:
                        pass
        return {k: int(v) for k, v in counts.items()}

    def delete_person_data(self, person_id: int) -> dict[str, int]:
        counts = {
            "embeddings": self.conn.execute(
                "SELECT COUNT(*) n FROM embeddings WHERE person_id = ?",
                (person_id,)).fetchone()["n"],
            "observations": self.conn.execute(
                "SELECT COUNT(*) n FROM observations WHERE person_id = ?",
                (person_id,)).fetchone()["n"],
        }
        paths = [r["image_path"] for r in self.conn.execute(
            "SELECT image_path FROM embeddings WHERE person_id = ? "
            "AND image_path IS NOT NULL", (person_id,)).fetchall()]
        self.delete_person(person_id)
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        return {k: int(v) for k, v in counts.items()}

    def stats(self) -> dict[str, int]:
        q = lambda s: int(self.conn.execute(s).fetchone()["n"])
        return {
            "persons": q("SELECT COUNT(*) n FROM persons"),
            "provisional": q(
                "SELECT COUNT(*) n FROM persons WHERE is_provisional = 1"),
            "embeddings": q("SELECT COUNT(*) n FROM embeddings"),
            "observations": q("SELECT COUNT(*) n FROM observations"),
            "sessions": q("SELECT COUNT(*) n FROM sessions"),
        }
