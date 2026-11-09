"""Dataset lokal: audio, metadata, dan fitur disimpan dalam transaksi SQLite."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from audio_analysis import FEATURE_VERSION

DB_PATH = Path(__file__).resolve().parent / "data" / "samples.sqlite3"


@contextmanager
def connect(path=DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY, label TEXT NOT NULL, notes TEXT NOT NULL,
                created_at TEXT NOT NULL, duration REAL NOT NULL,
                features TEXT NOT NULL, feature_version INTEGER NOT NULL,
                audio BLOB NOT NULL, digest TEXT UNIQUE NOT NULL
            )""")
            yield connection
    finally:
        connection.close()


def list_samples(path=DB_PATH):
    with connect(path) as connection:
        rows = connection.execute("SELECT id,label,notes,created_at,duration,features,feature_version FROM samples ORDER BY id DESC").fetchall()
    return [dict(row, features=json.loads(row["features"])) for row in rows]


def save_sample(label, notes, audio, features, duration, path=DB_PATH):
    label, notes = label.strip(), notes.strip()
    if not label or len(label) > 100 or len(notes) > 2000:
        raise ValueError("Label wajib diisi (maksimal 100 karakter); catatan maksimal 2.000 karakter.")
    with connect(path) as connection:
        connection.execute(
            "INSERT INTO samples (label,notes,created_at,duration,features,feature_version,audio,digest) VALUES (?,?,?,?,?,?,?,?)",
            (label, notes, datetime.now(timezone.utc).isoformat(), duration,
             json.dumps(features.tolist()), FEATURE_VERSION, audio, hashlib.sha256(audio).hexdigest()),
        )


def get_audio(sample_id, path=DB_PATH):
    with connect(path) as connection:
        row = connection.execute("SELECT audio FROM samples WHERE id=?", (sample_id,)).fetchone()
    if row is None:
        raise ValueError("Sampel tidak ditemukan.")
    return row["audio"]


def delete_sample(sample_id, path=DB_PATH):
    with connect(path) as connection:
        connection.execute("DELETE FROM samples WHERE id=?", (sample_id,))
