"""Small durable, process-safe SQLite store for per-browser scan history."""
import json
import sqlite3
from pathlib import Path


class HistoryStore:
    def __init__(self, path: Path, max_records: int = 100):
        self.path = Path(path)
        self.max_records = max_records
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                record_json TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_history_session_id ON prediction_history(session_key, id DESC)")

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def list(self, session_key):
        with self._connect() as db:
            rows = db.execute("SELECT record_json FROM prediction_history WHERE session_key=? ORDER BY id DESC LIMIT ?", (session_key, self.max_records)).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    def add(self, session_key, record):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO prediction_history(session_key, record_json) VALUES (?, ?)", (session_key, json.dumps(record, separators=(",", ":"))))
            db.execute("DELETE FROM prediction_history WHERE session_key=? AND id NOT IN (SELECT id FROM prediction_history WHERE session_key=? ORDER BY id DESC LIMIT ?)", (session_key, session_key, self.max_records))

    def clear(self, session_key):
        with self._connect() as db:
            db.execute("DELETE FROM prediction_history WHERE session_key=?", (session_key,))
