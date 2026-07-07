"""SQLite-backed dedup so the same item is never delivered twice.

Single table keyed by (source, item_id); rows older than 30 days are
purged on each open to keep the file small.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

_PURGE_AFTER_DAYS = 30


class Dedup:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notified (
                source TEXT NOT NULL,
                item_id TEXT NOT NULL,
                notified_at TEXT NOT NULL,
                PRIMARY KEY (source, item_id)
            )
            """
        )
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=_PURGE_AFTER_DAYS)
        ).isoformat()
        self._conn.execute("DELETE FROM notified WHERE notified_at < ?", (cutoff,))
        self._conn.commit()

    def seen(self, source: str, item_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM notified WHERE source=? AND item_id=? LIMIT 1",
            (source, item_id),
        ).fetchone()
        return row is not None

    def mark_many(self, keys: Iterable[tuple[str, str]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.executemany(
            "INSERT OR REPLACE INTO notified (source, item_id, notified_at) VALUES (?, ?, ?)",
            [(s, i, now) for s, i in keys],
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
