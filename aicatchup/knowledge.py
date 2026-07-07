"""Embedding knowledge base: SQLite for rows, numpy for similarity.

Why not a vector DB? At this scale (a few hundred items/day, purged
after 90 days ≈ tens of thousands of rows max) brute-force cosine over
an in-memory matrix runs in milliseconds. Plain SQLite + numpy keeps
the project install-anywhere with zero native-extension roulette.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

_PURGE_AFTER_DAYS = 90


class Knowledge:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                source      TEXT NOT NULL,
                item_id     TEXT NOT NULL,
                title       TEXT NOT NULL,
                url         TEXT NOT NULL,
                summary     TEXT NOT NULL DEFAULT '',
                tier        INTEGER NOT NULL DEFAULT 3,
                published_at TEXT,
                ingested_at TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                dim         INTEGER NOT NULL,
                PRIMARY KEY (source, item_id)
            )
            """
        )
        self._purge()
        self._conn.commit()

    def _purge(self) -> None:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=_PURGE_AFTER_DAYS)
        ).isoformat()
        self._conn.execute("DELETE FROM items WHERE ingested_at < ?", (cutoff,))

    def has(self, source: str, item_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM items WHERE source = ? AND item_id = ?", (source, item_id)
        ).fetchone()
        return row is not None

    def add(self, item: dict[str, Any], embedding: list[float]) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm  # store unit vectors so similarity is a dot product
        published = item.get("published_at")
        self._conn.execute(
            """
            INSERT OR REPLACE INTO items
            (source, item_id, title, url, summary, tier, published_at, ingested_at, embedding, dim)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["source"],
                str(item["item_id"]),
                item["title"],
                item["url"],
                (item.get("raw_summary") or "")[:1500],
                int(item.get("tier") or 3),
                published.isoformat() if published else None,
                datetime.now(timezone.utc).isoformat(),
                vec.tobytes(),
                int(vec.shape[0]),
            ),
        )

    def commit(self) -> None:
        self._conn.commit()

    def matrix(self, *, before: datetime | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Return (unit-vector matrix, row metadata), optionally only rows
        ingested before *before* — that cut gives 'what did we already
        know' for novelty scoring."""
        q = "SELECT source, item_id, title, tier, ingested_at, embedding, dim FROM items"
        args: tuple = ()
        if before is not None:
            q += " WHERE ingested_at < ?"
            args = (before.isoformat(),)
        rows = self._conn.execute(q, args).fetchall()
        metas: list[dict[str, Any]] = []
        vecs: list[np.ndarray] = []
        for source, item_id, title, tier, ingested_at, blob, dim in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape[0] != dim:
                continue  # corrupted row; skip
            vecs.append(v)
            metas.append(
                {
                    "source": source,
                    "item_id": item_id,
                    "title": title,
                    "tier": tier,
                    "ingested_at": ingested_at,
                }
            )
        if not vecs:
            return np.empty((0, 0), dtype=np.float32), []
        return np.vstack(vecs), metas

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
