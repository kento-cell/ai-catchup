"""Offline unit tests — no network, no Ollama required."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from aicatchup import signals
from aicatchup.dedup import Dedup
from aicatchup.digest import build
from aicatchup.knowledge import Knowledge


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_signals_corroboration_counts_distinct_sources():
    now = datetime.now(timezone.utc)
    items = [
        {"source": "A", "item_id": "1", "tier": 1, "published_at": now},
        {"source": "B", "item_id": "2", "tier": 2, "published_at": now},
        {"source": "C", "item_id": "3", "tier": 3, "published_at": now},
    ]
    # items 0 and 1 are the same story; 2 is orthogonal
    vecs = np.vstack([_unit([1, 0.05, 0]), _unit([1, 0, 0.05]), _unit([0, 1, 0])])
    signals.compute(items, vecs, np.empty((0, 0), dtype=np.float32))
    assert items[0]["corroboration"] == 2
    assert items[1]["corroboration"] == 2
    assert items[2]["corroboration"] == 1
    # nothing known → everything maximally novel
    assert all(it["novelty"] == 1.0 for it in items)


def test_signals_novelty_drops_for_known_topic():
    now = datetime.now(timezone.utc)
    items = [{"source": "A", "item_id": "1", "tier": 1, "published_at": now}]
    vec = _unit([1, 0, 0])
    known = np.vstack([_unit([1, 0.01, 0])])  # nearly identical known item
    signals.compute(items, vec.reshape(1, -1), known)
    assert items[0]["novelty"] < 0.05


def test_signals_recency_decays():
    old = datetime.now(timezone.utc) - timedelta(hours=48)
    fresh_item = {"source": "A", "item_id": "1", "tier": 1,
                  "published_at": datetime.now(timezone.utc)}
    old_item = {"source": "B", "item_id": "2", "tier": 1, "published_at": old}
    vecs = np.vstack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    signals.compute([fresh_item, old_item], vecs, np.empty((0, 0), dtype=np.float32))
    assert fresh_item["recency"] > old_item["recency"]


def test_dedup_roundtrip(tmp_path: Path):
    d = Dedup(tmp_path / "dedup.db")
    assert not d.seen("src", "id1")
    d.mark_many([("src", "id1")])
    assert d.seen("src", "id1")
    d.close()
    # persists across reopen
    d2 = Dedup(tmp_path / "dedup.db")
    assert d2.seen("src", "id1")
    d2.close()


def test_knowledge_store_and_matrix(tmp_path: Path):
    kb = Knowledge(tmp_path / "kb.db")
    item = {
        "source": "A", "item_id": "x", "title": "t", "url": "u",
        "raw_summary": "s", "tier": 1,
        "published_at": datetime.now(timezone.utc),
    }
    kb.add(item, [3.0, 4.0])  # gets unit-normalised on store
    kb.commit()
    assert kb.count() == 1
    mat, metas = kb.matrix()
    assert mat.shape == (1, 2)
    assert abs(float(np.linalg.norm(mat[0])) - 1.0) < 1e-5
    assert metas[0]["source"] == "A"
    kb.close()


def test_digest_merges_also_in_sources():
    item = {
        "title": "Big News", "url": "https://example.com", "source": "HN",
        "also_in": ["Techmeme", "Techmeme", "Bluesky"],
        "corroboration": 3, "novelty": 0.9, "usefulness": 9,
        "judge_reason": "major release", "category": "model_release",
    }
    text = build([item], lang="ja", knowledge_size=42)
    assert "HN, Techmeme, Bluesky" in text  # deduped, order kept
    assert "3ソースが同時報道" in text
    assert "https://example.com" in text
