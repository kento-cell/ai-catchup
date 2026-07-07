"""Deterministic 'fastest & useful' signals, computed before any LLM sees
the items. The LLM refines the ranking; it never decides it alone.

Per item:
- novelty        1 - max cosine similarity vs. everything already in the
                 knowledge base → high = we have never seen this topic.
- corroboration  number of *distinct sources* reporting the same story in
                 this batch (cosine >= SAME_STORY). Multiple independent
                 outlets within hours is the strongest breaking-news signal.
- tier_weight    source authority (tier 1 official lab > tier 3 community).
- recency        exponential decay over the item age.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

# Cosine similarity at which two items are considered the same story.
SAME_STORY = 0.80
# Novelty saturates against this many nearest neighbours.
_TIER_WEIGHT = {1: 1.0, 2: 0.8, 3: 0.6}
_RECENCY_HALF_LIFE_H = 24.0

# Pre-score weights — corroboration dominates, novelty second.
_W_CORROBORATION = 0.40
_W_NOVELTY = 0.30
_W_TIER = 0.15
_W_RECENCY = 0.15


def _recency(published_at: datetime | None) -> float:
    if published_at is None:
        return 0.5
    age_h = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600.0
    return float(0.5 ** (max(age_h, 0.0) / _RECENCY_HALF_LIFE_H))


def compute(
    items: list[dict[str, Any]],
    batch_vecs: np.ndarray,
    known_vecs: np.ndarray,
) -> list[dict[str, Any]]:
    """Annotate *items* (aligned with rows of *batch_vecs*) in place with
    novelty / corroboration / tier_weight / recency / pre_score."""
    n = len(items)
    if n == 0:
        return items

    # novelty vs. prior knowledge
    if known_vecs.size:
        sim_known = batch_vecs @ known_vecs.T  # unit vectors → cosine
        max_known = sim_known.max(axis=1)
    else:
        max_known = np.zeros(n, dtype=np.float32)

    # corroboration within the batch
    sim_batch = batch_vecs @ batch_vecs.T
    for i, item in enumerate(items):
        same = sim_batch[i] >= SAME_STORY
        sources = {items[j]["source"] for j in np.flatnonzero(same)}
        corroboration = len(sources)  # includes own source → min 1
        novelty = float(1.0 - max_known[i])
        tier_w = _TIER_WEIGHT.get(int(item.get("tier") or 3), 0.6)
        recency = _recency(item.get("published_at"))
        # 4 distinct sources saturate the corroboration term.
        corro_norm = min(corroboration - 1, 3) / 3.0
        item["novelty"] = round(novelty, 4)
        item["corroboration"] = corroboration
        item["tier_weight"] = tier_w
        item["recency"] = round(recency, 4)
        item["pre_score"] = round(
            _W_CORROBORATION * corro_norm
            + _W_NOVELTY * novelty
            + _W_TIER * tier_w
            + _W_RECENCY * recency,
            4,
        )
    return items
