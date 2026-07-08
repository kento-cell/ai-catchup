"""LangGraph pipeline: the whole catchup run as an explicit state graph.

    collect ─→ dedup ─→ embed_store ─→ signals ─→ judge ─→ rank ─→ digest ─→ deliver

Each node is a pure-ish function over CatchupState, so any stage can be
unit-tested alone, and the graph diagram *is* the architecture doc.
Source fetching inside `collect` fans out over a thread pool (network
bound); embedding is batched; only the top pre-scored candidates reach
the LLM judge so runtime stays in minutes on a laptop CPU/GPU.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from . import digest as digest_mod
from . import notify, signals
from .config import Config
from .dedup import Dedup
from .knowledge import Knowledge
from .judge import judge_items
from .ollama_client import Ollama
from .sources import SOURCES, fetch_all_parallel
from .summarize import summarize_items

logger = logging.getLogger(__name__)

# Cap the number of freshly-embedded items per run (arXiv alone can dump
# ~800 on a Monday). Highest-tier first so the cap never drops lab news.
_MAX_EMBED_PER_RUN = 400


class CatchupState(TypedDict, total=False):
    items: list[dict[str, Any]]          # raw fetched
    fresh: list[dict[str, Any]]          # after dedup
    batch_vecs: Any                      # np.ndarray aligned with fresh
    known_count: int
    ranked: list[dict[str, Any]]         # final top-N (story-deduped)
    absorbed: list[tuple[str, str]]      # (source, item_id) of story-duplicates
    digest_text: str
    dry_run: bool


def build_graph(cfg: Config, llm: Ollama, dedup: Dedup, knowledge: Knowledge):
    def collect(state: CatchupState) -> CatchupState:
        items = fetch_all_parallel()
        logger.info("collected %d items from %d sources", len(items), len(SOURCES))
        return {"items": items}

    def dedup_node(state: CatchupState) -> CatchupState:
        fresh = [
            it for it in state["items"]
            if not dedup.seen(it["source"], str(it["item_id"]))
        ]
        # tier ↑, then recency ↓ — so the embed cap drops the right tail.
        fresh.sort(
            key=lambda it: (
                it.get("tier") or 3,
                -(it.get("published_at") or datetime(1970, 1, 1, tzinfo=timezone.utc)).timestamp(),
            )
        )
        if len(fresh) > _MAX_EMBED_PER_RUN:
            logger.info("capping %d fresh items to %d", len(fresh), _MAX_EMBED_PER_RUN)
            fresh = fresh[:_MAX_EMBED_PER_RUN]
        logger.info("%d fresh after dedup", len(fresh))
        return {"fresh": fresh}

    def embed_store(state: CatchupState) -> CatchupState:
        fresh = state["fresh"]
        if not fresh:
            return {"batch_vecs": np.empty((0, 0), dtype=np.float32), "known_count": knowledge.count()}
        known_count = knowledge.count()
        texts = [f"{it['title']}\n{(it.get('raw_summary') or '')[:600]}" for it in fresh]
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 32):  # modest batches keep Ollama happy
            vecs.extend(llm.embed(texts[i : i + 32]))
        mat = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        for it, vec in zip(fresh, mat):
            it["_vec"] = vec  # kept on the item so rank can dedupe stories
            knowledge.add(it, vec.tolist())
        knowledge.commit()
        logger.info("embedded + stored %d items (knowledge: %d → %d)",
                    len(fresh), known_count, knowledge.count())
        return {"batch_vecs": mat, "known_count": known_count}

    def signals_node(state: CatchupState) -> CatchupState:
        fresh = state["fresh"]
        mat = state["batch_vecs"]
        if not fresh:
            return {}
        # "already known" = everything ingested before this run started;
        # approximate with rows other than this batch via ingested_at cut.
        known_vecs, _ = knowledge.matrix(before=_run_started)
        signals.compute(fresh, mat, known_vecs)
        return {"fresh": fresh}

    def judge_node(state: CatchupState) -> CatchupState:
        fresh = sorted(state["fresh"], key=lambda it: -it.get("pre_score", 0))
        candidates = fresh[: cfg.judge_candidates]
        logger.info("judging top %d of %d candidates with %s",
                    len(candidates), len(fresh), cfg.llm_model)
        judge_items(candidates, llm, lang=cfg.lang)
        return {"fresh": fresh}

    def rank_node(state: CatchupState) -> CatchupState:
        judged = [it for it in state["fresh"] if "final_score" in it]
        judged.sort(key=lambda it: -it["final_score"])
        # Story-level dedupe: the same story arriving via several sources is
        # a *signal* (corroboration), not something to deliver twice. Keep
        # the best-scoring representative; absorb the rest.
        ranked: list[dict[str, Any]] = []
        absorbed: list[tuple[str, str]] = []
        for it in judged:
            dup_of = None
            for kept in ranked:
                if it["url"] == kept["url"] or (
                    "_vec" in it and "_vec" in kept
                    and float(np.dot(it["_vec"], kept["_vec"])) >= signals.SAME_STORY
                ):
                    dup_of = kept
                    break
            if dup_of is not None:
                dup_of.setdefault("also_in", []).append(it["source"])
                absorbed.append((it["source"], str(it["item_id"])))
                continue
            ranked.append(it)
            if len(ranked) >= cfg.top_n:
                break
        return {"ranked": ranked, "absorbed": absorbed}

    def summarize_node(state: CatchupState) -> CatchupState:
        ranked = state.get("ranked", [])
        logger.info("writing dense summaries for %d delivered items", len(ranked))
        summarize_items(ranked, llm, lang=cfg.lang)
        return {"ranked": ranked}

    def digest_node(state: CatchupState) -> CatchupState:
        text = digest_mod.build(
            state.get("ranked", []), lang=cfg.lang, knowledge_size=knowledge.count()
        )
        return {"digest_text": text}

    def deliver(state: CatchupState) -> CatchupState:
        ranked = state.get("ranked", [])
        text = state["digest_text"]
        if cfg.slack_webhook_url and not state.get("dry_run"):
            notify.to_slack(text, cfg.slack_webhook_url)
        else:
            notify.to_stdout(text)
        if not state.get("dry_run"):
            # Mark delivered items AND their absorbed story-duplicates, so a
            # delivered story cannot resurface tomorrow via another source.
            # Undelivered fresh items stay unmarked on purpose — they may
            # return with more corroboration.
            keys = [(it["source"], str(it["item_id"])) for it in ranked]
            keys.extend(state.get("absorbed", []))
            dedup.mark_many(keys)
            # 2026-07-08: voice digest is fully opt-in — running the LLM
            # script + edge-tts synthesis on every delivery regardless of
            # whether the user plans to listen is wasted local compute
            # (user feedback: "毎回自動で回るのはもったいない"). Only
            # stash what was delivered; `python -m aicatchup.tts --last`
            # voices it later, on request.
            try:
                from . import tts as _tts
                _tts.stash_last_delivered(ranked, cfg)
            except Exception as _exc:  # noqa: BLE001
                logger.warning("tts stash skipped (%s)", _exc)
        return {}

    _run_started = datetime.now(timezone.utc)

    g = StateGraph(CatchupState)
    g.add_node("collect", collect)
    g.add_node("dedup", dedup_node)
    g.add_node("embed_store", embed_store)
    g.add_node("signals", signals_node)
    g.add_node("judge", judge_node)
    g.add_node("rank", rank_node)
    g.add_node("summarize", summarize_node)
    g.add_node("digest", digest_node)
    g.add_node("deliver", deliver)

    g.set_entry_point("collect")
    g.add_edge("collect", "dedup")
    g.add_edge("dedup", "embed_store")
    g.add_edge("embed_store", "signals")
    g.add_edge("signals", "judge")
    g.add_edge("judge", "rank")
    g.add_edge("rank", "summarize")
    g.add_edge("summarize", "digest")
    g.add_edge("digest", "deliver")
    g.add_edge("deliver", END)
    return g.compile()


def run(cfg: Config, *, dry_run: bool = False) -> dict:
    llm = Ollama(cfg.ollama_url, cfg.llm_model, cfg.embed_model)
    if not llm.ping():
        raise SystemExit(
            f"Ollama is not reachable at {cfg.ollama_url} — start it with `ollama serve` "
            f"and pull the models: `ollama pull {cfg.llm_model}` / `ollama pull {cfg.embed_model}`"
        )
    dedup = Dedup(cfg.data_dir / "catchup.db")
    knowledge = Knowledge(cfg.data_dir / "knowledge.db")
    try:
        app = build_graph(cfg, llm, dedup, knowledge)
        return app.invoke({"dry_run": dry_run})
    finally:
        knowledge.close()
        dedup.close()
