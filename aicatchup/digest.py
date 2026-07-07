"""Digest formatting — Slack mrkdwn-friendly plain text with a visible
"why this ranked" line per item, so the ranking is auditable at a glance."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

_CATEGORY_EMOJI = {
    "model_release": "🚀",
    "research": "📄",
    "product": "🛠",
    "funding": "💰",
    "policy": "🏛",
    "tooling": "🔧",
    "other": "📌",
}


def _why_ja(item: dict[str, Any]) -> str:
    parts: list[str] = []
    if item.get("corroboration", 1) > 1:
        parts.append(f"{item['corroboration']}ソースが同時報道")
    if item.get("novelty", 0) >= 0.55:
        parts.append("既知情報との重複なし")
    parts.append(f"LLM評価 {item.get('usefulness', '?')}/10")
    reason = item.get("judge_reason") or ""
    if reason:
        parts.append(reason)
    return " / ".join(parts)


def _why_en(item: dict[str, Any]) -> str:
    parts: list[str] = []
    if item.get("corroboration", 1) > 1:
        parts.append(f"{item['corroboration']} sources reporting")
    if item.get("novelty", 0) >= 0.55:
        parts.append("no overlap with known items")
    parts.append(f"LLM {item.get('usefulness', '?')}/10")
    if item.get("judge_reason"):
        parts.append(item["judge_reason"])
    return " / ".join(parts)


def build(ranked: list[dict[str, Any]], *, lang: str = "ja", knowledge_size: int = 0) -> str:
    now = datetime.now(timezone.utc).astimezone()
    stamp = now.strftime("%Y-%m-%d %H:%M")
    if lang == "ja":
        head = f"*🎯 AI Catchup — 今、最速で有益な{len(ranked)}件* ({stamp})"
        tail = f"_ナレッジ蓄積: {knowledge_size}件 / 全ソース無料・推論は完全ローカル_"
        why = _why_ja
    else:
        head = f"*🎯 AI Catchup — the {len(ranked)} items worth your time right now* ({stamp})"
        tail = f"_knowledge base: {knowledge_size} items / all sources free, inference fully local_"
        why = _why_en

    lines = [head, ""]
    for i, item in enumerate(ranked, 1):
        emoji = _CATEGORY_EMOJI.get(item.get("category", "other"), "📌")
        sources = item["source"]
        also = item.get("also_in") or []
        if also:
            sources += ", " + ", ".join(dict.fromkeys(also))  # dedupe, keep order
        lines.append(f"*{i}. {emoji} {item['title']}*")
        lines.append(f"   {item['url']}")
        lines.append(f"   `{sources}` — {why(item)}")
        lines.append("")
    lines.append(tail)
    return "\n".join(lines)
