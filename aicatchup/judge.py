"""LLM judging: a small local model scores each pre-ranked candidate
against a tight rubric and must justify itself. Structured output only —
the deterministic signals (signals.py) are passed in as context, and the
final score blends both, so a hallucinating judge cannot hijack the
ranking."""
from __future__ import annotations

import logging
from typing import Any

from .ollama_client import Ollama, OllamaError

logger = logging.getLogger(__name__)

_PROMPT_JA = """\
あなたはAI業界ニュースの目利きです。次の1件を評価してください。

タイトル: {title}
ソース: {source} (tier {tier})
要約: {summary}

機械シグナル (参考):
- 新規性: {novelty} (1に近いほど過去90日のナレッジに無い話題)
- 同時報道ソース数: {corroboration}
- 鮮度: {recency}

次のJSONだけを出力:
{{
  "usefulness": <0-10 の整数。実務者が今日知るべき度合い>,
  "category": "<model_release|research|product|funding|policy|tooling|other>",
  "reason": "<なぜその点数か、日本語で40字以内。具体的に>"
}}

採点基準:
- 9-10: 主要ラボの新モデル/重大発表、業界構造が変わる話
- 6-8: 有力な新技術・注目論文・大型資金調達
- 3-5: 興味深いが影響が限定的
- 0-2: 宣伝・雑談・ニュース価値なし
"""

_PROMPT_EN = """\
You are an expert curator of AI-industry news. Evaluate this single item.

Title: {title}
Source: {source} (tier {tier})
Summary: {summary}

Machine signals (context):
- novelty: {novelty} (near 1 = topic absent from the last 90 days of knowledge)
- corroborating sources: {corroboration}
- recency: {recency}

Output ONLY this JSON:
{{
  "usefulness": <integer 0-10: how much a practitioner needs this today>,
  "category": "<model_release|research|product|funding|policy|tooling|other>",
  "reason": "<why, max 15 words, concrete>"
}}

Rubric:
- 9-10: major-lab model release / industry-shifting news
- 6-8: strong new technique, notable paper, large funding round
- 3-5: interesting but limited impact
- 0-2: promo, chatter, no news value
"""

# LLM opinion vs. deterministic pre-score in the final blend.
_W_LLM = 0.55
_W_SIGNALS = 0.45


def judge_items(
    items: list[dict[str, Any]], llm: Ollama, *, lang: str = "ja"
) -> list[dict[str, Any]]:
    """Score candidates with the local LLM; annotate in place.

    Fail-soft per item: an unparseable/failed judgement falls back to a
    neutral 5/10 so one bad generation never kills the run.
    """
    template = _PROMPT_JA if lang == "ja" else _PROMPT_EN
    for item in items:
        prompt = template.format(
            title=item["title"][:300],
            source=item["source"],
            tier=item.get("tier", 3),
            summary=(item.get("raw_summary") or "(no summary)")[:800],
            novelty=item.get("novelty", "?"),
            corroboration=item.get("corroboration", 1),
            recency=item.get("recency", "?"),
        )
        try:
            verdict = llm.generate_json(prompt)
            usefulness = max(0, min(10, int(verdict.get("usefulness", 5))))
            item["usefulness"] = usefulness
            item["category"] = str(verdict.get("category") or "other")[:32]
            item["judge_reason"] = str(verdict.get("reason") or "").strip()[:120]
        except (OllamaError, ValueError, TypeError) as exc:
            logger.warning("judge failed for %r: %s", item["title"][:60], exc)
            item["usefulness"] = 5
            item["category"] = "other"
            item["judge_reason"] = ""
        item["final_score"] = round(
            _W_LLM * (item["usefulness"] / 10.0) + _W_SIGNALS * item["pre_score"], 4
        )
    return items
