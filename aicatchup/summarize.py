"""Dense per-item digest for the delivered top-N.

A ranked list of translated headlines is not a catchup — the reader
should get the substance without opening the link. Only the delivered
items (top-N, ~10) pass through here, so the extra LLM cost stays small.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .ollama_client import Ollama, OllamaError

logger = logging.getLogger(__name__)

_PROMPT_JA = """\
以下はAI業界ニュース/論文/リポジトリのタイトルと原文要約です。
日本語で、密度の高い解説ダイジェストに再構成してください。
目安は4〜7行(合計300〜450文字)。薄い一般論で埋めず、原文にある具体を最大限拾うこと。

タイトル: {title}
ソース: {source}
原文要約: {summary}

必ず含める要素(順に、各1〜2行):
1. 何が起きたか — 主語と動作を具体的に。誰が何を発表/公開/達成したか
2. 具体的な数字・固有名詞 — モデル名・パラメータ数・ベンチマーク・価格・社名を原文ママで
3. 技術的ポイント — 何が新しいのか、どういう仕組みか
4. 実務への含意 — なぜ今知るべきか、どんな用途・影響があるか

ルール:
- 固有名詞・数値は必ず原文ママ。曖昧化や丸めは禁止
- 「すごい」「画期的」等の空虚な煽りは禁止。事実で書く
- 原文に無い事実を創作しない。情報が無い要素はスキップしてよい
- 出力は解説本文のみ。前置き・後置き・見出しは禁止
- 原文要約が空でも、タイトルから読み取れる範囲で必ず書く。
  「情報が不足しています」等のお断り・メタ応答は絶対に出力しない
"""

_PROMPT_EN = """\
Rewrite this AI-industry item as a dense explanatory digest in English,
4-7 lines (~80-120 words). No filler — extract every concrete fact.

Title: {title}
Source: {source}
Original summary: {summary}

Cover in order: what happened (who did what), concrete numbers & proper
nouns verbatim, the technical point (what is new / how it works), and the
practical implication. Never invent facts; skip elements with no info.
Output the digest text only — no preamble, no meta-responses like
"insufficient information".
"""

_META_RE = re.compile(
    r"(情報が不足|テキストをご提供|作成できません|申し訳|cannot|insufficient)", re.IGNORECASE
)


def summarize_items(items: list[dict[str, Any]], llm: Ollama, *, lang: str = "ja") -> None:
    """Attach ``digest_summary`` to each item in place. Fail-soft: on a bad
    generation the item simply ships without a summary block."""
    template = _PROMPT_JA if lang == "ja" else _PROMPT_EN
    for item in items:
        prompt = template.format(
            title=item["title"][:300],
            source=item["source"],
            summary=(item.get("raw_summary") or "(なし)")[:1200],
        )
        try:
            text = llm.generate(prompt, temperature=0.3).strip()
        except OllamaError as exc:
            logger.warning("summarize failed for %r: %s", item["title"][:60], exc)
            continue
        if not text or _META_RE.search(text):
            logger.warning("meta/empty summary dropped for %r", item["title"][:60])
            continue
        item["digest_summary"] = text[:800]
