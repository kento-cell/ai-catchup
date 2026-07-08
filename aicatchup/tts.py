"""Voice digest: read the delivered digest aloud, on request.

Pipeline: digest items -> local-LLM reading script (English names ->
katakana, hard kanji -> hiragana, numbers spelled out) -> edge-tts
neural mp3 -> desktop shortcut the user double-clicks whenever they
want to listen.

Design history (2026-07-08):

* v1: windowless background auto-play right after every catchup run.
  User reported it never actually reached their ears in practice.
* v2: dropped autoplay, tried Slack upload instead — blocked on bot
  channel invite, and Slack has no auto-play anyway (still needs a
  manual tap, same friction as a local file).
* v3: desktop shortcut, but still ran gemma/edge-tts AUTOMATICALLY at
  the end of every run regardless of whether the user planned to
  listen that day.
* v4 (this revision — user: "毎回自動で回るのはもったいない"): voice
  generation is fully opt-in. ``graph.py``'s deliver node only stashes
  the delivered item list via :func:`stash_last_delivered`; the LLM
  script + edge-tts synthesis run ONLY when the user explicitly
  executes ``python -m aicatchup.tts --last``.

Env toggles:
  CATCHUP_TTS_RATE     e.g. "+25%" (default) / "+40%" / "-10%"
  CATCHUP_TTS_VOICE    default "ja-JP-NanamiNeural"

Requires the optional dependency: ``pip install aicatchup[tts]``
(edge-tts is free and needs no API key).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path

from .config import Config, load_dotenv
from .ollama_client import Ollama

logger = logging.getLogger(__name__)

_KEEP_FILES = 3
_SCRIPT_BATCH = 6
_SHORTCUT_NAME = "AIキャッチアップを聞く.lnk"
_LAST_DELIVERED_FILENAME = "last_delivered.json"

_SCRIPT_PROMPT = """以下の AI ニュース要約を、日本語の音声読み上げ用台本に変換してください。

ルール:
- 英語の固有名詞・略語・製品名はカタカナに変換する
  (例: Anthropic→アンソロピック、Claude→クロード、LLM→エルエルエム、GitHub→ギットハブ)
- 読み間違えやすい漢字はひらがなに開く
- 数値は読み仮名にする (例: 3.4倍→さんてんよんばい、27%→にじゅうななパーセント)
- URL・記号・箇条書き記号は除去し、自然な話し言葉の文章にする
- 各ニュースの間に「次のニュースです。」を挟む
- 台本の本文のみを出力する。前置き・後置き・見出しは不要

ニュース要約:
{items}"""


def _audio_dir(cfg: Config) -> Path:
    return cfg.data_dir / "audio"


# ----------------------------------------------------------------------
# Stash: what the most recent run actually delivered
# ----------------------------------------------------------------------
def stash_last_delivered(items: list[dict], cfg: Config | None = None) -> None:
    """Save the most recently delivered digest so an opt-in voice pass
    doesn't need to re-fetch/re-summarise anything. Overwrites — only
    the latest run is kept. Non-fatal."""
    cfg = cfg or Config()
    out_dir = _audio_dir(cfg)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / _LAST_DELIVERED_FILENAME).write_text(
            json.dumps(items, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("tts: stashed %d delivered items for opt-in voice", len(items))
    except OSError as exc:
        logger.warning("tts: stash failed (non-fatal): %s", exc)


def load_last_delivered(cfg: Config | None = None) -> list[dict] | None:
    cfg = cfg or Config()
    path = _audio_dir(cfg) / _LAST_DELIVERED_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------
# 1. Reading script via local LLM
# ----------------------------------------------------------------------
def build_script(items: list[dict], llm: Ollama) -> str:
    parts: list[str] = []
    for i in range(0, len(items), _SCRIPT_BATCH):
        batch = items[i : i + _SCRIPT_BATCH]
        lines = []
        for it in batch:
            summary = (it.get("digest_summary") or it.get("title") or "").strip()
            if summary:
                lines.append(f"- {it.get('title', '')}: {summary}")
        if not lines:
            continue
        try:
            out = llm.generate(
                _SCRIPT_PROMPT.format(items="\n".join(lines)),
                temperature=0.2,
            ).strip()
            if out:
                parts.append(out)
        except Exception as exc:  # noqa: BLE001 - a lost batch is acceptable
            logger.warning("tts: script batch %d failed: %s", i // _SCRIPT_BATCH, exc)
    header = "エーアイ、キャッチアップです。本日のニュースをお伝えします。"
    footer = "以上、本日のキャッチアップでした。"
    return "\n".join([header, *parts, footer])


# ----------------------------------------------------------------------
# 2. Synthesis via edge-tts
# ----------------------------------------------------------------------
def synthesize(script: str, cfg: Config) -> Path | None:
    try:
        import edge_tts
    except ImportError:
        logger.warning("tts: edge-tts not installed — pip install aicatchup[tts]")
        return None

    out_dir = _audio_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"catchup_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
    voice = os.getenv("CATCHUP_TTS_VOICE", "ja-JP-NanamiNeural").strip()
    rate = os.getenv("CATCHUP_TTS_RATE", "+25%").strip()

    async def _run() -> None:
        tts = edge_tts.Communicate(script, voice=voice, rate=rate)
        await tts.save(str(dest))

    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        logger.warning("tts: synthesis failed: %s", exc)
        return None
    if not dest.exists() or dest.stat().st_size < 10_000:
        logger.warning("tts: output missing/too small")
        return None
    logger.info("tts: %d bytes -> %s (voice=%s rate=%s)",
                dest.stat().st_size, dest.name, voice, rate)
    return dest


# ----------------------------------------------------------------------
# 3. Desktop shortcut — the ONE thing the user interacts with.
# ----------------------------------------------------------------------
def _desktop_dir() -> Path:
    candidates = [
        Path(os.environ.get("USERPROFILE", "")) / "Desktop",
        Path(os.environ.get("USERPROFILE", "")) / "OneDrive" / "デスクトップ",
        Path(os.environ.get("USERPROFILE", "")) / "OneDrive" / "Desktop",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def refresh_desktop_shortcut(mp3: Path) -> Path | None:
    """(Re)point the single desktop shortcut at *mp3*.

    Windows-only (WScript.Shell COM via PowerShell — no extra pip
    dependency). Overwrites the same .lnk every time.
    """
    if os.name != "nt":
        logger.info("tts: desktop shortcut skipped (non-Windows)")
        return None
    desktop = _desktop_dir()
    if not desktop.is_dir():
        logger.warning("tts: desktop dir not found — skipping shortcut")
        return None
    link_path = desktop / _SHORTCUT_NAME
    ps = (
        "$W = New-Object -ComObject WScript.Shell; "
        f"$S = $W.CreateShortcut('{link_path}'); "
        f"$S.TargetPath = '{mp3}'; "
        f"$S.Description = 'AIキャッチアップ 音声版 (聴き終わったら削除してOK)'; "
        "$S.Save()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            check=True, capture_output=True, timeout=15,
        )
        logger.info("tts: desktop shortcut refreshed -> %s", link_path)
        return link_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("tts: shortcut creation failed: %s", exc)
        return None


# ----------------------------------------------------------------------
# 4. Disk hygiene — prune OLD generations only, never the current one.
# ----------------------------------------------------------------------
def cleanup_old(cfg: Config, keep: int = _KEEP_FILES) -> int:
    out_dir = _audio_dir(cfg)
    if not out_dir.exists():
        return 0
    files = sorted(
        out_dir.glob("catchup_*.mp3"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for old in files[keep:]:
        try:
            old.unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        logger.info("tts: cleaned up %d old mp3(s)", removed)
    return removed


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def run_tts(items: list[dict], cfg: Config | None = None) -> Path | None:
    """Full TTS pass. Never raises. Returns mp3 path on success."""
    try:
        cfg = cfg or Config()
        llm = Ollama(cfg.ollama_url, cfg.llm_model, cfg.embed_model)
        script = build_script(items, llm)
        if len(script) < 100:
            logger.warning("tts: script too short — skipping")
            return None
        mp3 = synthesize(script, cfg)
        if mp3 is None:
            return None
        refresh_desktop_shortcut(mp3)
        cleanup_old(cfg)
        return mp3
    except Exception as exc:  # noqa: BLE001
        logger.warning("tts: pipeline failed (non-fatal): %s", exc)
        return None


def spawn_detached_for_last() -> bool:
    """Launch an opt-in voice pass over the most recently delivered
    catchup, in a detached child (returns in ~50ms instead of blocking
    the terminal for the ~1-2 min gemma/edge-tts takes).

    Does nothing (and spends nothing) unless the user explicitly calls
    this — it is never called from the delivery path automatically.
    """
    import sys

    if load_last_delivered() is None:
        logger.warning("tts: no stashed catchup run to voice — run catchup first")
        return False
    try:
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        subprocess.Popen(
            [sys.executable, "-m", "aicatchup.tts", "--last"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        logger.info("tts: detached worker spawned for last-delivered digest")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("tts: detach failed (%s) — falling back to sync", exc)
        items = load_last_delivered()
        if items:
            run_tts(items)
        return False


def _worker_main() -> int:
    """CLI entry: ``python -m aicatchup.tts --last`` — voice the most
    recently delivered catchup run. This is the only supported mode;
    LLM inference + synthesis run only when a human asks."""
    import sys

    if "--last" not in sys.argv:
        print("Usage: python -m aicatchup.tts --last", file=sys.stderr)
        return 1

    load_dotenv()
    cfg = Config()
    # spawn_detached_for_last() redirects this process's stdout/stderr
    # to DEVNULL, so basicConfig(stream=stdout) would be silent — log
    # to a file instead so the run is inspectable after the fact.
    log_dir = _audio_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=str(log_dir / "tts_worker.log"),
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s",
    )
    items = load_last_delivered(cfg)
    if not items:
        logger.warning("tts: no stashed catchup run found — run catchup first")
        return 1
    logger.info("tts worker started (%d items, opt-in --last)", len(items))
    mp3 = run_tts(items, cfg)
    logger.info("tts worker finished (mp3=%s)", mp3)
    return 0 if mp3 else 1


if __name__ == "__main__":
    raise SystemExit(_worker_main())
