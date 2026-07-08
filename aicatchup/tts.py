"""Voice digest: read the delivered digest aloud on this machine.

Pipeline: digest items -> local-LLM reading script (English names ->
katakana, hard kanji -> hiragana, numbers spelled out) -> edge-tts
neural mp3 -> windowless background playback -> old-file cleanup.

Runs as a DETACHED child process (``python -m aicatchup.tts items.json``)
so the main catchup run finishes at normal speed; audio generation
overlaps with whatever the user does next.

Env toggles:
  CATCHUP_TTS=0            disable entirely (default on)
  CATCHUP_TTS_RATE         e.g. "+25%" (default) / "+40%" / "-10%"
  CATCHUP_TTS_VOICE        default "ja-JP-NanamiNeural"
  CATCHUP_TTS_AUTOPLAY=0   generate mp3 but skip local playback

Requires the optional dependency: ``pip install aicatchup[tts]``
(edge-tts is free and needs no API key).
"""
from __future__ import annotations

import asyncio
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


def is_enabled() -> bool:
    return os.getenv("CATCHUP_TTS", "1").strip().lower() not in {
        "", "0", "false", "no", "off",
    }


def _is_autoplay() -> bool:
    return os.getenv("CATCHUP_TTS_AUTOPLAY", "1").strip().lower() not in {
        "", "0", "false", "no", "off",
    }


def _audio_dir(cfg: Config) -> Path:
    return cfg.data_dir / "audio"


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


_PS_PLAY = r"""
Add-Type -AssemblyName PresentationCore
$p = New-Object System.Windows.Media.MediaPlayer
$p.Open([uri]('file:///' + ($args[0] -replace '\\','/')))
$p.Play()
$deadline = (Get-Date).AddSeconds(20)
while (-not $p.NaturalDuration.HasTimeSpan) {
    if ((Get-Date) -gt $deadline) { exit 1 }
    Start-Sleep -Milliseconds 200
}
$dur = $p.NaturalDuration.TimeSpan.TotalSeconds
Start-Sleep -Seconds ([math]::Ceiling($dur) + 1)
$p.Close()
"""


def play_background(path: Path) -> bool:
    """Windowless playback; helper process exits when the clip ends.

    Windows-only fast path (PowerShell MediaPlayer). On other platforms
    fall back to afplay (macOS) / mpg123 (Linux) when available.
    """
    try:
        if os.name == "nt":
            subprocess.Popen(
                [
                    "powershell", "-NoProfile", "-WindowStyle", "Hidden",
                    "-Command", _PS_PLAY, str(path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            player = "afplay" if os.uname().sysname == "Darwin" else "mpg123"
            subprocess.Popen(
                [player, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        logger.info("tts: background playback started (%s)", path.name)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("tts: playback spawn failed: %s", exc)
        return False


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
        if _is_autoplay():
            play_background(mp3)
        cleanup_old(cfg)
        return mp3
    except Exception as exc:  # noqa: BLE001
        logger.warning("tts: pipeline failed (non-fatal): %s", exc)
        return None


def spawn_detached(items: list[dict]) -> bool:
    """Launch run_tts in a detached child so the main run's wall-clock
    time is unaffected. Falls back to sync on spawn failure."""
    import json
    import sys
    import tempfile

    try:
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="aicatchup_tts_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(items, fh, ensure_ascii=False, default=str)
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        subprocess.Popen(
            [sys.executable, "-m", "aicatchup.tts", tmp],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        logger.info("tts: detached worker spawned (%d items)", len(items))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("tts: detach failed (%s) — falling back to sync", exc)
        run_tts(items)
        return False


def _worker_main() -> int:
    import json
    import sys

    if len(sys.argv) < 2:
        return 1
    tmp = Path(sys.argv[1])
    try:
        items = json.loads(tmp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    run_tts(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
