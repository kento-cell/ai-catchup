"""Env-driven configuration. Everything has a sane default so
`aicatchup run` works out of the box with a stock Ollama install."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    ollama_url: str = field(
        default_factory=lambda: (os.getenv("OLLAMA_API_URL") or "http://localhost:11434").rstrip("/")
    )
    llm_model: str = field(default_factory=lambda: os.getenv("CATCHUP_LLM_MODEL") or "gemma4:e4b")
    embed_model: str = field(
        default_factory=lambda: os.getenv("CATCHUP_EMBED_MODEL") or "embeddinggemma"
    )
    slack_webhook_url: str = field(default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL") or "")
    top_n: int = field(default_factory=lambda: _env_int("CATCHUP_TOP_N", 10))
    judge_candidates: int = field(default_factory=lambda: _env_int("CATCHUP_JUDGE_CANDIDATES", 30))
    lang: str = field(default_factory=lambda: os.getenv("CATCHUP_LANG") or "ja")
    data_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data")


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env loader — no python-dotenv dependency.

    Existing environment variables always win over file values.
    """
    p = path or _REPO_ROOT / ".env"
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
