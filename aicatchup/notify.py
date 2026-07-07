"""Delivery sinks. Slack incoming-webhook when configured, stdout always
available — the pipeline is deliverable-agnostic by design so new sinks
(Discord, email, file) are one function each."""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_SLACK_CHUNK_CHARS = 3500
_TIMEOUT = 15


def to_stdout(text: str) -> None:
    print(text)


def to_slack(text: str, webhook_url: str) -> None:
    """Post in chunks split at item boundaries (blank lines) — Slack
    renders very long single messages poorly."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for block in text.split("\n\n"):
        if size + len(block) > _SLACK_CHUNK_CHARS and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(block)
        size += len(block) + 2
    if current:
        chunks.append("\n\n".join(current))

    for chunk in chunks:
        r = requests.post(webhook_url, json={"text": chunk}, timeout=_TIMEOUT)
        r.raise_for_status()
    logger.info("posted %d chunk(s) to Slack", len(chunks))
