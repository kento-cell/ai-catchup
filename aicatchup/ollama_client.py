"""Minimal Ollama HTTP client — generate + embed, nothing else.

Deliberately dependency-free (plain requests): the whole point of this
project is that inference never leaves the machine.
"""
from __future__ import annotations

import json
import logging
import re

import requests

logger = logging.getLogger(__name__)

_TIMEOUT_GENERATE = 180
_TIMEOUT_EMBED = 60


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, base_url: str, llm_model: str, embed_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model

    def generate(self, prompt: str, *, json_mode: bool = False, temperature: float = 0.3) -> str:
        payload: dict = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        r = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=_TIMEOUT_GENERATE
        )
        if r.status_code != 200:
            raise OllamaError(f"generate failed: HTTP {r.status_code}: {r.text[:200]}")
        return (r.json().get("response") or "").strip()

    def generate_json(self, prompt: str) -> dict:
        """Generate with JSON output mode and parse; salvage the first
        {...} block if the model wraps it in prose."""
        text = self.generate(prompt, json_mode=True, temperature=0.1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            raise OllamaError(f"non-JSON judge output: {text[:200]}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed via /api/embed (Ollama >= 0.3 accepts a list input)."""
        if not texts:
            return []
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embed_model, "input": texts},
            timeout=_TIMEOUT_EMBED,
        )
        if r.status_code != 200:
            raise OllamaError(f"embed failed: HTTP {r.status_code}: {r.text[:200]}")
        embeddings = r.json().get("embeddings") or []
        if len(embeddings) != len(texts):
            raise OllamaError(f"embed count mismatch: {len(embeddings)} != {len(texts)}")
        return embeddings

    def ping(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/api/version", timeout=5).status_code == 200
        except requests.RequestException:
            return False
