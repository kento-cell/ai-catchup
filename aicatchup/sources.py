"""Source fetchers: pull recent items from RSS, HN Algolia, Reddit, arXiv,
Bluesky, HF Daily Papers, Techmeme, and GitHub new-repo search.

Each fetcher returns a list of normalised dicts:
    {
      "source": "OpenAI Blog",
      "tier": 1,
      "item_id": "<unique id within source>",
      "title": "...",
      "url": "https://...",
      "published_at": datetime | None,
      "raw_summary": "<abstract or first paragraph, may be empty>",
    }

No paid APIs. All endpoints are free public feeds.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET  # kept for ET.Element type annotations
import defusedxml.ElementTree as _SafeET  # hardened parser (XXE / billion-laughs)
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

# (name, type, url, tier)
SOURCES: list[tuple[str, str, str, int]] = [
    # Official RSS — Anthropic & Meta AI dropped RSS; we compensate by
    # keyword-searching HN for "Anthropic" / "Meta AI" below.
    ("OpenAI", "rss", "https://openai.com/news/rss.xml", 1),
    ("DeepMind", "rss", "https://deepmind.google/blog/rss.xml", 1),
    ("NVIDIA Developer", "rss", "https://developer.nvidia.com/blog/feed/", 1),
    ("HuggingFace", "rss", "https://huggingface.co/blog/feed.xml", 2),
    # HN keyword searches — catches official-lab news even when no RSS exists.
    ("HN · AI", "hn", "https://hn.algolia.com/api/v1/search_by_date?tags=story&query=AI&hitsPerPage=25", 2),
    ("HN · Anthropic", "hn", "https://hn.algolia.com/api/v1/search_by_date?tags=story&query=Anthropic&hitsPerPage=10", 1),
    ("HN · OpenAI", "hn", "https://hn.algolia.com/api/v1/search_by_date?tags=story&query=OpenAI&hitsPerPage=10", 1),
    ("HN · Meta AI", "hn", "https://hn.algolia.com/api/v1/search_by_date?tags=story&query=%22Meta+AI%22&hitsPerPage=5", 1),
    # Community feeds.
    ("Reddit r/LocalLLaMA", "reddit", "https://www.reddit.com/r/LocalLLaMA/.rss", 3),
    ("Reddit r/MachineLearning", "reddit", "https://www.reddit.com/r/MachineLearning/.rss", 3),
    ("arXiv cs.AI", "arxiv", "https://export.arxiv.org/rss/cs.AI", 3),
    # Curated industry news — editor-picked, fastest general tech aggregator.
    ("Techmeme", "rss", "https://www.techmeme.com/feed.xml", 2),
    # Industry press with an AI-dedicated feed — funding / product launches
    # land here fast, free RSS. (VentureBeat's AI feed was evaluated too but
    # it stopped updating in 2026-05, so it is intentionally absent.)
    ("TechCrunch AI", "rss", "https://techcrunch.com/category/artificial-intelligence/feed/", 2),
    # Japanese engineer-oriented news — the digest is JP, yet every other
    # source is EN; Publickey covers cloud/OSS/AI for the JP audience.
    ("Publickey", "rss", "https://www.publickey1.jp/atom.xml", 2),
    # Community-curated top papers of the day — far better S/N than raw arXiv.
    ("HF Daily Papers", "hf_papers", "https://huggingface.co/api/daily_papers?limit=15", 2),
    # Bluesky public search — AI researchers post here first-hand; free, no auth.
    ("Bluesky · Anthropic", "bluesky", "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts?q=Anthropic&sort=top&limit=15", 3),
    ("Bluesky · OpenAI", "bluesky", "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts?q=OpenAI&sort=top&limit=15", 3),
    ("Bluesky · LLM", "bluesky", "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts?q=LLM&sort=top&limit=15", 3),
    # New AI repos gaining stars fast — official GitHub Search API, no auth.
    # {since} is replaced at fetch time with (now - 7 days).
    ("GitHub New AI Repos", "github", "https://api.github.com/search/repositories?q=llm+OR+ai+created:%3E{since}+stars:%3E50&sort=stars&order=desc&per_page=15", 3),
]

# Reject items older than this — widen to 72h so weekend news survives,
# and labs that publish 2-3 times / week still contribute.
_MAX_AGE = timedelta(hours=72)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; ai-catchup/0.1; +personal-use)"
)
_TIMEOUT = 15

# Strip namespace prefixes like "{http://www.w3.org/2005/Atom}entry" → "entry".
_NS_RE = re.compile(r"\{[^}]+\}")


def _strip_ns(tag: str) -> str:
    return _NS_RE.sub("", tag)


def _findtext(el: ET.Element, names: tuple[str, ...]) -> str:
    """Return text of the first matching child (case-insensitive, ns-stripped)."""
    wanted = {n.lower() for n in names}
    for child in el.iter():
        if _strip_ns(child.tag).lower() in wanted and child.text:
            return child.text.strip()
    return ""


def _findattr(el: ET.Element, names: tuple[str, ...], attr: str) -> str:
    wanted = {n.lower() for n in names}
    for child in el.iter():
        if _strip_ns(child.tag).lower() in wanted:
            v = child.get(attr)
            if v:
                return v.strip()
    return ""


def _parse_date(s: str) -> datetime | None:
    if not s:
        return None
    # ISO 8601 first (Bluesky / HF / GitHub all emit it, often with ms + "Z").
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except ValueError:
        pass
    try:
        d = parsedate_to_datetime(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            d = datetime.strptime(s, fmt)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d
        except ValueError:
            continue
    return None


def _is_fresh(d: datetime | None) -> bool:
    if d is None:
        return True  # keep undated items, dedup will sort it out
    now = datetime.now(timezone.utc)
    return (now - d) < _MAX_AGE


def _fetch(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.content


def _parse_rss(name: str, tier: int, url: str) -> list[dict[str, Any]]:
    body = _fetch(url)
    root = _SafeET.fromstring(body)
    items: list[dict[str, Any]] = []
    # Both RSS (channel/item) and Atom (entry) — iterate all entry-like nodes.
    for el in root.iter():
        tag = _strip_ns(el.tag).lower()
        if tag not in ("item", "entry"):
            continue
        title = _findtext(el, ("title",))
        link = _findtext(el, ("link",)) or _findattr(el, ("link",), "href")
        guid = (
            _findtext(el, ("guid", "id"))
            or link
            or title
        )
        published = _parse_date(
            _findtext(el, ("pubdate", "published", "updated", "dc:date"))
        )
        summary = (_findtext(el, ("description", "summary", "content")) or "")[:1500]
        if not title or not link:
            continue
        if not _is_fresh(published):
            continue
        items.append(
            {
                "source": name,
                "tier": tier,
                "item_id": guid,
                "title": title,
                "url": link,
                "published_at": published,
                "raw_summary": _strip_html(summary),
            }
        )
    return items


def _parse_hn(name: str, tier: int, url: str) -> list[dict[str, Any]]:
    r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    items: list[dict[str, Any]] = []
    for hit in data.get("hits", []):
        title = hit.get("title") or ""
        link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        published = _parse_date(hit.get("created_at", ""))
        if not title:
            continue
        if not _is_fresh(published):
            continue
        items.append(
            {
                "source": name,
                "tier": tier,
                "item_id": str(hit.get("objectID")),
                "title": title,
                "url": link,
                "published_at": published,
                "raw_summary": (hit.get("story_text") or "")[:1500],
                "score": hit.get("points"),
            }
        )
    return items


def _strip_html(s: str) -> str:
    """Cheap HTML strip — not security-grade, just for prompt sanity."""
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_hf_papers(name: str, tier: int, url: str) -> list[dict[str, Any]]:
    """HF Daily Papers — community-upvoted papers, JSON API, no auth."""
    r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT)
    r.raise_for_status()
    items: list[dict[str, Any]] = []
    for entry in r.json():
        paper = entry.get("paper") or {}
        pid = paper.get("id") or ""
        title = paper.get("title") or ""
        published = _parse_date(entry.get("publishedAt") or "")
        if not pid or not title:
            continue
        if not _is_fresh(published):
            continue
        items.append(
            {
                "source": name,
                "tier": tier,
                "item_id": pid,
                "title": title.strip(),
                "url": f"https://huggingface.co/papers/{pid}",
                "published_at": published,
                "raw_summary": (paper.get("summary") or "")[:1500],
                "score": paper.get("upvotes"),
            }
        )
    return items


def _parse_bluesky(name: str, tier: int, url: str) -> list[dict[str, Any]]:
    """Bluesky public search (app.bsky.feed.searchPosts) — no auth needed."""
    r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT)
    r.raise_for_status()
    items: list[dict[str, Any]] = []
    for post in r.json().get("posts", []):
        record = post.get("record") or {}
        text = (record.get("text") or "").strip()
        uri = post.get("uri") or ""
        handle = (post.get("author") or {}).get("handle") or ""
        published = _parse_date(record.get("createdAt") or "")
        if not text or not uri or not handle:
            continue
        if not _is_fresh(published):
            continue
        # at://did:plc:xxx/app.bsky.feed.post/<rkey> → public web URL.
        rkey = uri.rsplit("/", 1)[-1]
        # Single-line title for the digest; full text goes to raw_summary.
        title = re.sub(r"\s+", " ", text)[:120]
        items.append(
            {
                "source": name,
                "tier": tier,
                "item_id": uri,
                "title": f"{title} (@{handle})",
                "url": f"https://bsky.app/profile/{handle}/post/{rkey}",
                "published_at": published,
                "raw_summary": text[:1500],
                "score": (post.get("likeCount") or 0) + (post.get("repostCount") or 0),
            }
        )
    return items


# GitHub "trending" proxy: repos *created* in the last week that already
# crossed the star floor. Uses its own window instead of _MAX_AGE — a repo
# created 6 days ago that is exploding right now is exactly the signal.
_GITHUB_WINDOW_DAYS = 7


def _parse_github(name: str, tier: int, url: str) -> list[dict[str, Any]]:
    """GitHub Search API for fast-rising new AI repos — no auth needed."""
    since = (
        datetime.now(timezone.utc) - timedelta(days=_GITHUB_WINDOW_DAYS)
    ).date().isoformat()
    r = requests.get(
        url.format(since=since),
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    items: list[dict[str, Any]] = []
    for repo in r.json().get("items", []):
        full_name = repo.get("full_name") or ""
        link = repo.get("html_url") or ""
        if not full_name or not link:
            continue
        stars = repo.get("stargazers_count") or 0
        desc = (repo.get("description") or "").strip()
        items.append(
            {
                "source": name,
                "tier": tier,
                "item_id": full_name,
                "title": f"{full_name} (★{stars})",
                "url": link,
                "published_at": _parse_date(repo.get("created_at") or ""),
                "raw_summary": desc[:1500],
                "score": stars,
            }
        )
    return items


_PARSERS = {
    "rss": _parse_rss,
    "reddit": _parse_rss,
    "arxiv": _parse_rss,
    "hn": _parse_hn,
    "hf_papers": _parse_hf_papers,
    "bluesky": _parse_bluesky,
    "github": _parse_github,
}


def _fetch_one(name: str, kind: str, url: str, tier: int) -> list[dict[str, Any]]:
    parser = _PARSERS.get(kind)
    if parser is None:
        logger.warning("Unknown source kind: %s", kind)
        return []
    items = parser(name, tier, url)
    logger.info("[%s] %d items", name, len(items))
    return items


def fetch_all() -> list[dict[str, Any]]:
    """Fetch every configured source sequentially.

    Per-source failures are logged and skipped so one dead feed doesn't
    take down the whole catchup.
    """
    out: list[dict[str, Any]] = []
    for name, kind, url, tier in SOURCES:
        try:
            out.extend(_fetch_one(name, kind, url, tier))
        except Exception as exc:  # noqa: BLE001 - keep loop alive
            logger.warning("[%s] fetch failed: %s", name, exc)
    return out


def fetch_all_parallel(max_workers: int = 8) -> list[dict[str, Any]]:
    """Fetch every configured source concurrently (network bound — a
    thread pool cuts wall-clock from ~sum to ~max of source latencies).
    Same fail-soft semantics as :func:`fetch_all`."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, name, kind, url, tier): name
            for name, kind, url, tier in SOURCES
        }
        for fut in as_completed(futures):
            try:
                out.extend(fut.result())
            except Exception as exc:  # noqa: BLE001 - keep loop alive
                logger.warning("[%s] fetch failed: %s", futures[fut], exc)
    return out
