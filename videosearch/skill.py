"""
Video Search Skill — unified entry point and LLM tool definition.

Provides a single `search_videos()` function that queries multiple backends
(YouTube, Serper), deduplicates by URL, and returns ranked results.
The `TOOL_DEFINITION` dict can be registered directly with any
OpenAI-compatible function-calling LLM.
"""

from __future__ import annotations

import json
from typing import Any

from .search_video import youtube_search, serper_search

# ---------------------------------------------------------------------------
# Tool definition (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_videos",
        "description": (
            "Search for YouTube videos on a given topic. "
            "Uses YouTube scraping by default, with optional Serper video search. "
            "Returns deduplicated results. Use this to find relevant videos "
            "before processing them with the video memory module."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing the video topic you want to find.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default is 10.",
                    "default": 10,
                },
                "engine": {
                    "type": "string",
                    "enum": ["youtube", "serper_video"],
                    "description": (
                        "Which video search backend to use. Default is 'youtube' (free, no API key). "
                        "'serper_video' uses Serper.dev (requires SERPER_API_KEY)."
                    ),
                    "default": "youtube",
                },
            },
            "required": ["query"],
        },
    },
}

# ---------------------------------------------------------------------------
# Default engine list
# ---------------------------------------------------------------------------

DEFAULT_ENGINE = "youtube"

# ---------------------------------------------------------------------------
# Engine dispatcher
# ---------------------------------------------------------------------------

_ENGINE_MAP = {
    "youtube": youtube_search,
    "serper_video": serper_search,
}


def _normalize(result: dict) -> dict:
    """Ensure every result has the standard field set."""
    return {
        "title": result.get("title", ""),
        "link": result.get("link", "") or result.get("url", ""),
        "snippet": result.get("snippet", "") or result.get("content", ""),
        "duration": result.get("duration", ""),
        "imageurl": result.get("imageurl", "") or result.get("imageUrl", ""),
        "videourl": result.get("videourl", "") or result.get("videoUrl", ""),
        "source": result.get("source", ""),
        "channel": result.get("channel", ""),
        "date": result.get("date", ""),
    }


def _deduplicate(results: list[dict]) -> list[dict]:
    """Remove duplicates by normalized link URL."""
    seen: set[str] = set()
    unique: list[dict] = []
    for r in results:
        url = r.get("link", "").rstrip("/").lower()
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(r)
    return unique


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def search_videos(
    query: str,
    max_results: int = 10,
    engine: str = DEFAULT_ENGINE,
) -> list[dict[str, str]]:
    """
    Search for videos.

    Parameters
    ----------
    query : str
        Search query describing the video topic.
    max_results : int
        Maximum number of results to return (default 10).
    engine : str
        Backend to use. ``"youtube"`` (default, free) or ``"serper_video"``.

    Returns
    -------
    list[dict]
        Each dict has keys: title, link, snippet, duration, imageurl,
        videourl, source, channel, date.
    """
    fn = _ENGINE_MAP.get(engine)
    if fn is None:
        print(f"[search_videos] Unknown engine: {engine!r}, falling back to youtube.")
        fn = _ENGINE_MAP["youtube"]

    try:
        raw = fn(query)
    except Exception as exc:
        print(f"[search_videos] Engine {engine!r} failed: {exc}")
        return []

    if not isinstance(raw, list):
        return []

    results = [_normalize(item) for item in raw]
    return _deduplicate(results)[:max_results]


# ---------------------------------------------------------------------------
# LLM tool runner — call this from your agent framework's tool dispatcher
# ---------------------------------------------------------------------------


def run_tool(arguments: dict[str, Any]) -> str:
    """
    Execute the search_videos tool from LLM function-calling arguments.

    Parameters
    ----------
    arguments : dict
        The ``arguments`` dict from the LLM's function call
        (keys: query, max_results, engines).

    Returns
    -------
    str
        JSON string of search results, ready to return to the LLM.
    """
    results = search_videos(
        query=arguments["query"],
        max_results=arguments.get("max_results", 10),
        engine=arguments.get("engine", DEFAULT_ENGINE),
    )
    return json.dumps(results, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "machine learning tutorial"
    print(f"Searching: {q!r}\n")
    for r in search_videos(q, max_results=5):
        print(f"  [{r['source']}] {r['title']}")
        print(f"    {r['link']}")
        print(f"    {r['snippet'][:120]}")
        print()
