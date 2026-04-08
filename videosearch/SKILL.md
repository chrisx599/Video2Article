---
name: video-search
description: >
  Search for YouTube videos across multiple video search backends.
  Use this skill when you need to find relevant videos on a topic before
  processing them with the video memory module (mm-harness).
triggers:
  - search video
  - find video
  - youtube search
  - look for video
  - video about
  - find a lecture
  - find a tutorial
---

# Video Search

Use `video-search` when the user asks you to find videos on a topic, or when you need to discover relevant video URLs before processing them with the video memory module.

## What It Does

Searches across video search backends (YouTube scraping, Serper video search) and returns deduplicated, ranked video results with metadata (title, URL, duration, channel, snippet).

## When To Use It

- The user asks you to find videos about a topic
- You need YouTube URLs to feed into the video memory module (mm-harness)
- The user wants to compare multiple videos on a subject
- You need to find a specific lecture, tutorial, or talk

## How To Use It

### As a Python function

```python
from videosearch import search_videos

# Basic search
results = search_videos("transformer architecture explained")

# With options
results = search_videos(
    query="deep learning lecture",
    max_results=5,
    engines=["youtube", "serper_video"]
)

# Each result dict has:
# title, link, snippet, duration, imageurl, videourl, source, channel, date
```

### As an LLM tool call

```json
{
  "name": "search_videos",
  "arguments": {
    "query": "transformer architecture explained",
    "max_results": 5,
    "engines": ["youtube", "serper_video"]
  }
}
```

### Handle the tool call in your agent loop

```python
from videosearch.skill import run_tool

# When the LLM returns a function call for "search_videos":
result_json = run_tool(function_call.arguments)
# Feed result_json back to the LLM as tool output
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | yes | — | The search query |
| max_results | integer | no | 10 | Maximum results to return |
| engines | list[string] | no | ["youtube", "serper_video"] | Which backends to use |

### Available engines

| Engine | API Key Required | Returns |
|--------|-----------------|---------|
| youtube | No | YouTube videos with duration, channel, thumbnails |
| serper_video | SERPER_API_KEY | Google video results via Serper.dev |

## Return Format

A list of result dicts, each containing:

```json
{
  "title": "Video title",
  "link": "https://www.youtube.com/watch?v=...",
  "snippet": "Description or excerpt",
  "duration": "12:34",
  "imageurl": "https://...",
  "videourl": "https://...",
  "source": "YouTube",
  "channel": "Channel Name",
  "date": "2024-01-15"
}
```

## Typical Workflow

1. **Search** — Use this skill to find relevant videos
2. **Select** — Pick the best video(s) based on title, duration, channel
3. **Memorize** — Feed the URL to `mm-harness create --url <URL>` to build video memory
4. **Report** — Use the `write-report` skill to generate an interleaved report from memory

## Required Environment

Optional API keys (set in `.env` or environment):

```bash
export SERPER_API_KEY=...   # For serper_video engine
```

The `youtube` engine works without any API key.
