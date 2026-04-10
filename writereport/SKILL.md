---
name: writereport
description: >
  Generate a frame-text interleaved article from video memory (atlas directory).
  Use this skill after processing a video with mm-harness to produce a
  well-written article where an agent probes the video via a vision model to
  pick the best frames for each section.
---

# Write Report

Use `write-report` after a video has been processed into memory by mm-harness. It reads the atlas directory and produces a Markdown article: the writer LLM plans an outline, then drives an agent loop that probes the video via a vision model to pick the best frames for each section, then writes the article section-by-section.

## Pipeline

1. **Load** — parse the atlas directory into memory (segments, units, transcripts)
2. **Outline** — the writer LLM plans sections, hooks, and narrative arc (no frames yet)
3. **Agent frame selection** — the agent reads the outline, calls `probe_video(start, end, n)` on targeted time ranges, the VLM describes each frame, the agent accepts the good ones per section
4. **Write** — the writer LLM produces the article section-by-section, interleaving the accepted frames

## When To Use It

- After processing a video with `mm-harness create`
- When the user asks for a report, summary, or analysis of a processed video
- When you need to reference specific video moments with timestamps
- When building a study guide, meeting notes, or content breakdown

## Two Modes of Use

### Mode 1: Full article (`write_report`)

Generates a complete Markdown article automatically.

```python
from writereport import create_report

article = create_report(
    atlas_dir="./outputs/my_video_atlas",
    output_path="./reports/video_article.md",
    focus="key technical concepts",
    mllm_client=vlm_client,
    writer_client=writer_client,
)
```

### Mode 2: Raw memory load (`load_video_memory`)

Returns structured JSON of the video memory. Best when you (the LLM) want to reason over the content and write a custom report with your own analysis.

```python
from writereport.skill import run_load_memory_tool

json_str = run_load_memory_tool({"atlas_dir": "./outputs/my_video_atlas"})
```

## As an LLM tool call

```json
{
  "name": "write_report",
  "arguments": {
    "atlas_dir": "./outputs/my_video_atlas",
    "output_path": "./reports/video_article.md",
    "focus": "key technical concepts",
    "max_probes": 15
  }
}
```

Or load raw memory for custom processing:

```json
{
  "name": "load_video_memory",
  "arguments": {
    "atlas_dir": "./outputs/my_video_atlas"
  }
}
```

## Handle the tool call in your agent loop

```python
from writereport.skill import TOOL_RUNNERS, configure_clients

configure_clients(
    mllm_client=vlm_client,
    mllm_model="Qwen/Qwen3-VL-8B-Instruct",
    writer_client=writer_client,
    writer_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
)

runner = TOOL_RUNNERS[function_call.name]
result = runner(function_call.arguments)
```

## Parameters

### write_report

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| atlas_dir | string | yes | — | Path to the atlas output directory |
| output_path | string | no | — | File path to save the article |
| focus | string | no | "" | Focus angle for the article |
| max_probes | integer | no | 15 | Maximum frame-agent probes |

### load_video_memory

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| atlas_dir | string | yes | Path to the atlas output directory |

## Typical Workflow

1. **Search** — `video-search` skill finds relevant videos
2. **Memorize** — `mm-harness create --url <URL>` builds video memory
3. **Report** — this skill generates the article from memory
