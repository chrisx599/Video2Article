---
name: write-report
description: >
  Generate a detailed interleaved report from video memory (atlas directory).
  Use this skill after processing a video with mm-harness to produce a
  comprehensive report that references specific video segments, timestamps,
  and transcript excerpts.
triggers:
  - write report
  - generate report
  - video report
  - summarize video
  - report from memory
  - interleaved report
  - video analysis
---

# Write Report

Use `write-report` after a video has been processed into memory by mm-harness. It reads the atlas directory and generates a detailed, interleaved report that weaves together text analysis, video clip references, timestamps, and transcript excerpts.

## What It Does

Reads a video memory atlas directory and produces a Markdown report with:

- **Section headers** matching video segments with time ranges
- **Video clip references** — paths to extracted `.mp4` clips for each section
- **Summaries and descriptions** — from the atlas unit/segment analysis
- **Transcript excerpts** — key portions of subtitles for each section
- **Table of contents** — navigable structure for the full report

The result is an interleaved document where text and video references alternate, making it easy to jump between reading and watching specific moments.

## When To Use It

- After processing a video with `mm-harness create`
- When the user asks for a report, summary, or analysis of a processed video
- When you need to reference specific video moments with timestamps
- When building a study guide, meeting notes, or content breakdown

## Two Modes of Use

### Mode 1: Pre-formatted Report (`write_report`)

Generates a complete Markdown report automatically. Best when you want a ready-to-use document.

```python
from writereport import generate_report, load_atlas_memory

memory = load_atlas_memory("./outputs/my_video_atlas")
report = generate_report(memory, style="detailed")
```

### Mode 2: Raw Memory Load (`load_video_memory`)

Returns structured JSON of the video memory. Best when you (the LLM) want to reason over the content and write a custom report with your own analysis.

```python
from writereport.skill import run_load_memory_tool

json_str = run_load_memory_tool({"atlas_dir": "./outputs/my_video_atlas"})
# Feed this JSON to the LLM to write a custom report
```

## How To Use It

### As a Python function

```python
from writereport.write_report import create_report

# Full detailed report, saved to file
report = create_report(
    atlas_dir="./outputs/my_video_atlas",
    output_path="./reports/video_report.md",
    style="detailed",
    focus="key technical concepts",
    include_subtitles=True,
)

# Quick outline
outline = create_report(
    atlas_dir="./outputs/my_video_atlas",
    style="outline",
)
```

### As an LLM tool call

```json
{
  "name": "write_report",
  "arguments": {
    "atlas_dir": "./outputs/my_video_atlas",
    "output_path": "./reports/video_report.md",
    "focus": "key technical concepts",
    "style": "detailed",
    "include_subtitles": true
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

### Handle the tool call in your agent loop

```python
from writereport.skill import TOOL_RUNNERS

# Dispatch by tool name
tool_name = function_call.name           # "write_report" or "load_video_memory"
runner = TOOL_RUNNERS[tool_name]
result = runner(function_call.arguments)
# Feed result back to the LLM as tool output
```

## Parameters

### write_report

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| atlas_dir | string | yes | — | Path to the atlas output directory |
| output_path | string | no | — | File path to save the report |
| focus | string | no | "" | Focus topic for the report |
| style | string | no | "detailed" | "detailed", "summary", or "outline" |
| include_subtitles | boolean | no | true | Include transcript excerpts |
| max_subtitle_chars | integer | no | 500 | Max chars per transcript excerpt |

### load_video_memory

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| atlas_dir | string | yes | Path to the atlas output directory |

## Report Styles

### Detailed (default)

Full report with all unit breakdowns, descriptions, transcript excerpts, and video clip references. Best for comprehensive analysis.

### Summary

Segment-level summaries only, no unit breakdowns or transcripts. Best for quick overviews.

### Outline

Just segment and unit titles with time ranges. Best for navigation and content structure.

## Output Format

The report is generated in Markdown with this structure:

```
# Video Report: [Title]
**Duration**: HH:MM:SS
**Source Video**: path/to/video.mp4

## Overview
[Video abstract]

## Content Structure
[Segment/unit counts]

### Table of Contents
1. [Segment 1 Title] (00:00:00 - 00:05:30)
2. [Segment 2 Title] (00:05:30 - 00:12:15)
...

---

## Section 1: [Segment Title]
**Time Range**: 00:00:00 - 00:05:30 | **Duration**: 00:05:30
**Video Clip**: path/to/clip.mp4

### Summary
[Segment summary]

### Detailed Breakdown

#### 1.1 [Unit Title]
**Time**: 00:00:00 - 00:02:15
**Video Clip**: path/to/unit_clip.mp4

[Unit summary and description]

<details><summary>Transcript Excerpt</summary>
> [Subtitle text...]
</details>

---
(next section...)
```

## Typical Workflow

1. **Search** — `video-search` skill finds relevant videos
2. **Memorize** — `mm-harness create --url <URL>` builds video memory
3. **Report** — This skill generates an interleaved report from memory
4. **Refine** — Optionally use `load_video_memory` to get raw data and write a custom analysis
