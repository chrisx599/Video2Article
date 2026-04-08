"""
LLM-powered article writer.

Takes video memory + extracted frame paths and asks an LLM to write a
high-quality, frame-text interleaved article — not a mechanical dump of
metadata, but a real piece of writing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .write_report import AtlasMemory, _make_relative, _format_time_range


SYSTEM_PROMPT = """\
You are an expert technical writer who creates detailed, insightful articles \
based on video content. You write like a skilled journalist or educator — \
clear, engaging, well-structured prose that explains concepts thoroughly.

Your task is to write a frame-text interleaved Markdown article based on the \
video memory provided. This is NOT a summary or transcript dump. It should \
read like a high-quality blog post or educational article that someone would \
want to read from start to finish.

## Writing guidelines

- Write in a clear, authoritative voice with natural flow between sections
- Open with a compelling introduction that hooks the reader
- Explain concepts in depth — add context, analogies, and insight beyond what \
the video literally says
- Use the transcript excerpts as source material, but rewrite and synthesize — \
never just paste subtitles
- Build ideas progressively, connecting sections with smooth transitions
- End with a thoughtful conclusion that ties everything together

## Frame interleaving rules

You MUST interleave frames throughout the article using this exact syntax:

![caption](frame_path)

Where:
- caption: a descriptive caption you write for the frame (e.g., "The attention \
mechanism architecture showing encoder-decoder information flow")
- frame_path: the EXACT path from the frame list provided — do not modify paths

Place frames at natural points in the narrative:
- After introducing a concept that the frame illustrates
- Before detailed explanation of what the frame shows
- At visual transitions between topics

Every frame provided MUST appear exactly once in the article. Do not skip any.

## Format

- Use Markdown with ## for major sections, ### for subsections
- Keep paragraphs focused (3-5 sentences each)
- Use bold for key terms on first introduction
- Include timestamps in parentheses when referencing specific moments: (02:30)
"""


def _build_source_material(
    memory: AtlasMemory,
    report_dir: Path,
) -> str:
    """Build the source material block from video memory."""
    parts = []

    parts.append(f"# Video: {memory.video_title}")
    parts.append(f"Duration: {memory.video_duration}")
    parts.append("")

    if memory.abstract:
        parts.append(f"## Abstract")
        parts.append(memory.abstract)
        parts.append("")

    for i, seg in enumerate(memory.segments, 1):
        time_range = _format_time_range(seg.start_time, seg.end_time)
        parts.append(f"## Segment {i}: {seg.title} ({time_range})")
        if seg.summary:
            parts.append(f"Summary: {seg.summary}")
        parts.append("")

        for j, unit in enumerate(seg.units, 1):
            u_range = _format_time_range(unit.start_time, unit.end_time)
            parts.append(f"### Unit {i}.{j}: {unit.title} ({u_range})")

            if unit.summary:
                parts.append(f"Summary: {unit.summary}")

            if unit.detail:
                parts.append(f"Detail: {unit.detail}")

            # Frame references for this unit
            if unit.frame_paths:
                parts.append("Frames:")
                for fp in unit.frame_paths:
                    rel = _make_relative(fp, report_dir)
                    parts.append(f"  - {rel}")

            # Transcript
            if unit.subtitles:
                # Include full subtitles as source (the LLM will synthesize)
                parts.append(f"Transcript:")
                parts.append(unit.subtitles)

            parts.append("")

    return "\n".join(parts)


def write_article(
    memory: AtlasMemory,
    report_dir: Path,
    focus: str = "",
    client: OpenAI | None = None,
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
) -> str:
    """
    Use an LLM to write a high-quality article from video memory.

    Parameters
    ----------
    memory : AtlasMemory
        Parsed atlas memory with frame_paths populated.
    report_dir : Path
        Directory where the report will be saved (for relative frame paths).
    focus : str
        Optional focus topic — guides the article's angle.
    client : OpenAI
        OpenAI-compatible client.
    model : str
        LLM model name for writing.

    Returns
    -------
    str
        The article in Markdown format.
    """
    if client is None:
        raise ValueError("client is required for write_article")

    source_material = _build_source_material(memory, report_dir)

    # Collect all frame paths for the instruction
    all_frames = []
    for seg in memory.segments:
        for unit in seg.units:
            for fp in unit.frame_paths:
                rel = _make_relative(fp, report_dir)
                all_frames.append(rel)

    frame_list = "\n".join(f"  - {fp}" for fp in all_frames)

    user_prompt = f"""\
Write a detailed, well-crafted article based on the following video content.

{f'**Focus angle**: {focus}' if focus else ''}

## Frame paths to include (use these EXACT paths in ![caption](path) syntax)

{frame_list}

## Source material

{source_material}

---

Now write the article. Remember:
- Write like an expert educator, not a video summarizer
- Interleave ALL frames at natural points using ![caption](path) syntax
- Add insight, context, and explanation beyond what the video literally says
- Use smooth transitions between sections
- Make it something people would want to read
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8192,
        temperature=0.7,
    )

    article = response.choices[0].message.content

    # Strip any <think>...</think> blocks (Qwen thinking mode)
    article = re.sub(r"<think>.*?</think>", "", article, flags=re.DOTALL).strip()

    return article
