"""
Article outline generator (content-first).

Reads video memory and asks an LLM to plan the article structure.
Frames are NOT assigned here — they're matched to sections later by VLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .write_report import AtlasMemory, _format_time_range


OUTLINE_SYSTEM_PROMPT = """\
You are an expert article planner. Given structured notes from a video, \
you produce a detailed JSON outline for a high-quality article.

Think like a magazine editor planning a feature article: \
what is the hook, what is the arc, how do sections flow naturally?

Each section should cover a distinct concept and reference specific video units \
as source material (by unit_id).
"""

OUTLINE_SCHEMA_HINT = """\
Return ONLY valid JSON matching this schema (no other text):

{
  "title": "Article title — compelling, not generic",
  "hook": "1-2 sentence opening that draws the reader in",
  "narrative_arc": "Brief description of the story progression (e.g., problem → insight → mechanics → impact)",
  "sections": [
    {
      "title": "Section title",
      "key_points": ["point 1", "point 2", "point 3"],
      "source_units": ["unit_0001", "unit_0002"],
      "target_paragraphs": 3,
      "visual_needs": "What kind of visual would best illustrate this section (e.g., 'architecture diagram', 'step-by-step scoring process')"
    }
  ],
  "conclusion_points": ["takeaway 1", "takeaway 2", "takeaway 3"]
}
"""


def _build_outline_source(memory: AtlasMemory) -> str:
    """Build compact source material for the outline prompt."""
    parts = []
    parts.append(f"# Video: {memory.video_title}")
    parts.append(f"Duration: {memory.video_duration}")
    parts.append("")
    if memory.abstract:
        parts.append(f"Abstract: {memory.abstract}")
        parts.append("")

    for i, seg in enumerate(memory.segments, 1):
        time_range = _format_time_range(seg.start_time, seg.end_time)
        parts.append(f"## Segment {i}: {seg.title} ({time_range})")
        if seg.summary:
            parts.append(f"Summary: {seg.summary}")
        parts.append("")

        for j, unit in enumerate(seg.units, 1):
            u_range = _format_time_range(unit.start_time, unit.end_time)
            parts.append(f"### Unit {i}.{j}: {unit.title} ({u_range}) [ID: {unit.unit_id}]")
            if unit.summary:
                parts.append(f"Summary: {unit.summary}")
            if unit.detail:
                parts.append(f"Detail: {unit.detail[:300]}")
            parts.append("")

    return "\n".join(parts)


def _parse_json_from_response(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def generate_outline(
    memory: AtlasMemory,
    focus: str = "",
    client: OpenAI | None = None,
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
) -> dict:
    """
    Generate a structured article outline from video memory.
    No frames are assigned — that happens in the frame matching step.

    Returns dict with: title, hook, narrative_arc, sections, conclusion_points.
    """
    if client is None:
        raise ValueError("client is required")

    source = _build_outline_source(memory)

    user_prompt = f"""\
Plan a detailed article outline based on this video content.

{f"**Focus angle**: {focus}" if focus else ""}

## Source material

{source}

---

{OUTLINE_SCHEMA_HINT}"""

    print("  [outline] Generating article outline...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": OUTLINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0.5,
    )

    reply = response.choices[0].message.content
    outline = _parse_json_from_response(reply)

    if outline is None:
        raise ValueError(f"Failed to parse outline JSON:\n{reply[:500]}")

    sections = outline.get("sections", [])
    print(f"  [outline] Outline ready: {outline.get('title', '?')} ({len(sections)} sections)")
    return outline
