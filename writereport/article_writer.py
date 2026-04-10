"""
LLM-powered article writer.

Writes a full article section-by-section from a structured outline, using
frames previously selected by the frame agent.
"""

from __future__ import annotations

import re
from pathlib import Path

from openai import OpenAI

from .write_report import AtlasMemory, _format_time_range


# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------

_WRITER_PERSONA = """\
You are an expert technical writer who creates detailed, insightful articles \
based on video content. You write like a skilled journalist or educator — \
clear, engaging, well-structured prose that explains concepts thoroughly.

Guidelines:
- Write in a clear, authoritative voice with natural flow
- Explain concepts in depth — add context, analogies, and insight beyond \
what the video literally says
- Use the transcript as source material, but rewrite and synthesize — \
never paste subtitles directly
- Use bold for key terms on first introduction
- Include timestamps in parentheses when referencing moments, e.g. (02:30)
- Interleave frames using: ![caption](path)
"""


# ---------------------------------------------------------------------------
# Source material builder
# ---------------------------------------------------------------------------


def _build_section_source(
    memory: AtlasMemory,
    source_unit_ids: list[str],
    report_dir: Path,
) -> str:
    """Build source material for specific units only."""
    unit_map = {}
    for seg in memory.segments:
        for unit in seg.units:
            unit_map[unit.unit_id] = unit

    parts = []
    for uid in source_unit_ids:
        unit = unit_map.get(uid)
        if not unit:
            continue
        u_range = _format_time_range(unit.start_time, unit.end_time)
        parts.append(f"### {unit.title} ({u_range})")
        if unit.summary:
            parts.append(f"Summary: {unit.summary}")
        if unit.detail:
            parts.append(f"Detail: {unit.detail}")
        if unit.subtitles:
            parts.append(f"Transcript:\n{unit.subtitles}")
        parts.append("")

    return "\n".join(parts)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Section-by-section article writer (from outline)
# ---------------------------------------------------------------------------


def _write_introduction(
    outline: dict,
    client: OpenAI,
    model: str,
) -> str:
    prompt = f"""\
Write the opening of an article titled: "{outline['title']}"

Hook to use: {outline.get('hook', '')}
Narrative arc: {outline.get('narrative_arc', '')}
Sections coming: {', '.join(s['title'] for s in outline.get('sections', []))}

Write 2-3 engaging paragraphs that:
- Open with the hook
- Set up the problem or topic
- Preview what the reader will learn
- End with a natural transition to the first section

Use Markdown. Start with a # heading. Do NOT include frames in the introduction."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _WRITER_PERSONA},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0.7,
    )
    return _strip_thinking(response.choices[0].message.content)


def _write_section(
    section: dict,
    section_source: str,
    full_outline: dict,
    previous_ending: str,
    client: OpenAI,
    model: str,
) -> str:
    frame_lines = []
    for f in section.get("frames", []):
        desc = f.get("description", "")
        placement = f.get("placement", "")
        line = f"  - ![caption]({f['path']})"
        if desc:
            line += f" — shows: {desc}"
        if placement:
            line += f" — place: {placement}"
        frame_lines.append(line)
    frame_block = "\n".join(frame_lines) if frame_lines else "  (no frames for this section)"

    outline_context = "\n".join(
        f"  {'>>>' if s['title'] == section['title'] else '   '} {s['title']}"
        for s in full_outline.get("sections", [])
    )

    prompt = f"""\
Write the section titled: "{section['title']}"

## Article structure (you are writing the section marked with >>>)
{outline_context}

## Key points to cover
{chr(10).join(f"- {p}" for p in section.get('key_points', []))}

## Target length
About {section.get('target_paragraphs', 3)} paragraphs.

## Frames to interleave (use these EXACT paths in ![caption](path) syntax)
{frame_block}

## Source material for this section
{section_source}

## Previous section ending (continue naturally from this)
{previous_ending if previous_ending else "(This is the first body section)"}

---

Write this section now. Use ## for the section heading. \
Interleave ALL assigned frames at their suggested placement points. \
Write engaging prose, not a summary. Add insight beyond the source material."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _WRITER_PERSONA},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.7,
    )
    return _strip_thinking(response.choices[0].message.content)


def _write_conclusion(
    outline: dict,
    last_section_ending: str,
    client: OpenAI,
    model: str,
) -> str:
    points = outline.get("conclusion_points", [])
    prompt = f"""\
Write the conclusion for an article titled: "{outline['title']}"

## Key takeaways to weave in
{chr(10).join(f"- {p}" for p in points)}

## Last section ending (transition from this)
{last_section_ending}

Write 2-3 paragraphs that:
- Synthesize the key insights (don't just repeat section summaries)
- Connect back to the opening hook
- End with a forward-looking or thought-provoking final sentence

Use ## Conclusion as the heading. No frames in the conclusion."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _WRITER_PERSONA},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0.7,
    )
    return _strip_thinking(response.choices[0].message.content)


def _get_last_paragraphs(text: str, n: int = 2) -> str:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "\n\n".join(paragraphs[-n:]) if paragraphs else ""


def write_article_from_outline(
    memory: AtlasMemory,
    outline: dict,
    report_dir: Path,
    focus: str = "",
    client: OpenAI | None = None,
    model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
) -> str:
    """
    Write a full article section-by-section from a structured outline.

    Each section is written with awareness of the full outline and
    continuity from the previous section.
    """
    if client is None:
        raise ValueError("client is required")

    sections_out = outline.get("sections", [])
    total = len(sections_out)

    print(f"  [article] Writing introduction...")
    intro = _write_introduction(outline, client, model)
    parts = [intro]
    previous_ending = _get_last_paragraphs(intro)

    for i, section in enumerate(sections_out, 1):
        print(f"  [article] Writing section {i}/{total}: {section.get('title', '?')}...")
        source_units = section.get("source_units", [])
        section_source = _build_section_source(memory, source_units, report_dir)

        section_text = _write_section(
            section=section,
            section_source=section_source,
            full_outline=outline,
            previous_ending=previous_ending,
            client=client,
            model=model,
        )
        parts.append(section_text)
        previous_ending = _get_last_paragraphs(section_text)

    print(f"  [article] Writing conclusion...")
    conclusion = _write_conclusion(outline, previous_ending, client, model)
    parts.append(conclusion)

    article = "\n\n---\n\n".join(parts)

    all_frame_paths = set()
    for sec in sections_out:
        for f in sec.get("frames", []):
            all_frame_paths.add(f["path"])

    missing = [p for p in all_frame_paths if p not in article]
    if missing:
        print(f"  [article] Warning: {len(missing)} frames missing from article, appending...")
        appendix = "\n\n---\n\n## Additional Visuals\n\n"
        for p in missing:
            desc = ""
            for sec in sections_out:
                for f in sec.get("frames", []):
                    if f["path"] == p:
                        desc = f.get("description", "")
                        break
            caption = desc or "Video frame"
            appendix += f"![{caption}]({p})\n\n"
        article += appendix

    return article
