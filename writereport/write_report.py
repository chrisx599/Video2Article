"""
Write Report — load video memory (atlas) and orchestrate article generation.

The pipeline is:
  1. Load atlas memory (video + segments + units + transcript)
  2. Generate an article outline (outline_generator)
  3. Agent-driven frame selection via VLM probing (frame_agent)
  4. Write the article section-by-section (article_writer)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures for parsed atlas content
# ---------------------------------------------------------------------------


@dataclass
class UnitContent:
    unit_id: str = ""
    title: str = ""
    start_time: str = ""
    end_time: str = ""
    duration: str = ""
    summary: str = ""
    detail: str = ""
    subtitles: str = ""
    clip_path: str = ""


@dataclass
class SegmentContent:
    segment_id: str = ""
    title: str = ""
    start_time: str = ""
    end_time: str = ""
    duration: str = ""
    summary: str = ""
    composition_rationale: str = ""
    units: list[UnitContent] = field(default_factory=list)
    subtitles: str = ""
    clip_path: str = ""


@dataclass
class AtlasMemory:
    """Parsed representation of a video memory atlas directory."""

    atlas_dir: str = ""
    video_title: str = ""
    video_duration: str = ""
    abstract: str = ""
    num_units: int = 0
    num_segments: int = 0
    segments: list[SegmentContent] = field(default_factory=list)
    units: list[UnitContent] = field(default_factory=list)
    full_subtitles: str = ""
    video_path: str = ""


# ---------------------------------------------------------------------------
# README parser helpers
# ---------------------------------------------------------------------------


def _extract_field(text: str, field_name: str) -> str:
    """Extract a **Field**: value from README markdown."""
    pattern = rf"\*\*{re.escape(field_name)}\*\*:\s*(.+)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return ""


def _parse_unit_readme(readme_text: str) -> UnitContent:
    return UnitContent(
        unit_id=_extract_field(readme_text, "UnitID"),
        title=_extract_field(readme_text, "Title"),
        start_time=_extract_field(readme_text, "Start Time"),
        end_time=_extract_field(readme_text, "End Time"),
        duration=_extract_field(readme_text, "Duration"),
        summary=_extract_field(readme_text, "Summary"),
        detail=_extract_field(readme_text, "Detail Description"),
    )


def _parse_segment_readme(readme_text: str) -> SegmentContent:
    return SegmentContent(
        segment_id=_extract_field(readme_text, "SegID"),
        title=_extract_field(readme_text, "Title"),
        start_time=_extract_field(readme_text, "Start Time"),
        end_time=_extract_field(readme_text, "End Time"),
        duration=_extract_field(readme_text, "Duration"),
        summary=_extract_field(readme_text, "Summary"),
        composition_rationale=_extract_field(readme_text, "Composition Rationale"),
    )


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _hms_to_seconds(hms: str) -> float:
    parts = hms.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else 0.0


def _seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_time_range(start: str, end: str) -> str:
    if start and end:
        return f"{start} - {end}"
    return start or end or ""


def _make_relative(frame_path: str, report_dir: Path) -> str:
    """Make a frame path relative to the report file's directory."""
    try:
        return str(Path(frame_path).relative_to(report_dir))
    except ValueError:
        return str(frame_path)


# ---------------------------------------------------------------------------
# Atlas loader
# ---------------------------------------------------------------------------


def load_atlas_memory(atlas_dir: str | Path) -> AtlasMemory:
    """Load a CanonicalAtlas output directory into an AtlasMemory structure."""
    root = Path(atlas_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Atlas directory not found: {atlas_dir}")

    memory = AtlasMemory(atlas_dir=str(root))

    global_readme = _read_file(root / "README.md")
    memory.video_title = _extract_field(global_readme, "Title")
    memory.video_duration = _extract_field(global_readme, "Duration")
    memory.abstract = _extract_field(global_readme, "Abstract")

    memory.full_subtitles = _read_file(root / "SUBTITLES.md")

    video_path = root / "video.mp4"
    if not video_path.exists():
        input_dir = root / "input"
        if input_dir.is_dir():
            for ext in ("*.mp4", "*.webm", "*.mkv", "*.avi"):
                matches = list(input_dir.glob(ext))
                if matches:
                    video_path = matches[0]
                    break
    if video_path.exists():
        memory.video_path = str(video_path)

    units_dir = root / "units"
    if units_dir.is_dir():
        unit_folders = sorted(
            [d for d in units_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        for unit_folder in unit_folders:
            readme_text = _read_file(unit_folder / "README.md")
            if not readme_text:
                continue
            unit = _parse_unit_readme(readme_text)
            unit.subtitles = _read_file(unit_folder / "SUBTITLES.md")
            clip = unit_folder / "video_clip.mp4"
            if clip.exists():
                unit.clip_path = str(clip)
            memory.units.append(unit)
        memory.num_units = len(memory.units)

    segments_dir = root / "segments"
    if segments_dir.is_dir():
        seg_folders = sorted(
            [d for d in segments_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        for seg_folder in seg_folders:
            readme_text = _read_file(seg_folder / "README.md")
            if not readme_text:
                continue
            segment = _parse_segment_readme(readme_text)

            sub_unit_folders = sorted(
                [d for d in seg_folder.iterdir() if d.is_dir()],
                key=lambda d: d.name,
            )
            for sub_folder in sub_unit_folders:
                sub_readme = _read_file(sub_folder / "README.md")
                if not sub_readme:
                    continue
                sub_unit = _parse_unit_readme(sub_readme)
                sub_unit.subtitles = _read_file(sub_folder / "SUBTITLES.md")
                clip = sub_folder / "video_clip.mp4"
                if clip.exists():
                    sub_unit.clip_path = str(clip)
                segment.units.append(sub_unit)

            seg_subtitles = _read_file(seg_folder / "SUBTITLES.md")
            if seg_subtitles:
                segment.subtitles = seg_subtitles
            else:
                segment.subtitles = "\n\n".join(
                    u.subtitles for u in segment.units if u.subtitles
                )

            seg_clip = seg_folder / "video_clip.mp4"
            if seg_clip.exists():
                segment.clip_path = str(seg_clip)

            memory.segments.append(segment)
        memory.segments.sort(
            key=lambda s: _hms_to_seconds(s.start_time) if s.start_time else 0.0
        )
        memory.num_segments = len(memory.segments)

    return memory


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def create_report(
    atlas_dir: str | Path,
    output_path: str | Path | None = None,
    focus: str = "",
    mllm_client: Any = None,
    mllm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    writer_client: Any = None,
    writer_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    max_probes: int = 15,
) -> str:
    """
    Load video memory and generate a frame-text interleaved article.

    Pipeline:
      1. Generate outline from memory
      2. Agent probes the video via VLM to pick frames per section
      3. Write the article section-by-section

    Parameters
    ----------
    atlas_dir : str | Path
        Path to the atlas output directory.
    output_path : str | Path | None
        If provided, write the article to this file. Frames are saved into a
        ``frames/`` directory next to the article.
    focus : str
        Optional focus angle for the article.
    mllm_client : OpenAI
        Vision LLM client used by the frame agent for probing.
    mllm_model : str
        Vision model name.
    writer_client : OpenAI
        Writer LLM client. Drives the outline, the frame-selection agent, and
        the section writer. Falls back to ``mllm_client`` if not given.
    writer_model : str
        Writer model name.
    max_probes : int
        Maximum frame-agent probes.

    Returns
    -------
    str
        The generated Markdown article.
    """
    memory = load_atlas_memory(atlas_dir)

    w_client = writer_client or mllm_client
    if w_client is None:
        raise ValueError("writer_client or mllm_client is required")

    frames_dir = None
    report_dir = None
    if output_path:
        out = Path(output_path)
        report_dir = out.parent
        frames_dir = report_dir / "frames"
    rd = report_dir or Path(memory.atlas_dir)

    from .outline_generator import generate_outline
    outline = generate_outline(
        memory=memory,
        focus=focus,
        client=w_client,
        model=writer_model,
    )

    if frames_dir is not None and memory.video_path:
        from .frame_agent import select_frames_with_agent
        outline = select_frames_with_agent(
            outline=outline,
            memory=memory,
            frames_dir=frames_dir,
            report_dir=rd,
            agent_client=w_client,
            agent_model=writer_model,
            vlm_client=mllm_client,
            vlm_model=mllm_model,
            max_probes=max_probes,
        )

    from .article_writer import write_article_from_outline
    report = write_article_from_outline(
        memory=memory,
        outline=outline,
        report_dir=rd,
        focus=focus,
        client=w_client,
        model=writer_model,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")

    return report
