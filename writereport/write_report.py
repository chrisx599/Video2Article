"""
Write Report — read video memory (atlas) and generate frame-text interleaved reports.

This module reads a CanonicalAtlas output directory produced by mm-harness
and generates a Markdown report that interleaves extracted keyframes (images)
with text analysis, timestamps, and transcript excerpts.
"""

from __future__ import annotations

import re
import subprocess
import shlex
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
    frame_paths: list[str] = field(default_factory=list)  # extracted keyframe images


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
    """Read a text file, return empty string if missing."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return ""


def _parse_unit_readme(readme_text: str) -> UnitContent:
    """Parse a unit README.md into UnitContent."""
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
    """Parse a segment README.md into SegmentContent."""
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
# Frame extraction via ffmpeg
# ---------------------------------------------------------------------------


def _hms_to_seconds(hms: str) -> float:
    """Convert HH:MM:SS or MM:SS to seconds."""
    parts = hms.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else 0.0


def _seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_keyframes(
    video_path: str,
    output_dir: Path,
    prefix: str,
    n_frames: int = 3,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> list[str]:
    """
    Extract N evenly-spaced keyframes from a video clip using ffmpeg.

    Returns list of relative paths (relative to output_dir parent) for the
    saved frame images.
    """
    video = Path(video_path)
    if not video.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe video duration if end_time not given
    if end_time is None:
        probe_cmd = (
            f"ffprobe -v error -show_entries format=duration "
            f"-of default=noprint_wrappers=1:nokey=1 {shlex.quote(str(video))}"
        )
        try:
            result = subprocess.run(
                probe_cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            end_time = float(result.stdout.strip())
        except Exception:
            end_time = start_time + 60.0  # fallback

    clip_duration = end_time - start_time
    if clip_duration <= 0:
        return []

    # Compute timestamps for N evenly-spaced frames
    if n_frames == 1:
        timestamps = [start_time + clip_duration / 2]
    else:
        step = clip_duration / (n_frames + 1)
        timestamps = [start_time + step * (i + 1) for i in range(n_frames)]

    frame_paths = []
    for i, ts in enumerate(timestamps):
        fname = f"{prefix}_frame_{i+1}.jpg"
        out_path = output_dir / fname
        cmd = (
            f"ffmpeg -y -loglevel quiet -ss {ts:.2f} "
            f"-i {shlex.quote(str(video))} "
            f"-frames:v 1 -q:v 2 {shlex.quote(str(out_path))}"
        )
        try:
            subprocess.run(cmd, shell=True, timeout=10, check=False)
            if out_path.exists() and out_path.stat().st_size > 0:
                frame_paths.append(str(out_path))
        except Exception:
            pass

    return frame_paths


# ---------------------------------------------------------------------------
# Atlas loader
# ---------------------------------------------------------------------------


def load_atlas_memory(atlas_dir: str | Path) -> AtlasMemory:
    """
    Load a CanonicalAtlas output directory into an AtlasMemory structure.
    """
    root = Path(atlas_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Atlas directory not found: {atlas_dir}")

    memory = AtlasMemory(atlas_dir=str(root))

    # --- Global README ---
    global_readme = _read_file(root / "README.md")
    memory.video_title = _extract_field(global_readme, "Title")
    memory.video_duration = _extract_field(global_readme, "Duration")
    memory.abstract = _extract_field(global_readme, "Abstract")

    # --- Full subtitles ---
    memory.full_subtitles = _read_file(root / "SUBTITLES.md")

    # --- Video path ---
    video_path = root / "video.mp4"
    if not video_path.exists():
        # Check inside input/ directory for any video file
        input_dir = root / "input"
        if input_dir.is_dir():
            for ext in ("*.mp4", "*.webm", "*.mkv", "*.avi"):
                matches = list(input_dir.glob(ext))
                if matches:
                    video_path = matches[0]
                    break
    if video_path.exists():
        memory.video_path = str(video_path)

    # --- Units ---
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

    # --- Segments ---
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

            # Load sub-units within segment folder
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

            # Segment-level subtitles
            seg_subtitles = _read_file(seg_folder / "SUBTITLES.md")
            if seg_subtitles:
                segment.subtitles = seg_subtitles
            else:
                segment.subtitles = "\n\n".join(
                    u.subtitles for u in segment.units if u.subtitles
                )

            # Segment clip
            seg_clip = seg_folder / "video_clip.mp4"
            if seg_clip.exists():
                segment.clip_path = str(seg_clip)

            memory.segments.append(segment)
        # Sort segments chronologically by start_time
        memory.segments.sort(key=lambda s: _hms_to_seconds(s.start_time) if s.start_time else 0.0)
        memory.num_segments = len(memory.segments)

    return memory


# ---------------------------------------------------------------------------
# Frame extraction for the whole atlas
# ---------------------------------------------------------------------------


def extract_all_keyframes(
    memory: AtlasMemory,
    frames_dir: Path,
    frames_per_unit: int = 3,
    use_mllm: bool = False,
    mllm_client: Any = None,
    mllm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> None:
    """
    Extract keyframes for every unit in the atlas and populate
    unit.frame_paths.

    When use_mllm=True, samples dense candidates and uses a vision LLM
    to select the most informative frames. Otherwise, picks evenly-spaced frames.
    """
    source_video = memory.video_path

    has_source = source_video and Path(source_video).exists()

    for seg in memory.segments:
        for unit in seg.units:
            # Prefer source video (use absolute timestamps),
            # fall back to unit clip (use clip-relative timestamps 0..duration)
            if has_source:
                video_to_use = source_video
                t_start = unit.start_time
                t_end = unit.end_time
            elif unit.clip_path and Path(unit.clip_path).exists():
                video_to_use = unit.clip_path
                t_start = "00:00:00"
                t_end = ""  # let ffprobe detect duration
            else:
                continue

            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", unit.unit_id or unit.title[:20])

            if use_mllm:
                from .frame_selector import select_frames_for_unit
                paths = select_frames_for_unit(
                    video_path=video_to_use,
                    start_time=t_start,
                    end_time=t_end,
                    unit_title=unit.title,
                    unit_summary=unit.summary,
                    frames_dir=frames_dir,
                    prefix=safe_id,
                    n_select=frames_per_unit,
                    client=mllm_client,
                    model=mllm_model,
                )
            else:
                start = _hms_to_seconds(t_start) if t_start else 0.0
                end = _hms_to_seconds(t_end) if t_end else None
                paths = extract_keyframes(
                    video_path=video_to_use,
                    output_dir=frames_dir,
                    prefix=safe_id,
                    n_frames=frames_per_unit,
                    start_time=start,
                    end_time=end,
                )

            unit.frame_paths = paths


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _format_time_range(start: str, end: str) -> str:
    if start and end:
        return f"{start} - {end}"
    return start or end or ""


def _truncate_subtitles(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text.strip()
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        truncated = truncated[: last_period + 1]
    return truncated.strip() + " ..."


def _make_relative(frame_path: str, report_dir: Path) -> str:
    """Make a frame path relative to the report file's directory."""
    try:
        return str(Path(frame_path).relative_to(report_dir))
    except ValueError:
        return str(frame_path)


def generate_report(
    memory: AtlasMemory,
    focus: str = "",
    style: str = "detailed",
    include_subtitles: bool = True,
    max_subtitle_chars: int = 500,
    frames_dir: Path | None = None,
    frames_per_unit: int = 3,
    report_dir: Path | None = None,
    use_mllm: bool = False,
    mllm_client: Any = None,
    mllm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> str:
    """
    Generate a frame-text interleaved report from video memory.

    The report interleaves:
    - Extracted keyframe images from each video segment
    - Section headers with time ranges
    - Summaries and detailed descriptions
    - Transcript excerpts (from subtitles)

    Parameters
    ----------
    memory : AtlasMemory
        Parsed atlas memory from ``load_atlas_memory()``.
    focus : str
        Optional focus topic for the report.
    style : str
        "detailed", "summary", or "outline".
    include_subtitles : bool
        Whether to include transcript excerpts.
    max_subtitle_chars : int
        Max chars per subtitle excerpt.
    frames_dir : Path | None
        Directory to save extracted keyframes. If None, frames are skipped.
    frames_per_unit : int
        Number of keyframes to extract per unit (default 3).
    report_dir : Path | None
        Directory where the report will be saved, for computing relative paths.
    """
    # Extract keyframes if frames_dir is provided
    if frames_dir is not None:
        extract_all_keyframes(
            memory, frames_dir, frames_per_unit,
            use_mllm=use_mllm, mllm_client=mllm_client, mllm_model=mllm_model,
        )

    if report_dir is None:
        report_dir = frames_dir.parent if frames_dir else Path(memory.atlas_dir)

    lines: list[str] = []

    # === Header ===
    lines.append(f"# Video Report: {memory.video_title}")
    lines.append("")
    if memory.video_duration:
        lines.append(f"**Duration**: {memory.video_duration}")
    lines.append("")

    if focus:
        lines.append(f"> **Report Focus**: {focus}")
        lines.append("")

    # === Abstract ===
    lines.append("## Overview")
    lines.append("")
    lines.append(memory.abstract or "_No abstract available._")
    lines.append("")

    # === Structure summary ===
    lines.append("## Content Structure")
    lines.append("")
    lines.append(
        f"This video has been decomposed into **{memory.num_segments} segments** "
        f"containing **{memory.num_units} units**."
    )
    lines.append("")

    if style == "outline":
        for i, seg in enumerate(memory.segments, 1):
            time_range = _format_time_range(seg.start_time, seg.end_time)
            lines.append(f"{i}. **{seg.title}** ({time_range})")
            for unit in seg.units:
                u_range = _format_time_range(unit.start_time, unit.end_time)
                lines.append(f"   - {unit.title} ({u_range})")
        lines.append("")
        return "\n".join(lines)

    # === Table of Contents ===
    lines.append("### Table of Contents")
    lines.append("")
    for i, seg in enumerate(memory.segments, 1):
        time_range = _format_time_range(seg.start_time, seg.end_time)
        anchor = re.sub(r"[^a-z0-9-]", "", seg.title.lower().replace(" ", "-"))
        lines.append(f"{i}. [{seg.title}](#{anchor}) ({time_range})")
    lines.append("")

    # === Detailed sections ===
    lines.append("---")
    lines.append("")

    for i, seg in enumerate(memory.segments, 1):
        time_range = _format_time_range(seg.start_time, seg.end_time)

        # Segment header
        lines.append(f"## Section {i}: {seg.title}")
        lines.append("")
        lines.append(f"**Time Range**: {time_range} | **Duration**: {seg.duration}")
        lines.append("")

        # Segment summary
        if seg.summary:
            lines.append(seg.summary)
            lines.append("")

        if style == "summary":
            lines.append("---")
            lines.append("")
            continue

        # Composition rationale
        if seg.composition_rationale:
            lines.append(f"*{seg.composition_rationale}*")
            lines.append("")

        # --- Units: frame-text interleaved ---
        if seg.units:
            for j, unit in enumerate(seg.units, 1):
                u_range = _format_time_range(unit.start_time, unit.end_time)

                lines.append(f"### {i}.{j} {unit.title}")
                lines.append(f"*{u_range}*")
                lines.append("")

                # --- Interleave: frame → text → frame → text ---

                # First frame (opening keyframe for this unit)
                if unit.frame_paths:
                    rel = _make_relative(unit.frame_paths[0], report_dir)
                    lines.append(f"![{unit.title} - keyframe]({rel})")
                    lines.append("")

                # Summary text
                if unit.summary:
                    lines.append(unit.summary)
                    lines.append("")

                # Middle frame(s) interleaved with detail
                mid_frames = unit.frame_paths[1:-1] if len(unit.frame_paths) > 2 else []
                if unit.detail:
                    # Split detail into paragraphs and interleave frames
                    paragraphs = [p.strip() for p in unit.detail.split("\n\n") if p.strip()]
                    if not paragraphs:
                        paragraphs = [unit.detail]

                    for k, para in enumerate(paragraphs):
                        lines.append(para)
                        lines.append("")
                        # Insert a middle frame after a paragraph if available
                        if k < len(mid_frames):
                            rel = _make_relative(mid_frames[k], report_dir)
                            lines.append(f"![{unit.title} - detail]({rel})")
                            lines.append("")
                elif mid_frames:
                    for fp in mid_frames:
                        rel = _make_relative(fp, report_dir)
                        lines.append(f"![{unit.title}]({rel})")
                        lines.append("")

                # Last frame (closing keyframe)
                if len(unit.frame_paths) >= 2:
                    rel = _make_relative(unit.frame_paths[-1], report_dir)
                    lines.append(f"![{unit.title} - end]({rel})")
                    lines.append("")

                # Transcript excerpt
                if include_subtitles and unit.subtitles:
                    excerpt = _truncate_subtitles(unit.subtitles, max_subtitle_chars)
                    lines.append("<details>")
                    lines.append(f"<summary>Transcript ({u_range})</summary>")
                    lines.append("")
                    lines.append(f"> {excerpt}")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

        # Segment-level transcript (if no units)
        if include_subtitles and seg.subtitles and not seg.units:
            excerpt = _truncate_subtitles(seg.subtitles, max_subtitle_chars)
            lines.append("<details>")
            lines.append("<summary>Transcript</summary>")
            lines.append("")
            lines.append(f"> {excerpt}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined load + generate convenience function
# ---------------------------------------------------------------------------


def create_report(
    atlas_dir: str | Path,
    output_path: str | Path | None = None,
    focus: str = "",
    style: str = "article",
    include_subtitles: bool = True,
    max_subtitle_chars: int = 500,
    frames_per_unit: int = 3,
    use_mllm: bool = False,
    mllm_client: Any = None,
    mllm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    writer_client: Any = None,
    writer_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
) -> str:
    """
    Load video memory and generate a frame-text interleaved report.

    Parameters
    ----------
    atlas_dir : str | Path
        Path to the atlas output directory.
    output_path : str | Path | None
        If provided, write the report to this file. Keyframes are saved
        into a ``frames/`` directory next to the report.
    focus : str
        Optional focus topic for the report.
    style : str
        "article" (LLM-written, default), "detailed", "summary", or "outline".
        "article" produces a well-written piece using an LLM.
        Other styles use template-based generation.
    include_subtitles : bool
        Include transcript excerpts (for template styles only).
    max_subtitle_chars : int
        Max chars per transcript excerpt.
    frames_per_unit : int
        Number of keyframes to extract per unit (default 3).
    use_mllm : bool
        If True, use a vision LLM to select the most informative frames.
    mllm_client : OpenAI | None
        OpenAI-compatible client for vision LLM (frame selection).
    mllm_model : str
        Vision model name for frame selection.
    writer_client : OpenAI | None
        OpenAI-compatible client for article writing. If None, falls
        back to mllm_client. Required for style="article".
    writer_model : str
        LLM model name for article writing.

    Returns
    -------
    str
        The generated report in Markdown.
    """
    memory = load_atlas_memory(atlas_dir)

    # Determine frames directory
    frames_dir = None
    report_dir = None
    if output_path:
        out = Path(output_path)
        report_dir = out.parent
        frames_dir = report_dir / "frames"

    if style == "article":
        # Step 1: Extract frames (MLLM-selected or evenly-spaced)
        if frames_dir is not None:
            extract_all_keyframes(
                memory, frames_dir, frames_per_unit,
                use_mllm=use_mllm, mllm_client=mllm_client, mllm_model=mllm_model,
            )

        # Step 2: LLM writes the article
        w_client = writer_client or mllm_client
        if w_client is None:
            raise ValueError("writer_client or mllm_client required for style='article'")

        from .article_writer import write_article
        report = write_article(
            memory=memory,
            report_dir=report_dir or Path(memory.atlas_dir),
            focus=focus,
            client=w_client,
            model=writer_model,
        )
    else:
        report = generate_report(
            memory,
            focus=focus,
            style=style,
            include_subtitles=include_subtitles,
            max_subtitle_chars=max_subtitle_chars,
            frames_dir=frames_dir,
            frames_per_unit=frames_per_unit,
            report_dir=report_dir,
            use_mllm=use_mllm,
            mllm_client=mllm_client,
            mllm_model=mllm_model,
        )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")

    return report
