"""
MLLM-based keyframe selection.

Samples dense candidate frames from video clips, sends them to a
multimodal LLM, and asks it to select the most informative frames
for a report.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import shlex
from pathlib import Path
from openai import OpenAI


def _extract_candidates(
    video_path: str,
    start_time: float,
    end_time: float,
    output_dir: Path,
    prefix: str,
    interval_sec: float = 3.0,
) -> list[dict]:
    """
    Extract candidate frames at fixed intervals from a video.

    Returns list of dicts: {"index": int, "timestamp": float, "path": str}
    """
    video = Path(video_path)
    if not video.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    duration = end_time - start_time
    if duration <= 0:
        return []

    # Generate timestamps
    timestamps = []
    t = start_time + interval_sec / 2  # offset slightly from start
    while t < end_time:
        timestamps.append(t)
        t += interval_sec

    # Ensure at least 1 frame
    if not timestamps:
        timestamps = [start_time + duration / 2]

    candidates = []
    for i, ts in enumerate(timestamps):
        fname = f"{prefix}_cand_{i:03d}.jpg"
        out_path = output_dir / fname
        cmd = (
            f"ffmpeg -y -loglevel quiet -ss {ts:.2f} "
            f"-i {shlex.quote(str(video))} "
            f"-frames:v 1 -q:v 2 {shlex.quote(str(out_path))}"
        )
        try:
            subprocess.run(cmd, shell=True, timeout=10, check=False)
            if out_path.exists() and out_path.stat().st_size > 0:
                candidates.append({
                    "index": i,
                    "timestamp": ts,
                    "path": str(out_path),
                })
        except Exception:
            pass

    return candidates


def _encode_image_base64(path: str) -> str:
    """Read an image file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def select_keyframes_with_mllm(
    candidates: list[dict],
    unit_title: str,
    unit_summary: str,
    n_select: int = 3,
    client: OpenAI | None = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> list[dict]:
    """
    Use a multimodal LLM to select the best keyframes from candidates.

    Parameters
    ----------
    candidates : list[dict]
        Each dict has "index", "timestamp", "path".
    unit_title : str
        Title of the video unit (for context).
    unit_summary : str
        Summary of the video unit (for context).
    n_select : int
        Number of frames to select.
    client : OpenAI
        OpenAI-compatible client with vision support.
    model : str
        Vision model to use.

    Returns
    -------
    list[dict]
        Selected candidates (subset of input), ordered by timestamp.
    """
    if len(candidates) <= n_select:
        return candidates

    if client is None:
        # Fallback: evenly pick
        step = len(candidates) / n_select
        return [candidates[int(i * step)] for i in range(n_select)]

    # Build vision message with all candidate frames
    content = [
        {
            "type": "text",
            "text": (
                f"You are selecting keyframes for a video report.\n\n"
                f"**Unit title**: {unit_title}\n"
                f"**Unit summary**: {unit_summary}\n\n"
                f"Below are {len(candidates)} candidate frames extracted from this video segment. "
                f"Each frame is labeled with its index number.\n\n"
                f"Select the {n_select} most informative and visually distinct frames "
                f"that best represent the content of this segment. Prefer frames that:\n"
                f"- Show diagrams, slides, or visual explanations\n"
                f"- Are visually distinct from each other (avoid near-duplicates)\n"
                f"- Capture key moments or transitions in the content\n"
                f"- Are clear and not blurry or in mid-transition\n\n"
                f"Respond with ONLY a JSON array of the selected frame indices, e.g. [0, 3, 7]\n"
                f"No other text."
            ),
        }
    ]

    for cand in candidates:
        b64 = _encode_image_base64(cand["path"])
        content.append({
            "type": "text",
            "text": f"Frame {cand['index']} (timestamp: {_seconds_to_hms(cand['timestamp'])}):",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=100,
            temperature=0.1,
        )
        reply = response.choices[0].message.content.strip()

        # Parse JSON array from reply
        match = re.search(r"\[[\d\s,]+\]", reply)
        if match:
            selected_indices = json.loads(match.group())
        else:
            # Fallback: try to parse numbers
            selected_indices = [int(x) for x in re.findall(r"\d+", reply)]

        # Map back to candidates
        index_set = set(selected_indices[:n_select])
        selected = [c for c in candidates if c["index"] in index_set]

        # If parsing failed or too few, pad with evenly spaced
        if len(selected) < n_select:
            remaining = [c for c in candidates if c not in selected]
            step = max(1, len(remaining) // (n_select - len(selected)))
            for i in range(0, len(remaining), step):
                if len(selected) >= n_select:
                    break
                selected.append(remaining[i])

        selected.sort(key=lambda c: c["timestamp"])
        return selected[:n_select]

    except Exception as e:
        print(f"[frame_selector] MLLM call failed: {e}, falling back to even spacing")
        step = len(candidates) / n_select
        return [candidates[int(i * step)] for i in range(n_select)]


def select_frames_for_unit(
    video_path: str,
    start_time: str,
    end_time: str,
    unit_title: str,
    unit_summary: str,
    frames_dir: Path,
    prefix: str,
    n_select: int = 3,
    candidate_interval: float = 3.0,
    client: OpenAI | None = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> list[str]:
    """
    End-to-end: extract candidates → MLLM selection → return final frame paths.

    Returns list of paths to the selected keyframe images.
    """
    start_sec = _hms_to_seconds(start_time) if start_time else 0.0
    if end_time:
        end_sec = _hms_to_seconds(end_time)
    else:
        # Probe video duration
        try:
            probe = subprocess.run(
                f"ffprobe -v error -show_entries format=duration "
                f"-of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}",
                shell=True, capture_output=True, text=True, timeout=10,
            )
            end_sec = float(probe.stdout.strip())
        except Exception:
            end_sec = start_sec + 60.0

    candidates = _extract_candidates(
        video_path=video_path,
        start_time=start_sec,
        end_time=end_sec,
        output_dir=frames_dir / "_candidates",
        prefix=prefix,
        interval_sec=candidate_interval,
    )

    if not candidates:
        return []

    selected = select_keyframes_with_mllm(
        candidates=candidates,
        unit_title=unit_title,
        unit_summary=unit_summary,
        n_select=n_select,
        client=client,
        model=model,
    )

    # Copy selected frames to the main frames dir (clean names)
    final_paths = []
    for i, cand in enumerate(selected):
        src = Path(cand["path"])
        dst = frames_dir / f"{prefix}_frame_{i+1}.jpg"
        if src != dst:
            import shutil
            shutil.copy2(src, dst)
        final_paths.append(str(dst))

    return final_paths
