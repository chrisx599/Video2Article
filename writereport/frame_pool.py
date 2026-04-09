"""
Global frame pool — extracts, deduplicates, and scores frames from the
full video as a single pool, then matches frames to article sections.

This replaces the per-unit frame extraction approach.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import shlex
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from PIL import Image, ImageFilter
    import numpy as np
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Helpers
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


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Step 1: Extract dense candidates from full video
# ---------------------------------------------------------------------------


def extract_candidate_pool(
    video_path: str,
    duration: float,
    output_dir: Path,
    interval_sec: float = 2.0,
) -> list[dict]:
    """
    Extract candidate frames from the full video at fixed intervals.

    Returns list of dicts: {"index": int, "timestamp": float, "path": str}
    """
    video = Path(video_path)
    if not video.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = []
    t = interval_sec / 2
    while t < duration:
        timestamps.append(t)
        t += interval_sec

    if not timestamps:
        return []

    print(f"  [frame_pool] Extracting {len(timestamps)} candidates from full video...")
    candidates = []
    for i, ts in enumerate(timestamps):
        fname = f"pool_cand_{i:04d}.jpg"
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

    print(f"  [frame_pool] Extracted {len(candidates)} candidates")
    return candidates


# ---------------------------------------------------------------------------
# Step 2: Score + deduplicate → global pool
# ---------------------------------------------------------------------------


def _compute_richness_score(image_path: str) -> float:
    """Score visual information density (0.0 to 1.0)."""
    if not _HAS_PIL:
        return 0.5
    try:
        img = Image.open(image_path).convert("L")
        w, h = img.size
        if w < 640 or h < 360:
            return 0.05

        edges = img.filter(ImageFilter.FIND_EDGES)
        arr = np.array(edges)
        edge_ratio = float(np.mean(arr > 30))

        cell_h, cell_w = arr.shape[0] // 3, arr.shape[1] // 3
        active_cells = 0
        for r in range(3):
            for c in range(3):
                cell = arr[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                if float(np.mean(cell > 30)) > 0.02:
                    active_cells += 1
        spread = active_cells / 9.0

        score = (edge_ratio * 3.0) * (0.3 + 0.7 * spread)
        return min(score, 1.0)
    except Exception:
        return 0.5


def _compute_dhash(image_path: str, hash_size: int = 12) -> int:
    img = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size))
    pixels = list(img.getdata())
    w = hash_size + 1
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            if pixels[row * w + col] > pixels[row * w + col + 1]:
                bits |= 1 << (row * hash_size + col)
    return bits


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def build_frame_pool(
    candidates: list[dict],
    max_pool_size: int = 20,
    dedup_threshold: int = 15,
) -> list[dict]:
    """
    From raw candidates, produce a deduplicated, richness-ranked pool.

    Steps:
    1. Score all candidates for richness
    2. Drop bottom 50% by richness (remove blank/title/partial frames)
    3. Sort remaining by richness descending
    4. Deduplicate: when two frames look similar, keep the richer one
    5. Cap at max_pool_size
    """
    if not _HAS_PIL or not candidates:
        return candidates[:max_pool_size]

    # Score all candidates
    scored = []
    for c in candidates:
        score = _compute_richness_score(c["path"])
        try:
            h = _compute_dhash(c["path"])
        except Exception:
            h = 0
        scored.append({**c, "richness": score, "dhash": h})

    # Drop bottom 50% by richness — removes blank slides, title cards,
    # pre-animation frames, and other low-information content
    scored.sort(key=lambda c: c["richness"], reverse=True)
    cutoff = len(scored) // 2
    scored = scored[:max(cutoff, max_pool_size)]

    # Deduplicate: iterate from richest to poorest, keep unique
    pool: list[dict] = []
    pool_hashes: list[int] = []

    for cand in scored:
        is_dup = any(_hamming_distance(cand["dhash"], ph) < dedup_threshold for ph in pool_hashes)
        if not is_dup:
            pool.append(cand)
            pool_hashes.append(cand["dhash"])
        if len(pool) >= max_pool_size:
            break

    # Sort pool chronologically
    pool.sort(key=lambda c: c["timestamp"])

    print(f"  [frame_pool] {len(candidates)} candidates → {len(pool)} in pool (top 50% richness → deduped)")
    return pool


# ---------------------------------------------------------------------------
# Step 3: VLM matches frames to article sections
# ---------------------------------------------------------------------------


def match_frames_to_sections(
    pool: list[dict],
    outline: dict,
    memory: Any,
    frames_dir: Path,
    report_dir: Path,
    client: OpenAI | None = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_frames_per_section: int = 3,
) -> dict:
    """
    For each section in the outline, use VLM to pick the best matching
    frames from the pool based on the section's content.

    Returns the outline dict with frames populated in each section.
    """
    if not pool or client is None:
        return outline

    # Build base64-encoded pool for VLM
    pool_images = []
    for p in pool:
        try:
            b64 = _encode_image_base64(p["path"])
            pool_images.append({
                "index": p["index"],
                "timestamp": p["timestamp"],
                "path": p["path"],
                "richness": p.get("richness", 0.5),
                "b64": b64,
            })
        except Exception:
            pass

    if not pool_images:
        return outline

    # Build the full pool message content (images)
    image_content = []
    for pi in pool_images:
        image_content.append({
            "type": "text",
            "text": f"Frame {pi['index']} ({_seconds_to_hms(pi['timestamp'])}):",
        })
        image_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{pi['b64']}"},
        })

    # For each section, ask VLM to pick matching frames
    used_indices: set[int] = set()
    sections = outline.get("sections", [])

    for i, section in enumerate(sections):
        available = [pi for pi in pool_images if pi["index"] not in used_indices]
        if not available:
            break

        n_pick = min(max_frames_per_section, len(available))

        # Build section context
        section_context = f"**Section title**: {section.get('title', '')}\n"
        key_points = section.get("key_points", [])
        if key_points:
            section_context += "**Key points**:\n" + "\n".join(f"- {p}" for p in key_points)

        # Include source unit info for richer context
        source_units = section.get("source_units", [])
        unit_context = ""
        for seg in memory.segments:
            for unit in seg.units:
                if unit.unit_id in source_units:
                    unit_context += f"\nUnit: {unit.title} ({unit.start_time}-{unit.end_time})\n"
                    if unit.summary:
                        unit_context += f"Summary: {unit.summary}\n"

        visual_needs = section.get("visual_needs", "")
        content = [
            {
                "type": "text",
                "text": (
                    f"You are matching video frames to an article section.\n\n"
                    f"{section_context}\n"
                    f"{f'**Visual needs**: {visual_needs}' if visual_needs else ''}\n"
                    f"{unit_context}\n\n"
                    f"Available frames are shown below.\n"
                    f"Select 1 to {n_pick} frames that BEST illustrate this section.\n\n"
                    f"STRICT RULES:\n"
                    f"- ONLY pick frames with COMPLETE, fully-visible diagrams or explanations\n"
                    f"- REJECT frames that are: partially animated (elements still appearing), "
                    f"mostly blank, title-only slides, or just bullet points without content\n"
                    f"- Each frame MUST show a DIFFERENT concept\n"
                    f"- If no frames are good enough, return an EMPTY array []\n"
                    f"- Quality over quantity — 1 great frame is better than 3 mediocre ones\n\n"
                    f"Respond with ONLY a JSON array (no other text):\n"
                    f'[{{"index": 5, "description": "Complete attention architecture with encoder, decoder, and weight computation", '
                    f'"placement": "after explaining the scoring mechanism"}}]'
                ),
            }
        ]

        # Add only available frame images
        for pi in available:
            content.append({
                "type": "text",
                "text": f"Frame {pi['index']} ({_seconds_to_hms(pi['timestamp'])}):",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{pi['b64']}"},
            })

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=500,
                temperature=0.1,
            )
            reply = response.choices[0].message.content.strip()
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

            selected = _parse_frame_matches(reply, available, n_pick)

            # Save selected frames and update outline
            section_frames = []
            for sel in selected:
                idx = sel["index"]
                used_indices.add(idx)

                # Copy to clean name
                src = Path(sel["path"])
                dst = frames_dir / f"sec{i+1}_frame_{len(section_frames)+1}.jpg"
                shutil.copy2(src, dst)

                from .write_report import _make_relative
                rel = _make_relative(str(dst), report_dir)
                section_frames.append({
                    "path": rel,
                    "description": sel.get("description", ""),
                    "placement": sel.get("placement", "at an appropriate point"),
                })

            section["frames"] = section_frames
            print(f"  [frame_pool] Section {i+1} '{section.get('title', '')[:40]}': {len(section_frames)} frames matched")

        except Exception as e:
            print(f"  [frame_pool] Section {i+1} matching failed: {e}")
            section["frames"] = []

    return outline


def _parse_frame_matches(
    reply: str,
    available: list[dict],
    n_pick: int,
) -> list[dict]:
    """Parse VLM frame-matching response."""
    avail_by_index = {p["index"]: p for p in available}

    try:
        match = re.search(r"\[.*\]", reply, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                results = []
                for item in parsed[:n_pick]:
                    if isinstance(item, dict) and "index" in item:
                        idx = item["index"]
                        if idx in avail_by_index:
                            results.append({
                                **avail_by_index[idx],
                                "description": item.get("description", ""),
                                "placement": item.get("placement", ""),
                            })
                    elif isinstance(item, int) and item in avail_by_index:
                        results.append({**avail_by_index[item], "description": "", "placement": ""})
                if results:
                    return results
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fallback: pick top richness frames
    ranked = sorted(available, key=lambda p: p.get("richness", 0), reverse=True)
    return [{**ranked[i], "description": "", "placement": ""} for i in range(min(n_pick, len(ranked)))]
