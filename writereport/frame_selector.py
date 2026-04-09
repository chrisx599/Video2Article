"""
MLLM-based keyframe selection with perceptual deduplication and content scoring.

Pipeline per unit:
  1. Extract dense candidates (every 2s)
  2. Score each for visual richness (edge density)
  3. Remove near-duplicates via dHash
  4. Send to VLM with rich context → get selections + descriptions

Plus a global post-pass to remove cross-unit duplicates.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import shlex
from pathlib import Path
from openai import OpenAI

try:
    from PIL import Image, ImageFilter
    import numpy as np
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _extract_candidates(
    video_path: str,
    start_time: float,
    end_time: float,
    output_dir: Path,
    prefix: str,
    interval_sec: float = 2.0,
) -> list[dict]:
    """Extract candidate frames at fixed intervals. Returns list of dicts."""
    video = Path(video_path)
    if not video.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    duration = end_time - start_time
    if duration <= 0:
        return []

    timestamps = []
    t = start_time + interval_sec / 2
    while t < end_time:
        timestamps.append(t)
        t += interval_sec

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


# ---------------------------------------------------------------------------
# Content richness scoring
# ---------------------------------------------------------------------------


def _compute_richness_score(image_path: str) -> float:
    """
    Score a frame's visual information density (0.0 to 1.0).

    Combines:
    - Edge density: diagrams/text have many edges
    - Spatial spread: information should be distributed across the frame,
      not concentrated in one corner (which catches title slides with
      decorative shapes in one area only)
    - Resolution penalty: very small images score lower
    """
    if not _HAS_PIL:
        return 0.5

    try:
        img = Image.open(image_path).convert("L")
        w, h = img.size

        # Resolution penalty: reject thumbnails
        if w < 640 or h < 360:
            return 0.05

        edges = img.filter(ImageFilter.FIND_EDGES)
        arr = np.array(edges)

        # Edge density
        edge_ratio = float(np.mean(arr > 30))

        # Spatial spread: divide into a 3x3 grid, count how many cells
        # have significant edges. A good diagram fills multiple cells.
        cell_h, cell_w = arr.shape[0] // 3, arr.shape[1] // 3
        active_cells = 0
        for r in range(3):
            for c in range(3):
                cell = arr[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                if float(np.mean(cell > 30)) > 0.02:
                    active_cells += 1
        spread = active_cells / 9.0  # 0.0 to 1.0

        # Combined score: need both edge density AND spatial spread
        score = (edge_ratio * 3.0) * (0.3 + 0.7 * spread)
        return min(score, 1.0)
    except Exception:
        return 0.5


def _filter_low_richness(
    candidates: list[dict],
    min_keep: int = 1,
) -> list[dict]:
    """
    Remove frames with low content richness using adaptive thresholding.

    The threshold is relative to the best frame in this unit's candidates:
    - Hard floor at 0.005 (pure blank/white frames)
    - Keep frames scoring >= 40% of the unit's best frame
    - Always return at least min_keep (if any pass the hard floor)
    """
    if not _HAS_PIL or not candidates:
        return candidates

    scored = []
    for c in candidates:
        score = _compute_richness_score(c["path"])
        scored.append({**c, "_richness": score})

    # Sort by richness descending
    scored.sort(key=lambda c: c["_richness"], reverse=True)

    best = scored[0]["_richness"]

    # Hard floor: if the BEST frame is essentially blank, skip this unit
    if best < 0.005:
        return []

    # Adaptive threshold: keep frames scoring >= 40% of the best
    threshold = best * 0.4
    kept = [c for c in scored if c["_richness"] >= threshold]
    if len(kept) < min_keep:
        kept = scored[:min_keep]

    # Restore original timestamp order
    kept.sort(key=lambda c: c["timestamp"])
    return kept


# ---------------------------------------------------------------------------
# Perceptual hashing & deduplication
# ---------------------------------------------------------------------------


def _compute_dhash(image_path: str, hash_size: int = 12) -> int:
    """Compute dHash. Larger hash_size = more sensitive to differences."""
    img = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size))
    pixels = list(img.getdata())
    w = hash_size + 1

    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * w + col]
            right = pixels[row * w + col + 1]
            if left > right:
                bits |= 1 << (row * hash_size + col)
    return bits


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _deduplicate_candidates(
    candidates: list[dict],
    threshold: int = 15,
) -> list[dict]:
    """
    Remove near-duplicate frames via dHash.

    Iterates in REVERSE timestamp order so that for animated slides,
    the LATEST (most complete) version of each visual cluster is kept.
    """
    if not _HAS_PIL or len(candidates) <= 1:
        return candidates

    # Process latest-first so the most complete animation state wins
    reversed_cands = list(reversed(candidates))

    kept: list[dict] = []
    kept_hashes: list[int] = []

    for cand in reversed_cands:
        try:
            h = _compute_dhash(cand["path"])
        except Exception:
            kept.append(cand)
            continue

        is_dup = any(_hamming_distance(h, kh) < threshold for kh in kept_hashes)
        if not is_dup:
            kept.append(cand)
            kept_hashes.append(h)

    # Restore chronological order
    kept.sort(key=lambda c: c["timestamp"])
    return kept


def deduplicate_across_units(
    memory,
    threshold: int = 15,
) -> None:
    """
    Global post-pass: remove cross-unit duplicate frames.

    When a duplicate is found, keeps the version with higher richness score
    (more informative). Modifies memory in place.
    """
    if not _HAS_PIL:
        return

    # (hash, path, richness_score) for each globally-kept frame
    global_hashes: list[tuple[int, str, float]] = []

    for seg in memory.segments:
        for unit in seg.units:
            new_paths = []
            new_descs = {}
            for fp in unit.frame_paths:
                try:
                    h = _compute_dhash(fp)
                    score = _compute_richness_score(fp)
                except Exception:
                    new_paths.append(fp)
                    if fp in unit.frame_descriptions:
                        new_descs[fp] = unit.frame_descriptions[fp]
                    continue

                # Check against all global hashes
                dup_idx = None
                for i, (gh, gp, gs) in enumerate(global_hashes):
                    if _hamming_distance(h, gh) < threshold:
                        dup_idx = i
                        break

                if dup_idx is None:
                    # New unique frame
                    global_hashes.append((h, fp, score))
                    new_paths.append(fp)
                    if fp in unit.frame_descriptions:
                        new_descs[fp] = unit.frame_descriptions[fp]
                elif score > global_hashes[dup_idx][2]:
                    # This version is richer — replace the global entry
                    # (the old version stays in its unit but this one also stays)
                    global_hashes[dup_idx] = (h, fp, score)
                    new_paths.append(fp)
                    if fp in unit.frame_descriptions:
                        new_descs[fp] = unit.frame_descriptions[fp]
                # else: this version is worse, skip it
                    new_paths.append(fp)
                    if fp in unit.frame_descriptions:
                        new_descs[fp] = unit.frame_descriptions[fp]

            removed = len(unit.frame_paths) - len(new_paths)
            if removed > 0:
                print(f"  [global_dedup] {unit.unit_id}: removed {removed} cross-unit duplicate(s)")
            unit.frame_paths = new_paths
            unit.frame_descriptions = new_descs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image_base64(path: str) -> str:
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


# ---------------------------------------------------------------------------
# VLM frame selection (with descriptions)
# ---------------------------------------------------------------------------


def select_keyframes_with_mllm(
    candidates: list[dict],
    unit_title: str,
    unit_summary: str,
    n_select: int = 3,
    client: OpenAI | None = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    unit_detail: str = "",
    unit_subtitles: str = "",
) -> list[dict]:
    """
    Use a vision LLM to select the best keyframes and describe them.
    Returns list of dicts with keys: index, timestamp, path, description.
    """
    if len(candidates) <= n_select:
        return [{**c, "description": ""} for c in candidates]

    if client is None:
        step = len(candidates) / n_select
        return [{**candidates[int(i * step)], "description": ""} for i in range(n_select)]

    # Build rich context
    context_parts = [
        f"**Unit title**: {unit_title}",
        f"**Unit summary**: {unit_summary}",
    ]
    if unit_detail:
        context_parts.append(f"**Detail**: {unit_detail[:400]}")
    if unit_subtitles:
        excerpt = unit_subtitles[:500].strip()
        if len(unit_subtitles) > 500:
            excerpt += " ..."
        context_parts.append(f"**Transcript excerpt**: {excerpt}")

    context_block = "\n".join(context_parts)

    content = [
        {
            "type": "text",
            "text": (
                f"You are selecting keyframes for a technical article about a video.\n\n"
                f"{context_block}\n\n"
                f"Below are {len(candidates)} candidate frames from this video segment.\n\n"
                f"Select the {n_select} BEST frames. Follow these rules strictly:\n\n"
                f"REJECT frames that are:\n"
                f"- Title slides or speaker intro cards (just text, no diagrams)\n"
                f"- Partially animated — diagram elements still appearing/moving\n"
                f"- Mostly blank or empty space\n"
                f"- Blurry or mid-transition\n"
                f"- Showing the same diagram as another frame (even at different stages)\n\n"
                f"PREFER frames that are:\n"
                f"- Complete diagrams with all elements fully visible\n"
                f"- Information-dense: charts, formulas, architecture diagrams, flowcharts\n"
                f"- Visually distinct from each other (show DIFFERENT concepts)\n"
                f"- The FINAL state of an animated diagram (most complete version)\n\n"
                f"For each selected frame, describe what it shows.\n\n"
                f"Respond with ONLY a JSON array (no other text):\n"
                f'[{{"index": 0, "description": "Complete attention architecture diagram showing encoder, decoder, and weight computation"}}]'
            ),
        }
    ]

    for cand in candidates:
        b64 = _encode_image_base64(cand["path"])
        content.append({
            "type": "text",
            "text": f"Frame {cand['index']} ({_seconds_to_hms(cand['timestamp'])}):",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
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

        selected = _parse_vlm_response(reply, candidates, n_select)
        selected.sort(key=lambda c: c["timestamp"])
        return selected[:n_select]

    except Exception as e:
        print(f"[frame_selector] MLLM call failed: {e}, falling back to richness-based selection")
        # Fallback: pick by richness score
        scored = [(c, _compute_richness_score(c["path"]) if _HAS_PIL else 0.5) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{**scored[i][0], "description": ""} for i in range(min(n_select, len(scored)))]


def _parse_vlm_response(
    reply: str,
    candidates: list[dict],
    n_select: int,
) -> list[dict]:
    """Parse VLM response, trying rich format first, then index-only fallback."""
    cand_by_index = {c["index"]: c for c in candidates}

    # Try parsing as array of objects with index + description
    try:
        match = re.search(r"\[.*\]", reply, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and parsed:
                if isinstance(parsed[0], dict) and "index" in parsed[0]:
                    selected = []
                    for item in parsed[:n_select]:
                        idx = item["index"]
                        if idx in cand_by_index:
                            selected.append({
                                **cand_by_index[idx],
                                "description": item.get("description", ""),
                            })
                    if selected:
                        return _pad_if_needed(selected, candidates, n_select)

                if isinstance(parsed[0], int):
                    selected = []
                    for idx in parsed[:n_select]:
                        if idx in cand_by_index:
                            selected.append({**cand_by_index[idx], "description": ""})
                    if selected:
                        return _pad_if_needed(selected, candidates, n_select)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    indices = [int(x) for x in re.findall(r"\d+", reply)]
    selected = []
    for idx in indices[:n_select]:
        if idx in cand_by_index:
            selected.append({**cand_by_index[idx], "description": ""})
    return _pad_if_needed(selected, candidates, n_select)


def _pad_if_needed(
    selected: list[dict],
    candidates: list[dict],
    n_select: int,
) -> list[dict]:
    """Pad selection with richness-ranked candidates if too few selected."""
    if len(selected) >= n_select:
        return selected

    selected_indices = {c["index"] for c in selected}
    remaining = [c for c in candidates if c["index"] not in selected_indices]

    if _HAS_PIL:
        remaining.sort(key=lambda c: _compute_richness_score(c["path"]), reverse=True)

    for c in remaining:
        if len(selected) >= n_select:
            break
        selected.append({**c, "description": ""})

    return selected


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------


def select_frames_for_unit(
    video_path: str,
    start_time: str,
    end_time: str,
    unit_title: str,
    unit_summary: str,
    frames_dir: Path,
    prefix: str,
    n_select: int = 3,
    candidate_interval: float = 2.0,
    client: OpenAI | None = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    unit_detail: str = "",
    unit_subtitles: str = "",
) -> list[dict]:
    """
    End-to-end: extract → score → deduplicate → MLLM selection → return frames.

    Returns list of dicts: {"path": str, "description": str}
    """
    start_sec = _hms_to_seconds(start_time) if start_time else 0.0
    if end_time:
        end_sec = _hms_to_seconds(end_time)
    else:
        try:
            probe = subprocess.run(
                f"ffprobe -v error -show_entries format=duration "
                f"-of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}",
                shell=True, capture_output=True, text=True, timeout=10,
            )
            end_sec = float(probe.stdout.strip())
        except Exception:
            end_sec = start_sec + 60.0

    # 1. Extract dense candidates
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

    n0 = len(candidates)

    # 2. Filter low-richness frames (title slides, blank areas)
    candidates = _filter_low_richness(candidates, min_keep=max(n_select, 3))
    n1 = len(candidates)

    # 3. Deduplicate near-identical frames
    candidates = _deduplicate_candidates(candidates)
    n2 = len(candidates)

    if n0 != n2:
        print(f"  [frame_selector] {prefix}: {n0} candidates → {n1} after richness filter → {n2} after dedup")

    # 4. VLM selects best frames with descriptions
    selected = select_keyframes_with_mllm(
        candidates=candidates,
        unit_title=unit_title,
        unit_summary=unit_summary,
        n_select=n_select,
        client=client,
        model=model,
        unit_detail=unit_detail,
        unit_subtitles=unit_subtitles,
    )

    # 5. Copy selected frames to main dir with clean names
    results = []
    for i, cand in enumerate(selected):
        src = Path(cand["path"])
        dst = frames_dir / f"{prefix}_frame_{i+1}.jpg"
        if src != dst:
            shutil.copy2(src, dst)
        results.append({
            "path": str(dst),
            "description": cand.get("description", ""),
        })

    return results
