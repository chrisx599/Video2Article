"""
Agentic frame selection.

The writer LLM drives frame selection iteratively:
  1. Reads the article outline to understand what visuals are needed
  2. Probes specific video time ranges via VLM to "see" frames
  3. Evaluates VLM feedback, re-probes if needed
  4. Accepts the best frames for each section

The VLM is a tool the agent uses — the agent reasons, VLM sees.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from .outline_generator import _parse_json_from_response

try:
    from PIL import Image, ImageFilter
    import numpy as np
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Frame extraction and scoring
# ---------------------------------------------------------------------------


def _extract_candidates(
    video_path: str,
    start_time: float,
    end_time: float,
    output_dir: Path,
    prefix: str,
    interval_sec: float = 2.0,
) -> list[dict]:
    """Extract candidate frames at fixed intervals."""
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


def _compute_richness_score(image_path: str) -> float:
    """Score a frame's visual information density (0.0 to 1.0)."""
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


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function calling format)
# ---------------------------------------------------------------------------

TOOL_GET_TIMELINE = {
    "type": "function",
    "function": {
        "name": "get_video_timeline",
        "description": (
            "Get the video's structural timeline showing segments and units "
            "with timestamps, titles, and summaries. Call this FIRST to understand "
            "where topics are discussed so you can probe the right time ranges."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

TOOL_PROBE_VIDEO = {
    "type": "function",
    "function": {
        "name": "probe_video",
        "description": (
            "Extract frames from a specific time range and get visual descriptions "
            "from the vision model. Returns frame descriptions, quality assessments, "
            "and richness scores. Use targeted time ranges based on the timeline. "
            "Tip: probe the END of a time range to get complete (not mid-animation) slides."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_sec": {
                    "type": "number",
                    "description": "Start time in seconds.",
                },
                "end_sec": {
                    "type": "number",
                    "description": "End time in seconds.",
                },
                "num_frames": {
                    "type": "integer",
                    "description": "Frames to extract (1-8). Default 5.",
                    "default": 5,
                },
            },
            "required": ["start_sec", "end_sec"],
        },
    },
}

TOOL_ACCEPT_FRAME = {
    "type": "function",
    "function": {
        "name": "accept_frame",
        "description": (
            "Accept a frame as the final selection for an article section. "
            "Provide the frame_id from a probe_video result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "frame_id": {
                    "type": "string",
                    "description": "The frame_id from a probe_video result.",
                },
                "section_index": {
                    "type": "integer",
                    "description": "0-based section index in the outline.",
                },
                "caption": {
                    "type": "string",
                    "description": "Caption describing what the frame shows.",
                },
                "placement": {
                    "type": "string",
                    "description": "Where to place in the section (e.g., 'after explaining scoring').",
                    "default": "at an appropriate point",
                },
            },
            "required": ["frame_id", "section_index", "caption"],
        },
    },
}

AGENT_TOOLS = [TOOL_GET_TIMELINE, TOOL_PROBE_VIDEO, TOOL_ACCEPT_FRAME]


# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a visual editor selecting frames from a video for an article. \
You can probe specific time ranges in the video to see what appears on screen.

## Your tools

1. **get_video_timeline()** — See the video structure with timestamps. Call this FIRST.
2. **probe_video(start_sec, end_sec, num_frames)** — Look at frames in a time range. \
Each call costs 1 probe from your budget.
3. **accept_frame(frame_id, section_index, caption, placement)** — Save a frame for a section.

## Strategy

1. Call get_video_timeline() to see where topics are discussed.
2. For each section, identify the time range from source_units.
3. Probe that range. Review the vision model's descriptions.
4. If a frame is described as "complete" with richness >= 4, accept it.
5. If frames are "partial" or low quality, try probing the LAST few seconds \
of the time range (animations are complete at the end), or try adjacent ranges.
6. Accept 1-2 frames per section. Quality over quantity.
7. Skip sections where no good visual exists.

## Quality criteria

ACCEPT: complete diagrams, information-dense charts, clear architecture visuals.
REJECT: mid-animation frames, blank slides, title cards, speaker face shots.

When you're done selecting frames for all sections, just say "Done" with no tool call.\
"""


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

@dataclass
class AgentBudget:
    max_probes: int = 15
    max_probes_per_section: int = 3
    max_turns: int = 40
    probes_used: int = 0
    probes_per_section: dict = field(default_factory=dict)
    turns_used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.probes_used >= self.max_probes or self.turns_used >= self.max_turns

    def record_probe(self):
        self.probes_used += 1

    def status(self, n_sections: int, accepted: dict) -> str:
        filled = len([s for s in range(n_sections) if s in accepted])
        return (
            f"[Budget: {self.probes_used}/{self.max_probes} probes used | "
            f"Sections with frames: {filled}/{n_sections}]"
        )


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _handle_get_timeline(memory) -> str:
    """Return compact video timeline."""
    timeline = {
        "video_title": memory.video_title,
        "duration": memory.video_duration,
        "duration_sec": _hms_to_seconds(memory.video_duration) if memory.video_duration else 0,
        "segments": [],
    }
    for seg in memory.segments:
        seg_data = {
            "title": seg.title,
            "start": seg.start_time,
            "end": seg.end_time,
            "start_sec": _hms_to_seconds(seg.start_time) if seg.start_time else 0,
            "end_sec": _hms_to_seconds(seg.end_time) if seg.end_time else 0,
            "summary": seg.summary[:200] if seg.summary else "",
            "units": [],
        }
        for unit in seg.units:
            seg_data["units"].append({
                "unit_id": unit.unit_id,
                "title": unit.title,
                "start": unit.start_time,
                "end": unit.end_time,
                "start_sec": _hms_to_seconds(unit.start_time) if unit.start_time else 0,
                "end_sec": _hms_to_seconds(unit.end_time) if unit.end_time else 0,
                "summary": unit.summary[:150] if unit.summary else "",
            })
        timeline["segments"].append(seg_data)

    return json.dumps(timeline, ensure_ascii=False, indent=2)


def _handle_probe_video(
    start_sec: float,
    end_sec: float,
    num_frames: int,
    video_path: str,
    output_dir: Path,
    probe_counter: list,  # mutable counter [int]
    vlm_client: OpenAI,
    vlm_model: str,
    probe_results: dict,  # shared store: frame_id -> {path, timestamp, ...}
) -> str:
    """Extract frames, send to VLM, return descriptions."""
    probe_idx = probe_counter[0]
    probe_counter[0] += 1

    num_frames = max(1, min(num_frames, 8))

    # Extract frames
    candidates = _extract_candidates(
        video_path=video_path,
        start_time=start_sec,
        end_time=end_sec,
        output_dir=output_dir,
        prefix=f"probe_{probe_idx:03d}",
        interval_sec=max(0.5, (end_sec - start_sec) / (num_frames + 1)),
    )

    if not candidates:
        return json.dumps({"error": "No frames could be extracted from this range."})

    # Pre-filter blanks
    filtered = []
    for c in candidates:
        score = _compute_richness_score(c["path"])
        if score >= 0.005:
            c["richness_pixel"] = round(score, 4)
            c["frame_id"] = f"probe_{probe_idx:03d}_f{c['index']}"
            filtered.append(c)

    if not filtered:
        return json.dumps({"error": "All frames in this range are blank or nearly empty."})

    # Send to VLM
    vlm_content = [
        {
            "type": "text",
            "text": (
                f"Describe each video frame. For each, provide:\n"
                f"1. description: what is shown\n"
                f"2. quality: 'complete' | 'partial' | 'blank' | 'title'\n"
                f"3. richness: 1-5 (5 = dense diagram, 1 = blank)\n\n"
                f"Respond with ONLY a JSON array:\n"
                f'[{{"frame_id": "...", "description": "...", "quality": "complete", "richness": 4}}]'
            ),
        }
    ]

    for c in filtered:
        try:
            b64 = _encode_image_base64(c["path"])
        except Exception:
            continue
        vlm_content.append({
            "type": "text",
            "text": f"{c['frame_id']} ({_seconds_to_hms(c['timestamp'])}):",
        })
        vlm_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    try:
        response = vlm_client.chat.completions.create(
            model=vlm_model,
            messages=[{"role": "user", "content": vlm_content}],
            max_tokens=800,
            temperature=0.1,
        )
        reply = _strip_thinking(response.choices[0].message.content)
        vlm_results = _parse_json_from_response(reply)
    except Exception as e:
        # Fallback: return with pixel scores only
        vlm_results = None
        print(f"  [frame_agent] VLM call failed: {e}")

    # Build result
    frames_out = []
    if isinstance(vlm_results, list):
        vlm_map = {r.get("frame_id", ""): r for r in vlm_results if isinstance(r, dict)}
    else:
        vlm_map = {}

    for c in filtered:
        fid = c["frame_id"]
        vlm_info = vlm_map.get(fid, {})
        result = {
            "frame_id": fid,
            "timestamp": _seconds_to_hms(c["timestamp"]),
            "timestamp_sec": round(c["timestamp"], 1),
            "description": vlm_info.get("description", "No VLM description available"),
            "quality": vlm_info.get("quality", "unknown"),
            "richness": vlm_info.get("richness", 0),
            "richness_pixel": c.get("richness_pixel", 0),
        }
        frames_out.append(result)
        # Store for accept_frame
        probe_results[fid] = {
            "path": c["path"],
            "timestamp": c["timestamp"],
            "description": result["description"],
        }

    return json.dumps({"frames": frames_out}, ensure_ascii=False, indent=2)


def _handle_accept_frame(
    frame_id: str,
    section_index: int,
    caption: str,
    placement: str,
    probe_results: dict,
    accepted_frames: dict,  # section_index -> list[dict]
    frames_dir: Path,
    report_dir: Path,
) -> str:
    """Accept a frame for a section."""
    if frame_id not in probe_results:
        return json.dumps({"error": f"Unknown frame_id: {frame_id}. Check probe_video results."})

    # Reject if this exact frame was already accepted for any section
    for sec_frames in accepted_frames.values():
        for existing in sec_frames:
            if existing.get("_frame_id") == frame_id:
                return json.dumps({"error": f"Frame {frame_id} was already accepted. Pick a different frame."})

    info = probe_results[frame_id]

    # Copy to final location
    existing = accepted_frames.get(section_index, [])
    frame_num = len(existing) + 1
    dst = frames_dir / f"sec{section_index + 1}_frame_{frame_num}.jpg"
    shutil.copy2(info["path"], dst)

    from .write_report import _make_relative
    rel_path = _make_relative(str(dst), report_dir)

    frame_entry = {
        "path": rel_path,
        "description": caption,
        "placement": placement,
        "_frame_id": frame_id,
    }

    if section_index not in accepted_frames:
        accepted_frames[section_index] = []
    accepted_frames[section_index].append(frame_entry)

    return json.dumps({
        "status": "accepted",
        "frame_id": frame_id,
        "section": section_index,
        "saved_as": rel_path,
    })


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def select_frames_with_agent(
    outline: dict,
    memory,
    frames_dir: Path,
    report_dir: Path,
    agent_client: OpenAI,
    agent_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    vlm_client: OpenAI | None = None,
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_probes: int = 15,
) -> dict:
    """
    Agentic frame selection: the writer LLM probes the video iteratively.

    Returns the outline dict with section.frames populated.
    """
    if vlm_client is None:
        vlm_client = agent_client

    frames_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = frames_dir / "_agent_probes"
    cand_dir.mkdir(parents=True, exist_ok=True)

    video_path = memory.video_path
    if not video_path:
        print("  [frame_agent] No video path found, skipping frame selection")
        return outline

    sections = outline.get("sections", [])
    n_sections = len(sections)

    budget = AgentBudget(max_probes=max_probes)
    probe_counter = [0]  # mutable counter
    probe_results: dict[str, dict] = {}  # frame_id -> info
    accepted_frames: dict[int, list] = {}  # section_idx -> list[frame_entry]

    # Build outline summary for the agent
    outline_summary = json.dumps(outline, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Select frames for this article. Budget: {max_probes} probes.\n\n"
                f"## Article Outline\n\n```json\n{outline_summary}\n```\n\n"
                f"Start by calling get_video_timeline(), then probe and accept frames."
            ),
        },
    ]

    print(f"  [frame_agent] Starting agent loop (budget: {max_probes} probes, {n_sections} sections)")

    while not budget.exhausted:
        budget.turns_used += 1

        try:
            response = agent_client.chat.completions.create(
                model=agent_model,
                messages=messages,
                tools=AGENT_TOOLS,
            )
        except Exception as e:
            print(f"  [frame_agent] Agent LLM call failed: {e}")
            break

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)

            # Print any reasoning text
            if choice.message.content:
                text = _strip_thinking(choice.message.content)
                if text:
                    print(f"  [frame_agent] Agent: {text[:120]}")

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if name == "get_video_timeline":
                    result = _handle_get_timeline(memory)
                    print(f"  [frame_agent] → get_video_timeline()")

                elif name == "probe_video":
                    if budget.probes_used >= budget.max_probes:
                        result = json.dumps({"error": "Budget exhausted. No more probes available."})
                    else:
                        budget.record_probe()
                        start = args.get("start_sec", 0)
                        end = args.get("end_sec", start + 30)
                        n = args.get("num_frames", 5)
                        print(f"  [frame_agent] → probe_video({_seconds_to_hms(start)}-{_seconds_to_hms(end)}, n={n}) [probe {budget.probes_used}/{budget.max_probes}]")
                        result = _handle_probe_video(
                            start_sec=start, end_sec=end, num_frames=n,
                            video_path=video_path, output_dir=cand_dir,
                            probe_counter=probe_counter,
                            vlm_client=vlm_client, vlm_model=vlm_model,
                            probe_results=probe_results,
                        )

                elif name == "accept_frame":
                    fid = args.get("frame_id", "")
                    sec = args.get("section_index", 0)
                    cap = args.get("caption", "")
                    plc = args.get("placement", "at an appropriate point")
                    result = _handle_accept_frame(
                        frame_id=fid, section_index=sec,
                        caption=cap, placement=plc,
                        probe_results=probe_results,
                        accepted_frames=accepted_frames,
                        frames_dir=frames_dir, report_dir=report_dir,
                    )
                    print(f"  [frame_agent] → accept_frame({fid} → section {sec}): {cap[:60]}")

                else:
                    result = json.dumps({"error": f"Unknown tool: {name}"})

                # Append budget status
                status = budget.status(n_sections, accepted_frames)
                result_with_status = result + f"\n\n{status}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_with_status,
                })

        else:
            # Agent is done
            if choice.message.content:
                text = _strip_thinking(choice.message.content)
                if text:
                    print(f"  [frame_agent] Agent: {text[:200]}")
            break

    # Populate outline with accepted frames
    for i, section in enumerate(sections):
        section["frames"] = accepted_frames.get(i, [])

    total_frames = sum(len(v) for v in accepted_frames.values())
    print(f"  [frame_agent] Done: {total_frames} frames accepted across {len(accepted_frames)} sections ({budget.probes_used} probes used)")

    return outline
