"""
Write Report Skill — LLM tool definition and runner.

Provides a tool that reads video memory (atlas directory) and generates a
frame-text interleaved article. The pipeline uses an agent that probes the
video via a VLM to select section-appropriate frames, then writes the article
section-by-section.

`TOOL_DEFINITION` can be registered directly with any OpenAI-compatible
function-calling LLM.
"""

from __future__ import annotations

import json
from typing import Any

from .write_report import load_atlas_memory, create_report


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_report",
        "description": (
            "Generate a frame-text interleaved article from video memory (atlas directory). "
            "The pipeline plans an outline, uses an agent to probe the video via a vision "
            "model and pick the best frames for each section, then writes the article "
            "section-by-section. When output_path is provided, frames are saved to a "
            "frames/ directory next to the article. The atlas directory must have been "
            "previously created by the video memory module (mm-harness)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "atlas_dir": {
                    "type": "string",
                    "description": (
                        "Path to the atlas output directory produced by mm-harness. "
                        "This directory contains README.md, segments/, units/, "
                        "video clips, and subtitles."
                    ),
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Optional file path to save the article. "
                        "If omitted, the article is returned as text only."
                    ),
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "Optional focus angle for the article. Shapes the outline "
                        "around this topic."
                    ),
                    "default": "",
                },
                "max_probes": {
                    "type": "integer",
                    "description": (
                        "Maximum number of frame-agent probes. Each probe extracts and "
                        "inspects frames from one time range. Default is 15."
                    ),
                    "default": 15,
                },
            },
            "required": ["atlas_dir"],
        },
    },
}

# A second tool: load-only, returns structured memory for the LLM to
# reason over and write its own report.

LOAD_MEMORY_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "load_video_memory",
        "description": (
            "Load video memory from an atlas directory into a structured JSON object. "
            "Returns the video title, abstract, segments, units, timestamps, summaries, "
            "and transcript excerpts. Use this when you want to read the memory and "
            "write a custom report yourself, rather than using the pre-formatted report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "atlas_dir": {
                    "type": "string",
                    "description": "Path to the atlas output directory produced by mm-harness.",
                },
            },
            "required": ["atlas_dir"],
        },
    },
}


ALL_TOOL_DEFINITIONS = [TOOL_DEFINITION, LOAD_MEMORY_TOOL_DEFINITION]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _memory_to_dict(memory) -> dict[str, Any]:
    """Convert an AtlasMemory to a JSON-serializable dict."""
    def _unit_dict(u):
        d = {
            "unit_id": u.unit_id,
            "title": u.title,
            "start_time": u.start_time,
            "end_time": u.end_time,
            "duration": u.duration,
            "summary": u.summary,
            "detail": u.detail,
        }
        if u.subtitles:
            d["subtitles"] = u.subtitles[:1000]
        if u.clip_path:
            d["clip_path"] = u.clip_path
        return d

    def _seg_dict(s):
        d = {
            "segment_id": s.segment_id,
            "title": s.title,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "duration": s.duration,
            "summary": s.summary,
            "composition_rationale": s.composition_rationale,
            "units": [_unit_dict(u) for u in s.units],
        }
        if s.clip_path:
            d["clip_path"] = s.clip_path
        return d

    return {
        "atlas_dir": memory.atlas_dir,
        "video_title": memory.video_title,
        "video_duration": memory.video_duration,
        "abstract": memory.abstract,
        "num_segments": memory.num_segments,
        "num_units": memory.num_units,
        "segments": [_seg_dict(s) for s in memory.segments],
        "video_path": memory.video_path,
    }


# ---------------------------------------------------------------------------
# LLM tool runners
# ---------------------------------------------------------------------------


# Shared client config — set these before calling run_tool
_mllm_client = None
_mllm_model = "Qwen/Qwen3-VL-8B-Instruct"
_writer_client = None
_writer_model = "Qwen/Qwen3-235B-A22B-Instruct-2507"


def configure_clients(
    mllm_client=None,
    mllm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    writer_client=None,
    writer_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
):
    """Set the LLM clients used by run_tool."""
    global _mllm_client, _mllm_model, _writer_client, _writer_model
    _mllm_client = mllm_client
    _mllm_model = mllm_model
    _writer_client = writer_client or mllm_client
    _writer_model = writer_model


def run_tool(arguments: dict[str, Any]) -> str:
    """
    Execute the write_report tool from LLM function-calling arguments.

    Returns the Markdown article as a string.
    """
    report = create_report(
        atlas_dir=arguments["atlas_dir"],
        output_path=arguments.get("output_path"),
        focus=arguments.get("focus", ""),
        max_probes=arguments.get("max_probes", 15),
        mllm_client=_mllm_client,
        mllm_model=_mllm_model,
        writer_client=_writer_client,
        writer_model=_writer_model,
    )
    return report


def run_load_memory_tool(arguments: dict[str, Any]) -> str:
    """Execute the load_video_memory tool and return JSON."""
    memory = load_atlas_memory(arguments["atlas_dir"])
    return json.dumps(_memory_to_dict(memory), ensure_ascii=False, indent=2)


TOOL_RUNNERS = {
    "write_report": run_tool,
    "load_video_memory": run_load_memory_tool,
}
