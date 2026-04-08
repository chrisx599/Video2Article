"""
Full video research agent — search → process → report.

Usage:
    python test_skills.py "research attention mechanism in deep learning"
"""

import os
import subprocess
import sys
import json
import shlex
from pathlib import Path
from openai import OpenAI

from videosearch.skill import (
    TOOL_DEFINITION as SEARCH_TOOL,
    run_tool as run_search,
)
from writereport.skill import (
    TOOL_DEFINITION as REPORT_TOOL,
    LOAD_MEMORY_TOOL_DEFINITION as MEMORY_TOOL,
    TOOL_RUNNERS,
    configure_clients,
)

# ---------------------------------------------------------------------------
# Setup — SiliconFlow + Qwen3-235B
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.getenv("LLM_API_BASE_URL", "https://api.siliconflow.cn/v1"),
    api_key=os.getenv("LLM_API_KEY"),
)

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

OUTPUT_BASE = Path("./outputs")

# Configure write_report clients for MLLM frame selection + article writing
configure_clients(
    mllm_client=client,
    mllm_model="Qwen/Qwen3-VL-8B-Instruct",
    writer_client=client,
    writer_model=MODEL,
)

# ---------------------------------------------------------------------------
# Process video tool — wraps mm-harness create
# ---------------------------------------------------------------------------

PROCESS_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "process_video",
        "description": (
            "Process a video URL into structured memory using mm-harness. "
            "Downloads the video, transcribes it, segments it, and builds "
            "a structured atlas directory with summaries, video clips, and subtitles. "
            "Returns the atlas directory path for use with write_report or load_video_memory. "
            "This step takes a few minutes for long videos."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "YouTube video URL to process.",
                },
                "output_dir": {
                    "type": "string",
                    "description": (
                        "Directory to write the atlas output. "
                        "Defaults to ./outputs if not specified."
                    ),
                },
            },
            "required": ["url"],
        },
    },
}


HARNESS_DIR = Path(__file__).resolve().parent / "multimodal-harness"


def run_process_video(arguments: dict) -> str:
    url = arguments["url"]
    output_dir = str(Path(arguments.get("output_dir", str(OUTPUT_BASE))).resolve())

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "mm-harness", "create",
        "--url", url,
        "--output-dir", output_dir,
    ]

    print(f"  [process_video] Running: {shlex.join(cmd)}")
    print(f"  [process_video] cwd: {HARNESS_DIR}")
    print(f"  [process_video] This may take a few minutes...")

    env = {**os.environ}
    env.setdefault("LLM_API_BASE_URL", os.getenv("LLM_API_BASE_URL", "https://api.siliconflow.cn/v1"))
    env.setdefault("LLM_API_KEY", os.getenv("LLM_API_KEY", ""))
    env.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    env.setdefault("YOUTUBE_COOKIES_FILE", str(Path(__file__).resolve().parent / "cookies.txt"))
    # Ensure deno is on PATH for yt-dlp JS challenge solving
    deno_bin = Path.home() / ".deno" / "bin"
    if deno_bin.is_dir() and str(deno_bin) not in env.get("PATH", ""):
        env["PATH"] = f"{deno_bin}:{env.get('PATH', '')}"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=HARNESS_DIR,
        env=env,
        timeout=600,  # 10 min timeout
    )

    if result.returncode != 0:
        return json.dumps({
            "status": "error",
            "message": result.stderr[-500:] if result.stderr else "Unknown error",
        })

    # Find the atlas directory (mm-harness creates a subdirectory in output_dir)
    output_path = Path(output_dir)
    atlas_dirs = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and (d / "README.md").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if atlas_dirs:
        atlas_dir = str(atlas_dirs[0])
    else:
        atlas_dir = output_dir

    return json.dumps({
        "status": "success",
        "atlas_dir": atlas_dir,
        "message": f"Video processed. Atlas directory: {atlas_dir}",
    })


# ---------------------------------------------------------------------------
# All tools + dispatch
# ---------------------------------------------------------------------------

TOOLS = [SEARCH_TOOL, PROCESS_VIDEO_TOOL, REPORT_TOOL, MEMORY_TOOL]

TOOL_DISPATCH = {
    "search_videos": run_search,
    "process_video": run_process_video,
    "write_report": TOOL_RUNNERS["write_report"],
    "load_video_memory": TOOL_RUNNERS["load_video_memory"],
}

SYSTEM_PROMPT = """\
You are a video deep research assistant. You help users research topics by finding, \
processing, and analyzing YouTube videos.

You have 4 tools that form a pipeline:

1. **search_videos** — Search YouTube for relevant videos on a topic.
2. **process_video** — Download and process a video URL into structured memory \
(transcription, segmentation, summaries). This takes a few minutes.
3. **load_video_memory** — Load the structured memory as JSON so you can reason over it.
4. **write_report** — Generate a formatted interleaved report from the memory. \
IMPORTANT: Always pass output_path to save the report to a file (use ./reports/<topic>.md).

## Typical workflow

1. Use search_videos to find relevant videos for the user's topic.
2. Pick the best video (prefer shorter ones for faster processing, ~10-30 min).
3. Use process_video to build memory from that video.
4. Use load_video_memory to read the structured content.
5. Use write_report with output_path="./reports/<topic>.md" to save the report to a file.
6. Write your own detailed analysis based on the video memory, referencing specific timestamps.

Always tell the user what you're doing at each step. \
Always tell the user where the report file was saved.\
"""

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run(user_message: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    step = 0
    while True:
        step += 1
        print(f"\n{'='*60}")
        print(f"Step {step}: Calling LLM...")
        print(f"{'='*60}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]

        # If the model wants to call tools
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)

            # Print any text the model said before tool calls
            if choice.message.content:
                print(f"\n[LLM]: {choice.message.content}\n")

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"\n>>> Tool call: {name}")
                print(f"    Args: {json.dumps(args, ensure_ascii=False)[:200]}")

                runner = TOOL_DISPATCH.get(name)
                if runner:
                    result = runner(args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {name}"})

                preview = result[:300] + ("..." if len(result) > 300 else "")
                print(f"    Result: {preview}\n")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # If the model is done — print final response
        else:
            print(f"\n{'='*60}")
            print("FINAL RESPONSE")
            print(f"{'='*60}\n")
            print(choice.message.content)
            break

    return choice.message.content


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "find a short video about attention mechanism and write a report"
    run(query)
