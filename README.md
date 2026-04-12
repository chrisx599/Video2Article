# Video Deep Research

An agent-driven pipeline that turns YouTube videos into well-written, frame-interleaved articles. Search for videos, process them into structured memory, then generate articles where an LLM agent selects the best frames by probing the video through a vision model.

## Pipeline

```
Search → Process → Write
```

1. **Search** — find relevant YouTube videos (`videosearch`)
2. **Process** — download, transcribe, and segment into structured memory (`multimodal-harness`)
3. **Write** — generate an article with agent-selected keyframes (`writereport`)

## How It Works

The write step uses an agentic approach:

```
Atlas Memory → Generate Outline → Agent Frame Selection → Write Article
                                       ↕
                              probe_video (ffmpeg → VLM)
                              accept_frame (save if good)
```

The writer LLM plans an article outline, then drives a tool-calling loop: it reads the video timeline, probes specific time ranges via a vision model, evaluates the frame descriptions, and accepts the best ones per section. Finally it writes the article section-by-section, interleaving the chosen frames.

## Project Structure

```
videodeepresearch/
├── videosearch/          # YouTube video search skill
│   ├── search_video.py   # youtube_search() and serper_search()
│   └── skill.py          # LLM tool definition
├── writereport/          # Article generation skill
│   ├── write_report.py   # Atlas loader + pipeline orchestration
│   ├── outline_generator.py  # Content-first outline planning
│   ├── frame_agent.py    # Agent that probes video via VLM
│   ├── article_writer.py # Section-by-section writer
│   └── skill.py          # LLM tool definition
├── multimodal-harness/   # Video → structured memory (subtree)
├── test_skills.py        # Full agent loop: search → process → write
├── outputs/              # Processed video atlases
└── reports/              # Generated articles
```

## Quick Start

### Requirements

- Python 3.10+
- ffmpeg
- API access to an OpenAI-compatible endpoint (e.g. SiliconFlow)

```bash
uv pip install openai pillow numpy
```

### Run the full agent

```bash
export LLM_API_KEY="your-api-key"
export LLM_API_BASE_URL="https://api.siliconflow.cn/v1"
export GROQ_API_KEY="your-groq-key"  # for Whisper transcription

python test_skills.py "research attention mechanism in deep learning"
```

### Use individual skills

```python
from openai import OpenAI
from writereport import create_report
from writereport.skill import configure_clients

client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key="...")

configure_clients(
    mllm_client=client,
    mllm_model="Qwen/Qwen3-VL-8B-Instruct",
    writer_client=client,
    writer_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
)

# Generate article from a processed video
article = create_report(
    atlas_dir="./outputs/my_video_atlas",
    output_path="./reports/article.md",
    focus="transformer architecture",
)
```

## Models Used

| Role | Model | Purpose |
|------|-------|---------|
| Writer / Agent | Qwen3-235B-A22B-Instruct | Outline planning, frame selection decisions, article writing |
| Vision (VLM) | Qwen3-VL-8B-Instruct | Describing and evaluating video frames |
| Transcription | Whisper (Groq) | Video → text via multimodal-harness |

## LLM Tool Interface

Both skills expose OpenAI function-calling tool definitions that can be registered with any compatible agent:

```python
from videosearch.skill import TOOL_DEFINITION as SEARCH_TOOL
from writereport.skill import TOOL_DEFINITION as REPORT_TOOL, LOAD_MEMORY_TOOL_DEFINITION

tools = [SEARCH_TOOL, PROCESS_TOOL, REPORT_TOOL, LOAD_MEMORY_TOOL_DEFINITION]
```

See [test_skills.py](test_skills.py) for a complete agent loop example.
