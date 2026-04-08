# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MM Harness (`mm-harness`) is a Python package that transforms long-form video and audio into LLM-friendly structured workspaces. It's designed to be invoked by AI agents (Claude Code, Cursor, etc.) via a CLI or Python API. Agents learn to use it through the installed `SKILL.md`.

## Common Commands

```bash
# Install in development mode
uv pip install -e .

# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_cli.py

# Run a single test method
python -m pytest tests/test_cli.py::CliSmokeTest::test_info_entrypoint_smoke

# Check environment readiness
mm-harness doctor

# Run the main pipeline (URL source)
mm-harness create --url <URL> --output-dir <DIR>

# Run the main pipeline (local files)
mm-harness create --video-file <PATH> --output-dir <DIR>
```

Tests use `unittest` (not pytest fixtures) — test classes inherit from `unittest.TestCase`. CLI tests invoke the CLI via subprocess with `PYTHONPATH` set to `src/`.

## Architecture

**Package layout**: `src/video_atlas/` — installed as `video_atlas`, CLI entry point is `mm-harness`.

### Pipeline flow

```
URL or local file
  → source_acquisition/     # download video/audio/subtitles (YouTube, Xiaoyuzhou, local)
  → transcription/          # speech-to-text (Groq Whisper, Faster-Whisper, Aliyun)
  → workflows/              # TextFirstCanonicalAtlasWorkflow: plan → segment → compose structure
      uses generators/      # LLM calls via OpenAI-compatible API
      uses prompts/         # prompt templates for each pipeline stage
  → persistence/            # write structured workspace to disk
```

### Key layers

- **CLI** (`cli/main.py`): argparse-based, dispatches to application layer. Commands: `info`, `doctor`, `config`, `create`, `install`, `skill`.
- **Application** (`application/canonical_create.py`): Orchestrates end-to-end creation. Two entry points: `create_canonical_from_url()` and `create_canonical_from_local()`.
- **Workflows** (`workflows/text_first_canonical_atlas_workflow.py`): Main workflow class. Sub-modules in `workflows/text_first_canonical/` handle pipeline stages (subtitle prep, language resolution, execution plan building, structure composition).
- **Schemas** (`schemas/`): Pydantic models — `CanonicalAtlas`, `CanonicalRequest`, `SourceInfo`, `DerivedAtlas`, registries.
- **Config** (`config/models.py`): Dataclasses for model, transcriber, acquisition, and runtime config. Loaded from JSON via `load_canonical_pipeline_config()`. Default config: `configs/canonical/default.json`.
- **Settings** (`settings.py`): Reads API credentials from environment variables (`.env` supported).
- **Skill** (`skill/SKILL.md`): Agent-readable documentation bundled as package data, installed to agent skill directories via `skill_install.py`.

### LLM integration

All LLM calls go through `generators/openai_compatible.py` using the OpenAI-compatible protocol. The pipeline config defines separate model roles: `planner`, `text_segmentor`, `multimodal_segmentor`, `structure_composer`, `captioner`.

## Required Environment Variables

- `LLM_API_BASE_URL` — OpenAI-compatible API base URL
- `LLM_API_KEY` — API key for LLM service
- `GROQ_API_KEY` — Groq API key for Whisper transcription

## System Dependencies

`ffmpeg`, `yt-dlp`, `deno` — check with `mm-harness doctor`.
