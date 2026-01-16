# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

noScribe is an AI-powered audio transcription tool using faster-whisper and pyannote for speaker diarization. It runs completely locally with no cloud dependencies.

**Main app:** `noScribe.py` (tkinter/customtkinter GUI)

## Development Commands

```bash
# Setup virtual environment (macOS Apple Silicon)
python3 -m venv venv
source venv/bin/activate
pip install -r environments/requirements_macOS_arm64.txt

# Run the app (GUI mode)
python noScribe.py

# Run headless transcription (CLI mode)
python noScribe.py input.wav output.html --no-gui --language auto
python noScribe.py input.wav output.html --no-gui --speaker-detection auto --timestamps --model precise

# Run tests
pytest tests/

# Run a specific test
pytest tests/test_utils.py::test_str_to_ms
```

## Architecture

### Core Components

| File | Purpose |
|------|---------|
| `noScribe.py` | Main GUI app, orchestrates transcription pipeline |
| `whisper_mp_worker.py` | Whisper transcription in subprocess (faster-whisper) |
| `pyannote_mp_worker.py` | Speaker diarization in subprocess (pyannote.audio) |
| `utils.py` | Time conversion utilities (str_to_ms, ms_to_str, etc.) |
| `noScribeEdit/` | Separate transcript editor app (submodule from kaixxx/noScribeEditor) |

### Multiprocessing Design

Heavy ML inference runs in separate processes to avoid GUI freezing and enable clean resource management:

1. **Main process** (`noScribe.py`): GUI, audio preprocessing with ffmpeg, coordination
2. **Whisper subprocess** (`whisper_mp_worker.py`): Loads faster-whisper model, performs transcription
3. **Pyannote subprocess** (`pyannote_mp_worker.py`): Loads pyannote Pipeline, performs speaker diarization

Workers communicate via `multiprocessing.Queue` with JSON messages:
- `{"type": "log", "level": "info|warn|error|debug", "msg": str}`
- `{"type": "progress", "pct": float, "detail": str}`
- `{"type": "result", "ok": True|False, ...}`

### MeetingRecorder (macOS companion tool)

Located in `tools/`:

| File | Purpose |
|------|---------|
| `tools/meeting_recorder.py` | Menu bar app for recording control |
| `tools/transcribe_watcher.py` | Background service that auto-transcribes new recordings |
| `tools/install.sh` / `tools/uninstall.sh` | LaunchAgent management |

**Config:** `~/Documents/MeetingRecorder/config.yaml`

**Manual workflow trigger:** To manually test the n8n transcript processing workflow, use `source ~/noScribe/venv/bin/activate && python3 -c "import requests; requests.post('http://localhost:5678/webhook/transcript-mvp', json={'transcript_path': '/path/to/transcript.html', 'transcript_html': open('/path/to/transcript.html').read(), 'started_at': '2025-01-01T00:00:00Z', 'audio_duration_seconds': 600})"`

## Technical Notes

### PyTorch 2.6+ Compatibility

PyTorch 2.6 changed `torch.load()` default to `weights_only=True`, breaking pyannote model loading. The fix in `pyannote_mp_worker.py` adds safe globals:

```python
from pyannote.audio.core.task import Specifications, Problem, Resolution
from omegaconf import ListConfig, DictConfig
torch.serialization.add_safe_globals([Specifications, Problem, Resolution, ListConfig, DictConfig])
```

### launchd PATH Issues

When running under launchd (LaunchAgents), PATH is minimal. Use absolute paths for system tools:
- `/opt/homebrew/bin/ffmpeg`
- `/opt/homebrew/bin/ffprobe`

### Multi-Channel Audio Mixing

noScribe's ffmpeg uses `-ac 1` which only takes the first 2 channels. For 3+ channel recordings (e.g., stereo system audio + mono mic), pre-mix with:

```python
weight = 1.0 / channels
mix_filter = f"pan=mono|c0={'+'.join([f'{weight}*c{i}' for i in range(channels)])}"
```

See `_premix_audio()` in `tools/transcribe_watcher.py`.

## Requirements Files

Located in `environments/`:
- `requirements_macOS_arm64.txt` - Apple Silicon Macs
- `requirements_linux.txt` - Linux
- `requirements_win_cpu.txt` - Windows (CPU)
- `requirements_win_cuda.txt` - Windows (NVIDIA GPU)

**Key dependencies:**
- `pyannote.audio>=4` - Speaker diarization
- `faster-whisper` - Transcription
- `torch==2.8` / `torchaudio==2.8` - PyTorch (pyannote 4 not compatible with torch 2.9)
- `omegaconf` - Required for PyTorch 2.6+ safe globals fix

## Git Remotes

This fork setup:
```
origin   → https://github.com/MatthiasHeim/noScribe (this fork)
upstream → https://github.com/kaixxx/noScribe (original)
```

To sync with upstream:
```bash
git fetch upstream
git merge upstream/main
```

# Tool Usage Protocol
- **N8N Operations**: You are FORBIDDEN from calling `n8n_*` tools directly.
- Instead, you MUST delegate all n8n interactions to the `@n8n-expert` subagent.
- Example: "I need to check the Slack node." -> Delegate to `@n8n-expert`: "Check the Slack node configuration in workflow ID 5."
- **Documentation**: The n8n MCP provides `tools_documentation` for MCP usage guides and `get_node` with `mode='docs'` for node-specific markdown documentation. The `@n8n-expert` agent can look up both when needed.

## Models

Whisper models are stored in `models/` directory:
- `models/swiss-german` - Swiss German fine-tuned model (default, best for Swiss German)
- `models/fast` - faster-whisper-int8 (quick transcription)
- `models/precise` - faster-whisper-large-v3-turbo (accurate transcription)

Models are too large for GitHub; see `models/README.md` for download instructions.

## Logs

- noScribe: `~/Library/Application Support/noscribe/log/` (macOS)
- MeetingRecorder watcher: `~/Documents/MeetingRecorder/logs/watcher.log`
- Recordings: `~/Documents/MeetingRecorder/Recordings/`
- Transcripts: `~/Documents/MeetingRecorder/Transcripts/`