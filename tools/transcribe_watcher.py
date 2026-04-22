#!/usr/bin/env python3
"""
TranscribeWatcher - Automatic transcription service for MeetingRecorder

Monitors a folder for new audio files and automatically transcribes them
using either noScribe (Whisper) or Gemini Flash, then sends results to n8n webhook.

Processing modes:
- "whisper": Traditional noScribe transcription → n8n analysis
- "gemini": Direct Gemini 2.5 Flash transcription + analysis
- "both": Run both pipelines in parallel for comparison

Usage:
    python transcribe_watcher.py [--config PATH]
"""

import os
import sys
import time
import json
import queue
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import yaml
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from project root .env file
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# Add tools directory to path for sibling module imports
_tools_dir = Path(__file__).parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

# Import Gemini processing modules (optional - gracefully handle if not available)
try:
    from audio_converter import convert_for_gemini, get_audio_duration
    from gemini_processor import GeminiAudioProcessor, GeminiResult
    GEMINI_AVAILABLE = True
except ImportError as e:
    GEMINI_AVAILABLE = False
    _GEMINI_IMPORT_ERROR = str(e)


# Default config path
DEFAULT_CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "watcher.log"

    logger = logging.getLogger("TranscribeWatcher")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def html_to_text(html_content: str) -> str:
    """Extract plain text from HTML transcript."""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_content)
    # Decode HTML entities
    import html
    text = html.unescape(text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TranscriptionQueue:
    """Thread-safe queue for managing transcription jobs."""

    def __init__(self, logger: logging.Logger):
        self.queue = queue.Queue()
        self.processing = False
        self.current_file: Optional[Path] = None
        self.logger = logger

    def add(self, audio_file: Path):
        """Add a file to the transcription queue."""
        self.queue.put(audio_file)
        self.logger.info(f"Queued for transcription: {audio_file.name}")

    def get(self) -> Optional[Path]:
        """Get next file from queue, non-blocking."""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def is_empty(self) -> bool:
        return self.queue.empty()


class AudioFileHandler(FileSystemEventHandler):
    """Handles new audio file events."""

    def __init__(self, transcription_queue: TranscriptionQueue,
                 debounce_seconds: float, logger: logging.Logger):
        self.queue = transcription_queue
        self.debounce_seconds = debounce_seconds
        self.logger = logger
        self.pending_files = {}  # file_path -> timer

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process WAV files
        if file_path.suffix.lower() != '.wav':
            return

        self.logger.debug(f"File detected: {file_path.name}")

        # Cancel existing timer for this file
        if file_path in self.pending_files:
            self.pending_files[file_path].cancel()

        # Set debounce timer
        timer = threading.Timer(
            self.debounce_seconds,
            self._add_to_queue,
            args=[file_path]
        )
        self.pending_files[file_path] = timer
        timer.start()

    def _add_to_queue(self, file_path: Path):
        """Add file to queue after debounce period."""
        if file_path in self.pending_files:
            del self.pending_files[file_path]

        # Verify file still exists and has content
        if file_path.exists() and file_path.stat().st_size > 0:
            self.queue.add(file_path)
        else:
            self.logger.warning(f"File no longer exists or is empty: {file_path.name}")


class TranscribeWatcher:
    """Main watcher service that monitors folder and processes transcriptions."""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Expand paths
        self.recordings_dir = expand_path(config['paths']['recordings'])
        self.transcripts_dir = expand_path(config['paths']['transcripts'])
        self.noscribe_path = Path(config['noscribe']['path'])

        # Ensure directories exist
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        # Processing mode: "whisper", "gemini", or "both"
        self.processing_mode = config.get('processing', {}).get('mode', 'whisper')
        self.logger.info(f"Processing mode: {self.processing_mode}")

        # Validate Gemini availability if needed
        if self.processing_mode in ('gemini', 'both') and not GEMINI_AVAILABLE:
            self.logger.warning(
                "Gemini processing requested but modules not available. "
                "Falling back to whisper mode. Install: pip install google-genai"
            )
            self.processing_mode = 'whisper'

        # Initialize Gemini processor if needed
        self.gemini_processor = None
        if self.processing_mode in ('gemini', 'both') and GEMINI_AVAILABLE:
            gemini_config = config.get('gemini', {})
            api_key = os.environ.get(gemini_config.get('api_key_env', 'GEMINI_API_KEY'))
            if api_key:
                self.gemini_processor = GeminiAudioProcessor(
                    api_key=api_key,
                    model=gemini_config.get('model', 'gemini-2.5-flash'),
                    max_output_tokens=gemini_config.get('max_output_tokens', 65536),
                    temperature=gemini_config.get('temperature', 0.1),
                    timeout_seconds=gemini_config.get('timeout', 600)
                )
                self.logger.info(f"Gemini processor initialized with model: {gemini_config.get('model', 'gemini-2.5-flash')}")
            else:
                self.logger.error(f"GEMINI_API_KEY not found in environment. Gemini processing disabled.")
                if self.processing_mode == 'gemini':
                    self.processing_mode = 'whisper'

        # Initialize queue
        self.queue = TranscriptionQueue(logger)

        # Set up file watcher
        debounce = config.get('watcher', {}).get('debounce_seconds', 2)
        self.handler = AudioFileHandler(self.queue, debounce, logger)
        self.observer = Observer()

        self.running = False

    def start(self):
        """Start watching for new files."""
        self.logger.info(f"Starting TranscribeWatcher...")
        self.logger.info(f"Watching: {self.recordings_dir}")
        self.logger.info(f"Transcripts: {self.transcripts_dir}")

        # Check for existing unprocessed files
        self._process_existing_files()

        # Start file watcher
        self.observer.schedule(self.handler, str(self.recordings_dir), recursive=False)
        self.observer.start()

        self.running = True
        self.logger.info("Watcher started. Press Ctrl+C to stop.")

        # Process queue in main loop
        try:
            while self.running:
                self._process_queue()
                time.sleep(self.config.get('watcher', {}).get('poll_interval', 1))
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the watcher."""
        self.logger.info("Stopping watcher...")
        self.running = False

        # Terminate any running Claude sessions
        for proc in getattr(self, '_claude_pids', []):
            if proc.poll() is None:
                self.logger.info(f"Terminating Claude session PID={proc.pid}")
                proc.terminate()

        self.observer.stop()
        self.observer.join()
        self.logger.info("Watcher stopped.")

    def _process_existing_files(self):
        """Check for WAV files that don't have corresponding outputs."""
        for wav_file in self.recordings_dir.glob("*.wav"):
            needs_processing = False

            if self.processing_mode in ('whisper', 'both'):
                # Check for Whisper HTML transcript
                transcript_file = self.transcripts_dir / f"{wav_file.stem}.html"
                if not transcript_file.exists():
                    needs_processing = True

            if self.processing_mode in ('gemini', 'both'):
                # Check for Gemini JSON output
                gemini_json = self.transcripts_dir / f"{wav_file.stem}.json"
                if not gemini_json.exists():
                    needs_processing = True

            if needs_processing:
                self.logger.info(f"Found unprocessed file: {wav_file.name}")
                self.queue.add(wav_file)

    def _process_queue(self):
        """Process the next file in the queue based on processing mode."""
        audio_file = self.queue.get()
        if audio_file is None:
            return

        self.queue.current_file = audio_file
        self.queue.processing = True

        try:
            if self.processing_mode == 'whisper':
                self._process_with_whisper(audio_file)
            elif self.processing_mode == 'gemini':
                self._process_with_gemini(audio_file)
            elif self.processing_mode == 'both':
                # Run both pipelines
                self.logger.info(f"Running both pipelines for: {audio_file.name}")
                self._process_with_whisper(audio_file)
                self._process_with_gemini(audio_file)
            else:
                self.logger.error(f"Unknown processing mode: {self.processing_mode}")
        except Exception as e:
            self.logger.error(f"Error processing {audio_file.name}: {e}")
        finally:
            self.queue.processing = False
            self.queue.current_file = None

    def _premix_audio(self, audio_file: Path) -> Path:
        """Pre-mix multi-channel audio to mono, dropping silent channels.

        Background: the macOS capture pipeline sometimes writes a 3-channel WAV
        where only one channel contains audio (observed with BlackHole routing:
        audio lands in channel 2 (LFE); channels 0/1 are digital silence). A
        naive equal-weight mix across all channels attenuates the signal by
        ~10dB and pushes Swiss German ASR below its accuracy threshold.

        Strategy: per-channel volumedetect → keep channels with mean_volume
        above SILENCE_THRESHOLD_DB → mix only those channels.
        """
        import tempfile

        ffprobe_path = '/opt/homebrew/bin/ffprobe'
        ffmpeg_path = '/opt/homebrew/bin/ffmpeg'
        SILENCE_THRESHOLD_DB = -60.0  # channels below this are treated as silent

        # Check channel count
        probe_cmd = [ffprobe_path, '-v', 'error', '-select_streams', 'a:0',
                     '-show_entries', 'stream=channels', '-of', 'csv=p=0',
                     str(audio_file)]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            channels = int(result.stdout.strip())
            self.logger.debug(f"Detected {channels} channels in {audio_file.name}")
        except Exception as e:
            self.logger.warning(f"ffprobe failed: {e}, assuming 2 channels")
            channels = 2

        if channels <= 1:
            return audio_file

        # Measure per-channel mean volume (probe first 60s for speed)
        active_channels = []
        for i in range(channels):
            probe = subprocess.run(
                [ffmpeg_path, '-i', str(audio_file), '-t', '60',
                 '-af', f'pan=mono|c0=c{i},volumedetect', '-f', 'null', '-'],
                capture_output=True, text=True,
            )
            import re
            m = re.search(r'mean_volume:\s*(-?[0-9.]+)\s*dB', probe.stderr)
            mean_db = float(m.group(1)) if m else float('-inf')
            self.logger.debug(f"  channel {i}: mean_volume={mean_db} dB")
            if mean_db > SILENCE_THRESHOLD_DB:
                active_channels.append(i)

        if not active_channels:
            self.logger.warning(f"No active channels detected in {audio_file.name}, "
                                f"falling back to equal-weight mix of all {channels}")
            active_channels = list(range(channels))

        if channels == 2 and active_channels == [0, 1]:
            # Standard stereo with audio on both channels — noScribe's -ac 1 handles this fine
            return audio_file

        self.logger.info(
            f"Pre-mixing to mono: {channels} channels detected, "
            f"using active channels {active_channels}"
        )

        temp_dir = Path(tempfile.gettempdir())
        mixed_file = temp_dir / f"{audio_file.stem}_mixed.wav"

        weight = 1.0 / len(active_channels)
        mix_filter = (
            f"pan=mono|c0=" + '+'.join([f'{weight}*c{i}' for i in active_channels])
        )

        mix_cmd = [
            ffmpeg_path, '-y', '-i', str(audio_file),
            '-af', mix_filter,
            '-ar', '48000',
            '-c:a', 'pcm_s16le',
            str(mixed_file),
        ]

        try:
            result = subprocess.run(mix_cmd, capture_output=True, text=True)
            if result.returncode == 0 and mixed_file.exists():
                self.logger.debug(f"Pre-mixed audio saved to: {mixed_file}")
                return mixed_file
            else:
                self.logger.warning(f"Pre-mix failed, using original: {result.stderr}")
                return audio_file
        except Exception as e:
            self.logger.warning(f"Pre-mix error, using original: {e}")
            return audio_file

    def _process_with_whisper(self, audio_file: Path):
        """Transcribe a single audio file using noScribe (Whisper)."""
        transcript_file = self.transcripts_dir / f"{audio_file.stem}.html"

        self.logger.info(f"Starting transcription: {audio_file.name}")
        start_time = time.time()

        # Pre-mix multi-channel audio to ensure all channels are transcribed
        processed_audio = self._premix_audio(audio_file)

        # Build noScribe command
        cmd = [
            sys.executable,
            str(self.noscribe_path),
            str(processed_audio),
            str(transcript_file),
            "--no-gui",
            "--language", self.config['noscribe'].get('language', 'auto'),
            "--speaker-detection", str(self.config['noscribe'].get('speaker_detection', 'auto')),
        ]

        # Add optional flags
        if self.config['noscribe'].get('timestamps', True):
            cmd.append("--timestamps")

        pause = self.config['noscribe'].get('pause', 'none')
        if pause and pause != 'none':
            cmd.extend(["--pause", pause])

        model = self.config['noscribe'].get('model')
        if model:
            cmd.extend(["--model", model])

        self.logger.debug(f"Command: {' '.join(cmd)}")

        # Run noScribe
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.noscribe_path.parent
            )

            if result.returncode != 0:
                self.logger.error(f"noScribe failed with exit code {result.returncode}")
                self.logger.error(f"stdout: {result.stdout}")
                self.logger.error(f"stderr: {result.stderr}")
                return

        except Exception as e:
            self.logger.error(f"Failed to run noScribe: {e}")
            return

        processing_time = time.time() - start_time
        self.logger.info(f"Transcription complete: {transcript_file.name} ({processing_time:.1f}s)")

        # Cleanup temp mixed audio file if created
        if processed_audio != audio_file and processed_audio.exists():
            try:
                processed_audio.unlink()
                self.logger.debug(f"Cleaned up temp file: {processed_audio}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp file: {e}")

        # Send webhook notification
        if self.config.get('webhook', {}).get('enabled', False):
            self._send_webhook(audio_file, transcript_file, processing_time)

        # Trigger Claude for immediate processing
        self._trigger_claude(transcript_file)

    def _send_webhook(self, audio_file: Path, transcript_file: Path, processing_time: float):
        """Send transcription result to n8n webhook.

        Sends payload matching n8n workflow expectations:
        {
            "transcript_path": "/path/to/transcript.html",
            "transcript_html": "<html>...</html>",
            "started_at": "2025-12-05T15:56:47Z",
            "audio_duration_seconds": 2279
        }
        """
        webhook_url = self.config.get('webhook', {}).get('url', '')

        if not webhook_url:
            self.logger.debug("Webhook URL not configured, skipping notification")
            return

        try:
            # Read transcript HTML content
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_html = f.read()

            # Get audio duration (approximate from file size, assuming 16kHz mono 16-bit)
            audio_size = audio_file.stat().st_size
            # WAV header is ~44 bytes, 16kHz * 2 bytes = 32000 bytes/second
            duration_seconds = max(0, (audio_size - 44) / 32000)

            # Parse start time from filename (format: YYYY-MM-DD_HH-MM-SS.wav)
            # Example: 2025-12-05_15-56-47.wav
            # Note: Filename timestamp is in LOCAL time, need to convert to UTC
            filename_without_ext = audio_file.stem
            try:
                date_part, time_part = filename_without_ext.split('_')
                year, month, day = date_part.split('-')
                hour, minute, second = time_part.split('-')
                # Parse as local time (no timezone = naive datetime interpreted as local)
                local_dt = datetime(int(year), int(month), int(day),
                                   int(hour), int(minute), int(second))
                # Convert local time to UTC
                # astimezone() treats naive datetime as local time and converts to UTC
                utc_dt = local_dt.astimezone(timezone.utc)
                started_at = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                self.logger.debug(f"Parsed timestamp: local={local_dt}, utc={started_at}")
            except (ValueError, IndexError) as e:
                # Fallback: use current time if filename doesn't match expected format
                self.logger.warning(f"Could not parse timestamp from filename: {filename_without_ext} ({e})")
                started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Build payload matching n8n workflow expectations
            payload = {
                "transcript_path": str(transcript_file),
                "transcript_html": transcript_html,
                "started_at": started_at,
                "audio_duration_seconds": round(duration_seconds)
            }

            timeout = self.config.get('webhook', {}).get('timeout', 30)
            response = requests.post(webhook_url, json=payload, timeout=timeout)
            response.raise_for_status()

            self.logger.info(f"✅ Webhook sent successfully to n8n: {response.status_code}")
            self.logger.debug(f"Payload sent (HTML content: {len(transcript_html)} chars)")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Webhook request failed: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error preparing webhook: {e}")

    def _process_with_gemini(self, audio_file: Path):
        """Process audio file using Gemini 2.5 Flash for transcription + analysis."""
        if not self.gemini_processor:
            self.logger.error("Gemini processor not initialized, skipping Gemini processing")
            return

        self.logger.info(f"Starting Gemini processing: {audio_file.name}")
        start_time = time.time()

        try:
            # Step 1: Convert WAV to MP3 (extracts mic channel)
            self.logger.info("Converting WAV to MP3...")
            mp3_path = convert_for_gemini(audio_file, output_dir=self.transcripts_dir)
            self.logger.info(f"Converted to: {mp3_path.name} ({mp3_path.stat().st_size / 1024 / 1024:.1f} MB)")

            # Step 2: Get audio duration
            audio_duration = get_audio_duration(mp3_path)
            self.logger.info(f"Audio duration: {audio_duration / 60:.1f} minutes")

            # Step 3: Process with Gemini
            self.logger.info("Sending to Gemini API...")
            result = self.gemini_processor.process_audio(mp3_path)

            processing_time = time.time() - start_time
            self.logger.info(f"Gemini processing complete in {processing_time:.1f}s")

            if result.error:
                self.logger.error(f"Gemini processing error: {result.error}")
                return

            # Log some stats
            if result.input_tokens and result.output_tokens:
                self.logger.info(f"Tokens - Input: {result.input_tokens}, Output: {result.output_tokens}")

            # Step 4: Save JSON result alongside MP3
            json_path = mp3_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result.parsed_response, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved result to: {json_path.name}")

            # Step 5: Send webhook if configured
            if self.config.get('webhook_gemini', {}).get('enabled', False):
                self._send_gemini_webhook(audio_file, mp3_path, result, audio_duration, processing_time)

            # Trigger Claude for immediate processing
            self._trigger_claude(json_path)

        except Exception as e:
            self.logger.error(f"Gemini processing failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def _send_gemini_webhook(self, audio_file: Path, mp3_path: Path,
                             result: 'GeminiResult', audio_duration: float,
                             processing_time: float):
        """Send Gemini transcription result to n8n webhook.

        Sends payload optimized for the conversations_gemini table:
        {
            "source_audio_path": "/path/to/original.wav",
            "source_mp3_path": "/path/to/converted.mp3",
            "started_at": "2025-01-01T00:00:00Z",
            "duration_seconds": 1800,
            "processing_time_seconds": 45.2,
            "gemini_model": "gemini-2.5-flash",
            "gemini_input_tokens": 57600,
            "gemini_output_tokens": 8500,
            "transcript_text": "Full transcript...",
            "transcript_language": "de",
            "title": "Meeting Title",
            "summary": "Meeting summary...",
            "key_points": ["point1", "point2"],
            "tags": ["tag1", "tag2"],
            "participants": {...},
            "sentiment": "positive",
            "meeting_type": "client_call",
            "lailix_communication_score": 7,
            "lailix_communication_feedback": "...",
            "lailix_sales_score": 6,
            "lailix_sales_feedback": "...",
            "lailix_strategic_alignment": "...",
            "lailix_improvement_areas": ["area1", "area2"],
            "lailix_strengths": ["strength1", "strength2"]
        }
        """
        webhook_url = self.config.get('webhook_gemini', {}).get('url', '')

        if not webhook_url:
            self.logger.debug("Gemini webhook URL not configured, skipping notification")
            return

        try:
            # Parse start time from filename
            filename_without_ext = audio_file.stem
            try:
                date_part, time_part = filename_without_ext.split('_')
                year, month, day = date_part.split('-')
                hour, minute, second = time_part.split('-')
                local_dt = datetime(int(year), int(month), int(day),
                                   int(hour), int(minute), int(second))
                utc_dt = local_dt.astimezone(timezone.utc)
                started_at = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, IndexError):
                started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Extract data from Gemini result
            data = result.parsed_response or {}
            lailix_feedback = data.get('lailix_feedback', {})

            # Build payload for conversations_gemini table
            payload = {
                # Source info
                "source_audio_path": str(audio_file),
                "source_mp3_path": str(mp3_path),
                "started_at": started_at,
                "duration_seconds": round(audio_duration),

                # Processing metadata
                "processing_time_seconds": round(processing_time, 2),
                "gemini_model": result.model,
                "gemini_input_tokens": result.input_tokens,
                "gemini_output_tokens": result.output_tokens,

                # Transcript
                "transcript_text": data.get('transcript', ''),
                "transcript_language": data.get('language', ''),

                # Standard metadata
                "title": data.get('title', ''),
                "summary": data.get('summary', ''),
                "key_points": data.get('key_points', []),
                "tags": data.get('tags', []),
                "participants": data.get('participants', {}),
                "sentiment": data.get('sentiment', ''),
                "meeting_type": data.get('meeting_type', ''),

                # Lailix-specific feedback
                "lailix_communication_score": lailix_feedback.get('communication_score'),
                "lailix_communication_feedback": lailix_feedback.get('communication_feedback', ''),
                "lailix_sales_score": lailix_feedback.get('sales_score'),
                "lailix_sales_feedback": lailix_feedback.get('sales_feedback', ''),
                "lailix_strategic_alignment": lailix_feedback.get('strategic_alignment', ''),
                "lailix_improvement_areas": lailix_feedback.get('improvement_areas', []),
                "lailix_strengths": lailix_feedback.get('strengths', []),

                # Raw response for debugging
                "gemini_raw_response": data
            }

            timeout = self.config.get('webhook_gemini', {}).get('timeout', 60)
            response = requests.post(webhook_url, json=payload, timeout=timeout)
            response.raise_for_status()

            self.logger.info(f"✅ Gemini webhook sent successfully: {response.status_code}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Gemini webhook request failed: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error preparing Gemini webhook: {e}")

    def _trigger_claude(self, transcript_path: Path):
        """Fire-and-forget headless Claude session to process transcript."""
        claude_config = self.config.get('claude_trigger', {})
        if not claude_config.get('enabled', False):
            self.logger.debug("Claude trigger disabled, skipping")
            return

        claude_path = claude_config.get('claude_path', '/Users/Matthias/.local/bin/claude')
        brain_repo = claude_config.get('brain_repo', '/Users/Matthias/Desktop/Repos/Brain')
        command = claude_config.get('command', 'meeting-actions')

        prompt = f"Read .claude/commands/{command}.md and process transcript: {transcript_path}"

        # Log file for this Claude session
        log_dir = expand_path(self.config['paths']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_dir / f"claude-{timestamp}.log"

        env = os.environ.copy()
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/Users/Matthias/.local/bin"

        try:
            with open(log_file, 'w') as lf:
                proc = subprocess.Popen(
                    [claude_path, "-p", prompt, "--dangerously-skip-permissions"],
                    cwd=brain_repo,
                    env=env,
                    stdout=lf,
                    stderr=lf,
                )
            self.logger.info(f"Claude trigger fired: PID={proc.pid}, log={log_file.name}")
            if not hasattr(self, '_claude_pids'):
                self._claude_pids = []
            self._claude_pids.append(proc)
        except Exception as e:
            self.logger.error(f"Failed to trigger Claude: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Watch for audio files and automatically transcribe them"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})"
    )
    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Set up logging
    log_dir = expand_path(config['paths']['logs'])
    logger = setup_logging(log_dir)

    logger.info(f"Config loaded from: {args.config}")

    # Create and start watcher
    watcher = TranscribeWatcher(config, logger)
    watcher.start()


if __name__ == "__main__":
    main()
