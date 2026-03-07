#!/usr/bin/env python3
"""
MeetingRecorder - macOS menu bar app for recording meetings

A simple menu bar app that records audio from your microphone (or combined
mic + system audio via BlackHole) and saves it for automatic transcription.

Usage:
    python meeting_recorder.py [--config PATH]
"""

import os
import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
import numpy as np
import sounddevice as sd
import soundfile as sf
import rumps


# Default config path
DEFAULT_CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"

# Menu bar icons (using emoji as fallback)
ICON_IDLE = None  # Will use title instead
ICON_RECORDING = None
TITLE_IDLE = "🎙️"
TITLE_RECORDING = "🔴"


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        # Return defaults if config doesn't exist
        return {
            'audio': {
                'device': 'default',
                'sample_rate': 16000,
                'channels': 1
            },
            'paths': {
                'recordings': '~/Documents/MeetingRecorder/Recordings',
                'transcripts': '~/Documents/MeetingRecorder/Transcripts',
                'logs': '~/Documents/MeetingRecorder/logs'
            }
        }

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_audio_device_index(device_name: str) -> Optional[int]:
    """Get device index by name, or None for default."""
    if device_name.lower() == 'default':
        return None

    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if device_name.lower() in dev['name'].lower():
            return i

    return None


class AudioRecorder:
    """Handles audio recording in a background thread."""

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config['audio'].get('sample_rate', 16000)
        self.channels = config['audio'].get('channels', 1)
        self.device = get_audio_device_index(config['audio'].get('device', 'default'))

        self.recording = False
        self.audio_data = []
        self.stream: Optional[sd.InputStream] = None
        self.output_file: Optional[Path] = None

    def start(self, output_file: Path):
        """Start recording audio to the specified file."""
        if self.recording:
            return False

        self.output_file = output_file
        self.audio_data = []
        self.recording = True

        try:
            self.stream = sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.int16,
                callback=self._audio_callback
            )
            self.stream.start()
            return True
        except Exception as e:
            self.recording = False
            raise RuntimeError(f"Failed to start recording: {e}")

    def stop(self) -> Optional[Path]:
        """Stop recording and save to file."""
        if not self.recording:
            return None

        self.recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_data and self.output_file:
            # Concatenate all audio chunks
            audio_array = np.concatenate(self.audio_data, axis=0)

            # Ensure output directory exists
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save as WAV
            sf.write(
                str(self.output_file),
                audio_array,
                self.sample_rate,
                subtype='PCM_16'
            )

            return self.output_file

        return None

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)

        if self.recording:
            self.audio_data.append(indata.copy())

    @property
    def is_recording(self) -> bool:
        return self.recording


class MeetingRecorderApp(rumps.App):
    """macOS menu bar application for recording meetings."""

    def __init__(self, config: dict, config_path: Path):
        super().__init__(
            name="MeetingRecorder",
            title=TITLE_IDLE,
            icon=ICON_IDLE,
            quit_button=None  # We'll add our own
        )

        self.config = config
        self.config_path = config_path
        self.recordings_dir = expand_path(config['paths']['recordings'])
        self.transcripts_dir = expand_path(config['paths']['transcripts'])

        # Ensure directories exist
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize recorder
        self.recorder = AudioRecorder(config)
        self.recording_start_time: Optional[datetime] = None

        # Build menu
        self._build_menu()

    def _build_menu(self):
        """Build the menu bar menu."""
        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.toggle_recording),
            None,  # Separator
            rumps.MenuItem("Open Recordings Folder", callback=self.open_recordings),
            rumps.MenuItem("Open Transcripts Folder", callback=self.open_transcripts),
            None,  # Separator
            rumps.MenuItem("Preferences...", callback=self.open_preferences),
            rumps.MenuItem("List Audio Devices", callback=self.list_devices),
            None,  # Separator
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

    def toggle_recording(self, sender):
        """Start or stop recording."""
        if self.recorder.is_recording:
            self._stop_recording(sender)
        else:
            self._start_recording(sender)

    def _start_recording(self, sender):
        """Start a new recording."""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = self.recordings_dir / f"{timestamp}.wav"

        try:
            self.recorder.start(output_file)
            self.recording_start_time = datetime.now()

            # Update UI
            self.title = TITLE_RECORDING
            sender.title = "Stop Recording"

            rumps.notification(
                title="MeetingRecorder",
                subtitle="Recording started",
                message=f"Saving to: {output_file.name}"
            )

        except Exception as e:
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Error",
                message=str(e)
            )

    def _stop_recording(self, sender):
        """Stop the current recording."""
        output_file = self.recorder.stop()

        # Calculate duration
        duration = ""
        if self.recording_start_time:
            elapsed = datetime.now() - self.recording_start_time
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            duration = f" ({minutes}m {seconds}s)"

        # Update UI
        self.title = TITLE_IDLE
        sender.title = "Start Recording"
        self.recording_start_time = None

        if output_file and output_file.exists():
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Recording saved" + duration,
                message=f"File: {output_file.name}\nTranscription will start automatically."
            )
        else:
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Recording stopped",
                message="No audio was captured."
            )

    def open_recordings(self, _):
        """Open the recordings folder in Finder."""
        subprocess.run(["open", str(self.recordings_dir)])

    def open_transcripts(self, _):
        """Open the transcripts folder in Finder."""
        subprocess.run(["open", str(self.transcripts_dir)])

    def open_preferences(self, _):
        """Open the config file in the default editor."""
        subprocess.run(["open", str(self.config_path)])

    def list_devices(self, _):
        """Show available audio devices."""
        devices = sd.query_devices()
        input_devices = []

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                marker = " (current)" if i == self.recorder.device else ""
                input_devices.append(f"• {dev['name']}{marker}")

        device_list = "\n".join(input_devices[:10])  # Limit to 10
        if len(input_devices) > 10:
            device_list += f"\n... and {len(input_devices) - 10} more"

        rumps.alert(
            title="Available Audio Input Devices",
            message=device_list or "No input devices found",
            ok="OK"
        )

    def quit_app(self, _):
        """Quit the application, stopping any active recording."""
        if self.recorder.is_recording:
            self.recorder.stop()
        rumps.quit_application()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Menu bar app for recording meetings"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create and run app
    app = MeetingRecorderApp(config, args.config)
    app.run()


if __name__ == "__main__":
    main()
