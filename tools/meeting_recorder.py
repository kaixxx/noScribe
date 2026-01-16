#!/usr/bin/env python3
"""
MeetingRecorder - macOS menu bar app for recording meetings

A simple menu bar app that records audio from your microphone (or combined
mic + system audio via BlackHole) and saves it for automatic transcription.

Features:
- Manual start/stop recording
- Automatic Teams meeting detection (optional)
- Preferences GUI for configuration

Usage:
    python meeting_recorder.py [--config PATH]
"""

import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import rumps
import sounddevice as sd
import soundfile as sf
import yaml

from teams_detector import TeamsDetector, check_screen_recording_permission


# Default config path
DEFAULT_CONFIG_PATH = Path.home() / "Documents" / "MeetingRecorder" / "config.yaml"

# Menu bar icons (using emoji as fallback)
ICON_IDLE = None  # Will use title instead
ICON_RECORDING = None
TITLE_IDLE = "🎙️"
TITLE_RECORDING = "🔴"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MeetingRecorder")


def expand_path(path: str) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    defaults = {
        'audio': {
            'device': 'default',
            'sample_rate': 16000,
            'channels': 1
        },
        'paths': {
            'recordings': '~/Documents/MeetingRecorder/Recordings',
            'transcripts': '~/Documents/MeetingRecorder/Transcripts',
            'logs': '~/Documents/MeetingRecorder/logs'
        },
        'teams': {
            'auto_record': False,
            'confirm_before_record': True,
            'grace_period': 10,
            'poll_interval': 3
        }
    }

    if not config_path.exists():
        return defaults

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Merge with defaults (config values override defaults)
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue

    return config


def save_config(config: dict, config_path: Path):
    """Save configuration to YAML file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


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

        # Teams detection
        self.teams_detector: Optional[TeamsDetector] = None
        self._pending_meeting_title: Optional[str] = None
        self._auto_recording = False  # Track if current recording was auto-started

        # Build menu
        self._build_menu()

        # Start Teams detector if enabled
        if self.config['teams'].get('auto_record', False):
            self._start_teams_detector()

    def _build_menu(self):
        """Build the menu bar menu."""
        teams_enabled = self.config['teams'].get('auto_record', False)
        teams_label = "Teams Auto-Record: ON" if teams_enabled else "Teams Auto-Record: OFF"

        self.teams_menu_item = rumps.MenuItem(teams_label, callback=self.toggle_teams_detection)

        self.menu = [
            rumps.MenuItem("Start Recording", callback=self.toggle_recording),
            None,  # Separator
            self.teams_menu_item,
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

    def _start_recording_internal(self):
        """Start recording without a menu sender (for auto-start)."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = self.recordings_dir / f"{timestamp}.wav"

        try:
            self.recorder.start(output_file)
            self.recording_start_time = datetime.now()
            self._auto_recording = True

            # Update UI
            self.title = TITLE_RECORDING
            self.menu["Start Recording"].title = "Stop Recording"

            return True
        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            return False

    def _stop_recording(self, sender):
        """Stop the current recording."""
        output_file = self.recorder.stop()
        self._auto_recording = False

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

    def _stop_recording_internal(self):
        """Stop recording without a menu sender (for auto-stop)."""
        output_file = self.recorder.stop()
        was_auto = self._auto_recording
        self._auto_recording = False

        duration = ""
        if self.recording_start_time:
            elapsed = datetime.now() - self.recording_start_time
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            duration = f" ({minutes}m {seconds}s)"

        self.title = TITLE_IDLE
        self.menu["Start Recording"].title = "Start Recording"
        self.recording_start_time = None

        if output_file and output_file.exists() and was_auto:
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Auto-recording saved" + duration,
                message=f"File: {output_file.name}"
            )

    # --- Teams Detection ---

    def _start_teams_detector(self):
        """Initialize and start the Teams detector."""
        if self.teams_detector is not None:
            return

        # Check screen recording permission first
        if not check_screen_recording_permission():
            logger.warning("Screen recording permission may not be granted")

        teams_config = self.config.get('teams', {})
        self.teams_detector = TeamsDetector(
            on_meeting_start=self._on_teams_meeting_start,
            on_meeting_end=self._on_teams_meeting_end,
            poll_interval=teams_config.get('poll_interval', 3),
            grace_period=teams_config.get('grace_period', 10),
            logger=logger
        )
        self.teams_detector.start()

    def _stop_teams_detector(self):
        """Stop the Teams detector."""
        if self.teams_detector:
            self.teams_detector.stop()
            self.teams_detector = None

    def _on_teams_meeting_start(self, meeting_title: str):
        """Called when Teams meeting starts (from background thread)."""
        if self.recorder.is_recording:
            logger.info("Meeting detected but already recording, skipping auto-start")
            return

        self._pending_meeting_title = meeting_title

        if self.config['teams'].get('confirm_before_record', True):
            # Schedule confirmation dialog on main thread
            rumps.Timer(self._show_meeting_confirmation, 0).start()
        else:
            # Auto-start immediately
            rumps.Timer(self._auto_start_recording, 0).start()

    def _show_meeting_confirmation(self, _):
        """Show confirmation dialog for auto-recording (main thread)."""
        title = self._pending_meeting_title or "Teams Meeting"
        response = rumps.alert(
            title="Teams Meeting Detected",
            message=f"Start recording?\n\n{title}",
            ok="Start Recording",
            cancel="Don't Record"
        )
        if response == 1:  # OK clicked
            if self._start_recording_internal():
                rumps.notification(
                    title="MeetingRecorder",
                    subtitle="Recording Started",
                    message="Teams meeting detected"
                )

    def _auto_start_recording(self, _):
        """Auto-start recording without confirmation (main thread)."""
        if not self.recorder.is_recording:
            if self._start_recording_internal():
                rumps.notification(
                    title="MeetingRecorder",
                    subtitle="Auto-Recording Started",
                    message="Teams meeting detected"
                )

    def _on_teams_meeting_end(self):
        """Called when Teams meeting ends (from background thread)."""
        if self.recorder.is_recording and self._auto_recording:
            # Only auto-stop if we auto-started
            rumps.Timer(self._auto_stop_recording, 0).start()

    def _auto_stop_recording(self, _):
        """Auto-stop recording (main thread)."""
        if self.recorder.is_recording and self._auto_recording:
            self._stop_recording_internal()

    def toggle_teams_detection(self, sender):
        """Toggle Teams auto-detection on/off."""
        currently_enabled = self.config['teams'].get('auto_record', False)

        if currently_enabled:
            # Disable
            self._stop_teams_detector()
            self.config['teams']['auto_record'] = False
            sender.title = "Teams Auto-Record: OFF"
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Teams Auto-Record Disabled",
                message="Manual recording only"
            )
        else:
            # Enable
            self.config['teams']['auto_record'] = True
            self._start_teams_detector()
            sender.title = "Teams Auto-Record: ON"
            rumps.notification(
                title="MeetingRecorder",
                subtitle="Teams Auto-Record Enabled",
                message="Will auto-record when Teams meetings start"
            )

        # Save config
        save_config(self.config, self.config_path)

    # --- Preferences ---

    def open_preferences(self, _):
        """Open preferences dialog."""
        self._show_preferences_main()

    def _show_preferences_main(self):
        """Show main preferences menu."""
        response = rumps.alert(
            title="MeetingRecorder Preferences",
            message="Choose a category to configure:",
            ok="Teams Settings",
            cancel="Close",
            other="Open Config File"
        )

        if response == 1:  # Teams Settings
            self._show_teams_preferences()
        elif response == 0:  # Open Config File
            subprocess.run(["open", str(self.config_path)])

    def _show_teams_preferences(self):
        """Show Teams-specific preferences."""
        teams_config = self.config.get('teams', {})
        auto_record = teams_config.get('auto_record', False)
        confirm = teams_config.get('confirm_before_record', True)
        grace_period = teams_config.get('grace_period', 10)

        # Toggle auto-record
        status = "ON" if auto_record else "OFF"
        response = rumps.alert(
            title="Teams Auto-Record",
            message=f"Current status: {status}\n\nEnable automatic recording when Teams meetings are detected?",
            ok="Enable",
            cancel="Disable"
        )
        new_auto_record = response == 1

        # Toggle confirmation
        if new_auto_record:
            status = "ON" if confirm else "OFF"
            response = rumps.alert(
                title="Confirmation Dialog",
                message=f"Current status: {status}\n\nShow confirmation dialog before auto-starting recording?",
                ok="Yes, ask me",
                cancel="No, auto-start silently"
            )
            new_confirm = response == 1

            # Grace period
            window = rumps.Window(
                title="Grace Period",
                message="Seconds to wait before stopping after meeting ends.\n(Prevents premature stop if you briefly leave and rejoin)",
                default_text=str(grace_period),
                ok="Save",
                cancel="Cancel",
                dimensions=(100, 24)
            )
            result = window.run()

            if result.clicked == 1:
                try:
                    new_grace = max(0, min(60, int(result.text)))
                except ValueError:
                    new_grace = grace_period
            else:
                new_grace = grace_period
        else:
            new_confirm = confirm
            new_grace = grace_period

        # Apply changes
        old_auto_record = self.config['teams'].get('auto_record', False)
        self.config['teams']['auto_record'] = new_auto_record
        self.config['teams']['confirm_before_record'] = new_confirm
        self.config['teams']['grace_period'] = new_grace

        # Update detector if auto_record changed
        if new_auto_record != old_auto_record:
            if new_auto_record:
                self._start_teams_detector()
                self.teams_menu_item.title = "Teams Auto-Record: ON"
            else:
                self._stop_teams_detector()
                self.teams_menu_item.title = "Teams Auto-Record: OFF"

        # Save config
        save_config(self.config, self.config_path)

        rumps.notification(
            title="MeetingRecorder",
            subtitle="Preferences Saved",
            message="Teams settings updated"
        )

    # --- Other Menu Actions ---

    def open_recordings(self, _):
        """Open the recordings folder in Finder."""
        subprocess.run(["open", str(self.recordings_dir)])

    def open_transcripts(self, _):
        """Open the transcripts folder in Finder."""
        subprocess.run(["open", str(self.transcripts_dir)])

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
        # Stop Teams detector
        self._stop_teams_detector()

        # Stop recording if active
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
