"""Application logic for noScribe PyQt6 interface."""
from pathlib import Path


def handle_audio_file(path: str) -> str:
    """Handle selected audio file and return log message."""
    if path:
        return f"Selected audio file: {Path(path).name}"
    return "No audio file selected"


def handle_transcript_file(path: str) -> str:
    """Handle selected transcript file and return log message."""
    if path:
        return f"Selected transcript file: {Path(path).name}"
    return "No transcript file selected"


def start_transcription() -> str:
    """Placeholder for starting transcription."""
    return "Transcription started"


def stop_transcription() -> str:
    """Placeholder for stopping transcription."""
    return "Transcription stopped"
