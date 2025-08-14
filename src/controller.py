"""Application logic for noScribe PyQt6 interface."""
from pathlib import Path

from translator import t


def handle_audio_file(path: str) -> str:
    """Handle selected audio file and return log message."""
    if path:
        return t("Selected audio file: {name}", name=Path(path).name)
    return t("No audio file selected")


def handle_transcript_file(path: str) -> str:
    """Handle selected transcript file and return log message."""
    if path:
        return t("Selected transcript file: {name}", name=Path(path).name)
    return t("No transcript file selected")


def start_transcription() -> str:
    """Placeholder for starting transcription."""
    return t("Transcription started")


def stop_transcription() -> str:
    """Placeholder for stopping transcription."""
    return t("Transcription stopped")
