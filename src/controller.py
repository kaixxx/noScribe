"""Application logic and transcription worker for noScribe PyQt6 interface."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from faster_whisper import WhisperModel

from translator import set_language, t


def _format_timestamp(seconds: float) -> str:
    """Format seconds to hh:mm:ss.mmm."""
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000
    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{secs:02d}.{milliseconds:03d}"


class TranscriptionThread(QThread):
    """Background transcription worker."""

    log = pyqtSignal(str)

    def __init__(
        self,
        audio_path: str,
        transcript_path: str,
        language: str,
        diarization: bool = False,
    ) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.language = language
        self.diarization = diarization
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:  # type: ignore[override]
        try:
            self.log.emit(t("Transcription started"))
            model = WhisperModel(
                "models/fast", device="auto", compute_type="auto", local_files_only=True
            )
            segments, _ = model.transcribe(
                self.audio_path,
                language=None if self.language == "auto" else self.language,
                beam_size=5,
            )
            with open(self.transcript_path, "w", encoding="utf-8") as out:
                for seg in segments:
                    if self._stop_requested:
                        self.log.emit(t("Transcription stopped"))
                        break
                    line = f"[{_format_timestamp(seg.start)} -> {_format_timestamp(seg.end)}] {seg.text}\n"
                    out.write(line)
                    self.log.emit(seg.text)
                else:
                    self.log.emit(t("Transcription finished"))
        except Exception as exc:  # pragma: no cover - runtime safeguard
            self.log.emit(str(exc))


class NoScribeController(QObject):
    """Controller handling user actions and worker management."""

    log = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.audio_file: str = ""
        self.transcript_file: str = ""
        self.language: str = "en"
        self.diarization: bool = False
        self.worker: Optional[TranscriptionThread] = None

    # --------------------- setters ---------------------
    def handle_audio_file(self, path: str) -> None:
        if path:
            self.audio_file = path
            self.log.emit(t("Selected audio file: {name}", name=Path(path).name))
        else:
            self.log.emit(t("No audio file selected"))

    def handle_transcript_file(self, path: str) -> None:
        if path:
            self.transcript_file = path
            self.log.emit(t("Selected transcript file: {name}", name=Path(path).name))
        else:
            self.log.emit(t("No transcript file selected"))

    def set_language(self, lang: str) -> None:
        self.language = lang
        set_language("de" if lang == "de" else "en")

    def set_diarization(self, enabled: bool) -> None:
        self.diarization = enabled

    # --------------------- actions ---------------------
    def start_transcription(self) -> None:
        if not self.audio_file or not self.transcript_file:
            self.log.emit(t("Please select audio and transcript file"))
            return
        if self.worker is None:
            self.worker = TranscriptionThread(
                self.audio_file, self.transcript_file, self.language, self.diarization
            )
            self.worker.log.connect(self.log.emit)
            self.worker.finished.connect(self._worker_finished)
            self.worker.start()

    def _worker_finished(self) -> None:
        self.worker = None

    def stop_transcription(self) -> None:
        if self.worker:
            self.worker.stop()
