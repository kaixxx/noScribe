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


def _parse_timecode(value: str) -> Optional[float]:
    """Parse hh:mm:ss string into seconds."""
    try:
        parts = [int(p) for p in value.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    except Exception:
        return None


class TranscriptionThread(QThread):
    """Background transcription worker."""

    log = pyqtSignal(str)

    def __init__(
        self,
        audio_path: str,
        transcript_path: str,
        language: str,
        diarization: bool = False,
        start_time: float = 0.0,
        stop_time: float = float("inf"),
    ) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.language = language
        self.diarization = diarization
        self.start_time = start_time
        self.stop_time = stop_time
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
                    if seg.end < self.start_time or seg.start > self.stop_time:
                        continue
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
    def start_transcription(self, start: str, stop: str) -> None:
        if not self.audio_file or not self.transcript_file:
            self.log.emit(t("Please select audio and transcript file"))
            return
        start_sec = _parse_timecode(start)
        if start_sec is None:
            self.log.emit(t("Invalid time format: {time}", time=start))
            start_sec = 0.0
        stop_sec = _parse_timecode(stop)
        if stop_sec is None:
            self.log.emit(t("Invalid time format: {time}", time=stop))
            stop_sec = float("inf")
        if self.worker is None:
            self.worker = TranscriptionThread(
                self.audio_file,
                self.transcript_file,
                self.language,
                self.diarization,
                start_sec,
                stop_sec,
            )
            self.worker.log.connect(self.log.emit)
            self.worker.finished.connect(self._worker_finished)
            self.worker.start()

    def _worker_finished(self) -> None:
        self.worker = None

    def stop_transcription(self) -> None:
        if self.worker:
            self.worker.stop()
