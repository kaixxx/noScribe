"""UI-independent inference requests, results, and local worker backend."""

from __future__ import annotations

import multiprocessing as mp
import queue as pyqueue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional


LogCallback = Callable[[str, str], None]
TranscriptionProgressCallback = Callable[[float, Optional[str]], None]
DiarizationProgressCallback = Callable[[str, int], None]
CancelCallback = Callable[[], bool]


def _ignore_log(level: str, message: str) -> None:
    pass


def _ignore_transcription_progress(percent: float, detail: Optional[str]) -> None:
    pass


def _ignore_diarization_progress(step: str, percent: int) -> None:
    pass


def _never_cancel() -> bool:
    return False


@dataclass(frozen=True)
class TranscriptionWord:
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    probability: Optional[float] = None

    @classmethod
    def from_mapping(cls, value: dict) -> "TranscriptionWord":
        return cls(
            word=str(value.get("word") or ""),
            start=value.get("start"),
            end=value.get("end"),
            probability=value.get("prob"),
        )


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    words: tuple[TranscriptionWord, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: dict) -> "TranscriptionSegment":
        return cls(
            start=float(value.get("start") or 0.0),
            end=float(value.get("end") or 0.0),
            text=str(value.get("text") or ""),
            words=tuple(
                TranscriptionWord.from_mapping(word)
                for word in (value.get("words") or [])
            ),
        )


@dataclass(frozen=True)
class TranscriptionInfo:
    duration: Optional[float] = None
    language: Optional[str] = None
    language_probability: Optional[float] = None
    sample_rate: Optional[int] = None

    @classmethod
    def from_mapping(cls, value: dict) -> "TranscriptionInfo":
        return cls(
            duration=value.get("duration"),
            language=value.get("language"),
            language_probability=value.get("language_probability"),
            sample_rate=value.get("sample_rate"),
        )


@dataclass(frozen=True)
class DiarizationSegment:
    start_ms: int
    end_ms: int
    label: str

    @classmethod
    def from_mapping(cls, value: dict) -> "DiarizationSegment":
        return cls(
            start_ms=int(value.get("start") or 0),
            end_ms=int(value.get("end") or 0),
            label=str(value.get("label") or ""),
        )


@dataclass(frozen=True)
class TranscriptionRequest:
    audio_path: str
    model_path: str
    language_name: str
    language_code: Optional[str]
    device: str = "auto"
    compute_type: str = "default"
    cpu_threads: int = 4
    local_files_only: bool = True
    disfluencies: bool = True
    beam_size: int = 5
    word_timestamps: bool = True
    vad_filter: bool = True
    vad_threshold: float = 0.5
    locale: str = "en"

    def to_worker_args(self) -> dict:
        return {
            "model_path": self.model_path,
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "local_files_only": self.local_files_only,
            "audio_path": self.audio_path,
            "language_name": self.language_name,
            "language_code": self.language_code,
            "disfluencies": self.disfluencies,
            "beam_size": self.beam_size,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
            "vad_threshold": self.vad_threshold,
            "locale": self.locale,
        }


@dataclass(frozen=True)
class DiarizationRequest:
    audio_path: str
    num_speakers: Optional[int] = None
    device: str = ""

    def to_worker_args(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "num_speakers": self.num_speakers,
            "device": self.device,
        }


class InferenceCancelled(RuntimeError):
    """Raised when the active local inference operation is canceled."""


class InferenceWorkerError(RuntimeError):
    """An inference child process reported an error."""

    def __init__(self, message: str, trace: Optional[str] = None):
        super().__init__(message)
        self.trace = trace


class LocalInferenceBackend:
    """Run the existing Whisper and Pyannote workers outside the GUI layer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._queue = None
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation and promptly terminate the active child."""
        self._cancel_event.set()
        with self._lock:
            process = self._process
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
            except Exception:
                pass

    def close(self) -> None:
        self.cancel()

    def transcribe(
        self,
        request: TranscriptionRequest,
        on_segment: Callable[[TranscriptionSegment], None],
        on_log: LogCallback = _ignore_log,
        on_progress: TranscriptionProgressCallback = _ignore_transcription_progress,
        is_cancelled: CancelCallback = _never_cancel,
    ) -> TranscriptionInfo:
        from .whisper_mp_worker import whisper_proc_entrypoint

        self._cancel_event.clear()
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=whisper_proc_entrypoint,
            args=(request.to_worker_args(), result_queue),
        )
        process.start()
        self._set_active(process, result_queue)
        info = {}

        try:
            while True:
                message = self._next_message(process, result_queue, is_cancelled)
                message_type = message.get("type") if isinstance(message, dict) else None
                if message_type == "log":
                    on_log(message.get("level", "info"), str(message.get("msg", "")))
                elif message_type == "progress":
                    percent = message.get("pct")
                    if percent is not None:
                        on_progress(float(percent), message.get("detail"))
                elif message_type == "segment":
                    on_segment(TranscriptionSegment.from_mapping(message.get("segment") or {}))
                elif message_type == "result":
                    if not message.get("ok"):
                        raise InferenceWorkerError(
                            message.get("error", "Transcription failed"),
                            message.get("trace"),
                        )
                    info = message.get("info") or {}
                    break
        finally:
            self._cleanup(process, result_queue)

        return TranscriptionInfo.from_mapping(info)

    def diarize(
        self,
        request: DiarizationRequest,
        on_log: LogCallback = _ignore_log,
        on_progress: DiarizationProgressCallback = _ignore_diarization_progress,
        is_cancelled: CancelCallback = _never_cancel,
    ) -> list[DiarizationSegment]:
        from .pyannote_mp_worker import pyannote_proc_entrypoint

        self._cancel_event.clear()
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=pyannote_proc_entrypoint,
            args=(request.to_worker_args(), result_queue),
        )
        process.start()
        self._set_active(process, result_queue)
        segments = []

        try:
            while True:
                message = self._next_message(process, result_queue, is_cancelled)
                message_type = message.get("type") if isinstance(message, dict) else None
                if message_type == "log":
                    on_log(message.get("level", "info"), str(message.get("msg", "")))
                elif message_type == "progress":
                    on_progress(
                        str(message.get("step", "")),
                        int(message.get("pct", 0)),
                    )
                elif message_type == "result":
                    if not message.get("ok"):
                        raise InferenceWorkerError(
                            message.get("error", "Diarization failed"),
                            message.get("trace"),
                        )
                    segments = [
                        DiarizationSegment.from_mapping(segment)
                        for segment in (message.get("segments") or [])
                    ]
                    break
        finally:
            self._cleanup(process, result_queue)

        return segments

    def _next_message(self, process, result_queue, is_cancelled: CancelCallback):
        while True:
            if self._cancel_event.is_set() or is_cancelled():
                self.cancel()
                raise InferenceCancelled("Inference canceled")
            try:
                return result_queue.get(timeout=0.1)
            except pyqueue.Empty:
                if not process.is_alive():
                    raise InferenceWorkerError(
                        f"Inference worker exited unexpectedly (code {process.exitcode})."
                    )

    def _set_active(self, process, result_queue) -> None:
        with self._lock:
            self._process = process
            self._queue = result_queue

    def _cleanup(self, process, result_queue) -> None:
        try:
            process.join(timeout=0.2)
        except Exception:
            pass
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.2)
        except Exception:
            pass
        try:
            process.close()
        except Exception:
            pass
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        with self._lock:
            if self._process is process:
                self._process = None
                self._queue = None
