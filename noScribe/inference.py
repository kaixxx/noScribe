"""UI-independent inference requests, results, and local worker backend."""

from __future__ import annotations

import multiprocessing as mp
import queue as pyqueue
import threading
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from .models import ModelRef
from .plugins.protocol import validate_worker_event


LogCallback = Callable[[str, str], None]
StatusCallback = Callable[[str, dict, str], None]
TranscriptionProgressCallback = Callable[[float, Optional[str]], None]
DiarizationProgressCallback = Callable[[str, int], None]
CancelCallback = Callable[[], bool]


def _ignore_log(level: str, message: str) -> None:
    pass


def _ignore_status(message_id: str, params: dict, level: str) -> None:
    pass


def _ignore_transcription_progress(percent: float, detail: Optional[str]) -> None:
    pass


def _ignore_diarization_progress(step: str, percent: int) -> None:
    pass


def _never_cancel() -> bool:
    return False


LOCAL_WHISPER_BACKEND = "local-whisper"
LOCAL_DIARIZATION_BACKEND = "local-pyannote"


@dataclass
class LocalWorkerSettings:
    """Machine-specific settings shared by all local inference jobs."""

    cpu_threads: int = 4
    force_whisper_cpu: bool = False
    force_diarization_cpu: bool = False
    vad_threshold: float = 0.5


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
    model: ModelRef
    language: Optional[str] = None
    multilingual: bool = False
    include_disfluencies: bool = True


@dataclass(frozen=True)
class DiarizationRequest:
    audio_path: str
    model: ModelRef = field(
        default_factory=lambda: ModelRef(LOCAL_DIARIZATION_BACKEND, "default")
    )
    num_speakers: Optional[int] = None


@dataclass(frozen=True)
class InferenceWorkflowRequest:
    """One upload followed by one or both inference operations."""

    audio_path: str
    transcription: TranscriptionRequest | None = None
    diarization: DiarizationRequest | None = None

    def __post_init__(self) -> None:
        requests = tuple(
            request
            for request in (self.diarization, self.transcription)
            if request is not None
        )
        if not requests:
            raise ValueError("An inference workflow requires at least one operation.")
        if any(request.audio_path != self.audio_path for request in requests):
            raise ValueError("Workflow operations must use the same audio file.")
        if len({request.model.backend_id for request in requests}) != 1:
            raise ValueError("Workflow operations must use the same backend.")


@dataclass(frozen=True)
class InferenceWorkflowResult:
    transcription_info: TranscriptionInfo | None = None
    transcription_segments: tuple[TranscriptionSegment, ...] = ()
    diarization_segments: tuple[DiarizationSegment, ...] = ()


class InferenceCancelled(RuntimeError):
    """Raised when the active local inference operation is canceled."""


class InferenceWorkerError(RuntimeError):
    """An inference child process reported an error."""

    def __init__(self, message: str, trace: Optional[str] = None):
        super().__init__(message)
        self.trace = trace


class LocalInferenceBackend:
    """Run the existing Whisper and Pyannote workers outside the GUI layer."""

    def __init__(
        self,
        whisper_models: Optional[Mapping[str, str]] = None,
        settings: Optional[LocalWorkerSettings] = None,
    ):
        self._lock = threading.Lock()
        self._process = None
        self._queue = None
        self._cancel_event = threading.Event()
        self._whisper_models = dict(whisper_models or {})
        self.settings = settings or LocalWorkerSettings()

    def force_cpu(self, component: str) -> None:
        if component == "whisper":
            self.settings.force_whisper_cpu = True
        elif component == "pyannote":
            self.settings.force_diarization_cpu = True
        else:
            raise ValueError(f"Unknown inference component: {component}")

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
        on_status: StatusCallback = _ignore_status,
        on_progress: TranscriptionProgressCallback = _ignore_transcription_progress,
        is_cancelled: CancelCallback = _never_cancel,
    ) -> TranscriptionInfo:
        from .whisper_mp_worker import whisper_proc_entrypoint

        self._cancel_event.clear()
        if request.model.backend_id != LOCAL_WHISPER_BACKEND:
            raise ValueError(
                f"Backend {request.model.backend_id!r} cannot be handled locally."
            )
        try:
            model_path = self._whisper_models[request.model.model_id]
        except KeyError as error:
            raise ValueError(f"Unknown local Whisper model: {request.model}") from error

        worker_args = {
            "model_path": model_path,
            "device": "cpu" if self.settings.force_whisper_cpu else "auto",
            "compute_type": "auto",
            "cpu_threads": self.settings.cpu_threads,
            "local_files_only": True,
            "audio_path": request.audio_path,
            "language": request.language,
            "multilingual": request.multilingual,
            "include_disfluencies": request.include_disfluencies,
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
            "vad_threshold": self.settings.vad_threshold,
        }

        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=whisper_proc_entrypoint,
            args=(worker_args, result_queue),
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
                elif message_type == "status":
                    on_status(
                        str(message.get("message_id", "")),
                        message.get("params") or {},
                        str(message.get("level", "info")),
                    )
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

        if request.model.backend_id != LOCAL_DIARIZATION_BACKEND:
            raise ValueError(
                f"Backend {request.model.backend_id!r} cannot be handled locally."
            )
        worker_args = {
            "audio_path": request.audio_path,
            "num_speakers": request.num_speakers,
            "device": "cpu" if self.settings.force_diarization_cpu else "",
        }
        self._cancel_event.clear()
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=pyannote_proc_entrypoint,
            args=(worker_args, result_queue),
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
                return validate_worker_event(result_queue.get(timeout=0.1))
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
