"""UI-independent inference requests, results, and local worker backend."""

from __future__ import annotations

import multiprocessing as mp
import queue as pyqueue
import threading
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from .models import ModelRef
from .worker_errors import WorkerError
from .plugins.protocol import validate_extensions, validate_worker_event


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
    extensions: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict) -> "TranscriptionWord":
        return cls(
            word=str(value.get("word") or ""),
            start=value.get("start"),
            end=value.get("end"),
            probability=value.get("prob"),
            extensions=dict(value.get("extensions") or {}),
        )


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    words: tuple[TranscriptionWord, ...] = field(default_factory=tuple)
    extensions: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict) -> "TranscriptionSegment":
        return cls(
            start=float(value.get("start") or 0.0),
            end=float(value.get("end") or 0.0),
            text=str(value.get("text") or ""),
            extensions=dict(value.get("extensions") or {}),
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
    extensions: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict) -> "TranscriptionInfo":
        return cls(
            duration=value.get("duration"),
            language=value.get("language"),
            language_probability=value.get("language_probability"),
            sample_rate=value.get("sample_rate"),
            extensions=dict(value.get("extensions") or {}),
        )


@dataclass(frozen=True)
class DiarizationSegment:
    start_ms: int
    end_ms: int
    label: str
    extensions: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict) -> "DiarizationSegment":
        return cls(
            start_ms=int(value.get("start") or 0),
            end_ms=int(value.get("end") or 0),
            label=str(value.get("label") or ""),
            extensions=dict(value.get("extensions") or {}),
        )


@dataclass(frozen=True)
class DiarizationResult:
    """Segments and opaque, JSON-compatible backend evidence.

    Times refer to the supplied audio, before any desktop transcript offset.
    Sequence access supports consumers interested only in the segments.
    """

    segments: tuple[DiarizationSegment, ...] = ()
    extensions: dict = field(default_factory=dict)

    def __post_init__(self):
        validate_extensions(self.extensions)

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        return self.segments[index]

    @classmethod
    def from_mapping(cls, value: dict, *, seconds: bool = False):
        segments = []
        for item in value.get("segments") or []:
            if seconds:
                item = dict(item, start=round(float(item["start"]) * 1000),
                            end=round(float(item["end"]) * 1000),
                            label=item.get("speaker") or item.get("label") or "")
            segments.append(DiarizationSegment.from_mapping(item))
        return cls(tuple(segments), dict(value.get("extensions") or {}))

    def to_mapping(self, *, seconds: bool = False) -> dict:
        return {
            "segments": [
                {
                    "start": segment.start_ms / 1000 if seconds else segment.start_ms,
                    "end": segment.end_ms / 1000 if seconds else segment.end_ms,
                    ("speaker" if seconds else "label"): segment.label,
                    **({"extensions": segment.extensions} if segment.extensions else {}),
                }
                for segment in self.segments
            ],
            "extensions": self.extensions,
        }


@dataclass(frozen=True)
class TranscriptionRequest:
    audio_path: str
    model: ModelRef
    language: Optional[str] = None
    multilingual: bool = False
    include_disfluencies: bool = True
    diarization: DiarizationResult | None = None
    extensions: dict = field(default_factory=dict)

    def __post_init__(self):
        validate_extensions(self.extensions)


@dataclass(frozen=True)
class DiarizationRequest:
    audio_path: str
    model: ModelRef = field(
        default_factory=lambda: ModelRef(LOCAL_DIARIZATION_BACKEND, "default")
    )
    num_speakers: Optional[int] = None
    extensions: dict = field(default_factory=dict)

    def __post_init__(self):
        validate_extensions(self.extensions)


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
    diarization: DiarizationResult | None = None

    @property
    def diarization_segments(self) -> tuple[DiarizationSegment, ...]:
        return self.diarization.segments if self.diarization is not None else ()


class InferenceCancelled(RuntimeError):
    """Raised when the active local inference operation is canceled."""


class InferenceWorkerError(RuntimeError):
    """An inference backend reported an error.

    ``message`` and ``trace`` are diagnostic and may be logged to the console
    or log file.  Remote adapters can additionally provide a translation key
    for a short message that is safe and useful in the GUI.
    """

    def __init__(
        self,
        message: str,
        trace: Optional[str] = None,
        *,
        user_message_id: Optional[str] = None,
        user_message_params: Optional[Mapping[str, object]] = None,
    ):
        super().__init__(message)
        self.trace = trace
        self.user_message_id = user_message_id
        self.user_message_params = dict(user_message_params or {})


class LocalInferenceBackend:
    """Run the existing Whisper and Pyannote workers outside the GUI layer."""

    def __init__(
        self,
        whisper_models: Optional[Mapping[str, str]] = None,
        settings: Optional[LocalWorkerSettings] = None,
        *,
        whisper_settings=None,
        diarization_settings=None,
    ):
        from .plugins.local_whisper.settings import WhisperWorkerSettings
        from .plugins.local_pyannote.settings import PyannoteWorkerSettings

        self._lock = threading.Lock()
        self._process = None
        self._queue = None
        self._cancel_event = threading.Event()
        self._whisper_models = dict(whisper_models or {})
        self.settings = settings or LocalWorkerSettings()
        self.whisper_settings = whisper_settings or WhisperWorkerSettings(
            device="cpu" if self.settings.force_whisper_cpu else "auto",
            cpu_threads=self.settings.cpu_threads,
            vad_threshold=self.settings.vad_threshold,
        )
        self.diarization_settings = diarization_settings or PyannoteWorkerSettings(
            device="cpu" if self.settings.force_diarization_cpu else "auto",
        )

    def force_cpu(self, component: str) -> None:
        if component == "whisper":
            self.settings.force_whisper_cpu = True
            self.whisper_settings.device = "cpu"
        elif component == "pyannote":
            self.settings.force_diarization_cpu = True
            self.diarization_settings.device = "cpu"
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
        *,
        device_override: str | None = None,
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
            "device": device_override or self.whisper_settings.device,
            "compute_type": "auto",
            "cpu_threads": self.whisper_settings.cpu_threads,
            "local_files_only": True,
            "audio_path": request.audio_path,
            "language": request.language,
            "multilingual": request.multilingual,
            "include_disfluencies": request.include_disfluencies,
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
            "vad_threshold": self.whisper_settings.vad_threshold,
            "diarization": request.diarization.to_mapping() if request.diarization is not None else None,
            "extensions": request.extensions,
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
        worker_state = {'device': 'cpu' if worker_args['device'] == 'cpu' else None,
                        'stage': 'starting'}

        try:
            while True:
                message = self._next_message(process, result_queue, is_cancelled, worker_state)
                message_type = message.get("type") if isinstance(message, dict) else None
                if message_type == "state":
                    worker_state = {'device': message.get('device'), 'stage': message.get('stage')}
                elif message_type == "log":
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
                        raise WorkerError(message)
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
        *,
        device_override: str | None = None,
    ) -> DiarizationResult:
        from .pyannote_mp_worker import pyannote_proc_entrypoint

        if request.model.backend_id != LOCAL_DIARIZATION_BACKEND:
            raise ValueError(
                f"Backend {request.model.backend_id!r} cannot be handled locally."
            )
        worker_args = {
            "audio_path": request.audio_path,
            "num_speakers": request.num_speakers,
            "device": device_override or self.diarization_settings.device,
            "extensions": request.extensions,
        }
        return self.run_diarization_worker(
            pyannote_proc_entrypoint, worker_args, on_log, on_progress, is_cancelled)

    def run_diarization_worker(
        self,
        entrypoint,
        worker_args: dict,
        on_log: LogCallback = _ignore_log,
        on_progress: DiarizationProgressCallback = _ignore_diarization_progress,
        is_cancelled: CancelCallback = _never_cancel,
    ) -> DiarizationResult:
        """Run one diarization worker that speaks the local worker protocol.

        Shared by every bundled diarization plugin."""
        self._cancel_event.clear()
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=entrypoint,
            args=(worker_args, result_queue),
        )
        process.start()
        self._set_active(process, result_queue)
        result = DiarizationResult()
        worker_state = {'device': 'cpu' if worker_args.get('device') == 'cpu' else None,
                        'stage': 'starting'}

        try:
            while True:
                message = self._next_message(process, result_queue, is_cancelled, worker_state)
                message_type = message.get("type") if isinstance(message, dict) else None
                if message_type == "state":
                    worker_state = {'device': message.get('device'), 'stage': message.get('stage')}
                elif message_type == "log":
                    on_log(message.get("level", "info"), str(message.get("msg", "")))
                elif message_type == "progress":
                    on_progress(
                        str(message.get("step", "")),
                        int(message.get("pct", 0)),
                    )
                elif message_type == "result":
                    if not message.get("ok"):
                        raise WorkerError(message)
                    result = DiarizationResult.from_mapping(message)
                    break
        finally:
            self._cleanup(process, result_queue)

        return result

    def _next_message(self, process, result_queue, is_cancelled: CancelCallback, worker_state):
        while True:
            if self._cancel_event.is_set() or is_cancelled():
                self.cancel()
                raise InferenceCancelled("Inference canceled")
            try:
                return validate_worker_event(result_queue.get(timeout=0.1))
            except pyqueue.Empty:
                if not process.is_alive():
                    raise WorkerError.unexpected_exit(process.exitcode, worker_state)

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
