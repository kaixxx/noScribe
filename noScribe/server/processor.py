"""Adapter from queued server workflows to noScribe's backend registry."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..audio.convert import ToWav
from ..inference import (
    DiarizationRequest, DiarizationResult, TranscriptionRequest, InferenceCancelled,
    LOCAL_WHISPER_BACKEND, LOCAL_DIARIZATION_BACKEND,
)
from ..models import ModelDescriptor, ModelRef
from ..plugins.registry import BackendRegistry
from ..plugins.protocol import validate_extensions
from ..worker_errors import WorkerError, WorkerRetryError
from .jobs import JobSnapshot, JobTask


class RegistryWorkflowProcessor:
    """Execute an atomic workflow with the same plugins as the desktop app."""

    def __init__(
        self,
        registry: BackendRegistry,
        *,
        wav_converter: Callable[[Path, Path], None] | None = None,
        max_audio_seconds: float = 24 * 3600,
        audio_validator: Callable[[Path, float], float] | None = None,
    ):
        self.registry = registry
        self._wav_converter = wav_converter or _convert_to_wav
        self._max_audio_seconds = max_audio_seconds
        self._audio_validator = audio_validator or validate_flac_audio
        self._models = self._build_model_map(registry.list_models())

    def list_models(self) -> list[dict[str, object]]:
        return [
            {
                "id": public_id,
                "name": model.display_name,
                "engine": model.engine,
                "capabilities": sorted(model.capabilities),
            }
            for public_id, model in self._models.items()
        ]

    def validate_tasks(self, tasks: tuple[JobTask, ...]) -> None:
        for task in tasks:
            if task.operation == "transcription":
                self._model(task, "transcription")
                options = _options(
                    task,
                    allowed={"language", "multilingual", "include_disfluencies", "extensions", "diarization"},
                )
                _optional_string(options.get("language"))
                _boolean(options.get("multilingual"), False)
                _boolean(options.get("include_disfluencies"), True)
                if options.get("diarization") is not None:
                    try:
                        DiarizationResult.from_mapping(options["diarization"], seconds=True)
                    except (TypeError, KeyError, AttributeError, OverflowError) as error:
                        raise ValueError("Invalid diarization context.") from error
            else:
                self._model(task, "diarization")
                options = _options(task, allowed={"num_speakers", "extensions"})
                _optional_positive_int(options.get("num_speakers"), "num_speakers")
            validate_extensions(options.get("extensions", {}))

    def process(
        self,
        job: JobSnapshot,
        audio_path: Path,
        emit: Callable[[dict], None],
        is_cancelled: Callable[[], bool],
    ) -> None:
        wav_path: Path | None = None
        try:
            self._audio_validator(audio_path, self._max_audio_seconds)
            # Decode the lossless transport once. Both workers then receive the
            # same 16 kHz mono PCM input as the local desktop pipeline.
            wav_path = audio_path.with_suffix(".wav")
            self._wav_converter(audio_path, wav_path)

            diarization = None
            for task_index, task in enumerate(job.tasks):
                if is_cancelled():
                    return
                emit({
                    "type": "task_started",
                    "task_index": task_index,
                    "operation": task.operation,
                })
                if task.operation == "transcription":
                    self._transcribe(
                        task_index, task, wav_path, emit, is_cancelled, diarization
                    )
                else:
                    diarization = self._diarize(
                        task_index, task, wav_path, emit, is_cancelled
                    )
        finally:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)

    def cancel(self) -> None:
        self.registry.cancel()

    def close(self) -> None:
        self.registry.close()

    def _transcribe(self, index, task, audio_path, emit, is_cancelled, diarization=None) -> None:
        model = self._model(task, "transcription")
        options = _options(
            task,
            allowed={"language", "multilingual", "include_disfluencies", "extensions", "diarization"},
        )
        request = TranscriptionRequest(
            audio_path=str(audio_path),
            model=model.ref,
            extensions=dict(options.get("extensions") or {}),
            diarization=(diarization if diarization is not None else
                         DiarizationResult.from_mapping(options["diarization"], seconds=True)
                         if options.get("diarization") is not None else None),
            language=_optional_string(options.get("language")),
            multilingual=_boolean(options.get("multilingual"), False),
            include_disfluencies=_boolean(
                options.get("include_disfluencies"), True
            ),
        )

        def event(event_type: str, **payload) -> None:
            emit({
                "type": event_type,
                "task_index": index,
                "operation": "transcription",
                **payload,
            })

        info = self._run_with_cpu_retry(
            self.registry.transcribe,
            request,
            on_retry=lambda: event("task_started", retry=True),
            on_segment=lambda segment: event(
                "segment",
                segment={
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "extensions": segment.extensions,
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "prob": word.probability,
                            "extensions": word.extensions,
                        }
                        for word in segment.words
                    ],
                },
            ),
            on_log=lambda level, message: event("log", level=level, msg=message),
            on_status=lambda message_id, params, level: event(
                "status", message_id=message_id, params=params, level=level
            ),
            on_progress=lambda percent, detail: event(
                "progress", pct=percent, detail=detail
            ),
            is_cancelled=is_cancelled,
        )
        event(
            "task_result",
            ok=True,
            info={
                "duration": info.duration,
                "language": info.language,
                "language_probability": info.language_probability,
                "sample_rate": info.sample_rate,
                "extensions": info.extensions,
            },
        )

    def _diarize(self, index, task, audio_path, emit, is_cancelled) -> DiarizationResult:
        model = self._model(task, "diarization")
        options = _options(task, allowed={"num_speakers", "extensions"})
        num_speakers = _optional_positive_int(
            options.get("num_speakers"), "num_speakers"
        )
        request = DiarizationRequest(
            audio_path=str(audio_path),
            model=model.ref,
            num_speakers=num_speakers,
            extensions=dict(options.get("extensions") or {}),
        )

        def event(event_type: str, **payload) -> None:
            emit({
                "type": event_type,
                "task_index": index,
                "operation": "diarization",
                **payload,
            })

        result = self._run_with_cpu_retry(
            self.registry.diarize,
            request,
            on_retry=lambda: event("task_started", retry=True),
            on_log=lambda level, message: event("log", level=level, msg=message),
            on_progress=lambda step, percent: event(
                "progress", step=step, pct=percent
            ),
            is_cancelled=is_cancelled,
        )
        event(
            "task_result",
            ok=True,
            **result.to_mapping(seconds=True),
        )
        return result

    def _run_with_cpu_retry(self, operation, request, *, on_retry, **callbacks):
        """Retry a built-in auto-device worker once without changing its settings."""
        try:
            return operation(request, **callbacks)
        except WorkerError as gpu_error:
            if (callbacks['is_cancelled']() or not gpu_error.allows_cpu_retry
                    or request.model.backend_id not in {LOCAL_WHISPER_BACKEND, LOCAL_DIARIZATION_BACKEND}
                    or self.registry.get(request.model.backend_id).settings.device != 'auto'):
                raise
            on_retry()
            callbacks.get('on_log', lambda *_: None)(
                'info', 'GPU inference failed; retrying this task once on CPU.')
            try:
                return operation(request, device_override='cpu', **callbacks)
            except Exception as cpu_error:
                if callbacks['is_cancelled']():
                    raise
                if isinstance(cpu_error, InferenceCancelled):
                    raise
                raise WorkerRetryError(gpu_error, cpu_error) from cpu_error

    def _model(self, task: JobTask, capability: str) -> ModelDescriptor:
        try:
            model = self._models[task.model]
        except KeyError as error:
            raise ValueError(f"Unknown server model: {task.model!r}") from error
        if capability not in model.capabilities:
            raise ValueError(
                f"Model {task.model!r} does not support {capability}."
            )
        return model

    @staticmethod
    def _build_model_map(
        models: list[ModelDescriptor],
    ) -> dict[str, ModelDescriptor]:
        """Use stable slash-qualified IDs across the server boundary."""
        result = {}
        for model in models:
            public_id = f"{model.ref.backend_id}/{model.ref.model_id}"
            if public_id in result:
                raise ValueError(f"Duplicate server model ID: {public_id}")
            result[public_id] = model
        return result


def create_server_registry(config) -> BackendRegistry:
    """Discover configured models and construct the shared built-in plugins."""
    from ..plugins.factory import create_builtin_registry

    model_paths = discover_whisper_models(config.whisper_models_dir)
    registry = create_builtin_registry(
        whisper_models=model_paths,
        settings_dir=config.plugin_settings_dir,
    )
    return registry


def validate_flac_audio(path: Path, max_duration_seconds: float) -> float:
    """Validate the uploaded container and return its declared duration."""
    import av

    try:
        with av.open(str(path), mode="r") as container:
            audio_streams = list(container.streams.audio)
            if len(audio_streams) != 1:
                raise ValueError("Audio must contain exactly one audio stream.")
            stream = audio_streams[0]
            codec_name = getattr(stream.codec_context.codec, "name", "")
            if codec_name != "flac" or "flac" not in container.format.name.split(","):
                raise ValueError("Uploaded audio must be FLAC.")
            if stream.sample_rate != 16000 or stream.channels != 1:
                raise ValueError("Uploaded FLAC must be 16 kHz mono audio.")
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            else:
                raise ValueError("Audio duration could not be determined.")
    except (OSError, av.error.FFmpegError) as error:
        raise ValueError("Uploaded audio could not be decoded.") from error
    if duration <= 0:
        raise ValueError("Uploaded audio is empty.")
    if duration > max_duration_seconds:
        raise ValueError("Uploaded audio exceeds the configured duration limit.")
    return duration


def discover_whisper_models(directory: Path) -> dict[str, str]:
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {
        path.name: str(path.resolve())
        for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        if path.is_dir() and (path / "model.bin").is_file()
    }


def _convert_to_wav(source: Path, target: Path) -> None:
    with ToWav(source, target, force=True) as converter:
        while converter.convert():
            pass


def _options(task: JobTask, *, allowed: set[str]) -> dict[str, object]:
    unknown = set(task.options) - allowed
    if unknown:
        raise ValueError(
            f"Unsupported {task.operation} options: {', '.join(sorted(unknown))}"
        )
    return dict(task.options)


def _boolean(value: object, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("Boolean job options must be true or false.")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Language must be a string or null.")
    return value or None


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or null.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value
