"""Adapter from queued server workflows to noScribe's backend registry."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..audio.convert import ToWav
from ..inference import DiarizationRequest, TranscriptionRequest
from ..models import ModelDescriptor, ModelRef
from ..plugins.registry import BackendRegistry
from .jobs import JobSnapshot, JobTask


class RegistryWorkflowProcessor:
    """Execute an atomic workflow with the same plugins as the desktop app."""

    def __init__(
        self,
        registry: BackendRegistry,
        *,
        wav_converter: Callable[[Path, Path], None] | None = None,
    ):
        self.registry = registry
        self._wav_converter = wav_converter or _convert_to_wav
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
                    allowed={"language", "multilingual", "include_disfluencies"},
                )
                _optional_string(options.get("language"))
                _boolean(options.get("multilingual"), False)
                _boolean(options.get("include_disfluencies"), True)
            else:
                self._model(task, "diarization")
                options = _options(task, allowed={"num_speakers"})
                _optional_positive_int(options.get("num_speakers"), "num_speakers")

    def process(
        self,
        job: JobSnapshot,
        audio_path: Path,
        emit: Callable[[dict], None],
        is_cancelled: Callable[[], bool],
    ) -> None:
        wav_path: Path | None = None
        try:
            if any(task.operation == "diarization" for task in job.tasks):
                wav_path = audio_path.with_suffix(".wav")
                self._wav_converter(audio_path, wav_path)

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
                        task_index, task, audio_path, emit, is_cancelled
                    )
                else:
                    assert wav_path is not None
                    self._diarize(
                        task_index, task, wav_path, emit, is_cancelled
                    )
        finally:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)

    def cancel(self) -> None:
        self.registry.cancel()

    def close(self) -> None:
        self.registry.close()

    def _transcribe(self, index, task, audio_path, emit, is_cancelled) -> None:
        model = self._model(task, "transcription")
        options = _options(
            task,
            allowed={"language", "multilingual", "include_disfluencies"},
        )
        request = TranscriptionRequest(
            audio_path=str(audio_path),
            model=model.ref,
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

        info = self.registry.transcribe(
            request,
            on_segment=lambda segment: event(
                "segment",
                segment={
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "prob": word.probability,
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
            },
        )

    def _diarize(self, index, task, audio_path, emit, is_cancelled) -> None:
        model = self._model(task, "diarization")
        options = _options(task, allowed={"num_speakers"})
        num_speakers = _optional_positive_int(
            options.get("num_speakers"), "num_speakers"
        )
        request = DiarizationRequest(
            audio_path=str(audio_path),
            model=model.ref,
            num_speakers=num_speakers,
        )

        def event(event_type: str, **payload) -> None:
            emit({
                "type": event_type,
                "task_index": index,
                "operation": "diarization",
                **payload,
            })

        segments = self.registry.diarize(
            request,
            on_log=lambda level, message: event("log", level=level, msg=message),
            on_progress=lambda step, percent: event(
                "progress", step=step, pct=percent
            ),
            is_cancelled=is_cancelled,
        )
        event(
            "task_result",
            ok=True,
            segments=[
                {
                    "speaker": segment.label,
                    "start": segment.start_ms / 1000.0,
                    "end": segment.end_ms / 1000.0,
                }
                for segment in segments
            ],
        )

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
    from ..inference import LocalWorkerSettings
    from ..plugins.factory import create_builtin_registry

    model_paths = discover_whisper_models(config.whisper_models_dir)
    registry = create_builtin_registry(
        whisper_models=model_paths,
        settings=LocalWorkerSettings(
            cpu_threads=config.cpu_threads,
            force_whisper_cpu=config.force_cpu,
            force_diarization_cpu=config.force_cpu,
            vad_threshold=config.vad_threshold,
        ),
    )
    return registry


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
