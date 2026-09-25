"""Construction of the backend plugins shipped with noScribe."""

from __future__ import annotations

from typing import Mapping

from ..inference import LocalInferenceBackend, LocalWorkerSettings
from .local_pyannote import LocalPyannotePlugin
from .local_whisper import LocalWhisperPlugin
from .registry import BackendRegistry


def create_builtin_registry(
    whisper_models: Mapping[str, str],
    settings: LocalWorkerSettings,
) -> BackendRegistry:
    """Create the registry without treating built-ins as special at runtime."""
    host = LocalInferenceBackend(whisper_models=whisper_models, settings=settings)
    return BackendRegistry([
        LocalWhisperPlugin(host, whisper_models),
        LocalPyannotePlugin(host),
    ])
