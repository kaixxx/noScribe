"""Construction of the backend plugins shipped with noScribe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..inference import LocalInferenceBackend, LocalWorkerSettings
from .local_pyannote import LocalPyannotePlugin
from .local_whisper import LocalWhisperPlugin
from .remote_http import NOSCRIBE_HTTP_DRIVER, RemoteHttpPlugin
from .remote_profiles import RemoteBackendProfile
from .registry import BackendRegistry


@dataclass(frozen=True)
class RemoteBackendRegistrationError:
    profile: RemoteBackendProfile
    message: str


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


def register_remote_profiles(
    registry: BackendRegistry,
    profiles: tuple[RemoteBackendProfile, ...],
) -> tuple[RemoteBackendRegistrationError, ...]:
    """Connect enabled profiles without letting one failure block the rest."""
    errors = []
    for profile in profiles:
        if not profile.enabled:
            continue
        try:
            if profile.driver != NOSCRIBE_HTTP_DRIVER:
                raise ValueError(
                    f"Unsupported remote backend driver: {profile.driver!r}"
                )
            registry.register(RemoteHttpPlugin(profile))
        except Exception as error:
            errors.append(RemoteBackendRegistrationError(profile, str(error)))
    return tuple(errors)
