"""Construction of the backend plugins shipped with noScribe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from pathlib import Path

from ..inference import LocalInferenceBackend, LocalWorkerSettings
from .local_pyannote import LocalPyannotePlugin
from .local_whisper import LocalWhisperPlugin
from .local_whisper.settings import WhisperWorkerSettings
from .local_pyannote.settings import PyannoteWorkerSettings
from .settings import PluginSettingsStore
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
    settings_dir: Path | None = None,
) -> BackendRegistry:
    """Create the registry without treating built-ins as special at runtime."""
    store = PluginSettingsStore(settings_dir) if settings_dir is not None else None
    whisper_settings = WhisperWorkerSettings.from_mapping(
        store.load("local-whisper") if store else {},
        defaults=WhisperWorkerSettings(
            device="cpu" if settings.force_whisper_cpu else "auto",
            cpu_threads=settings.cpu_threads,
            vad_threshold=settings.vad_threshold,
        ),
    )
    diarization_settings = PyannoteWorkerSettings.from_mapping(
        store.load("local-pyannote") if store else {},
        defaults=PyannoteWorkerSettings(
            device="cpu" if settings.force_diarization_cpu else "auto",
        ),
    )
    # Keep legacy desktop consumers (including its preliminary VAD) in sync.
    settings.cpu_threads = whisper_settings.cpu_threads
    settings.vad_threshold = whisper_settings.vad_threshold
    settings.force_whisper_cpu = whisper_settings.device == "cpu"
    settings.force_diarization_cpu = diarization_settings.device == "cpu"
    host = LocalInferenceBackend(
        whisper_models=whisper_models, settings=settings,
        whisper_settings=whisper_settings, diarization_settings=diarization_settings,
    )
    return BackendRegistry([
        LocalWhisperPlugin(host, whisper_models, settings_store=store),
        LocalPyannotePlugin(host, settings_store=store),
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


def refresh_remote_profiles(
    registry: BackendRegistry,
    profiles: tuple[RemoteBackendProfile, ...],
) -> tuple[RemoteBackendRegistrationError, ...]:
    """Refresh connected remotes and retry profiles unavailable at startup."""
    errors = []
    for profile in profiles:
        if not profile.enabled:
            continue
        try:
            try:
                plugin = registry.get(profile.id)
            except ValueError:
                plugin = RemoteHttpPlugin(profile)
                registry.register(plugin)
                continue
            if not isinstance(plugin, RemoteHttpPlugin):
                raise ValueError(
                    f"Backend ID {profile.id!r} is already used by another plugin."
                )
            plugin.refresh_models()
        except Exception as error:
            errors.append(RemoteBackendRegistrationError(profile, str(error)))
    return tuple(errors)


def sync_remote_profiles(
    registry: BackendRegistry,
    profiles: tuple[RemoteBackendProfile, ...],
) -> tuple[RemoteBackendRegistrationError, ...]:
    """Make registered remote plugins match the persisted profile set."""
    desired = {profile.id: profile for profile in profiles if profile.enabled}

    for manifest in registry.list_plugins():
        try:
            plugin = registry.get(manifest.id)
        except ValueError:
            continue
        if not isinstance(plugin, RemoteHttpPlugin):
            continue
        profile = desired.get(manifest.id)
        if profile is None or plugin.profile != profile:
            registry.unregister(manifest.id)

    return refresh_remote_profiles(registry, tuple(desired.values()))
