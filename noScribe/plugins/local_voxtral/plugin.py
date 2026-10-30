"""Plugin adapter for the bundled Voxtral engine (noScribe/voxtral_engine.py).

Voxtral runs on MLX, so it is offered on Apple Silicon only, and has no device
setting and no CUDA fallback. Besides the request's own fields it reads:

- ``extensions["noscribe"]``, the job data main.py adds to local requests:
  ``needs_timing`` (False when the output carries no times, which lets the
  engine skip the forced alignment) and ``speaker_names`` (whose spellings the
  engine corrects mis-heard names to).
- ``diarization``: its turns are where the engine cuts a chunk that loops,
  instead of at the quietest pause.
"""

from __future__ import annotations

import json
import logging
import platform
from importlib import resources

from ...inference import LocalInferenceBackend
from ...models import ModelDescriptor, ModelRef
from ..manifest import PluginManifest
from .settings import VoxtralWorkerSettings

logger = logging.getLogger(__name__)

# What main.py passes in TranscriptionRequest.extensions for every job.
JOB_EXTENSION = "noscribe"

# The RAM hint follows the model name in the model menu, so a model that would
# not fit is noticed before it starts swapping.
MENU_HINT_SEP = '   ·   '


def _load_manifest() -> PluginManifest:
    resource = resources.files(__package__) / "backend.json"
    with resource.open("r", encoding="utf-8") as stream:
        return PluginManifest.from_mapping(json.load(stream))


def _apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class LocalVoxtralPlugin:
    def __init__(self, host: LocalInferenceBackend, *, corrections_dir,
                 settings: VoxtralWorkerSettings | None = None, settings_store=None):
        self.manifest = _load_manifest()
        self._host = host
        self._corrections_dir = str(corrections_dir)
        self.settings = settings or VoxtralWorkerSettings()
        self._settings_store = settings_store
        self.settings_path = settings_store.path_for(self.manifest.id) if settings_store else None

    def list_models(self) -> list[ModelDescriptor]:
        """The shipped builds whose weights are present or downloadable. Asked
        again on every menu opening, so a build converted while the app runs
        (tools/quantize_voxtral.py) appears without a restart."""
        if not _apple_silicon():
            return []
        try:
            from ... import voxtral_engine
            if not voxtral_engine.is_available():
                return []
            return [
                ModelDescriptor(
                    ref=ModelRef(self.manifest.id, name),
                    display_name=name,
                    engine=self.manifest.engine or "voxtral",
                    capabilities=self.manifest.capabilities,
                )
                for name in voxtral_engine.VOXTRAL_MODELS
                if voxtral_engine.has_local_build(name)
            ]
        except Exception as error:
            logger.warning("Could not list the Voxtral models: %s", error)
            return []

    def menu_hint(self, model: ModelDescriptor) -> str:
        """The RAM the model needs, marked when it exceeds this machine's.

        Choosing a model that does not fit does not merely run slowly: the
        machine starts swapping and the run stops making progress, which is
        impossible to diagnose from the outside."""
        try:
            from ... import voxtral_engine
            need = voxtral_engine.min_ram_gb(self._repo(model.ref.model_id))
            mark = '' if need <= voxtral_engine.total_ram_gb() else '  ⚠'
            return f'{MENU_HINT_SEP}{need:.0f} GB RAM{mark}'
        except Exception as error:
            # The hint is a memory-safety cue, so its silent absence would be
            # undiagnosable; the menu itself must not break over it.
            logger.warning("Could not compute RAM hint for %s: %s", model.ref, error)
            return ''

    def _repo(self, name: str) -> str:
        from ... import voxtral_engine
        if name not in voxtral_engine.VOXTRAL_MODELS:
            raise ValueError(f"Unknown Voxtral model: {name!r}")
        return voxtral_engine.resolve_model(name, voxtral_engine.VOXTRAL_MODELS[name])

    def transcribe(self, request, on_segment, *, on_log=None, on_status=None,
                   on_progress=None, is_cancelled=None, device_override=None):
        from ... import transcript_corrections
        from ...voxtral_mp_worker import voxtral_proc_entrypoint

        if request.model.backend_id != self.manifest.id:
            raise ValueError(f"Backend {request.model.backend_id!r} is not {self.manifest.id!r}.")
        callbacks = {name: callback for name, callback in (
            ("on_log", on_log), ("on_status", on_status),
            ("on_progress", on_progress), ("is_cancelled", is_cancelled))
            if callback is not None}
        status = on_status or (lambda message_id, params, level: None)

        job = request.extensions.get(JOB_EXTENSION) or {}
        need_timestamps = bool(job.get("needs_timing", True))
        status('voxtral_path_long' if need_timestamps else 'voxtral_path_short', {}, 'debug')
        if request.include_disfluencies:
            # Voxtral has no prompt/hotword hook, so the disfluencies option
            # cannot steer it; say so instead of silently ignoring the setting.
            status('voxtral_no_disfluencies', {}, 'info')  # the user chose an option it cannot honour

        args = {
            "audio_path": request.audio_path,
            # None for Auto/Multilingual: Voxtral detects the language itself.
            "language_code": request.language,
            "need_timestamps": need_timestamps,
            "voxtral_repo": self._repo(request.model.model_id),
            "chunk_sec": self.settings.chunk_sec or None,
            "corrections_path": transcript_corrections.ensure_default_file(self._corrections_dir),
            "speaker_names": job.get("speaker_names") or None,
            "ram_reserve_gb": self.settings.ram_reserve_gb or None,
            # Same converted-WAV timeline the engine reads, in seconds.
            "speaker_turns": (
                [[segment.start_ms / 1000.0, segment.end_ms / 1000.0, segment.label]
                 for segment in request.diarization] if request.diarization else None),
        }
        return self._host.run_transcription_worker(
            voxtral_proc_entrypoint, args, on_segment, **callbacks)

    def cancel(self) -> None:
        self._host.cancel()

    def close(self) -> None:
        self._host.close()
