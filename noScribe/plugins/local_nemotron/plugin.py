"""Plugin adapter for the bundled Nemotron 3 Diarization worker
(noScribe/nemotron_mp_worker.py).

Its result carries the model's per-frame speaker probabilities in
``extensions["local-nemotron"]["probabilities"]`` (see the worker for the
format and nemotron_mp_worker.decode_probabilities to read them), which the
voice check uses as its second opinion. Nemotron finds the number of speakers
itself and cannot be held to one, so a request with a fixed number is refused.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib import resources

from ...inference import LocalInferenceBackend
from ...models import ModelDescriptor, ModelRef
from ..manifest import PluginManifest
from .settings import NemotronWorkerSettings


def _load_manifest() -> PluginManifest:
    resource = resources.files(__package__) / "backend.json"
    with resource.open("r", encoding="utf-8") as stream:
        return PluginManifest.from_mapping(json.load(stream))


class LocalNemotronPlugin:
    def __init__(self, host: LocalInferenceBackend, *,
                 settings: NemotronWorkerSettings | None = None, settings_store=None):
        self.manifest = _load_manifest()
        self._host = host
        self.settings = settings or NemotronWorkerSettings()
        self._settings_store = settings_store
        self.settings_path = settings_store.path_for(self.manifest.id) if settings_store else None

    def list_models(self) -> list[ModelDescriptor]:
        """The model, where transformers and librosa can run it; its weights may
        still have to be downloaded (ready() says whether they are on disk)."""
        from ...nemotron_mp_worker import installed
        if not installed():
            return []
        return [
            ModelDescriptor(
                ref=ModelRef(self.manifest.id, "default"),
                display_name="Nemotron",
                engine=self.manifest.engine or "nemotron",
                capabilities=self.manifest.capabilities,
            )
        ]

    def ready(self) -> bool:
        """Runnable without a download: what `auto` asks before choosing it."""
        from ...nemotron_mp_worker import available
        return available()

    def diarize(self, request, *, on_log=None, on_progress=None, is_cancelled=None,
                device_override=None):
        from ...nemotron_mp_worker import nemotron_proc_entrypoint

        if request.num_speakers is not None:
            raise ValueError("Nemotron finds the number of speakers itself and "
                             "cannot be held to a fixed one.")
        callbacks = {name: callback for name, callback in (
            ("on_log", on_log), ("on_progress", on_progress), ("is_cancelled", is_cancelled))
            if callback is not None}
        args = {
            "audio_path": request.audio_path,
            "device": device_override or self.settings.device,
            "extensions": request.extensions,
        }
        return self._host.run_diarization_worker(nemotron_proc_entrypoint, args, **callbacks)

    def force_cpu(self) -> None:
        self.settings.device = "cpu"
        if self._settings_store is not None:
            values = self._settings_store.load(self.manifest.id)
            values.update(asdict(self.settings))
            self._settings_store.save(self.manifest.id, values)

    def cancel(self) -> None:
        self._host.cancel()

    def close(self) -> None:
        self._host.close()
