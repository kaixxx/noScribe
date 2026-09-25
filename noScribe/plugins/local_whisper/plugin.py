"""Plugin adapter for the bundled Faster Whisper worker."""

from __future__ import annotations

import json
from importlib import resources
from typing import Mapping

from ...inference import LocalInferenceBackend
from ...models import ModelDescriptor, ModelRef
from ..manifest import PluginManifest


def _load_manifest() -> PluginManifest:
    resource = resources.files(__package__) / "backend.json"
    with resource.open("r", encoding="utf-8") as stream:
        return PluginManifest.from_mapping(json.load(stream))


class LocalWhisperPlugin:
    def __init__(self, host: LocalInferenceBackend, model_paths: Mapping[str, str]):
        self.manifest = _load_manifest()
        self._host = host
        self._model_paths = dict(model_paths)

    def list_models(self) -> list[ModelDescriptor]:
        return [
            ModelDescriptor(
                ref=ModelRef(self.manifest.id, model_id),
                display_name=model_id,
                engine=self.manifest.engine or "whisper",
                capabilities=self.manifest.capabilities,
            )
            for model_id in self._model_paths
        ]

    def transcribe(self, request, **callbacks):
        return self._host.transcribe(request, **callbacks)

    def force_cpu(self) -> None:
        self._host.force_cpu("whisper")

    def cancel(self) -> None:
        self._host.cancel()

    def close(self) -> None:
        self._host.close()
