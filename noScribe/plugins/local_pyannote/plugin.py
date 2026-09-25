"""Plugin adapter for the bundled Pyannote worker."""

from __future__ import annotations

import json
from importlib import resources

from ...inference import LocalInferenceBackend
from ...models import ModelDescriptor, ModelRef
from ..manifest import PluginManifest


def _load_manifest() -> PluginManifest:
    resource = resources.files(__package__) / "backend.json"
    with resource.open("r", encoding="utf-8") as stream:
        return PluginManifest.from_mapping(json.load(stream))


class LocalPyannotePlugin:
    def __init__(self, host: LocalInferenceBackend):
        self.manifest = _load_manifest()
        self._host = host

    def list_models(self) -> list[ModelDescriptor]:
        return [
            ModelDescriptor(
                ref=ModelRef(self.manifest.id, "default"),
                display_name="Pyannote",
                engine=self.manifest.engine or "pyannote",
                capabilities=self.manifest.capabilities,
            )
        ]

    def diarize(self, request, **callbacks):
        return self._host.diarize(request, **callbacks)

    def force_cpu(self) -> None:
        self._host.force_cpu("pyannote")

    def cancel(self) -> None:
        self._host.cancel()

    def close(self) -> None:
        self._host.close()
