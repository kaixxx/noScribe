"""Backend plugin registry and model-based inference routing."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Iterable, Protocol

from ..models import ModelDescriptor, ModelRef
from .manifest import ExecutionType, PluginManifest


class BackendPlugin(Protocol):
    manifest: PluginManifest

    def list_models(self) -> list[ModelDescriptor]: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


class BackendRegistry:
    """Register plugins and route requests using their qualified model refs."""

    def __init__(self, plugins: Iterable[BackendPlugin] = ()):
        self._plugins: dict[str, BackendPlugin] = {}
        self._active: BackendPlugin | None = None
        self._lock = threading.RLock()
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: BackendPlugin) -> None:
        backend_id = plugin.manifest.id
        with self._lock:
            if backend_id in self._plugins:
                raise ValueError(f"Backend {backend_id!r} is already registered.")
            self._plugins[backend_id] = plugin

    def get(self, backend_id: str) -> BackendPlugin:
        with self._lock:
            try:
                return self._plugins[backend_id]
            except KeyError as error:
                raise ValueError(f"Unknown inference backend: {backend_id}") from error

    def list_plugins(self) -> tuple[PluginManifest, ...]:
        with self._lock:
            return tuple(plugin.manifest for plugin in self._plugins.values())

    def list_models(self, capability: str | None = None) -> list[ModelDescriptor]:
        models = []
        with self._lock:
            plugins = tuple(self._plugins.values())
        for plugin in plugins:
            for model in plugin.list_models():
                if capability is None or capability in model.capabilities:
                    models.append(model)
        return models

    def model_options(
        self, capability: str | None = None
    ) -> dict[str, ModelDescriptor]:
        """Return unique, human-readable model labels for user interfaces.

        Qualified model references remain the stable internal identifiers. A
        local backend is only shown when two local models have the same
        display name. Remote models always show their profile name so users
        can see that selecting them sends data to a server.
        """
        models = self.list_models(capability)
        local_name_counts = Counter(
            model.display_name
            for model in models
            if self.get(model.ref.backend_id).manifest.execution.type
            is not ExecutionType.REMOTE
        )

        candidates: list[tuple[str, ModelDescriptor]] = []
        for model in models:
            manifest = self.get(model.ref.backend_id).manifest
            show_backend = (
                manifest.execution.type is ExecutionType.REMOTE
                or local_name_counts[model.display_name] > 1
            )
            label = (
                f"{model.display_name} ({manifest.name})"
                if show_backend
                else model.display_name
            )
            candidates.append((label, model))

        candidate_counts = Counter(label for label, _model in candidates)
        options: dict[str, ModelDescriptor] = {}
        for label, model in candidates:
            if candidate_counts[label] > 1:
                label = f"{model.display_name} ({model.ref.backend_id})"
            if label in options:
                label = f"{model.display_name} ({model.ref})"
            options[label] = model
        return options

    def get_model(self, ref: ModelRef) -> ModelDescriptor:
        plugin = self.get(ref.backend_id)
        for model in plugin.list_models():
            if model.ref == ref:
                return model
        raise ValueError(f"Backend {ref.backend_id!r} has no model {ref.model_id!r}.")

    def transcribe(self, request, **callbacks):
        plugin = self._select(request.model, "transcription")
        operation = getattr(plugin, "transcribe", None)
        if operation is None:
            raise ValueError(f"Backend {plugin.manifest.id!r} cannot transcribe.")
        return self._run(plugin, operation, request, **callbacks)

    def diarize(self, request, **callbacks):
        plugin = self._select(request.model, "diarization")
        operation = getattr(plugin, "diarize", None)
        if operation is None:
            raise ValueError(f"Backend {plugin.manifest.id!r} cannot diarize.")
        return self._run(plugin, operation, request, **callbacks)

    def supports_workflow(self, *refs: ModelRef) -> bool:
        if not refs or len({ref.backend_id for ref in refs}) != 1:
            return False
        plugin = self.get(refs[0].backend_id)
        return bool(getattr(plugin, "supports_workflows", False))

    def run_workflow(self, request, **callbacks):
        refs = [
            operation.model
            for operation in (request.diarization, request.transcription)
            if operation is not None
        ]
        if not self.supports_workflow(*refs):
            raise ValueError("The selected backend does not support workflows.")
        plugin = self.get(refs[0].backend_id)
        return self._run(
            plugin, plugin.run_workflow, request, **callbacks
        )

    def cancel(self) -> None:
        with self._lock:
            active = self._active
        if active is not None:
            active.cancel()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None

    def close(self) -> None:
        self.cancel()
        seen = set()
        with self._lock:
            plugins = tuple(self._plugins.values())
        for plugin in plugins:
            identity = id(plugin)
            if identity not in seen:
                plugin.close()
                seen.add(identity)

    def _select(self, ref: ModelRef, capability: str) -> BackendPlugin:
        model = self.get_model(ref)
        if capability not in model.capabilities:
            raise ValueError(f"Model {ref} does not support {capability}.")
        return self.get(ref.backend_id)

    def _run(self, plugin, operation, request, **callbacks):
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Another inference backend is already active.")
            self._active = plugin
        try:
            return operation(request, **callbacks)
        finally:
            with self._lock:
                if self._active is plugin:
                    self._active = None
