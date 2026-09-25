from dataclasses import dataclass

import pytest

from noScribe.inference import TranscriptionRequest
from noScribe.models import ModelDescriptor, ModelRef
from noScribe.plugins.factory import create_builtin_registry
from noScribe.plugins.manifest import (
    ExecutionType,
    PluginManifest,
    PLUGIN_PROTOCOL_VERSION,
)
from noScribe.plugins.protocol import WorkerRequest, validate_worker_event
from noScribe.plugins.registry import BackendRegistry
from noScribe.inference import LocalWorkerSettings


def _manifest(
    backend_id,
    execution_type="builtin",
    capability="transcription",
    name=None,
):
    execution = {
        "type": execution_type,
        {
            "builtin": "adapter",
            "external": "command",
            "remote": "driver",
        }[execution_type]: "test",
    }
    return PluginManifest.from_mapping({
        "id": backend_id,
        "name": name or backend_id,
        "version": "1.0.0",
        "protocol_version": PLUGIN_PROTOCOL_VERSION,
        "engine": "test",
        "execution": execution,
        "capabilities": [capability],
    })


@dataclass
class _Plugin:
    manifest: PluginManifest
    model_id: str = "same-name"
    canceled: bool = False
    closed: bool = False

    def list_models(self):
        return [ModelDescriptor(
            ref=ModelRef(self.manifest.id, self.model_id),
            display_name=self.model_id,
            engine=self.manifest.engine or "test",
            capabilities=self.manifest.capabilities,
        )]

    def transcribe(self, request, **callbacks):
        return str(request.model)

    def cancel(self):
        self.canceled = True

    def close(self):
        self.closed = True


def test_manifest_supports_all_planned_execution_types():
    assert _manifest("builtin", "builtin").execution.type is ExecutionType.BUILTIN
    assert _manifest("external", "external").execution.type is ExecutionType.EXTERNAL
    assert _manifest("remote", "remote").execution.type is ExecutionType.REMOTE


def test_manifest_rejects_incompatible_protocol_version():
    value = {
        "id": "future",
        "name": "Future",
        "version": "1",
        "protocol_version": PLUGIN_PROTOCOL_VERSION + 1,
        "engine": "test",
        "execution": {"type": "builtin", "adapter": "test"},
    }

    with pytest.raises(ValueError, match="uses protocol"):
        PluginManifest.from_mapping(value)


def test_registry_distinguishes_same_model_id_by_backend():
    first = _Plugin(_manifest("first"))
    second = _Plugin(_manifest("second"))
    registry = BackendRegistry([first, second])

    assert {str(model.ref) for model in registry.list_models("transcription")} == {
        "first:same-name",
        "second:same-name",
    }
    request = TranscriptionRequest("audio.opus", ModelRef("second", "same-name"))
    assert registry.transcribe(request) == "second:same-name"


def test_model_options_hide_unique_local_backend_names():
    registry = BackendRegistry([
        _Plugin(_manifest("whisper", name="Local Whisper"), "precise"),
        _Plugin(_manifest("voxtral", name="Local Voxtral"), "voxtral-small"),
    ])

    assert list(registry.model_options("transcription")) == [
        "precise",
        "voxtral-small",
    ]


def test_model_options_disambiguate_duplicate_local_names():
    registry = BackendRegistry([
        _Plugin(_manifest("whisper", name="Whisper"), "small"),
        _Plugin(_manifest("voxtral", name="Voxtral"), "small"),
    ])

    assert list(registry.model_options("transcription")) == [
        "small (Whisper)",
        "small (Voxtral)",
    ]


def test_model_options_always_identify_remote_profile():
    registry = BackendRegistry([
        _Plugin(_manifest("local", name="Local Whisper"), "precise"),
        _Plugin(
            _manifest("ifs-server", "remote", name="IfS-Server"),
            "precise",
        ),
    ])

    assert list(registry.model_options("transcription")) == [
        "precise",
        "precise (IfS-Server)",
    ]


def test_registry_rejects_unknown_backend_and_duplicate_registration():
    plugin = _Plugin(_manifest("one"))
    registry = BackendRegistry([plugin])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(plugin)
    with pytest.raises(ValueError, match="Unknown inference backend"):
        registry.get("missing")


def test_builtin_plugins_load_manifests_and_advertise_models():
    registry = create_builtin_registry(
        {"precise": "models/precise", "custom": "models/custom"},
        LocalWorkerSettings(),
    )

    manifests = {manifest.id: manifest for manifest in registry.list_plugins()}
    assert manifests["local-whisper"].execution.type is ExecutionType.BUILTIN
    assert manifests["local-pyannote"].execution.type is ExecutionType.BUILTIN
    assert {str(model.ref) for model in registry.list_models("transcription")} == {
        "local-whisper:precise",
        "local-whisper:custom",
    }
    assert [
        str(model.ref) for model in registry.list_models("diarization")
    ] == ["local-pyannote:default"]


def test_external_worker_protocol_has_versioned_request_and_shared_events():
    request = WorkerRequest("job-1", "transcribe", {"model": "precise"})

    assert request.to_mapping() == {
        "protocol_version": PLUGIN_PROTOCOL_VERSION,
        "id": "job-1",
        "method": "transcribe",
        "params": {"model": "precise"},
    }
    assert validate_worker_event({"type": "status", "message_id": "vad"}) == {
        "type": "status",
        "message_id": "vad",
    }
    with pytest.raises(ValueError, match="Unsupported worker event"):
        validate_worker_event({"type": "surprise"})
