from dataclasses import dataclass

import pytest

from noScribe.inference import (
    InferenceWorkflowRequest,
    TranscriptionRequest,
)
from noScribe.models import ModelDescriptor, ModelRef
from noScribe.plugins import factory
from noScribe.plugins.factory import create_builtin_registry, sync_remote_profiles
from noScribe.plugins.manifest import (
    ExecutionType,
    PluginManifest,
    PLUGIN_PROTOCOL_VERSION,
)
from noScribe.plugins.protocol import WorkerRequest, validate_worker_event
from noScribe.plugins.remote_profiles import (
    REMOTE_PROFILE_SCHEMA_VERSION,
    RemoteBackendProfile,
    delete_remote_profile,
    load_remote_profiles,
    new_remote_profile_id,
    save_remote_profile,
)
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
    supports_workflows: bool = False

    def list_models(self):
        return [ModelDescriptor(
            ref=ModelRef(self.manifest.id, self.model_id),
            display_name=self.model_id,
            engine=self.manifest.engine or "test",
            capabilities=self.manifest.capabilities,
        )]

    def transcribe(self, request, **callbacks):
        return str(request.model)

    def run_workflow(self, request, **callbacks):
        return str(request.transcription.model)

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


def test_registry_unregisters_and_closes_idle_plugin():
    plugin = _Plugin(_manifest("remote", "remote"))
    registry = BackendRegistry([plugin])

    registry.unregister("remote")

    assert plugin.closed is True
    assert registry.list_plugins() == ()


def test_registry_routes_supported_atomic_workflow():
    plugin = _Plugin(_manifest("remote", "remote"), supports_workflows=True)
    registry = BackendRegistry([plugin])
    transcription = TranscriptionRequest(
        "audio.opus", ModelRef("remote", "same-name")
    )

    result = registry.run_workflow(InferenceWorkflowRequest(
        "audio.opus", transcription=transcription
    ))

    assert result == "remote:same-name"


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


def test_remote_profile_validates_and_normalizes_connection():
    profile = RemoteBackendProfile.from_mapping({
        "schema_version": REMOTE_PROFILE_SCHEMA_VERSION,
        "id": "ifs-server",
        "name": "IfS-Server",
        "driver": "noscribe-http-v1",
        "enabled": True,
        "url": "https://noscribe.example.org/",
        "api_key": "secret",
    })

    assert profile.id == "ifs-server"
    assert profile.url == "https://noscribe.example.org"
    assert profile.enabled is True

    local_profile = RemoteBackendProfile.from_mapping({
        "schema_version": REMOTE_PROFILE_SCHEMA_VERSION,
        "id": "development",
        "name": "Development",
        "driver": "noscribe-http-v1",
        "url": "http://127.0.0.1:8000",
        "api_key": "secret",
    })
    assert local_profile.url == "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    "host",
    [
        "10.0.0.1",
        "10.255.255.254",
        "172.16.0.1",
        "172.31.255.254",
        "192.168.0.1",
        "192.168.255.254",
    ],
)
def test_remote_profile_allows_http_for_rfc1918_addresses(host):
    profile = RemoteBackendProfile.from_mapping({
        "schema_version": REMOTE_PROFILE_SCHEMA_VERSION,
        "id": "lan-server",
        "name": "LAN server",
        "driver": "noscribe-http-v1",
        "url": f"http://{host}:8765/noscribe",
        "api_key": "secret",
    })

    assert profile.url == f"http://{host}:8765/noscribe"


@pytest.mark.parametrize(
    "host",
    [
        "9.255.255.255",
        "11.0.0.1",
        "172.15.255.255",
        "172.32.0.1",
        "192.167.255.255",
        "192.169.0.1",
        "100.64.0.1",
        "169.254.1.1",
        "server.lan",
        "[fc00::1]",
    ],
)
def test_remote_profile_rejects_http_outside_loopback_and_rfc1918(host):
    with pytest.raises(ValueError, match="require HTTPS"):
        RemoteBackendProfile.from_mapping({
            "schema_version": REMOTE_PROFILE_SCHEMA_VERSION,
            "id": "insecure-server",
            "name": "Insecure server",
            "driver": "noscribe-http-v1",
            "url": f"http://{host}:8765/noscribe",
            "api_key": "secret",
        })


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "schema version"),
        ("schema_version", True, "schema version"),
        ("id", "bad:id", "must not contain"),
        ("url", "file:///tmp/server", "HTTP"),
        ("url", "http://example.org", "require HTTPS"),
        ("url", "https://user:password@example.org", "credentials"),
        ("enabled", "yes", "boolean"),
        ("api_key", "", "non-empty string"),
    ],
)
def test_remote_profile_rejects_invalid_values(field, value, match):
    config = {
        "schema_version": REMOTE_PROFILE_SCHEMA_VERSION,
        "id": "ifs-server",
        "name": "IfS-Server",
        "driver": "noscribe-http-v1",
        "enabled": True,
        "url": "https://noscribe.example.org",
        "api_key": "secret",
    }
    config[field] = value

    with pytest.raises(ValueError, match=match):
        RemoteBackendProfile.from_mapping(config)


def test_remote_profile_loader_isolates_invalid_files_and_duplicate_ids(tmp_path):
    profiles_dir = tmp_path / "backends"
    profiles_dir.mkdir()
    (profiles_dir / "01-first.yml").write_text(
        "\n".join([
            "schema_version: 1",
            "id: ifs-server",
            "name: IfS-Server",
            "driver: noscribe-http-v1",
            "url: https://noscribe.example.org",
            "api_key: secret",
        ]),
        encoding="utf-8",
    )
    (profiles_dir / "02-duplicate.yaml").write_text(
        "\n".join([
            "schema_version: 1",
            "id: ifs-server",
            "name: Duplicate",
            "driver: noscribe-http-v1",
            "url: https://duplicate.example.org",
            "api_key: secret",
        ]),
        encoding="utf-8",
    )
    (profiles_dir / "invalid.yml").write_text("- not-a-mapping\n", encoding="utf-8")

    result = load_remote_profiles(profiles_dir)

    assert [profile.name for profile in result.profiles] == ["IfS-Server"]
    assert len(result.errors) == 2
    assert {error.path.name for error in result.errors} == {
        "02-duplicate.yaml",
        "invalid.yml",
    }


def test_remote_profile_loader_creates_missing_directory(tmp_path):
    profiles_dir = tmp_path / "backends"

    result = load_remote_profiles(profiles_dir)

    assert profiles_dir.is_dir()
    assert result.profiles == ()
    assert result.errors == ()


def test_remote_profile_manager_saves_updates_and_deletes_atomically(tmp_path):
    profiles_dir = tmp_path / "backends"
    profile_id = new_remote_profile_id("IfS Server", set())
    assert profile_id == "ifs-server"
    assert new_remote_profile_id("IfS Server", {profile_id}) == "ifs-server-2"

    profile = RemoteBackendProfile.from_mapping({
        "schema_version": REMOTE_PROFILE_SCHEMA_VERSION,
        "id": profile_id,
        "name": "IfS Server",
        "driver": "noscribe-http-v1",
        "enabled": True,
        "url": "https://noscribe.example.org/",
        "api_key": "visible-secret",
    })
    saved = save_remote_profile(profiles_dir, profile)

    assert saved.source == profiles_dir / "ifs-server.yml"
    assert saved.url == "https://noscribe.example.org"
    assert load_remote_profiles(profiles_dir).profiles == (saved,)

    updated = RemoteBackendProfile(
        **{**saved.__dict__, "name": "Updated server", "enabled": False}
    )
    updated = save_remote_profile(profiles_dir, updated)
    assert len(tuple(profiles_dir.glob("*.yml"))) == 1
    assert load_remote_profiles(profiles_dir).profiles == (updated,)

    delete_remote_profile(profiles_dir, updated)
    assert load_remote_profiles(profiles_dir).profiles == ()


def test_remote_profile_sync_replaces_changed_and_removes_disabled(monkeypatch):
    created = []

    class FakeRemotePlugin(_Plugin):
        def __init__(self, profile):
            super().__init__(_manifest(
                profile.id, "remote", name=profile.name
            ))
            self.profile = profile
            self.refresh_count = 0
            created.append(self)

        def refresh_models(self):
            self.refresh_count += 1

    monkeypatch.setattr(factory, "RemoteHttpPlugin", FakeRemotePlugin)
    profile = RemoteBackendProfile(
        id="server",
        name="Server",
        driver="noscribe-http-v1",
        url="https://server.example.org",
        api_key="first",
    )
    registry = BackendRegistry()

    assert sync_remote_profiles(registry, (profile,)) == ()
    first = registry.get("server")
    assert first is created[0]

    changed = RemoteBackendProfile(
        **{**profile.__dict__, "api_key": "second"}
    )
    assert sync_remote_profiles(registry, (changed,)) == ()
    assert first.closed is True
    assert registry.get("server").profile.api_key == "second"

    disabled = RemoteBackendProfile(
        **{**changed.__dict__, "enabled": False}
    )
    assert sync_remote_profiles(registry, (disabled,)) == ()
    with pytest.raises(ValueError, match="Unknown inference backend"):
        registry.get("server")
