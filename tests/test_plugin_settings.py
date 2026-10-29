import pytest

from noScribe.inference import LocalWorkerSettings
from noScribe.plugins.factory import create_builtin_registry
from noScribe.plugins.settings import PluginSettingsStore
from noScribe.plugins.local_whisper.settings import WhisperWorkerSettings
from noScribe.plugins.local_pyannote.settings import PyannoteWorkerSettings
from noScribe.server.config import load_server_config
from noScribe.server.processor import create_server_registry


def test_plugin_files_override_legacy_defaults_independently(tmp_path):
    store = PluginSettingsStore(tmp_path)
    store.save("local-whisper", {"device": "cpu", "cpu_threads": 2})
    store.save("local-pyannote", {"device": "cuda"})
    legacy = LocalWorkerSettings(cpu_threads=8, force_diarization_cpu=True, vad_threshold=0.4)
    registry = create_builtin_registry({}, legacy, settings_dir=tmp_path)

    whisper = registry.get("local-whisper")
    pyannote = registry.get("local-pyannote")
    assert whisper.settings.device == "cpu"
    assert whisper.settings.cpu_threads == 2
    assert whisper.settings.vad_threshold == 0.4
    assert pyannote.settings.device == "cuda"
    assert whisper.settings_path == tmp_path / "local-whisper.yml"
    assert pyannote.settings_path == tmp_path / "local-pyannote.yml"
    # The preliminary desktop VAD still reads the legacy settings object.
    assert legacy.cpu_threads == 2


def test_missing_files_use_legacy_defaults_without_creating_files(tmp_path):
    directory = tmp_path / "settings"
    registry = create_builtin_registry({}, LocalWorkerSettings(force_whisper_cpu=True), directory)
    assert registry.get("local-whisper").settings.device == "cpu"
    assert registry.get("local-pyannote").settings.device == "auto"
    assert not directory.exists()


def test_cpu_fallback_is_saved_only_for_affected_plugin(tmp_path):
    store = PluginSettingsStore(tmp_path)
    store.save("local-whisper", {"device": "cuda"})
    store.save("local-pyannote", {"device": "cuda"})
    registry = create_builtin_registry({}, LocalWorkerSettings(), tmp_path)
    registry.get("local-whisper").force_cpu()
    reloaded = create_builtin_registry({}, LocalWorkerSettings(), tmp_path)
    assert reloaded.get("local-whisper").settings.device == "cpu"
    assert reloaded.get("local-pyannote").settings.device == "cuda"


def test_server_settings_are_resolved_beside_server_config(tmp_path):
    directory = tmp_path / "admin-settings"
    PluginSettingsStore(directory).save("local-whisper", {"device": "cpu"})
    PluginSettingsStore(directory).save("local-pyannote", {"device": "cuda"})
    config_path = tmp_path / "server.yml"
    config_path.write_text("plugin_settings_dir: admin-settings\n", encoding="utf-8")
    config = load_server_config(config_path)
    registry = create_server_registry(config)
    assert config.plugin_settings_dir == directory
    assert registry.get("local-whisper").settings.device == "cpu"
    assert registry.get("local-pyannote").settings.device == "cuda"

    config_path.write_text("{}", encoding="utf-8")
    assert load_server_config(config_path).plugin_settings_dir == tmp_path / "plugin-settings"


@pytest.mark.parametrize("plugin_id", ["../outside", "a/b", "C:\\elsewhere", "..", ""])
def test_settings_ids_cannot_escape_directory(tmp_path, plugin_id):
    with pytest.raises(ValueError):
        PluginSettingsStore(tmp_path).path_for(plugin_id)


@pytest.mark.parametrize("text", ["[]", "device: [", "42"])
def test_malformed_settings_report_the_file(tmp_path, text):
    path = tmp_path / "local-whisper.yml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="local-whisper.yml"):
        PluginSettingsStore(tmp_path).load("local-whisper")


@pytest.mark.parametrize("values", [
    {"device": "mps"}, {"cpu_threads": True}, {"cpu_threads": 0},
    {"vad_threshold": float("nan")}, {"vad_threshold": 2}, {"unknown": 1},
])
def test_whisper_validates_its_own_settings(values):
    with pytest.raises(ValueError):
        WhisperWorkerSettings.from_mapping(values)


def test_pyannote_owns_a_different_schema():
    assert PyannoteWorkerSettings.from_mapping({"device": "mps"}).device == "mps"
    with pytest.raises(ValueError, match="Unknown plugin settings"):
        PyannoteWorkerSettings.from_mapping({"cpu_threads": 8})
