from pathlib import Path

import pytest

from noScribe.server.config import ServerConfig, load_server_config


def test_server_config_loads_paths_from_yaml(tmp_path):
    config_path = tmp_path / "server.yml"
    config_path.write_text(
        "\n".join([
            "host: 127.0.0.1",
            "port: 9000",
            "whisper_models_dir: /opt/models",
            "runtime_dir: /run/noscribe",
            "require_tmpfs: true",
        ]),
        encoding="utf-8",
    )

    config = load_server_config(config_path)

    assert config.whisper_models_dir == Path("/opt/models")
    assert config.runtime_dir == Path("/run/noscribe")


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.2", "example.org"])
def test_server_config_refuses_non_loopback_binding(host):
    with pytest.raises(ValueError, match="loopback"):
        ServerConfig(host=host)


def test_server_config_refuses_string_booleans():
    with pytest.raises(ValueError, match="require_tmpfs must be a boolean"):
        ServerConfig(require_tmpfs="false")


@pytest.mark.parametrize("name", ["force_cpu", "cpu_threads", "vad_threshold"])
def test_server_config_rejects_central_worker_settings(name):
    with pytest.raises(ValueError, match="Unknown server configuration fields"):
        ServerConfig.from_mapping({name: 1})
