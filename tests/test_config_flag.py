"""On/off settings in a hand-edited config.yml, where an unquoted true or False
is a YAML boolean and 0 an integer, not the strings noScribe writes.
`force_whisper_cpu: true` used to stop noScribe at startup, `auto_save: False`
was ignored, and `check_for_update: True` switched the update check off."""
import os
import subprocess
import sys
import textwrap

import pytest
import yaml

import noScribe.main as m


@pytest.fixture
def config(monkeypatch):
    config = yaml.safe_load(
        "quoted: 'False'\nbare: true\nzero: 0\nquoted_off: 'off'\ngarbage: maybe\n"
    )
    monkeypatch.setattr(m, "config", config)
    return config


@pytest.mark.parametrize(
    "key, default, expected",
    [
        ("quoted", True, False),
        ("bare", False, True),
        ("zero", True, False),
        ("quoted_off", True, False),
        ("garbage", True, True),
        ("garbage", False, False),
    ],
)
def test_flag_reads_strings_and_yaml_booleans(config, key, default, expected):
    assert m.get_config_flag(key, default) is expected


@pytest.mark.parametrize("default", [True, False])
def test_missing_flag_is_stored_as_noscribe_writes_it(config, default):
    assert m.get_config_flag("missing", default) is default
    assert config["missing"] == str(default)


def test_job_honours_bare_false_auto_save(config):
    config["auto_save"] = False
    assert m.create_transcription_job(cli_mode=True).auto_save is False


def test_startup_reads_a_bare_true_force_whisper_cpu(tmp_path):
    """noScribe reads force_whisper_cpu while it is imported, from its config
    directory; so this runs in a child whose appdirs points into tmp_path.
    (Environment variables would not do: on Windows appdirs asks the shell.)"""
    child = textwrap.dedent(f"""
        import appdirs
        appdirs.user_config_dir = lambda *args, **kwargs: {str(tmp_path)!r}
        with open({str(tmp_path / 'config.yml')!r}, 'w') as f:
            f.write('force_whisper_cpu: true\\n')
        import noScribe.main as m
        print(m.force_whisper_cpu, m.force_pyannote_cpu)
    """)
    result = subprocess.run([sys.executable, "-c", child], capture_output=True,
                            text=True, cwd=os.getcwd())
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ["True", "False"]
