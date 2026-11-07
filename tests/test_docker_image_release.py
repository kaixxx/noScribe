"""Release images must not replace latest with a prerelease or an older build."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location(
    "docker_image_release", Path(__file__).resolve().parents[1] / "docker/image_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize("ref,tags,expected", [
    ("refs/tags/v0.8.1", ["v0.8", "v0.8.1"], True),
    ("refs/tags/v0.8", ["v0.8", "v0.8.1"], False),
    ("refs/tags/v0.9-rc1", ["v0.8", "v0.9-rc1"], False),
    ("refs/tags/v0.8", ["v0.8", "v0.9-rc1", "v1.1-enhanced"], True),
    ("refs/tags/v0.10", ["v0.9", "v0.10"], True),
    ("refs/tags/v0.8-r2", ["v0.8-r2", "v0.8-r10"], False),
    ("refs/tags/v0.8-r10", ["v0.8-r2", "v0.8-r10"], True),
    ("refs/tags/v0.8-r2", ["v0.8-r2", "v0.8.0-r1"], True),
    ("refs/tags/v0.8.1", ["v0.8-r10", "v0.8.1"], True),
    ("refs/heads/v0.8", ["v0.8"], False),
    ("refs/heads/main", ["v0.8"], False),
    ("refs/tags/v0.8", [], False),
])
def test_latest_stable_release(ref, tags, expected):
    assert release.is_latest_release(ref, tags) is expected


def test_workflow_output_uses_remote_tags(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v0.8")
    refs = iter(["abc\trefs/tags/v0.8\n", "abc\trefs/tags/v0.8\ndef\trefs/tags/v0.9\n"])

    def remote_tags(command, *, text):
        assert command == ["git", "ls-remote", "--tags", "--refs", "origin", "refs/tags/v*"]
        assert text is True
        return next(refs)

    monkeypatch.setattr(release.subprocess, "check_output", remote_tags)
    release.main()
    assert capsys.readouterr().out == "latest=true\n"
    release.main()
    assert capsys.readouterr().out == "latest=false\n"


def test_remote_failure_stops_publication(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v0.8")

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git ls-remote")

    monkeypatch.setattr(release.subprocess, "check_output", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        release.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/v0.8", "refs/tags/v0.9-rc1"])
def test_non_release_refs_do_not_query_remote(ref, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_REF", ref)
    monkeypatch.setattr(release.subprocess, "check_output",
                        lambda *args, **kwargs: pytest.fail("Unexpected remote query"))
    release.main()
    assert capsys.readouterr().out == "latest=false\n"
