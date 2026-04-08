"""
Integration tests.
"""

import importlib.resources as impres
import shlex

from noScribe import noScribe


def test_transcribe_to_txt(tmp_path):
    path_input = impres.files("tests") / "data" / "interview.mp3"
    path_output = tmp_path / "output.txt"

    noScribe.noScribeMain(
        shlex.split(
            "--no-gui --start 00:00:00 --stop 00:01:00 --model ci "
            f"\"{str(path_input)}\" \"{str(path_output)}\""
        )
    )

    # Check whether output file is present.
    path_output.exists()
    with path_output.open("r", encoding="utf-8") as fd:
        tmp = fd.read()
        assert "Transcribed with noScribe vers." in tmp


def test_transcribe_to_vtt(tmp_path):
    path_input = impres.files("tests") / "data" / "interview.mp3"
    path_output = tmp_path / "output.vtt"

    noScribe.noScribeMain(
        shlex.split(
            "--no-gui --start 00:00:00 --stop 00:01:00 --model ci "
            f"\"{str(path_input)}\" \"{str(path_output)}\""
        )
    )

    # Check whether output file is present.
    path_output.exists()
    with path_output.open("r", encoding="utf-8") as fd:
        tmp = fd.read()
        assert "Transcribed with noScribe vers." in tmp
        tmp.startswith("WEBVTT")


def test_transcribe_to_html(tmp_path):
    path_input = impres.files("tests") / "data" / "interview.mp3"
    path_output = tmp_path / "output.html"

    noScribe.noScribeMain(
        shlex.split(
            "--no-gui --start 00:00:00 --stop 00:01:00 --model ci "
            f"\"{str(path_input)}\" \"{str(path_output)}\""
        )
    )

    # Check whether output file is present.
    path_output.exists()
    with path_output.open("r", encoding="utf-8") as fd:
        tmp = fd.read()
        assert "Transcribed with noScribe vers." in tmp
        tmp.startswith("<!DOCTYPE HTML PUBLIC")
