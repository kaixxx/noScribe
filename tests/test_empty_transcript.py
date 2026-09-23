"""A recording without speech still gets its transcript file.

The transcript used to be written only once the first segment arrived. When
Whisper returned none, the log still said "transcript saved" and linked the
file, and for a single html job the editor was opened on it -- but the file did
not exist. It now holds the header, like any other transcript.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("tkinter")

import noScribe.main as m


def _run(tmp_path, monkeypatch, whisper):
    monkeypatch.setattr(m, "config_dir", str(tmp_path))
    monkeypatch.setattr(m, "config", {})
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(16000 * 3, dtype=np.int16), 16000)
    transcript = tmp_path / "transcript.html"
    app = m.HeadlessApp()
    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", whisper)
    job = m.create_transcription_job(
        audio_file=str(audio), transcript_file=str(transcript),
        speaker_detection="none", cli_mode=True,
    )
    app._process_single_job(job)
    return transcript


def test_transcript_without_segments_is_saved(tmp_path, monkeypatch):
    transcript = _run(
        tmp_path, monkeypatch, lambda *args: SimpleNamespace(duration=3.0)
    )

    assert "silence" in transcript.read_text(encoding="utf-8")


def test_failed_transcription_writes_no_file(tmp_path, monkeypatch):
    def whisper(*args):
        raise RuntimeError("worker died")

    with pytest.raises(RuntimeError, match="worker died"):
        _run(tmp_path, monkeypatch, whisper)

    assert not (tmp_path / "transcript.html").exists()
