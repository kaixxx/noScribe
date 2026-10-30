"""Silero's pause map is computed beside the transcription worker, not before it,
and the pause adjustment still uses it."""
import threading

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("tkinter")

import noScribe.main as m
from noScribe.inference import TranscriptionInfo, TranscriptionSegment


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "config_dir", str(tmp_path))
    monkeypatch.setattr(m, "config", {})
    instance = m.HeadlessApp()
    for name in ("log", "logn", "set_progress", "update_queue_table", "update_queue_controls"):
        monkeypatch.setattr(instance, name, lambda *args, **kwargs: None)
    return instance


def test_the_worker_starts_before_silero_has_finished(app, tmp_path, monkeypatch):
    """Silero is held until the transcription has started: run before the worker,
    as it used to, this job would wait out the timeout. Its pauses still reach
    the transcript: the segment that starts inside a pause is moved out of it."""
    started = threading.Event()

    def silero(audio, options):
        assert started.wait(timeout=20), "the transcription did not start before Silero finished"
        return [{"start": 0, "end": 16000}, {"start": 32000, "end": 48000}]  # speech 0-1 s and 2-3 s

    def whisper(path, job, on_segment, diarization=None, **kwargs):
        started.set()
        on_segment(TranscriptionSegment(1.1, 2.6, " Hallo"))  # starts in the pause 1-2 s
        return TranscriptionInfo(duration=3.0)

    monkeypatch.setattr(m, "get_speech_timestamps", silero)
    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", whisper)
    audio = tmp_path / "a.wav"
    sf.write(audio, np.zeros(16000 * 3, dtype=np.int16), 16000)
    job = m.create_transcription_job(audio_file=str(audio), transcript_file=str(tmp_path / "a.html"),
                                     speaker_detection="none", pause="none", cli_mode=True)
    app.queue.add_job(job)
    app.transcription_worker()
    assert job.status == m.JobStatus.FINISHED, job.error_message
    html = (tmp_path / "a.html").read_text(encoding="utf-8")
    assert "Hallo" in html
    # moved out of the pause to where the speech resumes, 2.0 s (without Silero: 1.1 s)
    assert "ts_2000_2600" in html
