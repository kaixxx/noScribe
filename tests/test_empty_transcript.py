"""Empty results fail without saving; meaningful partial transcripts survive."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

pytest.importorskip("tkinter")

import noScribe.main as m


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "config_dir", str(tmp_path))
    monkeypatch.setattr(m, "config", {})
    monkeypatch.setattr(m, "get_speech_timestamps", lambda *args: [])
    messages = yaml.safe_load(
        (Path(m.__file__).parent.parent / "trans/noScribe.en.yml").read_text(encoding="utf-8")
    )["en"]
    monkeypatch.setattr(m, "t", lambda key, **kwargs: messages.get(key, key))
    instance = m.HeadlessApp()
    instance.logs = []
    monkeypatch.setattr(instance, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(instance, "logn", lambda text="", *args, **kwargs: instance.logs.append(text))
    monkeypatch.setattr(instance, "set_progress", lambda *args: None)
    monkeypatch.setattr(instance, "update_queue_table", lambda: None)
    monkeypatch.setattr(instance, "update_queue_controls", lambda: None)
    monkeypatch.setattr(instance, "_run_diarize_subprocess", lambda *args: [
        {"start": 0, "end": 3000, "label": "SPEAKER_00"}
    ])
    return instance


def _job(tmp_path, suffix="html", detection="none", name="transcript"):
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(16000 * 3, dtype=np.int16), 16000)
    return m.create_transcription_job(
        audio_file=str(audio), transcript_file=str(tmp_path / f"{name}.{suffix}"),
        speaker_detection=detection, timestamps=True, pause=1, cli_mode=True,
    )


def _segment(text, start=1.0):
    return {"start": start, "end": start + 0.5, "text": text, "words": []}


@pytest.mark.parametrize("suffix", ["html", "txt", "vtt"])
@pytest.mark.parametrize("detection", ["none", "auto"])
@pytest.mark.parametrize("texts", [[], ["", " \t\n ", "\u2003"]])
def test_empty_result_fails_without_saving(app, tmp_path, monkeypatch, suffix, detection, texts):
    def whisper(path, job, on_segment, *speech_map):
        for text in texts:
            on_segment(_segment(text))
        return {}

    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", whisper)
    job = _job(tmp_path, suffix, detection)
    app.queue.add_job(job)
    app.transcription_worker()

    assert job.status == m.JobStatus.ERROR
    assert "No text was recognized" in job.error_message
    assert not job.has_partial_transcript
    assert not Path(job.transcript_file).exists()
    assert not job.speaker_name_map
    assert not any("Saved to:" in line or "Transcription finished" in line for line in app.logs)


def test_empty_job_does_not_open_editor(app, tmp_path, monkeypatch):
    app._headless = False
    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", lambda *args: {})
    monkeypatch.setattr(app, "launch_editor", lambda *args: pytest.fail("Empty job opened editor"))
    app.queue.add_job(_job(tmp_path))
    app.transcription_worker()
    assert app.queue.jobs[0].status == m.JobStatus.ERROR


def test_batch_continues_after_empty_job(app, tmp_path, monkeypatch):
    empty = _job(tmp_path, name="empty")
    valid = _job(tmp_path, name="valid")

    def whisper(path, job, on_segment, *speech_map):
        if job is valid:
            on_segment(_segment(" Hello"))
        return {}

    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", whisper)
    app.queue.add_job(empty)
    app.queue.add_job(valid)
    app.transcription_worker()
    assert empty.status == m.JobStatus.ERROR
    assert not Path(empty.transcript_file).exists()
    assert valid.status == m.JobStatus.FINISHED
    assert "Hello" in Path(valid.transcript_file).read_text(encoding="utf-8")


@pytest.mark.parametrize("suffix", ["html", "txt", "vtt"])
@pytest.mark.parametrize("canceled", [False, True])
def test_failed_or_canceled_job_keeps_real_partial_text(app, tmp_path, monkeypatch, suffix, canceled):
    def whisper(path, job, on_segment, *speech_map):
        on_segment(_segment(" Hello"))
        on_segment(_segment(" \t ", start=2.0))
        if canceled:
            app.cancel = True
            raise RuntimeError(m.t("err_user_cancelation"))
        raise RuntimeError("worker died")

    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", whisper)
    job = _job(tmp_path, suffix)
    app.queue.add_job(job)
    app.transcription_worker()
    assert job.status == (m.JobStatus.CANCELED if canceled else m.JobStatus.ERROR)
    assert job.has_partial_transcript
    assert "Hello" in Path(job.transcript_file).read_text(encoding="utf-8")


def test_empty_cli_result_returns_failure(app, tmp_path, monkeypatch, capsys):
    job = _job(tmp_path)
    app.whisper_models = {"test": object()}
    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", lambda *args: {})
    monkeypatch.setattr(m, "HeadlessApp", lambda: app)
    monkeypatch.setattr(m, "create_job_from_cli_args", lambda args: job)

    assert m.run_cli_mode(SimpleNamespace(model="test")) == 1
    output = capsys.readouterr().out
    assert "No text was recognized" in output
    assert "Output saved to:" not in output
    assert not Path(job.transcript_file).exists()


@pytest.mark.parametrize("detection", ["none", "auto"])
def test_segment_logs_metadata_without_text(app, tmp_path, monkeypatch, capsys, detection):
    # Exercise the actual file logging as well as the live output.
    monkeypatch.setattr(app, "log", m.App.log.__get__(app))
    monkeypatch.setattr(app, "logn", m.App.logn.__get__(app))
    text = " Confidential interview content."

    def whisper(path, job, on_segment, *speech_map):
        on_segment(_segment(text))
        on_segment(_segment(" \t ", start=2.0))
        return {}

    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", whisper)
    job = _job(tmp_path, detection=detection)
    app.queue.add_job(job)
    app.transcription_worker()

    assert job.status == m.JobStatus.FINISHED
    assert text.strip() in capsys.readouterr().out
    assert text.strip() in Path(job.transcript_file).read_text(encoding="utf-8")
    log = (tmp_path / "log" / "transcript.log").read_text(encoding="utf-8")
    assert text.strip() not in log
    segments = [line for line in log.splitlines() if line.startswith("Segment received:")]
    assert segments == [
        f"Segment received: start=00:00:01.000 end=00:00:01.500 chars={len(text)}"
    ]
