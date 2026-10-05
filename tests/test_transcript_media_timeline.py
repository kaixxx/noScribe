import re

import pytest

import noScribe.main as m
from noScribe.models import ModelRef
from noScribe.inference import (
    DiarizationSegment, InferenceWorkflowResult, TranscriptionInfo, TranscriptionSegment,
)


@pytest.mark.parametrize("workflow", [False, True])
@pytest.mark.parametrize("suffix", ["html", "txt", "vtt"])
@pytest.mark.parametrize("detection", ["none", "auto"])
def test_transcript_uses_actual_converted_media_position(
    monkeypatch, tmp_path, delayed_audio_video, suffix, detection, workflow
):
    monkeypatch.setattr(m, "config_dir", str(tmp_path))
    monkeypatch.setattr(m, "config", {})
    # Exercise conversion and the real transcript writers, without loading models.
    monkeypatch.setattr(m, "get_speech_timestamps", lambda *args: [])
    output = tmp_path / f"transcript.{suffix}"
    job = m.create_transcription_job(
        audio_file=str(delayed_audio_video), transcript_file=str(output),
        start_time=500, stop_time=6000, speaker_detection=detection,
        timestamps=True, pause=1, cli_mode=True,
    )
    logs = []

    def stream(path, job, on_segment):
        # Leave a one-second pause before the first phrase to cover pause anchors.
        on_segment(TranscriptionSegment(1.0, 1.5, " Hello"))
        return {}

    app = m.HeadlessApp()
    monkeypatch.setattr(app, "set_progress", lambda *args: None)
    monkeypatch.setattr(app, "update_queue_table", lambda: None)
    monkeypatch.setattr(app, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "logn", lambda text="", *args, **kwargs: logs.append(text))
    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", stream)
    monkeypatch.setattr(app, "_run_diarize_subprocess", lambda *args: [
        DiarizationSegment(1000, 1500, "SPEAKER_00")
    ])
    if workflow and detection == "auto":
        job.transcription_model = ModelRef("test-server", "whisper")
        job.diarization_model = ModelRef("test-server", "pyannote")
        monkeypatch.setattr(m, "model_uses_remote_backend", lambda *args: True)
        monkeypatch.setattr(app.inference_backend, "supports_workflow", lambda *args: True)

        def run_workflow(request, **callbacks):
            assert request.audio_path.endswith(".flac")
            callbacks["on_diarization_result"]([
                DiarizationSegment(1000, 1500, "SPEAKER_00")
            ])
            callbacks["on_transcription_segment"](TranscriptionSegment(1.0, 1.5, " Hello"))
            return InferenceWorkflowResult(transcription_info=TranscriptionInfo())

        monkeypatch.setattr(app.inference_backend, "run_workflow", run_workflow)
    m.App._process_single_job(app, job)

    text = output.read_text(encoding="utf-8")
    if suffix == "html":
        anchors = re.findall(r'name="ts_(\d+)_(\d+)_', text)
        assert len(anchors) == 2
        pause_start, pause_end = map(int, anchors[0])
        speech_start, speech_end = map(int, anchors[1])
        assert pause_start == pytest.approx(2000, abs=30)
        assert pause_end == speech_start == pytest.approx(3000, abs=30)
        assert speech_end == pytest.approx(3500, abs=30)
    elif suffix == "vtt":
        cue = next(line for line in text.splitlines() if " --> " in line)
        # The pause may be included in the same cue as the following speech.
        start, end = cue.split(" --> ")
        assert start.startswith("00:00:0")
        seconds = lambda value: float(value.split(":")[-1])
        assert seconds(start) >= 1.9
        assert seconds(end) == pytest.approx(3.5, abs=0.03)
    else:
        assert "[00:00:02]" in text or "[00:00:03]" in text
    if detection == "auto":
        assert any("SPEAKER_00" in line and line.startswith("00:00:02.") for line in logs)
    assert job.start == 500  # Repeating the job must keep the user's requested range.
