from dataclasses import dataclass
from pathlib import Path

import pytest

from noScribe.inference import (
    DiarizationSegment,
    TranscriptionInfo,
    TranscriptionSegment,
    TranscriptionWord,
)
from noScribe.models import ModelDescriptor, ModelRef
from noScribe.server.jobs import JobSnapshot, JobState, JobTask
from noScribe.server.processor import (
    RegistryWorkflowProcessor,
    create_server_registry,
    discover_whisper_models,
    validate_opus_audio,
)
from noScribe.server.config import ServerConfig
from noScribe.audio.convert import ToOpus


class FakeRegistry:
    def __init__(self):
        self.calls = []
        self.cancelled = False

    def list_models(self, capability=None):
        models = [
            ModelDescriptor(
                ModelRef("local-whisper", "precise"),
                "precise",
                "whisper",
                frozenset({"transcription", "word_timestamps"}),
            ),
            ModelDescriptor(
                ModelRef("local-pyannote", "default"),
                "Pyannote",
                "pyannote",
                frozenset({"diarization"}),
            ),
        ]
        if capability:
            return [model for model in models if capability in model.capabilities]
        return models

    def diarize(self, request, **callbacks):
        self.calls.append(("diarization", request))
        callbacks["on_progress"]("segmentation", 50)
        return [DiarizationSegment(500, 1750, "SPEAKER_00")]

    def transcribe(self, request, **callbacks):
        self.calls.append(("transcription", request))
        callbacks["on_segment"](TranscriptionSegment(
            0.5,
            1.5,
            " Hello",
            (TranscriptionWord("Hello", 0.5, 1.5, 0.9),),
        ))
        return TranscriptionInfo(duration=2.0, language="en")

    def cancel(self):
        self.cancelled = True

    def close(self):
        pass


def _job(tasks):
    return JobSnapshot(
        "job-1",
        JobState.RUNNING,
        0,
        tuple(tasks),
        "audio.opus",
        10,
    )


def test_processor_runs_diarization_then_transcription_with_one_audio(tmp_path):
    source = tmp_path / "audio.opus"
    source.write_bytes(b"opus")

    def fake_wav_converter(input_path, output_path):
        assert input_path == source
        output_path.write_bytes(b"wav")

    registry = FakeRegistry()
    processor = RegistryWorkflowProcessor(
        registry,
        wav_converter=fake_wav_converter,
        audio_validator=lambda _path, _limit: 2.0,
    )
    events = []
    processor.process(
        _job([
            JobTask("diarization", "local-pyannote/default", {"num_speakers": 2}),
            JobTask("transcription", "local-whisper/precise", {"language": "en"}),
        ]),
        source,
        events.append,
        lambda: False,
    )

    assert [call[0] for call in registry.calls] == [
        "diarization",
        "transcription",
    ]
    assert registry.calls[0][1].audio_path.endswith(".wav")
    assert registry.calls[1][1].audio_path.endswith(".opus")
    assert not source.with_suffix(".wav").exists()
    assert [event["type"] for event in events].count("task_result") == 2
    diarization_result = next(
        event for event in events if event["type"] == "task_result"
        and event["operation"] == "diarization"
    )
    assert diarization_result["segments"] == [{
        "speaker": "SPEAKER_00", "start": 0.5, "end": 1.75
    }]


def test_processor_catalogue_uses_stable_slash_qualified_ids():
    processor = RegistryWorkflowProcessor(
        FakeRegistry(), audio_validator=lambda _path, _limit: 2.0
    )

    assert [model["id"] for model in processor.list_models()] == [
        "local-whisper/precise",
        "local-pyannote/default",
    ]


def test_processor_rejects_unknown_options(tmp_path):
    source = tmp_path / "audio.opus"
    source.write_bytes(b"opus")
    processor = RegistryWorkflowProcessor(
        FakeRegistry(), audio_validator=lambda _path, _limit: 2.0
    )

    with pytest.raises(ValueError, match="Unsupported transcription options"):
        processor.process(
            _job([JobTask(
                "transcription", "local-whisper/precise", {"device": "cuda"}
            )]),
            source,
            lambda _event: None,
            lambda: False,
        )


def test_model_discovery_ignores_incomplete_directories(tmp_path):
    complete = tmp_path / "precise"
    complete.mkdir()
    (complete / "model.bin").write_bytes(b"model")
    (tmp_path / "incomplete").mkdir()

    assert discover_whisper_models(tmp_path) == {
        "precise": str(complete.resolve())
    }


def test_opus_validator_checks_codec_and_duration(tmp_path):
    source = Path(__file__).parent / "data" / "interview.mp3"
    opus = tmp_path / "interview.opus"
    with ToOpus(source, opus) as converter:
        converter.stop_after(1000)
        while converter.convert():
            pass

    assert validate_opus_audio(opus, 2.0) == pytest.approx(1.0, abs=0.1)
    with pytest.raises(ValueError, match="duration limit"):
        validate_opus_audio(opus, 0.5)


def test_force_cpu_configures_both_builtin_workers(tmp_path):
    model = tmp_path / "precise"
    model.mkdir()
    (model / "model.bin").write_bytes(b"model")
    registry = create_server_registry(ServerConfig(
        whisper_models_dir=tmp_path,
        runtime_dir=tmp_path / "runtime",
        force_cpu=True,
    ))

    whisper_host = registry.get("local-whisper")._host
    pyannote_host = registry.get("local-pyannote")._host
    assert whisper_host is pyannote_host
    assert whisper_host.settings.force_whisper_cpu is True
    assert whisper_host.settings.force_diarization_cpu is True
