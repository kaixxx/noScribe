from collections import deque
from queue import Empty

import pytest

import noScribe.inference as inference
from noScribe.inference import (
    DiarizationRequest,
    DiarizationSegment,
    LOCAL_WHISPER_BACKEND,
    LocalWorkerSettings,
    ModelRef,
    TranscriptionInfo,
    TranscriptionRequest,
    TranscriptionSegment,
)


class _FakeQueue:
    def __init__(self, messages):
        self.messages = deque(messages)
        self.closed = False

    def get(self, timeout):
        if not self.messages:
            raise Empty
        return self.messages.popleft()

    def close(self):
        self.closed = True

    def join_thread(self):
        pass


class _FakeProcess:
    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.exitcode = 0
        self.alive = False
        self.closed = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.alive = False

    def terminate(self):
        self.alive = False

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self, messages):
        self.messages = messages
        self.queue = None
        self.process = None

    def Queue(self):
        self.queue = _FakeQueue(self.messages)
        return self.queue

    def Process(self, target, args):
        self.process = _FakeProcess(target, args)
        return self.process


def test_model_reference_round_trip_and_validation():
    model = ModelRef.parse("local-whisper:my-model")

    assert model == ModelRef("local-whisper", "my-model")
    assert str(model) == "local-whisper:my-model"

    with pytest.raises(ValueError):
        ModelRef.parse("missing-backend")


def test_transcription_request_contains_only_job_options():
    request = TranscriptionRequest(
        audio_path="recording.opus",
        model=ModelRef(LOCAL_WHISPER_BACKEND, "precise"),
        language="de",
        include_disfluencies=False,
    )

    assert request.audio_path == "recording.opus"
    assert str(request.model) == "local-whisper:precise"
    assert request.language == "de"
    assert not request.include_disfluencies
    assert not hasattr(request, "device")
    assert not hasattr(request, "locale")


def test_transcription_segment_converts_worker_message_and_remains_adjustable():
    segment = TranscriptionSegment.from_mapping(
        {
            "start": 1.25,
            "end": 2.5,
            "text": " Hello",
            "words": [
                {"word": "Hello", "start": 1.25, "end": 2.0, "prob": 0.9}
            ],
        }
    )

    segment.start = 1.5

    assert segment.start == 1.5
    assert segment.text == " Hello"
    assert segment.words[0].word == "Hello"
    assert segment.words[0].probability == 0.9


def test_transcription_info_converts_optional_worker_metadata():
    info = TranscriptionInfo.from_mapping(
        {
            "duration": 12.3,
            "language": "de",
            "language_probability": 0.98,
            "sample_rate": 16000,
        }
    )

    assert info.duration == 12.3
    assert info.language == "de"
    assert info.language_probability == 0.98
    assert info.sample_rate == 16000


def test_diarization_request_and_segment_use_backend_independent_values():
    request = DiarizationRequest(
        audio_path="recording.opus",
        num_speakers=3,
    )
    segment = DiarizationSegment.from_mapping(
        {"start": 520, "end": 4730, "label": "SPEAKER_00"}
    )

    assert request.audio_path == "recording.opus"
    assert request.num_speakers == 3
    assert str(request.model) == "local-pyannote:default"
    assert segment == DiarizationSegment(520, 4730, "SPEAKER_00")


def test_local_backend_translates_worker_stream(monkeypatch):
    context = _FakeContext(
        [
            {"type": "log", "level": "info", "msg": "loaded"},
            {
                "type": "status",
                "level": "info",
                "message_id": "language_detect",
                "params": {"lang": "en", "prob": "0.99"},
            },
            {"type": "progress", "pct": 25, "detail": "audio"},
            {
                "type": "segment",
                "segment": {"start": 0.5, "end": 1.5, "text": "Hello"},
            },
            {
                "type": "result",
                "ok": True,
                "info": {"duration": 1.5, "language": "en"},
            },
        ]
    )
    monkeypatch.setattr(inference.mp, "get_context", lambda method: context)
    logs = []
    progress = []
    segments = []
    statuses = []

    backend = inference.LocalInferenceBackend(
        whisper_models={"model": "models/model"},
        settings=LocalWorkerSettings(cpu_threads=8, vad_threshold=0.42),
    )
    result = backend.transcribe(
        TranscriptionRequest(
            "audio.opus", ModelRef(LOCAL_WHISPER_BACKEND, "model")
        ),
        on_segment=segments.append,
        on_log=lambda level, message: logs.append((level, message)),
        on_status=lambda message_id, params, level: statuses.append(
            (message_id, params, level)
        ),
        on_progress=lambda percent, detail: progress.append((percent, detail)),
    )

    assert result == TranscriptionInfo(duration=1.5, language="en")
    assert segments == [TranscriptionSegment(0.5, 1.5, "Hello")]
    assert logs == [("info", "loaded")]
    assert statuses == [
        ("language_detect", {"lang": "en", "prob": "0.99"}, "info")
    ]
    assert progress == [(25.0, "audio")]
    worker_args = context.process.args[0]
    assert worker_args["model_path"] == "models/model"
    assert worker_args["cpu_threads"] == 8
    assert worker_args["vad_threshold"] == 0.42
    assert worker_args["word_timestamps"] is True
    assert worker_args["local_files_only"] is True
    assert "locale" not in worker_args
    assert context.process.closed
    assert context.queue.closed


def test_local_backend_preserves_worker_error_trace(monkeypatch):
    context = _FakeContext(
        [
            {
                "type": "result",
                "ok": False,
                "error": "model failed",
                "trace": "worker traceback",
            }
        ]
    )
    monkeypatch.setattr(inference.mp, "get_context", lambda method: context)

    with pytest.raises(inference.InferenceWorkerError) as caught:
        inference.LocalInferenceBackend().diarize(
            DiarizationRequest("audio.opus")
        )

    assert str(caught.value) == "model failed"
    assert caught.value.trace == "worker traceback"
    assert context.process.closed
    assert context.queue.closed
