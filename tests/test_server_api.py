import json
import asyncio
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from noScribe.server import api as server_api
from noScribe.server.api import MAX_RESERVATION_JSON_BYTES, create_app
from noScribe.server.config import ServerConfig
from noScribe.server.jobs import JobScheduler
from noScribe.server.jobs import JobState, JobTask
import pytest


class FakeProcessor:
    def __init__(self):
        self.calls = []
        self.cancelled = False

    def list_models(self):
        return [{
            "id": "precise",
            "name": "precise",
            "engine": "fake",
            "capabilities": ["transcription"],
        }]

    def validate_tasks(self, tasks):
        pass

    def process(self, job, audio_path: Path, emit, is_cancelled):
        self.calls.append((job, audio_path.read_bytes()))
        emit({"type": "status", "message_id": "server.test", "params": {}})

    def cancel(self):
        self.cancelled = True


def _client(tmp_path, **config_values):
    processor = FakeProcessor()
    config = ServerConfig(
        runtime_dir=tmp_path, require_tmpfs=False, **config_values
    )
    scheduler = JobScheduler(max_queued=config.max_queued_jobs)
    return TestClient(create_app(config, processor, scheduler=scheduler)), processor


def _reserve(client, payload=b"fLaC-audio"):
    response = client.post("/v1/audio/jobs", json={
        "tasks": [{"type": "transcription", "model": "precise"}],
        "audio": {"filename": "interview.flac", "size": len(payload)},
    })
    assert response.status_code == 202
    return response.json(), payload


def test_api_reserves_before_upload_and_streams_without_retaining_audio(tmp_path):
    client, processor = _client(tmp_path)
    job, payload = _reserve(client)

    status_response = client.get(
        f"/v1/audio/jobs/{job['job_id']}",
        headers={"X-noScribe-Job-Token": job["job_token"]},
    )
    assert status_response.json()["state"] == "ready_for_upload"

    response = client.post(
        f"/v1/audio/jobs/{job['job_id']}/audio",
        content=payload,
        headers={
            "Content-Type": "audio/flac",
            "X-noScribe-Job-Token": job["job_token"],
        },
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1] == {"type": "result", "ok": True}
    assert processor.calls[0][1] == payload
    assert list(tmp_path.iterdir()) == []


def test_api_does_not_accept_upload_until_job_reaches_front(tmp_path):
    client, _processor = _client(tmp_path)
    first, _ = _reserve(client, b"fLaC-first")
    second, payload = _reserve(client, b"fLaC-second")

    response = client.post(
        f"/v1/audio/jobs/{second['job_id']}/audio",
        content=payload,
        headers={
            "Content-Type": "audio/flac",
            "X-noScribe-Job-Token": second["job_token"],
        },
    )

    assert response.status_code == 409
    assert first["state"] == "ready_for_upload"
    assert second["state"] == "queued"


def test_api_enforces_size_before_and_during_upload(tmp_path):
    client, _processor = _client(tmp_path, max_upload_bytes=10)
    too_large = client.post("/v1/audio/jobs", json={
        "tasks": [{"type": "transcription", "model": "precise"}],
        "audio": {"filename": "large.flac", "size": 11},
    })
    assert too_large.status_code == 413

    job, _ = _reserve(client, b"12345")
    mismatch = client.post(
        f"/v1/audio/jobs/{job['job_id']}/audio",
        content=b"123456",
        headers={
            "Content-Type": "audio/flac",
            "X-noScribe-Job-Token": job["job_token"],
        },
    )
    assert mismatch.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_api_hides_jobs_without_the_capability_token(tmp_path):
    client, _processor = _client(tmp_path)
    job, _ = _reserve(client)

    response = client.get(
        f"/v1/audio/jobs/{job['job_id']}",
        headers={"X-noScribe-Job-Token": "wrong"},
    )

    assert response.status_code == 404


def test_queue_limit_returns_429(tmp_path):
    client, _processor = _client(tmp_path, max_queued_jobs=1)
    _reserve(client, b"one")
    _reserve(client, b"two")

    response = client.post("/v1/audio/jobs", json={
        "tasks": [{"type": "transcription", "model": "precise"}],
        "audio": {"filename": "three.flac", "size": 5},
    })
    assert response.status_code == 429


def test_reservation_limits_json_size_and_task_count(tmp_path):
    client, _processor = _client(tmp_path)
    too_many_tasks = client.post("/v1/audio/jobs", json={
        "tasks": [
            {"type": "transcription", "model": "precise"},
            {"type": "diarization", "model": "speakers"},
            {"type": "transcription", "model": "precise"},
        ],
        "audio": {"filename": "interview.flac", "size": 5},
    })
    assert too_many_tasks.status_code == 422

    duplicate_tasks = client.post("/v1/audio/jobs", json={
        "tasks": [
            {"type": "transcription", "model": "precise"},
            {"type": "transcription", "model": "precise"},
        ],
        "audio": {"filename": "interview.flac", "size": 5},
    })
    assert duplicate_tasks.status_code == 422

    oversized = json.dumps({
        "tasks": [{
            "type": "transcription",
            "model": "precise",
            "options": {"padding": "x" * MAX_RESERVATION_JSON_BYTES},
        }],
        "audio": {"filename": "interview.flac", "size": 5},
    })
    too_large = client.post(
        "/v1/audio/jobs",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert too_large.status_code == 413


def test_tempfile_creation_failure_releases_processing_slot(tmp_path, monkeypatch):
    client, _processor = _client(tmp_path)
    first, first_payload = _reserve(client, b"first")
    second, _ = _reserve(client, b"second")

    def fail_mkstemp(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(server_api.tempfile, "mkstemp", fail_mkstemp)
    response = client.post(
        f"/v1/audio/jobs/{first['job_id']}/audio",
        content=first_payload,
        headers={
            "Content-Type": "audio/flac",
            "X-noScribe-Job-Token": first["job_token"],
        },
    )
    assert response.status_code == 507

    second_status = client.get(
        f"/v1/audio/jobs/{second['job_id']}",
        headers={"X-noScribe-Job-Token": second["job_token"]},
    )
    assert second_status.json()["state"] == "ready_for_upload"


def test_direct_transcription_endpoint_runs_only_when_slot_is_free(tmp_path):
    client, processor = _client(tmp_path)
    payload = b"fLaC-direct"

    response = client.post(
        "/v1/audio/transcriptions",
        params={"model": "precise", "language": "de"},
        content=payload,
        headers={"Content-Type": "audio/flac"},
    )

    assert response.status_code == 200
    assert response.headers["X-noScribe-Job-ID"].startswith("job-")
    assert response.headers["X-noScribe-Job-Token"]
    assert processor.calls[0][0].tasks[0].operation == "transcription"
    assert processor.calls[0][0].tasks[0].options["language"] == "de"


def test_direct_endpoint_rejects_upload_instead_of_queueing_it(tmp_path):
    client, _processor = _client(tmp_path)
    _reserve(client, b"active")

    response = client.post(
        "/v1/audio/transcriptions",
        params={"model": "precise"},
        content=b"fLaC-direct",
        headers={"Content-Type": "audio/flac"},
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"


@pytest.mark.parametrize('timeout', [False, True])
def test_cancellation_keeps_slot_until_processor_exits(tmp_path, timeout):
    started = threading.Event()
    cancelled = threading.Event()
    release = threading.Event()

    class BlockingProcessor(FakeProcessor):
        def process(self, job, audio_path, emit, is_cancelled):
            started.set()
            assert release.wait(5)
            assert is_cancelled()

        def cancel(self):
            cancelled.set()

    scheduler = JobScheduler()
    processor = BlockingProcessor()
    service = server_api.InferenceService(
        ServerConfig(runtime_dir=tmp_path, job_timeout_seconds=0.05 if timeout else 30),
        scheduler, processor,
    )
    first = scheduler.submit([JobTask('transcription', 'precise')], audio_filename='one.flac', audio_size=1)
    second = scheduler.submit([JobTask('transcription', 'precise')], audio_filename='two.flac', audio_size=1)
    path = tmp_path / 'one.flac'
    path.write_bytes(b'x')
    snapshot = scheduler.begin_upload(first.job_id, first.token)
    response = service.stream_processing(snapshot, first.token, path)
    try:
        assert started.wait(3)
        if timeout:
            assert cancelled.wait(3)
        else:
            service.cancel(first.job_id, first.token)
        assert scheduler.get(first.job_id, first.token).state is JobState.CANCELING
        assert scheduler.get(second.job_id, second.token).state is JobState.QUEUED
        assert scheduler.active_job_id == first.job_id
    finally:
        release.set()

    promoted = scheduler.wait_for_change(second.job_id, second.token, JobState.QUEUED, 3)
    assert promoted.state is JobState.READY_FOR_UPLOAD
    assert not path.exists()

    async def collect():
        return [json.loads(line) async for line in response.body_iterator]

    events = asyncio.run(collect())
    assert events[-1]['ok'] is False
    assert events[-1]['error'] == ('job_timeout' if timeout else 'job_cancelled')
    # A late repeat cancel must not touch the processor used by a newer job.
    cancelled.clear()
    service.cancel(first.job_id, first.token)
    assert not cancelled.is_set()
