import json
from pathlib import Path

from fastapi.testclient import TestClient

from noScribe.server.api import create_app
from noScribe.server.config import ServerConfig
from noScribe.server.jobs import JobScheduler


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
    config = ServerConfig(runtime_dir=tmp_path, **config_values)
    scheduler = JobScheduler(max_queued=config.max_queued_jobs)
    return TestClient(create_app(config, processor, scheduler=scheduler)), processor


def _reserve(client, payload=b"OggS-audio"):
    response = client.post("/v1/audio/jobs", json={
        "tasks": [{"type": "transcription", "model": "precise"}],
        "audio": {"filename": "interview.opus", "size": len(payload)},
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
            "Content-Type": "audio/ogg",
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
    first, _ = _reserve(client, b"OggS-first")
    second, payload = _reserve(client, b"OggS-second")

    response = client.post(
        f"/v1/audio/jobs/{second['job_id']}/audio",
        content=payload,
        headers={"X-noScribe-Job-Token": second["job_token"]},
    )

    assert response.status_code == 409
    assert first["state"] == "ready_for_upload"
    assert second["state"] == "queued"


def test_api_enforces_size_before_and_during_upload(tmp_path):
    client, _processor = _client(tmp_path, max_upload_bytes=10)
    too_large = client.post("/v1/audio/jobs", json={
        "tasks": [{"type": "transcription", "model": "precise"}],
        "audio": {"filename": "large.opus", "size": 11},
    })
    assert too_large.status_code == 413

    job, _ = _reserve(client, b"12345")
    mismatch = client.post(
        f"/v1/audio/jobs/{job['job_id']}/audio",
        content=b"123456",
        headers={"X-noScribe-Job-Token": job["job_token"]},
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
        "audio": {"filename": "three.opus", "size": 5},
    })
    assert response.status_code == 429
