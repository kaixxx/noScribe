"""FastAPI transport for queued, zero-retention inference workflows."""

from __future__ import annotations

import json
import asyncio
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Protocol

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import ServerConfig
from .jobs import (
    InvalidJobState,
    JobNotFound,
    JobScheduler,
    JobSnapshot,
    JobState,
    JobTask,
    QueueFull,
)
from .storage import prepare_runtime_directory


SERVER_PROTOCOL_VERSION = 1
_END = object()


class WorkflowProcessor(Protocol):
    def list_models(self) -> list[Mapping[str, object]]: ...

    def validate_tasks(self, tasks: tuple[JobTask, ...]) -> None: ...

    def process(
        self,
        job: JobSnapshot,
        audio_path: Path,
        emit: Callable[[dict], None],
        is_cancelled: Callable[[], bool],
    ) -> None: ...

    def cancel(self) -> None: ...


class TaskBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    model: str = Field(min_length=1)
    options: dict[str, object] = Field(default_factory=dict)


class AudioBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1)
    size: int = Field(gt=0)


class JobBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[TaskBody] = Field(min_length=1)
    audio: AudioBody


class InferenceService:
    def __init__(
        self,
        config: ServerConfig,
        scheduler: JobScheduler,
        processor: WorkflowProcessor,
    ):
        self.config = config
        self.scheduler = scheduler
        self.processor = processor
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        prepare_runtime_directory(
            self.config.runtime_dir, require_tmpfs=self.config.require_tmpfs
        )

    async def receive_audio(
        self,
        request: Request,
        snapshot: JobSnapshot,
        token: str,
    ) -> Path:
        if request.headers.get("content-type", "").split(";", 1)[0] not in {
            "audio/ogg",
            "audio/opus",
            "application/octet-stream",
        }:
            raise HTTPException(415, "Expected an Opus audio request body.")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as error:
                raise HTTPException(400, "Invalid Content-Length header.") from error
            if declared_length != snapshot.audio_size:
                raise HTTPException(400, "Upload size differs from the reservation.")

        self.scheduler.begin_upload(snapshot.job_id, token)
        descriptor, filename = tempfile.mkstemp(
            prefix="upload-", suffix=".opus", dir=self.config.runtime_dir
        )
        path = Path(filename)
        received = 0
        started = time.monotonic()
        try:
            with os.fdopen(descriptor, "wb") as stream:
                chunks = request.stream().__aiter__()
                while True:
                    remaining = (
                        self.config.upload_timeout_seconds
                        - (time.monotonic() - started)
                    )
                    if remaining <= 0:
                        raise HTTPException(408, "Audio upload timed out.")
                    try:
                        chunk = await asyncio.wait_for(
                            anext(chunks), timeout=remaining
                        )
                    except StopAsyncIteration:
                        break
                    except TimeoutError as error:
                        raise HTTPException(408, "Audio upload timed out.") from error
                    received += len(chunk)
                    if received > self.config.max_upload_bytes:
                        raise HTTPException(413, "Audio upload is too large.")
                    if received > snapshot.audio_size:
                        raise HTTPException(400, "Upload exceeds its reserved size.")
                    stream.write(chunk)
            if received != snapshot.audio_size:
                raise HTTPException(400, "Upload size differs from the reservation.")
            return path
        except BaseException:
            path.unlink(missing_ok=True)
            try:
                self.scheduler.fail(snapshot.job_id, token, "upload_failed")
            except InvalidJobState:
                # Uploading is not RUNNING yet; cancellation releases the slot.
                self.scheduler.cancel(snapshot.job_id, token)
            raise

    def stream_processing(
        self, snapshot: JobSnapshot, token: str, audio_path: Path
    ) -> StreamingResponse:
        events: queue.Queue[object] = queue.Queue(maxsize=100)
        cancel_event = threading.Event()
        timed_out = threading.Event()
        finished = threading.Event()
        with self._lock:
            self._cancel_events[snapshot.job_id] = cancel_event

        def emit(event: dict) -> None:
            while not cancel_event.is_set():
                try:
                    events.put(event, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def run() -> None:
            def timeout_job() -> None:
                timed_out.set()
                cancel_event.set()
                self.processor.cancel()

            timer = threading.Timer(self.config.job_timeout_seconds, timeout_job)
            timer.daemon = True
            timer.start()
            try:
                self.scheduler.begin_processing(snapshot.job_id, token)
                self.processor.process(
                    snapshot, audio_path, emit, cancel_event.is_set
                )
                if cancel_event.is_set():
                    self.scheduler.cancel(snapshot.job_id, token)
                else:
                    self.scheduler.complete(snapshot.job_id, token)
                    emit({"type": "result", "ok": True})
            except Exception:
                if timed_out.is_set():
                    error_code = "job_timeout"
                elif cancel_event.is_set():
                    error_code = "job_cancelled"
                else:
                    error_code = "server_job_failed"
                try:
                    if error_code == "job_cancelled":
                        self.scheduler.cancel(snapshot.job_id, token)
                    else:
                        self.scheduler.fail(snapshot.job_id, token, error_code)
                except (InvalidJobState, JobNotFound):
                    pass
                emit({"type": "result", "ok": False, "error": error_code})
            finally:
                timer.cancel()
                audio_path.unlink(missing_ok=True)
                finished.set()
                with self._lock:
                    self._cancel_events.pop(snapshot.job_id, None)
                try:
                    events.put(_END, timeout=1)
                except queue.Full:
                    pass

        worker = threading.Thread(
            target=run, name=f"noscribe-{snapshot.job_id}", daemon=True
        )
        worker.start()

        def generate():
            completed_stream = False
            try:
                while True:
                    event = events.get()
                    if event is _END:
                        completed_stream = True
                        return
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            finally:
                if not completed_stream and not finished.is_set():
                    cancel_event.set()
                    self.processor.cancel()

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    def cancel(self, job_id: str, token: str) -> JobSnapshot:
        snapshot = self.scheduler.cancel(job_id, token)
        with self._lock:
            event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()
            self.processor.cancel()
        return snapshot


def create_app(
    config: ServerConfig,
    processor: WorkflowProcessor,
    *,
    scheduler: JobScheduler | None = None,
) -> FastAPI:
    scheduler = scheduler or JobScheduler(
        max_queued=config.max_queued_jobs,
        reservation_ttl=config.reservation_ttl_seconds,
        upload_ready_ttl=config.upload_ready_ttl_seconds,
        terminal_ttl=config.terminal_job_ttl_seconds,
    )
    service = InferenceService(config, scheduler, processor)
    app = FastAPI(title="noScribe inference server", version="0.1.0")
    app.state.inference_service = service

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(JobNotFound)
    async def unknown_job(_request: Request, _error: JobNotFound):
        return JSONResponse(status_code=404, content={"error": "unknown_job"})

    @app.exception_handler(InvalidJobState)
    async def invalid_state(_request: Request, error: InvalidJobState):
        return JSONResponse(
            status_code=409,
            content={"error": "invalid_job_state", "detail": str(error)},
        )

    @app.get("/health")
    def health():
        return {"ok": True, "active_job": scheduler.active_job_id is not None}

    @app.get("/v1/models")
    def models():
        return {
            "protocol_version": SERVER_PROTOCOL_VERSION,
            "server_version": app.version,
            "features": ["queued_workflows"],
            "data": processor.list_models(),
        }

    @app.post("/v1/audio/jobs", status_code=status.HTTP_202_ACCEPTED)
    def reserve_job(body: JobBody):
        if body.audio.size > config.max_upload_bytes:
            raise HTTPException(413, "Reserved audio is too large.")
        if not body.audio.filename.casefold().endswith(".opus"):
            raise HTTPException(400, "Server workflows require Opus audio.")
        try:
            tasks = tuple(
                JobTask(task.type, task.model, task.options) for task in body.tasks
            )
            processor.validate_tasks(tasks)
            admission = scheduler.submit(
                tasks,
                audio_filename=Path(body.audio.filename).name,
                audio_size=body.audio.size,
            )
        except QueueFull as error:
            raise HTTPException(429, "The inference queue is full.") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {
            "job_id": admission.job_id,
            "job_token": admission.token,
            "state": admission.state,
            "position": admission.position,
        }

    @app.get("/v1/audio/jobs/{job_id}")
    def job_status(
        job_id: str,
        x_noscribe_job_token: str = Header(alias="X-noScribe-Job-Token"),
    ):
        return _snapshot_response(scheduler.get(job_id, x_noscribe_job_token))

    @app.delete("/v1/audio/jobs/{job_id}")
    def cancel_job(
        job_id: str,
        x_noscribe_job_token: str = Header(alias="X-noScribe-Job-Token"),
    ):
        return _snapshot_response(service.cancel(job_id, x_noscribe_job_token))

    @app.post("/v1/audio/jobs/{job_id}/audio")
    async def upload_audio(
        job_id: str,
        request: Request,
        x_noscribe_job_token: str = Header(alias="X-noScribe-Job-Token"),
    ):
        snapshot = scheduler.get(job_id, x_noscribe_job_token, renew=False)
        if snapshot.audio_size > config.max_upload_bytes:
            raise HTTPException(413, "Reserved audio is too large.")
        path = await service.receive_audio(
            request, snapshot, x_noscribe_job_token
        )
        return service.stream_processing(snapshot, x_noscribe_job_token, path)

    async def direct_audio_request(
        request: Request,
        task: JobTask,
    ):
        content_length = request.headers.get("content-length")
        if content_length is None:
            raise HTTPException(411, "Content-Length is required.")
        try:
            audio_size = int(content_length)
        except ValueError as error:
            raise HTTPException(400, "Invalid Content-Length header.") from error
        if audio_size <= 0:
            raise HTTPException(400, "Audio upload is empty.")
        if audio_size > config.max_upload_bytes:
            raise HTTPException(413, "Audio upload is too large.")
        try:
            processor.validate_tasks((task,))
            admission = scheduler.submit(
                (task,),
                audio_filename="audio.opus",
                audio_size=audio_size,
                allow_queue=False,
            )
        except QueueFull as error:
            raise HTTPException(
                429,
                "The inference slot is busy; reserve a queued workflow instead.",
                headers={"Retry-After": "5"},
            ) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        snapshot = scheduler.get(admission.job_id, admission.token, renew=False)
        path = await service.receive_audio(request, snapshot, admission.token)
        response = service.stream_processing(snapshot, admission.token, path)
        response.headers["X-noScribe-Job-ID"] = admission.job_id
        response.headers["X-noScribe-Job-Token"] = admission.token
        return response

    @app.post("/v1/audio/transcriptions")
    async def direct_transcription(
        request: Request,
        model: str,
        language: str | None = None,
        multilingual: bool = False,
        include_disfluencies: bool = True,
        response_format: str = "noscribe_jsonl",
    ):
        if response_format != "noscribe_jsonl":
            raise HTTPException(400, "Only response_format=noscribe_jsonl is supported.")
        options = {
            "multilingual": multilingual,
            "include_disfluencies": include_disfluencies,
        }
        if language:
            options["language"] = language
        try:
            task = JobTask("transcription", model, options)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return await direct_audio_request(request, task)

    @app.post("/v1/audio/diarizations")
    async def direct_diarization(
        request: Request,
        model: str,
        num_speakers: int | None = None,
        response_format: str = "noscribe_jsonl",
    ):
        if response_format != "noscribe_jsonl":
            raise HTTPException(400, "Only response_format=noscribe_jsonl is supported.")
        options = {}
        if num_speakers is not None:
            options["num_speakers"] = num_speakers
        try:
            task = JobTask("diarization", model, options)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return await direct_audio_request(request, task)

    return app


def _snapshot_response(snapshot: JobSnapshot) -> dict:
    value = {
        "job_id": snapshot.job_id,
        "state": snapshot.state,
        "position": snapshot.position,
        "tasks": [
            {
                "type": task.operation,
                "model": task.model,
                "options": dict(task.options),
            }
            for task in snapshot.tasks
        ],
    }
    if snapshot.upload_before is not None:
        value["upload_expires_in"] = max(
            0.0, snapshot.upload_before - time.monotonic()
        )
    if snapshot.error_code:
        value["error"] = snapshot.error_code
    return value
