"""Thread-safe, transport-neutral scheduling for server inference jobs.

The scheduler deliberately stores neither uploaded audio nor inference results.
It owns only reservations and their lifecycle.  One reservation at a time is
allowed to upload and run, regardless of whether its workers use a GPU or CPU.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Mapping, Sequence


class JobState(StrEnum):
    QUEUED = "queued"
    READY_FOR_UPLOAD = "ready_for_upload"
    UPLOADING = "uploading"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_STATES = frozenset({
    JobState.COMPLETED,
    JobState.FAILED,
    JobState.CANCELLED,
    JobState.EXPIRED,
})


class QueueFull(RuntimeError):
    """Raised when the configured number of waiting jobs is reached."""


class JobNotFound(LookupError):
    """Raised for an unknown job or an invalid capability token."""


class InvalidJobState(RuntimeError):
    """Raised when a lifecycle transition is not valid for the job."""


@dataclass(frozen=True)
class JobTask:
    """One inference operation in an atomic audio workflow."""

    operation: str
    model: str
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation not in {"transcription", "diarization"}:
            raise ValueError(f"Unsupported job operation: {self.operation!r}")
        if not self.model or ":" in self.model:
            raise ValueError("Server model IDs must be non-empty and unqualified.")


@dataclass(frozen=True)
class JobAdmission:
    job_id: str
    token: str
    state: JobState
    position: int


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    state: JobState
    position: int
    tasks: tuple[JobTask, ...]
    audio_filename: str
    audio_size: int
    upload_before: float | None = None
    error_code: str | None = None


@dataclass
class _Job:
    job_id: str
    token_digest: bytes
    tasks: tuple[JobTask, ...]
    audio_filename: str
    audio_size: int
    state: JobState
    created_at: float
    expires_at: float
    upload_before: float | None = None
    error_code: str | None = None
    terminal_at: float | None = None


class JobScheduler:
    """FIFO reservations with one CPU/GPU-independent execution slot."""

    def __init__(
        self,
        *,
        max_queued: int = 50,
        reservation_ttl: float = 300.0,
        upload_ready_ttl: float = 30.0,
        terminal_ttl: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_queued < 0:
            raise ValueError("max_queued must not be negative")
        if reservation_ttl <= 0 or upload_ready_ttl <= 0 or terminal_ttl <= 0:
            raise ValueError("Job timeouts must be positive")
        self.max_queued = max_queued
        self.reservation_ttl = reservation_ttl
        self.upload_ready_ttl = upload_ready_ttl
        self.terminal_ttl = terminal_ttl
        self._clock = clock
        self._jobs: dict[str, _Job] = {}
        self._waiting: deque[str] = deque()
        self._active_id: str | None = None
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)

    def submit(
        self,
        tasks: Sequence[JobTask],
        *,
        audio_filename: str,
        audio_size: int,
        allow_queue: bool = True,
    ) -> JobAdmission:
        workflow = tuple(tasks)
        if not workflow:
            raise ValueError("A job requires at least one task.")
        if not audio_filename:
            raise ValueError("An audio filename is required.")
        if audio_size <= 0:
            raise ValueError("Audio size must be positive.")

        with self._changed:
            self._reap_expired_locked()
            if self._active_id is not None and not allow_queue:
                raise QueueFull("The inference slot is busy.")
            if self._active_id is not None and len(self._waiting) >= self.max_queued:
                raise QueueFull("The inference queue is full.")

            now = self._clock()
            job_id = f"job-{secrets.token_hex(16)}"
            token = secrets.token_urlsafe(32)
            job = _Job(
                job_id=job_id,
                token_digest=_token_digest(token),
                tasks=workflow,
                audio_filename=audio_filename,
                audio_size=audio_size,
                state=JobState.QUEUED,
                created_at=now,
                expires_at=now + self.reservation_ttl,
            )
            self._jobs[job_id] = job
            self._waiting.append(job_id)
            self._promote_locked()
            self._changed.notify_all()
            return JobAdmission(
                job_id=job_id,
                token=token,
                state=job.state,
                position=self._position_locked(job),
            )

    def get(self, job_id: str, token: str, *, renew: bool = True) -> JobSnapshot:
        with self._changed:
            self._reap_expired_locked()
            job = self._authorized_job_locked(job_id, token)
            if renew and job.state is JobState.QUEUED:
                job.expires_at = self._clock() + self.reservation_ttl
            return self._snapshot_locked(job)

    def begin_upload(self, job_id: str, token: str) -> JobSnapshot:
        with self._changed:
            self._reap_expired_locked()
            job = self._authorized_job_locked(job_id, token)
            self._require_state(job, JobState.READY_FOR_UPLOAD)
            job.state = JobState.UPLOADING
            job.upload_before = None
            self._changed.notify_all()
            return self._snapshot_locked(job)

    def begin_processing(self, job_id: str, token: str) -> JobSnapshot:
        with self._changed:
            job = self._authorized_job_locked(job_id, token)
            self._require_state(job, JobState.UPLOADING)
            job.state = JobState.RUNNING
            self._changed.notify_all()
            return self._snapshot_locked(job)

    def complete(self, job_id: str, token: str) -> JobSnapshot:
        return self._finish(job_id, token, JobState.COMPLETED)

    def fail(self, job_id: str, token: str, error_code: str) -> JobSnapshot:
        if not error_code:
            raise ValueError("Failed jobs require a non-empty error code.")
        return self._finish(job_id, token, JobState.FAILED, error_code)

    def cancel(self, job_id: str, token: str) -> JobSnapshot:
        with self._changed:
            self._reap_expired_locked()
            job = self._authorized_job_locked(job_id, token)
            if job.state in TERMINAL_STATES:
                return self._snapshot_locked(job)
            job.state = JobState.CANCELLED
            job.terminal_at = self._clock()
            self._release_locked(job)
            self._changed.notify_all()
            return self._snapshot_locked(job)

    def wait_for_change(
        self,
        job_id: str,
        token: str,
        previous_state: JobState,
        timeout: float,
    ) -> JobSnapshot:
        """Wait until a job changes state, while periodically reaping timeouts."""
        deadline = self._clock() + max(timeout, 0.0)
        with self._changed:
            while True:
                self._reap_expired_locked()
                job = self._authorized_job_locked(job_id, token)
                if job.state is not previous_state:
                    return self._snapshot_locked(job)
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return self._snapshot_locked(job)
                self._changed.wait(min(remaining, 0.5))

    def reap_expired(self) -> None:
        with self._changed:
            if self._reap_expired_locked():
                self._changed.notify_all()

    @property
    def queued_count(self) -> int:
        with self._lock:
            return len(self._waiting)

    @property
    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_id

    def _finish(
        self,
        job_id: str,
        token: str,
        state: JobState,
        error_code: str | None = None,
    ) -> JobSnapshot:
        with self._changed:
            job = self._authorized_job_locked(job_id, token)
            self._require_state(job, JobState.RUNNING)
            job.state = state
            job.error_code = error_code
            job.terminal_at = self._clock()
            self._release_locked(job)
            self._changed.notify_all()
            return self._snapshot_locked(job)

    def _authorized_job_locked(self, job_id: str, token: str) -> _Job:
        job = self._jobs.get(job_id)
        if job is None or not hmac.compare_digest(
            job.token_digest, _token_digest(token)
        ):
            # Do not reveal whether a job ID or its capability token was wrong.
            raise JobNotFound("Unknown job.")
        return job

    def _require_state(self, job: _Job, expected: JobState) -> None:
        if job.state is not expected:
            raise InvalidJobState(
                f"Job {job.job_id} is {job.state}, expected {expected}."
            )

    def _position_locked(self, job: _Job) -> int:
        if job.job_id == self._active_id:
            return 0
        try:
            return list(self._waiting).index(job.job_id) + 1
        except ValueError:
            return 0

    def _snapshot_locked(self, job: _Job) -> JobSnapshot:
        return JobSnapshot(
            job_id=job.job_id,
            state=job.state,
            position=self._position_locked(job),
            tasks=job.tasks,
            audio_filename=job.audio_filename,
            audio_size=job.audio_size,
            upload_before=job.upload_before,
            error_code=job.error_code,
        )

    def _promote_locked(self) -> None:
        if self._active_id is not None:
            return
        while self._waiting:
            job = self._jobs[self._waiting.popleft()]
            if job.state is not JobState.QUEUED:
                continue
            now = self._clock()
            job.state = JobState.READY_FOR_UPLOAD
            job.upload_before = now + self.upload_ready_ttl
            self._active_id = job.job_id
            return

    def _release_locked(self, job: _Job) -> None:
        if self._active_id == job.job_id:
            self._active_id = None
        else:
            try:
                self._waiting.remove(job.job_id)
            except ValueError:
                pass
        self._promote_locked()

    def _reap_expired_locked(self) -> bool:
        now = self._clock()
        changed = False
        purge = []
        for job in self._jobs.values():
            if (
                job.state in TERMINAL_STATES
                and job.terminal_at is not None
                and now >= job.terminal_at + self.terminal_ttl
            ):
                purge.append(job.job_id)
                continue
            expired = (
                job.state is JobState.QUEUED and now >= job.expires_at
            ) or (
                job.state is JobState.READY_FOR_UPLOAD
                and job.upload_before is not None
                and now >= job.upload_before
            )
            if expired:
                job.state = JobState.EXPIRED
                job.terminal_at = now
                self._release_locked(job)
                changed = True
        for job_id in purge:
            self._jobs.pop(job_id, None)
            changed = True
        return changed


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()
