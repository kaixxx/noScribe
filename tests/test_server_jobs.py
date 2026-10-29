import threading

import pytest

from noScribe.server.jobs import (
    InvalidJobState,
    JobNotFound,
    JobScheduler,
    JobState,
    JobTask,
    QueueFull,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _task(operation="transcription", model="precise"):
    return JobTask(operation, model)


def test_scheduler_runs_atomic_workflows_in_fifo_order():
    scheduler = JobScheduler(max_queued=2)
    first = scheduler.submit(
        [_task("diarization", "speakers"), _task()],
        audio_filename="first.opus",
        audio_size=100,
    )
    second = scheduler.submit(
        [_task()], audio_filename="second.opus", audio_size=200
    )

    assert first.state is JobState.READY_FOR_UPLOAD
    assert second.state is JobState.QUEUED
    assert scheduler.get(second.job_id, second.token).position == 1

    scheduler.begin_upload(first.job_id, first.token)
    scheduler.begin_processing(first.job_id, first.token)
    scheduler.complete(first.job_id, first.token)

    promoted = scheduler.get(second.job_id, second.token)
    assert promoted.state is JobState.READY_FOR_UPLOAD
    assert promoted.position == 0


def test_scheduler_rejects_more_than_configured_waiting_jobs():
    scheduler = JobScheduler(max_queued=1)
    scheduler.submit([_task()], audio_filename="active.opus", audio_size=1)
    scheduler.submit([_task()], audio_filename="waiting.opus", audio_size=1)

    with pytest.raises(QueueFull):
        scheduler.submit([_task()], audio_filename="rejected.opus", audio_size=1)


def test_non_queueing_submission_is_rejected_while_slot_is_active():
    scheduler = JobScheduler()
    scheduler.submit([_task()], audio_filename="active.opus", audio_size=1)

    with pytest.raises(QueueFull, match="busy"):
        scheduler.submit(
            [_task()],
            audio_filename="direct.opus",
            audio_size=1,
            allow_queue=False,
        )

    assert scheduler.queued_count == 0


def test_ready_job_expires_and_releases_the_single_slot():
    clock = Clock()
    scheduler = JobScheduler(upload_ready_ttl=30, clock=clock)
    first = scheduler.submit([_task()], audio_filename="one.opus", audio_size=1)
    second = scheduler.submit([_task()], audio_filename="two.opus", audio_size=1)

    clock.advance(30)
    scheduler.reap_expired()

    assert scheduler.get(first.job_id, first.token).state is JobState.EXPIRED
    assert scheduler.get(second.job_id, second.token).state is JobState.READY_FOR_UPLOAD


def test_queued_polling_renews_reservation_but_ready_deadline_does_not():
    clock = Clock()
    scheduler = JobScheduler(
        reservation_ttl=20, upload_ready_ttl=30, clock=clock
    )
    first = scheduler.submit([_task()], audio_filename="one.opus", audio_size=1)
    second = scheduler.submit([_task()], audio_filename="two.opus", audio_size=1)

    clock.advance(15)
    assert scheduler.get(second.job_id, second.token).state is JobState.QUEUED
    clock.advance(10)
    scheduler.reap_expired()
    assert scheduler.get(second.job_id, second.token).state is JobState.QUEUED

    clock.advance(5)
    scheduler.reap_expired()
    assert scheduler.get(first.job_id, first.token).state is JobState.EXPIRED


def test_tokens_are_required_and_lifecycle_transitions_are_checked():
    scheduler = JobScheduler()
    job = scheduler.submit([_task()], audio_filename="audio.opus", audio_size=1)

    with pytest.raises(JobNotFound):
        scheduler.get(job.job_id, "wrong-token")
    with pytest.raises(InvalidJobState):
        scheduler.begin_processing(job.job_id, job.token)


def test_canceling_a_waiting_job_does_not_disturb_the_active_job():
    scheduler = JobScheduler()
    first = scheduler.submit([_task()], audio_filename="one.opus", audio_size=1)
    second = scheduler.submit([_task()], audio_filename="two.opus", audio_size=1)

    scheduler.cancel(second.job_id, second.token)

    assert scheduler.active_job_id == first.job_id
    assert scheduler.queued_count == 0
    assert scheduler.get(second.job_id, second.token).state is JobState.CANCELLED


def test_scheduler_serializes_concurrent_submissions():
    scheduler = JobScheduler(max_queued=20)
    admissions = []
    lock = threading.Lock()

    def submit(index):
        admission = scheduler.submit(
            [_task()], audio_filename=f"{index}.opus", audio_size=1
        )
        with lock:
            admissions.append(admission)

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(admissions) == 10
    assert sum(job.state is JobState.READY_FOR_UPLOAD for job in admissions) == 1
    assert scheduler.queued_count == 9


def test_terminal_job_metadata_is_purged_after_short_retention():
    clock = Clock()
    scheduler = JobScheduler(clock=clock, terminal_ttl=10)
    job = scheduler.submit([_task()], audio_filename="audio.opus", audio_size=1)
    scheduler.begin_upload(job.job_id, job.token)
    scheduler.begin_processing(job.job_id, job.token)
    scheduler.complete(job.job_id, job.token)

    clock.advance(10)
    scheduler.reap_expired()

    with pytest.raises(JobNotFound):
        scheduler.get(job.job_id, job.token)
