"""The clear-all button must never touch the job that is being processed --
removing it from the list would orphan the running worker.
"""
import pytest

pytest.importorskip("tkinter")  # noScribe.main pulls in the GUI stack

from noScribe.main import JobStatus, TranscriptionJob, TranscriptionQueue


def _queue(*statuses):
    q = TranscriptionQueue()
    for i, status in enumerate(statuses):
        job = TranscriptionJob()
        job.audio_file = f'{i}.wav'
        job.status = status
        q.jobs.append(job)
    return q


def test_keeps_the_running_job_and_its_order():
    q = _queue(JobStatus.FINISHED, JobStatus.TRANSCRIPTION, JobStatus.WAITING,
               JobStatus.CANCELED, JobStatus.AUDIO_CONVERSION, JobStatus.ERROR)
    q.clear_inactive()
    assert [j.audio_file for j in q.jobs] == ['1.wav', '4.wav']


def test_clears_everything_when_nothing_runs():
    q = _queue(JobStatus.WAITING, JobStatus.FINISHED, JobStatus.ERROR)
    q.clear_inactive()
    assert q.is_empty()


def test_canceling_job_survives():
    # CANCELING is still attached to a worker that is winding down.
    q = _queue(JobStatus.CANCELING, JobStatus.WAITING)
    q.clear_inactive()
    assert [j.status for j in q.jobs] == [JobStatus.CANCELING]
