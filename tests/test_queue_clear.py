"""The clear-all button must never touch the job that is being processed --
removing it from the list would orphan the running worker.
"""
from noScribe.jobs import JobStatus, TranscriptionJob, TranscriptionQueue


def _queue(*statuses):
    q = TranscriptionQueue()
    for i, status in enumerate(statuses):
        job = TranscriptionJob()
        job.audio_file = f'{i}.wav'
        job.status = status
        q.jobs.append(job)
    return q


def test_keeps_the_running_job_and_its_order():
    q = _queue(
        JobStatus.FINISHED,
        JobStatus.TRANSCRIPTION,
        JobStatus.WAITING,
        JobStatus.CANCELED,
        JobStatus.AUDIO_CONVERSION,
        JobStatus.WAITING_FOR_SERVER,
        JobStatus.ERROR,
    )
    q.clear_inactive()
    assert [j.audio_file for j in q.jobs] == ['1.wav', '4.wav', '5.wav']


def test_clears_everything_when_nothing_runs():
    q = _queue(JobStatus.WAITING, JobStatus.FINISHED, JobStatus.ERROR)
    q.clear_inactive()
    assert q.is_empty()


def test_canceling_job_survives():
    # CANCELING is still attached to a worker that is winding down.
    q = _queue(JobStatus.CANCELING, JobStatus.WAITING)
    q.clear_inactive()
    assert [j.status for j in q.jobs] == [JobStatus.CANCELING]


def test_waiting_for_server_is_a_running_job():
    q = _queue(JobStatus.WAITING_FOR_SERVER)

    assert q.is_running()
    assert q.get_waiting_jobs() == []


def test_has_inactive_jobs_matches_what_clearing_does():
    # Drives both the button state and the click-time guard, so it must agree
    # with clear_inactive() rather than approximate it.
    for statuses in ([], [JobStatus.TRANSCRIPTION], [JobStatus.WAITING],
                     [JobStatus.TRANSCRIPTION, JobStatus.FINISHED]):
        q = _queue(*statuses)
        before = len(q.jobs)
        expected = q.has_inactive_jobs()
        q.clear_inactive()
        assert (len(q.jobs) < before) is expected


def test_output_conflict_is_a_model_rule_without_gui(tmp_path):
    q = TranscriptionQueue()
    existing = TranscriptionJob()
    existing.transcript_file = str(tmp_path / 'transcript.html')
    q.add_job(existing)

    assert q.has_output_conflict(str(tmp_path / 'transcript.html'))
    assert not q.has_output_conflict(
        str(tmp_path / 'transcript.html'), ignore_job=existing)


def test_restarting_job_resets_speaker_mapping():
    job = TranscriptionJob()
    job.speaker_name_map = {'SPEAKER_00': 'Alice'}

    job.set_running()

    assert job.status == JobStatus.AUDIO_CONVERSION
    assert job.speaker_name_map == {}
    assert job.started_at is not None
