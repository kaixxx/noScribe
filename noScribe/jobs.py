"""UI-independent transcription job state and queue management."""

from __future__ import annotations

import datetime
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from .inference import LOCAL_DIARIZATION_BACKEND, LOCAL_WHISPER_BACKEND
from .models import ModelRef


class JobStatus(Enum):
    WAITING = "waiting"
    AUDIO_CONVERSION = "audio_conversion"
    SPEAKER_IDENTIFICATION = "speaker_identification"
    TRANSCRIPTION = "transcription"
    CANCELING = "canceling"
    CANCELED = "canceled"
    FINISHED = "finished"
    ERROR = "error"


class TranscriptionJob:
    """State and processing options for one transcription job.

    This model deliberately contains no GUI or localization logic so it can
    be shared by the desktop application, CLI, and future server code.
    """

    def __init__(self):
        self.status: JobStatus = JobStatus.WAITING
        self.error_message: Optional[str] = None
        self.error_tb: Optional[str] = None
        self.created_at: datetime.datetime = datetime.datetime.now()
        self.started_at: Optional[datetime.datetime] = None
        self.finished_at: Optional[datetime.datetime] = None

        self.progress: float = 0.0

        self.audio_file: str = ''
        self.transcript_file: str = ''
        self.has_partial_transcript: bool = False

        self.start: int = 0
        self.stop: int = 0

        self.language: Optional[str] = None
        self.multilingual: bool = False
        self.transcription_model = ModelRef(LOCAL_WHISPER_BACKEND, 'precise')
        self.diarization_model = ModelRef(LOCAL_DIARIZATION_BACKEND, 'default')

        self.diarization_enabled: bool = True
        self.num_speakers: Optional[int] = None
        self.speaker_names: list[str] = []
        self.speaker_name_map: dict[str, str] = {}
        self.overlapping: bool = True
        self.timestamps: bool = False
        self.disfluencies: bool = True
        self.pause: int = 0

    @property
    def output_format(self) -> str:
        return Path(self.transcript_file).suffix.lstrip('.').lower()

    def set_running(self) -> None:
        self.status = JobStatus.AUDIO_CONVERSION
        self.started_at = datetime.datetime.now()
        # A repeated job must rebuild the order-dependent speaker mapping.
        self.speaker_name_map = {}

    def set_finished(self) -> None:
        self.status = JobStatus.FINISHED
        self.finished_at = datetime.datetime.now()

    def set_error(self, error_message: str, error_tb: str = '') -> None:
        self.status = JobStatus.ERROR
        self.error_message = error_message
        self.error_tb = error_tb
        self.finished_at = datetime.datetime.now()

    def set_canceled(self, message: Optional[str] = None) -> None:
        self.status = JobStatus.CANCELED
        self.error_message = message
        self.finished_at = datetime.datetime.now()

    def get_duration(self) -> Optional[datetime.timedelta]:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None


class TranscriptionQueue:
    """UI-independent queue rules for transcription jobs."""

    RUNNING_STATUSES = frozenset({
        JobStatus.AUDIO_CONVERSION,
        JobStatus.SPEAKER_IDENTIFICATION,
        JobStatus.TRANSCRIPTION,
        JobStatus.CANCELING,
    })

    def __init__(self):
        self.jobs: list[TranscriptionJob] = []
        self.current_job: Optional[TranscriptionJob] = None

    def add_job(self, job: TranscriptionJob) -> None:
        self.jobs.append(job)

    def get_waiting_jobs(self) -> list[TranscriptionJob]:
        return [job for job in self.jobs if job.status == JobStatus.WAITING]

    def get_running_jobs(self) -> list[TranscriptionJob]:
        return [job for job in self.jobs if job.status in self.RUNNING_STATUSES]

    def get_finished_jobs(self) -> list[TranscriptionJob]:
        return [job for job in self.jobs if job.status == JobStatus.FINISHED]

    def get_failed_jobs(self) -> list[TranscriptionJob]:
        return [job for job in self.jobs if job.status == JobStatus.ERROR]

    def get_canceled_jobs(self) -> list[TranscriptionJob]:
        return [job for job in self.jobs if job.status == JobStatus.CANCELED]

    def has_pending_jobs(self) -> bool:
        return bool(self.get_waiting_jobs())

    def is_running(self) -> bool:
        return bool(self.get_running_jobs())

    def get_next_waiting_job(self) -> Optional[TranscriptionJob]:
        waiting_jobs = self.get_waiting_jobs()
        return waiting_jobs[0] if waiting_jobs else None

    def get_queue_summary(self) -> dict[str, int]:
        return {
            'total': len(self.jobs),
            'waiting': len(self.get_waiting_jobs()),
            'running': len(self.get_running_jobs()),
            'finished': len(self.get_finished_jobs()),
            'errors': len(self.get_failed_jobs()),
            'canceled': len(self.get_canceled_jobs()),
        }

    def is_empty(self) -> bool:
        return not self.jobs

    def has_inactive_jobs(self) -> bool:
        return len(self.jobs) > len(self.get_running_jobs())

    def clear_inactive(self) -> None:
        self.jobs = self.get_running_jobs()

    def has_output_conflict(
        self,
        transcript_file: str,
        ignore_job: Optional[TranscriptionJob] = None,
    ) -> bool:
        """Return whether another active job targets the same output path."""
        try:
            target = os.path.abspath(transcript_file)
        except Exception:
            return False

        ignored_statuses = {
            JobStatus.ERROR,
            JobStatus.CANCELING,
            JobStatus.CANCELED,
        }
        for job in self.jobs:
            try:
                if not job or job is ignore_job or not job.transcript_file:
                    continue
                if (os.path.abspath(job.transcript_file) == target
                        and job.status not in ignored_statuses):
                    return True
            except Exception:
                continue
        return False
