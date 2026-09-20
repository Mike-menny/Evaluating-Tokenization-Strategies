from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock
from uuid import uuid4


@dataclass(frozen=True)
class LiveJob:
    job_id: str
    cancel_event: Event


class LiveJobRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, LiveJob] = {}

    def create(self) -> LiveJob:
        with self._lock:
            while True:
                job = LiveJob(uuid4().hex[:12], Event())
                if job.job_id not in self._jobs:
                    self._jobs[job.job_id] = job
                    return job

    def register(self, job: LiveJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancel_event.set()
        return True

    def unregister(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
