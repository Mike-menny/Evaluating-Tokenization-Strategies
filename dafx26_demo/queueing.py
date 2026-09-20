from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import TypeVar

T = TypeVar("T")


class QueueBusyError(RuntimeError):
    pass


class GenerationTimeoutError(RuntimeError):
    pass


class GenerationLease:
    def __init__(self, lock: Lock) -> None:
        self._lock = lock
        self._state_lock = Lock()
        self._released = False

    def release(self) -> None:
        with self._state_lock:
            if self._released:
                return
            self._released = True
        self._lock.release()

    def __enter__(self) -> GenerationLease:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class GenerationQueue:
    def __init__(self, timeout_sec: float = 120.0) -> None:
        self.timeout_sec = timeout_sec
        self._lock = Lock()

    def acquire(self) -> GenerationLease:
        if not self._lock.acquire(blocking=False):
            raise QueueBusyError("A generation is already in progress")
        return GenerationLease(self._lock)

    def run(self, fn: Callable[[], T]) -> T:
        with self.acquire():
            return fn()
