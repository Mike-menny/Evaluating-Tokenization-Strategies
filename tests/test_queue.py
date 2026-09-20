from __future__ import annotations

import threading
import time

import pytest

from dafx26_demo.queueing import GenerationQueue, QueueBusyError


def test_queue_allows_one_in_flight_and_rejects_a_second() -> None:
    q = GenerationQueue(timeout_sec=2.0)
    started = threading.Event()
    finished = threading.Event()

    def slow() -> str:
        started.set()
        time.sleep(0.3)
        finished.set()
        return "ok"

    def runner() -> None:
        q.run(slow)

    t = threading.Thread(target=runner)
    t.start()
    assert started.wait(1.0)
    try:
        q.run(lambda: "nope")
        raise AssertionError("expected QueueBusyError")
    except QueueBusyError:
        pass
    t.join(2.0)
    assert finished.is_set()
    assert q.run(lambda: "later") == "later"


def test_lease_blocks_until_idempotent_release() -> None:
    queue = GenerationQueue()
    lease = queue.acquire()
    with pytest.raises(QueueBusyError):
        queue.acquire()
    lease.release()
    lease.release()
    queue.acquire().release()


def test_concurrent_release_unlocks_exactly_once() -> None:
    queue = GenerationQueue()
    lease = queue.acquire()
    threads = [threading.Thread(target=lease.release) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with queue.acquire():
        pass
