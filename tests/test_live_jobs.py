from __future__ import annotations

import threading

from dafx26_demo.live_jobs import LiveJob, LiveJobRegistry


def test_registry_cancel_then_unregister() -> None:
    registry = LiveJobRegistry()
    job = LiveJob("abc", threading.Event())
    registry.register(job)
    assert registry.cancel("abc") is True
    assert job.cancel_event.is_set()
    registry.unregister("abc")
    assert registry.cancel("abc") is False


def test_create_registers_unique_jobs() -> None:
    registry = LiveJobRegistry()
    first = registry.create()
    second = registry.create()
    assert first.job_id != second.job_id
    assert registry.cancel(first.job_id) is True
    assert registry.cancel(second.job_id) is True


def test_cancel_is_idempotent_while_registered() -> None:
    registry = LiveJobRegistry()
    job = registry.create()
    assert registry.cancel(job.job_id) is True
    assert registry.cancel(job.job_id) is True
    assert job.cancel_event.is_set()
    registry.unregister(job.job_id)
    assert registry.cancel(job.job_id) is False


def test_unregister_is_safe_under_concurrent_stop() -> None:
    registry = LiveJobRegistry()
    job = registry.create()
    errors: list[BaseException] = []

    def cancel_loop() -> None:
        try:
            for _ in range(50):
                registry.cancel(job.job_id)
        except BaseException as exc:  # pragma: no cover - collected below
            errors.append(exc)

    def unregister_loop() -> None:
        try:
            for _ in range(50):
                registry.unregister(job.job_id)
        except BaseException as exc:  # pragma: no cover - collected below
            errors.append(exc)

    threads = [
        threading.Thread(target=cancel_loop),
        threading.Thread(target=unregister_loop),
        threading.Thread(target=cancel_loop),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert registry.cancel(job.job_id) is False
