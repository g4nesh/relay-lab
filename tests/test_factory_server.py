"""Session lifecycle controls operate on committed boundaries."""

import time

import pytest

from relay_lab.factory.server import FactorySession


def until(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Session did not reach expected state")
        time.sleep(0.01)


def test_pause_resume_reset_do_not_create_parallel_workers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = FactorySession(coordination="memory")
    try:
        session.start({"planner": "rules", "orders": 1, "policy": "model-based"})
        until(lambda: session.view()["tick"] >= 2)
        with pytest.raises(ValueError, match="already active"):
            session.start({"planner": "rules"})
        session.pause()
        until(lambda: session.view()["status"] == "paused")
        tick = session.view()["tick"]
        time.sleep(0.08)
        assert session.view()["tick"] == tick
        session.resume()
        until(lambda: session.view()["tick"] > tick)
        session.reset()
        until(lambda: session.view()["status"] == "ready")
        assert session.view()["tick"] == 0
    finally:
        session.close()


def test_session_rejects_invalid_inputs_before_starting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = FactorySession(coordination="memory")
    for options in ({"orders": 7}, {"orders": True}, {"planner": "remote"}, {"seed": -1}):
        with pytest.raises(ValueError):
            session.start(options)
    for speed in (float("nan"), float("inf"), 0, -1, True, 13):
        with pytest.raises(ValueError):
            session.set_speed(speed)
    assert session.view()["status"] == "ready"


def test_overlapping_reset_and_start_preserve_the_new_worker(tmp_path, monkeypatch):
    """A reset paused at old-worker inspection must not orphan a concurrent start."""
    import threading

    monkeypatch.chdir(tmp_path)
    session = FactorySession(coordination="memory")
    inspected, release_reset, start_done = threading.Event(), threading.Event(), threading.Event()
    failures, captured = [], []

    class FinishedWorker:
        def is_alive(self):
            if threading.current_thread().name == "reset-regression":
                inspected.set()
                if not release_reset.wait(3):
                    raise AssertionError("Reset test barrier was not released")
            return False

        def join(self, timeout=None):
            pass

    def reset():
        try:
            session.reset()
        except Exception as error:  # noqa: BLE001 - return worker failures to the test thread
            failures.append(error)

    def start():
        try:
            session.start({"planner": "rules", "orders": 1, "policy": "model-based"})
            captured.append((session._run, session._worker))
        except Exception as error:  # noqa: BLE001 - return worker failures to the test thread
            failures.append(error)
        finally:
            start_done.set()

    session._worker = FinishedWorker()
    reset_thread = threading.Thread(target=reset, name="reset-regression")
    start_thread = threading.Thread(target=start, name="start-regression")
    try:
        reset_thread.start()
        assert inspected.wait(2)
        start_thread.start()
        # With the old race this start completes while reset is paused, then
        # reset erases the new worker. With serialized lifecycle changes it
        # waits here until reset releases the lock, then starts normally.
        start_done.wait(0.15)
        release_reset.set()
        reset_thread.join(3)
        start_thread.join(3)
        assert not reset_thread.is_alive() and not start_thread.is_alive()
        assert not failures
        assert len(captured) == 1
        run, worker = captured[0]
        assert run is not None and worker is not None
        assert session._run is run
        assert session._worker is worker
        assert worker.is_alive()
        until(lambda: session.view()["tick"] >= 1)
    finally:
        release_reset.set()
        session._stop.set()
        session._wake.set()
        reset_thread.join(3)
        if start_thread.ident is not None:
            start_thread.join(3)
        for _, worker in captured:
            if worker is not None:
                worker.join(3)
        session.close()


def test_closed_session_cannot_start_an_untracked_worker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = FactorySession(coordination="memory")
    session.close()
    with pytest.raises((RuntimeError, ValueError), match="closed"):
        session.start({"planner": "rules", "orders": 1, "policy": "model-based"})
    assert session._run is None
    assert session._worker is None


def test_portable_compressed_recording_is_available_without_starting_world(tmp_path):
    import gzip
    import json

    path = tmp_path / "replay.json.gz"
    recording = {"schema": "relay.factory.replay.v1", "frames": [{"tick": 7}]}
    with gzip.open(path, "wt") as stream:
        json.dump(recording, stream)
    session = FactorySession(replay=path)
    try:
        assert session.replay() == recording
        assert session.view()["replay_available"]
        assert session._worker is None
    finally:
        session.close()
