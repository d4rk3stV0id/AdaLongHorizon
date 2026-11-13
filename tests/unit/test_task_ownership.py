"""Process-level task ownership tests (also exercised on Windows)."""

import multiprocessing
import pytest

from agent.state.store import StateStore, TaskOwnershipError


def _hold_lock(root, task_id, ready, release):
    with StateStore(root).task_lock(task_id):
        ready.put("acquired")
        release.wait(20)


def _try_lock(root, task_id, result):
    try:
        with StateStore(root).task_lock(task_id):
            result.put("acquired")
    except TaskOwnershipError:
        result.put("busy")


def _acquire_then_exit(root, task_id, result):
    with StateStore(root).task_lock(task_id):
        result.put("acquired")


def _ctx():
    return multiprocessing.get_context("spawn")


def test_competing_process_cannot_own_same_task(tmp_path):
    ctx = _ctx()
    with StateStore(tmp_path).task_lock("same-task"):
        result = ctx.Queue()
        process = ctx.Process(target=_try_lock, args=(str(tmp_path), "same-task", result))
        process.start()
        assert result.get(timeout=10) == "busy"
        process.join(timeout=10)
        assert process.exitcode == 0


def test_different_task_ids_have_independent_process_locks(tmp_path):
    ctx = _ctx()
    with StateStore(tmp_path).task_lock("task-a"):
        result = ctx.Queue()
        process = ctx.Process(target=_try_lock, args=(str(tmp_path), "task-b", result))
        process.start()
        assert result.get(timeout=10) == "acquired"
        process.join(timeout=10)
        assert process.exitcode == 0


def test_process_termination_releases_task_ownership(tmp_path):
    ctx = _ctx()
    ready, release = ctx.Queue(), ctx.Event()
    process = ctx.Process(target=_hold_lock, args=(str(tmp_path), "task", ready, release))
    process.start()
    assert ready.get(timeout=10) == "acquired"
    process.terminate()
    process.join(timeout=10)
    assert process.exitcode is not None
    with StateStore(tmp_path).task_lock("task"):
        pass


def test_normal_process_exit_releases_task_ownership(tmp_path):
    ctx = _ctx()
    result = ctx.Queue()
    process = ctx.Process(target=_acquire_then_exit, args=(str(tmp_path), "task", result))
    process.start()
    assert result.get(timeout=10) == "acquired"
    process.join(timeout=10)
    assert process.exitcode == 0
    with StateStore(tmp_path).task_lock("task"):
        pass


def test_lock_releases_and_can_be_reacquired(tmp_path):
    store = StateStore(tmp_path)
    with store.task_lock("task"):
        with store.task_lock("task"):  # resume() may call recover() on this thread
            pass
    with store.task_lock("task"):
        pass


def test_lock_releases_after_exception(tmp_path):
    store = StateStore(tmp_path)
    with pytest.raises(RuntimeError, match="interrupted"):
        with store.task_lock("task"):
            raise RuntimeError("interrupted")
    with store.task_lock("task"):
        pass


def test_different_thread_cannot_reenter_owned_task(tmp_path):
    import threading

    store = StateStore(tmp_path)
    result = []
    with store.task_lock("task"):
        def contend():
            try:
                with store.task_lock("task"):
                    result.append("acquired")
            except TaskOwnershipError:
                result.append("busy")
        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
    assert result == ["busy"]
