import asyncio

from RL_Framework.infra.execution.async_runner import AsyncTaskRunner


async def _return_value(value):
    await asyncio.sleep(0.01)
    return value


async def _wait_forever():
    await asyncio.Event().wait()


def test_async_task_runner_returns_completed_result():
    runner = AsyncTaskRunner()
    runner.initialize()
    try:
        runner.submit(_return_value, {"ok": True}, task_id=1)

        result = runner.wait(count=1, timeout=2.0)[0]

        assert result.task_id == 1
        assert result.data == {"ok": True}
    finally:
        runner.destroy()


def test_async_task_runner_destroy_cancels_wrapped_running_tasks_cleanly():
    runner = AsyncTaskRunner()
    runner.initialize()
    runner.submit(_wait_forever, task_id=2)

    runner.destroy(timeout=2.0)

    assert not runner.thread.is_alive()
    assert runner._thread_exception is None


def test_reset_runtime_state_cancels_old_tasks_and_accepts_new_work():
    runner = AsyncTaskRunner()
    runner.initialize()
    try:
        runner.submit(_wait_forever, task_id=5)

        runner.reset_runtime_state(timeout=2.0)
        runner.submit(_return_value, "new-pool", task_id=6)
        result = runner.wait(count=1, timeout=2.0)[0]

        assert result.task_id == 6
        assert result.data == "new-pool"
        assert runner.get_queue_sizes() == (0, 0)
    finally:
        runner.destroy()


def test_pause_drains_already_accepted_tasks():
    runner = AsyncTaskRunner()
    runner.initialize()
    try:
        runner.submit(_return_value, "done", task_id=3)
        runner.pause()

        runner.wait_until_idle(timeout=2.0)
        result = runner.wait(count=1, timeout=2.0)[0]

        assert result.data == "done"
    finally:
        runner.destroy()


def test_wait_until_idle_times_out_for_running_task():
    runner = AsyncTaskRunner()
    runner.initialize()
    try:
        runner.submit(_wait_forever, task_id=4)
        runner.pause()

        try:
            runner.wait_until_idle(timeout=0.1)
        except TimeoutError as exc:
            assert "draining rollout tasks" in str(exc)
        else:
            raise AssertionError("wait_until_idle should time out")
    finally:
        runner.destroy()
