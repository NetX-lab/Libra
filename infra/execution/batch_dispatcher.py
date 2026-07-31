"""Support code for Batch dispatcher."""

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .async_runner import AsyncTaskRunner, TaskResult
from RL_Framework.infra.sync.staleness import StalenessManager

logger = logging.getLogger(__name__)


_MAX_FETCH_BATCH_SIZE = 100
_SHUTDOWN_TIMEOUT_SECONDS = 2.0
_DEFAULT_WAIT_TIMEOUT_SECONDS = float(7 * 24 * 3600)

TInput = TypeVar("TInput")
TResult = TypeVar("TResult")


@dataclass
class TaskInput:
    """Task input implementation."""

    task_id: int
    data: Any
    version: int = 0
    group_id: str | None = None
    rollout_index: int = 0


class BatchTaskDispatcher(Generic[TInput, TResult]):
    """Batch task dispatcher implementation."""

    def __init__(
        self,
        async_runner: AsyncTaskRunner,
        staleness_manager: StalenessManager,
        task_factory: Callable[[TInput], Callable[..., Awaitable[TResult | None]]],
        enable_tracing: bool = False,
    ):
        self.runner = async_runner
        self.staleness_manager = staleness_manager
        self.task_factory = task_factory
        self.enable_tracing = enable_tracing


        self._pending_inputs: deque[TInput] = deque()
        self._pending_results: dict[int, TaskResult] = {}
        self._active_task_ids: set[int] = set()


        self._input_lock = threading.Lock()
        self._input_cv = threading.Condition(self._input_lock)
        self._result_lock = threading.Lock()
        self._result_cv = threading.Condition(self._result_lock)
        self._reconfigure_lock = threading.RLock()
        self._resetting_runner = threading.Event()


        self._shutdown_event = threading.Event()
        self._commit_thread: threading.Thread | None = None
        self._fetch_thread: threading.Thread | None = None


        self._thread_exception: Exception | None = None
        self._thread_exception_lock = threading.Lock()

    def _set_thread_exception(self, exc: Exception):
        """Set thread exception."""
        with self._thread_exception_lock:
            if self._thread_exception is None:
                self._thread_exception = exc

    def _check_thread_exception(self):
        """Check thread exception."""
        with self._thread_exception_lock:
            if self._thread_exception is not None:
                raise RuntimeError(
                    f"background thread error: {self._thread_exception}"
                ) from self._thread_exception

    def _has_runner_capacity(self) -> bool:
        """Has runner capacity."""
        return (
            not self.runner.paused.is_set()
            and self.staleness_manager.get_capacity() > 0
            and self.runner.input_queue.qsize() < self.runner.max_queue_size
        )

    def initialize(self):
        """Initialize."""
        self._shutdown_event.clear()

        self._commit_thread = threading.Thread(
            target=self._commit_loop, daemon=True, name="dispatcher-producer"
        )
        self._commit_thread.start()

        self._fetch_thread = threading.Thread(
            target=self._fetch_loop, daemon=True, name="dispatcher-consumer"
        )
        self._fetch_thread.start()

        if self.enable_tracing:
            logger.info("BatchTaskDispatcher initialized; producer and consumer threads started")

    def destroy(self):
        """Destroy."""
        self._shutdown_event.set()

        with self._input_cv:
            self._input_cv.notify()
        with self._result_cv:
            self._result_cv.notify_all()

        if self._commit_thread and self._commit_thread.is_alive():
            self._commit_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if self._commit_thread.is_alive():
                logger.warning("Producer thread did not exit before the timeout")

        if self._fetch_thread and self._fetch_thread.is_alive():
            self._fetch_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            if self._fetch_thread.is_alive():
                logger.warning("Consumer thread did not exit before the timeout")

        self.runner.destroy()



    def _commit_loop(self) -> None:
        """Commit loop."""
        while not self._shutdown_event.is_set():
            try:
                self._check_thread_exception()

                task_input = self._get_next_task_for_submission()
                if task_input is None:
                    continue


                with self._reconfigure_lock:
                    task_fn = self.task_factory(task_input)

                    try:
                        self.runner.submit(
                            task_fn, task_id=task_input.task_id
                        )
                        self.staleness_manager.on_rollout_submitted()
                        if self.enable_tracing:
                            stats = self.staleness_manager.get_stats()
                            logger.info(
                                f"submitted rollout task_id={task_input.task_id}, "
                                f"running={stats.running}, accepted={stats.accepted}"
                            )
                    except queue.Full:

                        with self._input_cv:
                            self._pending_inputs.appendleft(task_input)
                            self._input_cv.wait_for(
                                lambda: (
                                    self._shutdown_event.is_set()
                                    or self._has_runner_capacity()
                                ),
                                timeout=1.0,
                            )
                        continue

            except Exception as e:
                logger.error(f"Producer thread error: {e}", exc_info=True)
                self._set_thread_exception(e)
                with self._result_cv:
                    self._result_cv.notify_all()
                break

    def _get_next_task_for_submission(self) -> TInput | None:
        """Get next task for submission."""
        with self._input_cv:
            while not self._shutdown_event.is_set():
                self._check_thread_exception()


                if (
                    not self.runner.paused.is_set()
                    and self.staleness_manager.get_capacity() > 0
                    and self._pending_inputs
                ):
                    return self._pending_inputs.popleft()


                self._input_cv.wait(timeout=0.5)

        return None



    def _fetch_loop(self) -> None:
        """Fetch loop."""
        while not self._shutdown_event.is_set():
            try:
                self._check_thread_exception()


                output_qsize = self.runner.output_queue.qsize()
                count = max(1, min(output_qsize, _MAX_FETCH_BATCH_SIZE))

                try:
                    results = self.runner.wait(count=count, timeout=0.05)
                except TimeoutError:
                    continue

                with self._result_cv:
                    for result in results:
                        self._pending_results[result.task_id] = result
                    self._result_cv.notify_all()


                with self._input_cv:
                    self._input_cv.notify()

            except Exception as e:
                if self._resetting_runner.is_set() and isinstance(e, RuntimeError):
                    if "AsyncTaskRunner is shutting down" in str(e):
                        continue
                logger.error(f"Consumer thread error: {e}", exc_info=True)
                self._set_thread_exception(e)
                with self._result_cv:
                    self._result_cv.notify_all()
                with self._input_cv:
                    self._input_cv.notify()
                break



    def submit_task_input(self, task_input: TInput) -> None:
        """Submit task input."""
        self._check_thread_exception()
        with self._input_cv:
            self._pending_inputs.append(task_input)
            self.staleness_manager.on_rollout_enqueued()
            self._input_cv.notify()
        with self._result_cv:
            self._active_task_ids.add(task_input.task_id)

    def wait_results(
        self,
        count: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> list[TaskResult]:
        """Wait results."""
        if count <= 0:
            raise ValueError(f"count must be positive; got {count}")

        start_time = time.perf_counter()
        if timeout is None:
            timeout = _DEFAULT_WAIT_TIMEOUT_SECONDS

        with self._result_cv:
            while len(self._pending_results) < count:
                self._check_thread_exception()

                elapsed = time.perf_counter() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    if raise_timeout:
                        raise TimeoutError(
                            f"waited{count}resultstimed out, "
                            f"received only {len(self._pending_results)}"
                        )
                    return []

                self._result_cv.wait(timeout=min(remaining, 1.0))


            drained = list(self._pending_results.values())
            self._pending_results.clear()


        drained.sort(key=lambda x: x.create_time)
        selected, pending = drained[:count], drained[count:]

        with self._result_cv:
            if pending:
                for result in pending:
                    self._pending_results[result.task_id] = result
                self._result_cv.notify_all()
            for r in selected:
                self._active_task_ids.discard(r.task_id)

        return selected

    def active_submit_and_wait(
        self,
        input_generator: Generator[TInput, None, None],
        batch_size: int,
        timeout: float | None = None,
    ) -> list[Any]:
        """Active submit and wait."""
        accepted_cnt = 0
        results = []
        start_time = time.monotonic()

        while True:
            if timeout is not None and timeout > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        "active batch collection timed out: "
                        f"accepted={accepted_cnt}/{batch_size}, "
                        f"runtime_metrics={self.get_runtime_metrics()}"
                    )

            with self._input_cv:
                pending_inputs = len(self._pending_inputs)
            cap_staleness = self.staleness_manager.get_pending_limit() - pending_inputs
            cap_queue = self.runner.max_queue_size - (
                self.runner.input_queue.qsize() + batch_size
            )
            capacity = min(cap_staleness, cap_queue)

            if capacity > 0:
                for _ in range(min(batch_size, capacity)):
                    try:
                        self.submit_task_input(next(input_generator))
                    except StopIteration:
                        raise RuntimeError(
                            "The input generator is exhausted; use an infinite generator"
                        ) from None


            try:
                arrived = self.wait_results(
                    count=batch_size - accepted_cnt,
                    timeout=1.0,
                    raise_timeout=False,
                )
            except TimeoutError:
                arrived = []

            for task_result in arrived:

                if task_result.data is None:
                    continue

                accepted_cnt += 1
                results.append(task_result.data)

                if accepted_cnt >= batch_size:
                    break

            if accepted_cnt >= batch_size:
                break

        return results

    def pause(self):
        """Pause."""
        self.runner.pause()
        with self._input_cv:
            self._input_cv.notify()

    def wait_until_idle(self, timeout: float = 3600.0) -> None:
        """Drain tasks accepted before pause without submitting new work."""
        if not self.is_paused():
            raise RuntimeError("pause() must be called before wait_until_idle()")
        self.runner.wait_until_idle(timeout=timeout)

    def reset_after_reconfigure(self) -> None:
        """Drop dispatcher-side state that belongs to the old rollout pool."""
        self._resetting_runner.set()
        try:
            with self._reconfigure_lock:
                with self._input_cv:
                    self._pending_inputs.clear()
                    self._input_cv.notify_all()
                with self._result_cv:
                    self._pending_results.clear()
                    self._active_task_ids.clear()
                    self._result_cv.notify_all()
                if hasattr(self.runner, "reset_runtime_state"):
                    self.runner.reset_runtime_state()
                if hasattr(self.staleness_manager, "reset_runtime_state"):
                    self.staleness_manager.reset_runtime_state()
        finally:
            self._resetting_runner.clear()
            with self._input_cv:
                self._input_cv.notify_all()
            with self._result_cv:
                self._result_cv.notify_all()

    def resume(self):
        """Resume."""
        self.runner.resume()
        with self._input_cv:
            self._input_cv.notify()

    def is_paused(self) -> bool:
        """Is paused."""
        return self.runner.paused.is_set()

    def get_runtime_metrics(self) -> dict[str, Any]:
        """Return a thread-safe snapshot for online resource planning."""
        with self._input_lock:
            pending_inputs = len(self._pending_inputs)
        with self._result_lock:
            pending_results = len(self._pending_results)
            active_tasks = len(self._active_task_ids)
        input_qsize, output_qsize = self.runner.get_queue_sizes()
        staleness_stats = self.staleness_manager.get_stats()
        return {
            "pending_inputs": pending_inputs,
            "pending_results": pending_results,
            "active_tasks": active_tasks,
            "runner_input_queue": input_qsize,
            "runner_output_queue": output_qsize,
            "runner_max_queue_size": self.runner.max_queue_size,
            "staleness_capacity": self.staleness_manager.get_capacity(),
            "staleness_pending_limit": self.staleness_manager.get_pending_limit(),
            "staleness_running": staleness_stats.running,
            "staleness_accepted": staleness_stats.accepted,
            "staleness_rejected": staleness_stats.rejected,
            "paused": self.is_paused(),
        }
