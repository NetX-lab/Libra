"""Support code for Batch dispatcher."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
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


def _is_successful(data: Any) -> bool:
    return data is not None and (not isinstance(data, dict) or data.get("status") != "failed")


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
        self._task_inputs: dict[int, TInput] = {}
        self._input_generator = None


        self._input_lock = threading.Lock()
        self._input_cv = threading.Condition(self._input_lock)
        self._result_lock = threading.Lock()
        self._result_cv = threading.Condition(self._result_lock)
        self._reconfigure_lock = threading.RLock()


        self._shutdown_event = threading.Event()
        self._resetting_runner = threading.Event()
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

    def report_fatal_error(self, exc: Exception):
        """Stop admission and wake callers after a task reports invalid evidence."""
        self._set_thread_exception(exc)
        self.runner.pause()
        # Do not hold the exception lock while taking either condition lock:
        # waiting callers check that exception while holding the conditions.
        with self._input_cv:
            self._input_cv.notify_all()
        with self._result_cv:
            self._result_cv.notify_all()

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
                    # Keep the selected task owned by the pending queue until
                    # commit, so cancel_queued() can still return its identity
                    # if pause wins the race for this lock.
                    with self._input_cv:
                        if (
                            not self._pending_inputs
                            or self._pending_inputs[0] is not task_input
                            or self.runner.paused.is_set()
                        ):
                            continue
                        self._pending_inputs.popleft()
                    task_fn = self.task_factory(task_input)

                    try:
                        self.runner.submit(
                            task_fn, task_id=task_input.task_id
                        )
                        if self.staleness_manager.group_reservations_enabled:
                            self.staleness_manager.on_member_submitted(str(task_input.group_id), task_input.task_id)
                        else:
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
        """Wait for a candidate, retaining queue ownership until commit."""
        with self._input_cv:
            while not self._shutdown_event.is_set():
                self._check_thread_exception()


                if not self.runner.paused.is_set():
                    if not self._pending_inputs:
                        self._admit_group()
                    if self.staleness_manager.get_capacity() > 0 and self._pending_inputs:
                        return self._pending_inputs[0]


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

                with self._reconfigure_lock:
                    if self.staleness_manager.group_reservations_enabled:
                        for result in results:
                            self._accept_group_result(result)
                        with self._result_cv:
                            self._result_cv.notify_all()
                    else:
                        with self._result_cv:
                            for result in results:
                                if result.task_id not in self._active_task_ids:
                                    continue
                                self.staleness_manager.on_rollout_finished(_is_successful(result.data))
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



    def start_production(self, input_generator, group_size: int) -> None:
        """Start the producer once; ready results and training do not stop it."""
        self._check_thread_exception()
        with self._input_cv:
            if self._input_generator is not None:
                if self._input_generator is not input_generator:
                    raise RuntimeError("rollout production already owns another input generator")
                return
            self.staleness_manager.configure_groups(group_size)
            self._input_generator = input_generator
            self._input_cv.notify_all()

    def _admit_group(self) -> None:
        # Called with _input_cv held by the single producer. Read the generator
        # only when a complete reservation can be made; no hidden prefetch.
        manager = self.staleness_manager
        if self._input_generator is None or not manager.can_reserve_group():
            return
        try:
            members = [next(self._input_generator) for _ in range(manager.group_size)]
        except StopIteration:
            raise RuntimeError("rollout input generator exhausted before a complete group") from None
        group_id = str(members[0].group_id)
        if (members[0].group_id is None or any(str(member.group_id) != group_id for member in members)
                or {member.rollout_index for member in members} != set(range(manager.group_size))):
            raise ValueError("rollout generator must yield contiguous complete groups")
        promise = manager.reserve_group(group_id, [member.task_id for member in members])
        if promise is None:
            raise RuntimeError("group admission changed outside the producer boundary")
        for member in members:
            member.version = promise
            self._pending_inputs.append(member)
            self._task_inputs[member.task_id] = member
            self._active_task_ids.add(member.task_id)

    def _accept_group_result(self, result: TaskResult) -> None:
        task = self._task_inputs.get(result.task_id)
        if task is None:
            return  # A terminated group's cancellation/late result cannot revive it.
        if result.error is not None:
            raise result.error
        packet = result.data
        if isinstance(packet, dict) and packet.get("failure_reason") == "task_timeout":
            self._terminate_group(str(task.group_id))
            return
        if not isinstance(packet, dict) or packet.get("status") == "failed":
            raise RuntimeError(f"rollout task {result.task_id} failed without a task_timeout result")
        if str(packet.get("grpo_group_id")) != str(task.group_id):
            raise ValueError("rollout result group differs from its reservation")
        revision = packet.get("sample_revision")
        if type(revision) is not int:
            raise ValueError("rollout result requires its actual integer sample_revision")
        with self._result_cv:
            self.staleness_manager.on_member_completed(str(task.group_id), result.task_id, revision)
            self._pending_results[result.task_id] = result

    def _terminate_group(self, group_id: str) -> None:
        task_ids = set(self.staleness_manager.terminate_group(group_id))
        if not task_ids:
            return
        with self._input_cv:
            self._pending_inputs = deque(item for item in self._pending_inputs if item.task_id not in task_ids)
            self._input_cv.notify_all()
        with self._result_cv:
            for task_id in task_ids:
                self._pending_results.pop(task_id, None)
                self._task_inputs.pop(task_id, None)
                self._active_task_ids.discard(task_id)
            self._result_cv.notify_all()
        self.runner.cancel_tasks(task_ids)

    def collect_batch(self, batch_size: int, optimizer_revision: int) -> list[Any]:
        """Claim complete groups by deadline without releasing their slots."""
        manager = self.staleness_manager
        if not manager.group_reservations_enabled or batch_size != manager.consumer_batch_size:
            raise ValueError("collection must request the configured complete training batch")
        with self._result_cv:
            while not self._shutdown_event.is_set():
                self._check_thread_exception()
                group_ids = manager.select_groups(batch_size // manager.group_size, optimizer_revision)
                if group_ids is not None:
                    selected = []
                    for group_id in group_ids:
                        task_ids = manager.get_group_task_ids(group_id)
                        members = sorted((self._pending_results[task_id] for task_id in task_ids),
                                         key=lambda result: result.data["grpo_rollout_index"])
                        for member in members:
                            selected.append(self._pending_results.pop(member.task_id).data)
                            self._active_task_ids.discard(member.task_id)
                    return selected
                self._result_cv.wait(timeout=.5)
        raise RuntimeError("dispatcher stopped before a complete batch was available")

    def retire_batch(self, trajectories: list[dict], optimizer_revision: int) -> None:
        group_ids = {str(trajectory["grpo_group_id"]) for trajectory in trajectories}
        with self._reconfigure_lock, self._input_cv:
            self.staleness_manager.retire_batch(group_ids, optimizer_revision)
            for trajectory in trajectories:
                self._task_inputs.pop(trajectory["task_id"], None)
            self._input_cv.notify_all()

    def on_publication(self, published_revision: int) -> None:
        with self._input_cv:
            self.staleness_manager.on_publication(published_revision)
            self._input_cv.notify_all()

    def submit_task_input(self, task_input: TInput) -> None:
        """Queue an input for the legacy pull-based collector."""
        self._check_thread_exception()
        with self._input_cv, self._result_cv:
            self._pending_inputs.append(task_input)
            self._active_task_ids.add(task_input.task_id)
            self.staleness_manager.on_rollout_enqueued()
            self._input_cv.notify()

    def wait_results(
        self,
        count: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> list[TaskResult]:
        """Wait results."""
        if count <= 0:
            raise ValueError(f"count must be positive; got {count}")
        self._check_thread_exception()

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


            self._check_thread_exception()
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

    def peek_ready_data(self) -> list[Any]:
        """Snapshot ready payloads in consumption order without changing queues."""
        with self._result_cv:
            ready = sorted(self._pending_results.values(), key=lambda result: result.create_time)
            return [result.data for result in ready]

    def active_submit_and_wait(
        self,
        input_generator: Iterator[TInput],
        batch_size: int,
        timeout: float | None = None,
    ) -> list[Any]:
        """Keep the legacy batch collector supplied until it receives enough results."""
        results = []
        started = time.monotonic()
        while len(results) < batch_size:
            remaining = None if timeout is None else timeout - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                # This call has already removed these successes from the result
                # queue. Release their credit if timing out discards them.
                self.staleness_manager.on_batch_consumed(sum(
                    _is_successful(result) for result in results
                ))
                raise TimeoutError(f"active batch collection timed out: {len(results)}/{batch_size}")

            with self._input_cv:
                pending_inputs = len(self._pending_inputs)
            capacity = min(
                self.staleness_manager.get_pending_limit() - pending_inputs,
                self.runner.max_queue_size - self.runner.input_queue.qsize(),
            )
            for _ in range(min(batch_size - len(results), capacity)):
                try:
                    self.submit_task_input(next(input_generator))
                except StopIteration:
                    raise RuntimeError("rollout input generator exhausted") from None

            arrived = self.wait_results(
                count=batch_size - len(results),
                timeout=1.0 if remaining is None else min(1.0, remaining),
                raise_timeout=False,
            )
            results.extend(result.data for result in arrived if result.data is not None)
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

    def cancel_queued(self) -> dict[str, Any]:
        """Discard rollout work that has not started executing on vLLM."""
        if not self.is_paused():
            raise RuntimeError("pause() must be called before cancel_queued()")
        with self._reconfigure_lock:
            with self._input_cv:
                dispatcher_ids = [item.task_id for item in self._pending_inputs]
                self._pending_inputs.clear()
                self._input_cv.notify_all()
            runner_ids = self.runner.cancel_queued()
            cancelled_ids = sorted(set(dispatcher_ids + runner_ids))
            with self._result_cv:
                for task_id in cancelled_ids:
                    self._active_task_ids.discard(task_id)
                self._result_cv.notify_all()
            if self.staleness_manager.group_reservations_enabled:
                group_ids = {str(self._task_inputs[task_id].group_id) for task_id in cancelled_ids
                             if task_id in self._task_inputs}
                for group_id in group_ids:
                    self._terminate_group(group_id)
            else:
                self.staleness_manager.on_rollouts_cancelled(
                    enqueued=len(dispatcher_ids), running=len(runner_ids),
                )
        return {
            "dispatcher_queue": len(dispatcher_ids),
            "runner_queue": len(runner_ids),
            "task_ids": cancelled_ids,
        }

    def reset_after_reconfigure(self) -> None:
        """Drop admitted work after an explicit rollout runtime cutover."""
        self._resetting_runner.set()
        try:
            with self._reconfigure_lock:
                with self._input_cv, self._result_cv:
                    self._pending_inputs.clear()
                    self._pending_results.clear()
                    self._active_task_ids.clear()
                    self._task_inputs.clear()
                    self._input_cv.notify_all()
                    self._result_cv.notify_all()
                if hasattr(self.runner, "reset_runtime_state"):
                    self.runner.reset_runtime_state()
                self.staleness_manager.reset_runtime_state()
        finally:
            self._resetting_runner.clear()

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
            "group_reservations": self.staleness_manager.get_group_reservations(),
            "published_revision": self.staleness_manager.published_revision,
            "optimizer_revision": self.staleness_manager.optimizer_revision,
        }
