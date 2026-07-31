"""Support code for Async runner."""

import asyncio
import queue
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class TaskResult(Generic[T]):
    """Task result implementation."""
    data: T
    task_id: int
    create_time: float
    complete_time: float


class AsyncTaskRunner(Generic[T]):
    """Async task runner implementation."""

    def __init__(
        self,
        max_queue_size: int = 100,
        enable_tracing: bool = False,
    ):
        self.max_queue_size = max_queue_size
        self.enable_tracing = enable_tracing


        self.exiting = threading.Event()
        self.paused = threading.Event()


        self.input_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)


        self._loop_ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._input_event: asyncio.Event | None = None
        self._generation = 0
        self._generation_lock = threading.Lock()


        self._active_task_ids: set[int] = set()
        self._active_task_ids_lock = threading.Lock()


        self._thread_exception: Exception | None = None
        self._thread_exception_lock = threading.Lock()

        self.thread: threading.Thread | None = None

    def initialize(self):
        """Initialize."""
        self.exiting.clear()
        self.paused.clear()
        self._loop_ready = threading.Event()

        self.thread = threading.Thread(
            target=self._run_thread,
            args=(
                self.exiting,
                self.paused,
                self.input_queue,
                self.output_queue,
                self._generation,
                self._loop_ready,
            ),
            daemon=True,
        )
        self.thread.start()
        self._loop_ready.wait()

        if self._thread_exception:
            raise RuntimeError(f"AsyncTaskRunner initialization failed: {self._thread_exception}")

    def destroy(self, timeout: float = 10.0):
        """Destroy."""
        self.exiting.set()
        self.paused.clear()
        self._signal_new_input()

        if self.thread is not None:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                print("WARNING: AsyncTaskRunnerthread did not exit before the timeout")

    def reset_runtime_state(self, timeout: float = 10.0) -> None:
        """Cancel in-flight tasks and restart the runner with empty queues.

        Runtime rollout reconfiguration can replace vLLM instances while
        asynchronous requests are still in flight.  Those requests must not
        survive into the new rollout pool because they keep references to the
        old endpoint clients.  A full runner restart gives the dispatcher a
        clean async loop while preserving the paused/unpaused state expected by
        the caller.
        """
        was_paused = self.paused.is_set()
        with self._generation_lock:
            self._generation += 1
        self.destroy(timeout=timeout)

        self.exiting = threading.Event()
        self.paused = threading.Event()
        self.input_queue = queue.Queue(maxsize=self.max_queue_size)
        self.output_queue = queue.Queue(maxsize=self.max_queue_size)
        with self._active_task_ids_lock:
            self._active_task_ids.clear()
        with self._thread_exception_lock:
            self._thread_exception = None

        self.thread = None
        self._loop = None
        self._input_event = None
        self.initialize()
        if was_paused:
            self.pause()

    def _run_thread(
        self,
        exit_event: threading.Event,
        pause_event: threading.Event,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        generation: int,
        ready_event: threading.Event,
    ):
        """Run thread."""
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            if generation == self._generation:
                self._loop = loop
            ready_event.set()
            loop.run_until_complete(
                self._run_async_loop(
                    exit_event=exit_event,
                    pause_event=pause_event,
                    input_queue=input_queue,
                    output_queue=output_queue,
                    generation=generation,
                )
            )
        except Exception as e:
            with self._thread_exception_lock:
                if generation == self._generation:
                    self._thread_exception = e
            print(f"ERROR: AsyncTaskRunnerthread error: {e}")
        finally:
            exit_event.set()
            ready_event.set()
            if loop is not None:
                loop.close()
            if generation == self._generation:
                self._loop = None
                self._input_event = None

    async def _run_async_loop(
        self,
        *,
        exit_event: threading.Event,
        pause_event: threading.Event,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        generation: int,
    ):
        """Run async loop."""
        input_event = asyncio.Event()
        if generation == self._generation:
            self._input_event = input_event
        input_event.set()

        running_tasks: dict[str, dict[str, Any]] = {}

        try:
            while not exit_event.is_set():
                # Pausing blocks new dispatcher submissions. Tasks already
                # accepted by this runner must still be started and reaped so
                # callers can drain the rollout pipeline before restarting
                # serving processes for a weight update.
                self._drain_pending_inputs(running_tasks, input_queue)


                if not running_tasks:
                    if pause_event.is_set():
                        await asyncio.sleep(0.05)
                        continue
                    await self._wait_for_new_tasks(
                        exit_event=exit_event,
                        pause_event=pause_event,
                        input_queue=input_queue,
                        input_event=input_event,
                    )
                    continue


                tasks = [t["task"] for t in running_tasks.values()]
                done, _ = await asyncio.wait(
                    tasks,
                    timeout=0.05,
                    return_when=asyncio.FIRST_COMPLETED,
                )


                for async_task in done:
                    tid = async_task.get_name()
                    task_obj = running_tasks.pop(tid)

                    try:
                        result = await async_task
                    except asyncio.CancelledError:
                        result = None
                    except Exception as e:
                        print(f"ERROR: task {tid} execution failed: {e}")
                        result = None


                    try:
                        if generation != self._generation:
                            with self._active_task_ids_lock:
                                self._active_task_ids.discard(task_obj["task_id"])
                            continue
                        task_result = TaskResult(
                            data=result,
                            task_id=task_obj["task_id"],
                            create_time=task_obj["create_time"],
                            complete_time=time.monotonic_ns(),
                        )
                        output_queue.put_nowait(task_result)


                        with self._active_task_ids_lock:
                            self._active_task_ids.discard(task_obj["task_id"])

                    except queue.Full:
                        print("ERROR: output queue is full")
                        raise

        finally:

            pending_tasks = []
            for task_obj in running_tasks.values():
                task = task_obj["task"]
                if not task.done():
                    task.cancel()
                    pending_tasks.append(task)
                with self._active_task_ids_lock:
                    self._active_task_ids.discard(task_obj["task_id"])
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            if generation == self._generation:
                self._input_event = None

    def _drain_pending_inputs(
        self,
        running_tasks: dict[str, dict[str, Any]],
        input_queue: queue.Queue,
    ):
        """Drain pending inputs."""
        while True:
            try:
                task_input = input_queue.get_nowait()
            except queue.Empty:
                break

            tid = str(task_input["task_id"])

            if tid in running_tasks:
                raise ValueError(f"duplicate task ID: {tid}")


            async_task = asyncio.create_task(
                task_input["async_fn"](*task_input["args"], **task_input["kwargs"]),
                name=tid,
            )

            running_tasks[tid] = {
                "task": async_task,
                "task_id": task_input["task_id"],
                "create_time": task_input["create_time"],
            }

    async def _wait_for_new_tasks(
        self,
        *,
        exit_event: threading.Event,
        pause_event: threading.Event,
        input_queue: queue.Queue,
        input_event: asyncio.Event,
    ):
        """Wait for new tasks."""
        while not exit_event.is_set() and not pause_event.is_set():
            if input_queue.qsize() > 0:
                return
            input_event.clear()
            if input_queue.qsize() > 0 or exit_event.is_set():
                return
            await input_event.wait()

    def submit(
        self,
        async_fn: Callable[..., Awaitable[T]],
        *args,
        task_id: int,
        **kwargs,
    ) -> int:
        """Submit."""

        with self._thread_exception_lock:
            if self._thread_exception is not None:
                raise RuntimeError(f"AsyncTaskRunner is unusable: {self._thread_exception}")


        with self._active_task_ids_lock:
            if task_id in self._active_task_ids:
                raise ValueError(f"duplicate task ID: {task_id}")
            self._active_task_ids.add(task_id)


        fn_kwargs = {k: v for k, v in kwargs.items() if k != 'task_id'}

        task_input = {
            "async_fn": async_fn,
            "args": args,
            "kwargs": fn_kwargs,
            "task_id": task_id,
            "create_time": time.monotonic_ns(),
        }

        try:
            self.input_queue.put_nowait(task_input)
        except queue.Full:
            with self._active_task_ids_lock:
                self._active_task_ids.discard(task_id)
            raise queue.Full("input queue is full")

        self._signal_new_input()
        return task_id

    def wait(
        self,
        count: int,
        timeout: float | None = None,
    ) -> list[TaskResult[T]]:
        """Wait."""
        if timeout is None:
            timeout = 3600.0

        start_time = time.time()
        results = []

        while len(results) < count:

            with self._thread_exception_lock:
                if self._thread_exception is not None:
                    raise RuntimeError(f"AsyncTaskRunner error: {self._thread_exception}")

            if self.exiting.is_set():
                raise RuntimeError("AsyncTaskRunner is shutting down")


            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"waited{count}tasks; received only{len(results)}results"
                )


            try:
                wait_time = min(0.1, timeout - elapsed)
                result = self.output_queue.get(timeout=wait_time)
                results.append(result)
            except queue.Empty:
                continue

        return results

    def pause(self):
        """Pause."""
        self.paused.set()

    def resume(self):
        """Resume."""
        self.paused.clear()
        self._signal_new_input()

    def wait_until_idle(self, timeout: float = 3600.0) -> None:
        """Wait until every task accepted before pause has completed."""
        deadline = time.monotonic() + timeout
        while True:
            with self._thread_exception_lock:
                if self._thread_exception is not None:
                    raise RuntimeError(
                        f"AsyncTaskRunner error: {self._thread_exception}"
                    )
            with self._active_task_ids_lock:
                active_count = len(self._active_task_ids)
            if active_count == 0 and self.input_queue.empty():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out draining rollout tasks before weight reload: "
                    f"active={active_count}, queued={self.input_queue.qsize()}"
                )
            time.sleep(0.05)

    def get_queue_sizes(self) -> tuple[int, int]:
        """Get queue sizes."""
        return self.input_queue.qsize(), self.output_queue.qsize()

    def _signal_new_input(self):
        """Signal new input."""
        if self._loop is None or self._input_event is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._input_event.set)
        except RuntimeError:
            pass
