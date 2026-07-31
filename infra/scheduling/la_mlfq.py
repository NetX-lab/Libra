"""Support code for La mlfq."""

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from RL_Framework.infra.scheduling.base import (
    BaseScheduler,
    InstanceHandle,
    LoadBalanceStrategy,
    RoutingRule,
    SchedulerStats,
    SchedulingResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class ScoutStatus(Enum):
    """Scout status implementation."""
    RUNNING = "running"
    MIGRATED = "migrated"
    COMPLETED = "completed"


@dataclass
class ScoutState:
    """Scout state implementation."""
    prompt_id: str
    instance_index: int
    initial_bucket: str
    current_bucket: str
    status: ScoutStatus = ScoutStatus.RUNNING
    created_at: float = field(default_factory=time.time)
    generated_tokens: int = 0


@dataclass
class WaitingRequest:
    """Waiting request implementation."""
    prompt_id: str
    input_tokens: int
    n_samples: int
    epoch: int
    sample_index: int
    created_at: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None


@dataclass
class MigrationDecision:
    """Migration decision implementation."""
    should_migrate: bool
    reason: str
    current_bucket: str
    target_bucket: str = ""
    generated_tokens: int = 0


@dataclass
class RequestMigrationState:
    """Request migration state implementation."""
    request_id: str
    prompt_id: str
    instance_index: int
    current_bucket: str
    generated_tokens: int = 0
    is_scout: bool = False


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class HistoryTable:
    """History table implementation."""

    def __init__(self, ttl_epochs: int = 5):
        """Initialize the instance."""
        self._table: dict[str, dict[str, Any]] = {}  # prompt_id -> {bucket, epoch}
        self._ttl_epochs = ttl_epochs
        self._lock = threading.Lock()

    def lookup(self, prompt_id: str, current_epoch: int = -1) -> Optional[str]:
        """Lookup."""
        with self._lock:
            entry = self._table.get(prompt_id)
            if entry is None:
                return None

            if current_epoch >= 0 and self._ttl_epochs > 0:
                if current_epoch - entry["epoch"] > self._ttl_epochs:
                    del self._table[prompt_id]
                    return None
            return entry["bucket"]

    def update(self, prompt_id: str, bucket: str, epoch: int = -1):
        """Update."""
        with self._lock:
            self._table[prompt_id] = {"bucket": bucket, "epoch": epoch}

    def on_epoch_end(self, epoch: int):
        """On epoch end."""
        if self._ttl_epochs <= 0:
            return
        with self._lock:
            expired = [
                pid for pid, entry in self._table.items()
                if epoch - entry["epoch"] > self._ttl_epochs
            ]
            for pid in expired:
                del self._table[pid]
            if expired:
                logger.info(
                    f"[HistoryTable] Epoch {epoch} removed {len(expired)} expired records, "
                    f"{len(self._table)} remaining"
                )

    @property
    def size(self) -> int:
        return len(self._table)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class ScoutManager:
    """Scout manager implementation."""

    def __init__(self, scout_timeout: float = 30.0):
        """Initialize the instance."""
        self._scouts: dict[str, ScoutState] = {}
        self._waiting: dict[str, list[WaitingRequest]] = defaultdict(list)
        self._scout_timeout = scout_timeout
        self._lock = threading.Lock()

    def should_scout(self, prompt_id: str) -> bool:
        """Should scout."""
        with self._lock:
            return prompt_id not in self._scouts

    def register_scout(
        self,
        prompt_id: str,
        instance_index: int,
        bucket: str,
    ):
        """Register scout."""
        with self._lock:
            self._scouts[prompt_id] = ScoutState(
                prompt_id=prompt_id,
                instance_index=instance_index,
                initial_bucket=bucket,
                current_bucket=bucket,
            )
        logger.debug(
            f"[ScoutManager] registered scout: prompt={prompt_id}, "
            f"instance={instance_index}, bucket={bucket}"
        )

    def add_waiting(self, waiting: WaitingRequest):
        """Add waiting."""
        with self._lock:
            self._waiting[waiting.prompt_id].append(waiting)
        logger.debug(
            f"[ScoutManager] queued request: prompt={waiting.prompt_id}, "
            f"sample={waiting.sample_index}"
        )

    def is_scout(self, prompt_id: str, instance_index: int) -> bool:
        """Is scout."""
        with self._lock:
            scout = self._scouts.get(prompt_id)
            if scout is None:
                return False
            return scout.instance_index == instance_index

    def on_scout_migrated(self, prompt_id: str, new_bucket: str) -> list[WaitingRequest]:
        """On scout migrated."""
        with self._lock:
            scout = self._scouts.get(prompt_id)
            if scout is None:
                return []
            scout.current_bucket = new_bucket
            scout.status = ScoutStatus.MIGRATED
            waiting = self._waiting.pop(prompt_id, [])
        logger.info(
            f"[ScoutManager] scout migration: prompt={prompt_id}, "
            f"{scout.initial_bucket} -> {new_bucket}, "
            f"released {len(waiting)} waiting requests"
        )
        return waiting

    def on_scout_completed(self, prompt_id: str, final_bucket: str) -> list[WaitingRequest]:
        """On scout completed."""
        with self._lock:
            scout = self._scouts.get(prompt_id)
            if scout is None:
                return []
            scout.current_bucket = final_bucket
            scout.status = ScoutStatus.COMPLETED
            waiting = self._waiting.pop(prompt_id, [])
        logger.info(
            f"[ScoutManager] scout completed: prompt={prompt_id}, "
            f"final_bucket={final_bucket}, "
            f"released {len(waiting)} waiting requests"
        )
        return waiting

    def get_scout_bucket(self, prompt_id: str) -> Optional[str]:
        """Get scout bucket."""
        with self._lock:
            scout = self._scouts.get(prompt_id)
            return scout.current_bucket if scout else None

    def get_scout_status(self, prompt_id: str) -> Optional[ScoutStatus]:
        """Get scout status."""
        with self._lock:
            scout = self._scouts.get(prompt_id)
            return scout.status if scout else None

    def check_timeouts(self) -> list[str]:
        """Check timeouts."""
        now = time.time()
        timed_out = []
        with self._lock:
            for pid, scout in self._scouts.items():
                if scout.status == ScoutStatus.RUNNING:
                    if now - scout.created_at > self._scout_timeout:
                        timed_out.append(pid)
        return timed_out

    def force_release(self, prompt_id: str) -> list[WaitingRequest]:
        """Force release."""
        with self._lock:
            waiting = self._waiting.pop(prompt_id, [])
            scout = self._scouts.get(prompt_id)
            if scout:
                scout.status = ScoutStatus.COMPLETED
        if waiting:
            logger.warning(
                f"[ScoutManager] forced release: prompt={prompt_id}, "
                f"{len(waiting)} waiting requests"
            )
        return waiting

    def reset(self):
        """Reset the current state."""
        with self._lock:

            for pid, waiters in self._waiting.items():
                if waiters:
                    logger.warning(
                        f"[ScoutManager] released residual waiters during reset: "
                        f"prompt={pid}, count={len(waiters)}"
                    )
            self._scouts.clear()
            self._waiting.clear()

    @property
    def active_scouts(self) -> int:
        with self._lock:
            return sum(
                1 for s in self._scouts.values()
                if s.status == ScoutStatus.RUNNING
            )

    @property
    def total_waiting(self) -> int:
        with self._lock:
            return sum(len(w) for w in self._waiting.values())


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class MigrationController:
    """Migration controller implementation."""

    def __init__(
        self,
        bucket_thresholds: dict[str, int] | None = None,
    ):
        """Initialize the instance."""
        self._bucket_thresholds = bucket_thresholds or {
            "short": 3000,
            "medium": 10000,
        }

        self._demotion_map: dict[str, str] = {
            "short": "long",
            "medium": "long",
        }

        self._active: dict[str, RequestMigrationState] = {}
        self._lock = threading.Lock()

        self._migration_count = 0
        self._check_count = 0

    def register_request(
        self,
        request_id: str,
        prompt_id: str,
        instance_index: int,
        bucket: str,
        is_scout: bool = False,
    ):
        """Register request."""
        with self._lock:
            self._active[request_id] = RequestMigrationState(
                request_id=request_id,
                prompt_id=prompt_id,
                instance_index=instance_index,
                current_bucket=bucket,
                is_scout=is_scout,
            )

    def check_migration(
        self,
        request_id: str,
        generated_tokens: int,
    ) -> MigrationDecision:
        """Check migration."""
        self._check_count += 1

        with self._lock:
            state = self._active.get(request_id)
            if state is None:
                return MigrationDecision(
                    should_migrate=False,
                    reason="request_not_tracked",
                    current_bucket="unknown",
                )
            state.generated_tokens = generated_tokens

        threshold = self._bucket_thresholds.get(state.current_bucket)
        if threshold is None:

            return MigrationDecision(
                should_migrate=False,
                reason="no_threshold_for_bucket",
                current_bucket=state.current_bucket,
            )

        if generated_tokens <= threshold:
            return MigrationDecision(
                should_migrate=False,
                reason="below_threshold",
                current_bucket=state.current_bucket,
                generated_tokens=generated_tokens,
            )

        target_bucket = self._demotion_map.get(state.current_bucket, "long")
        return MigrationDecision(
            should_migrate=True,
            reason="threshold_exceeded",
            current_bucket=state.current_bucket,
            target_bucket=target_bucket,
            generated_tokens=generated_tokens,
        )

    def on_migration_executed(self, request_id: str, new_bucket: str):
        """On migration executed."""
        with self._lock:
            state = self._active.get(request_id)
            if state:
                state.current_bucket = new_bucket
                self._migration_count += 1

    def on_request_done(self, request_id: str):
        """On request done."""
        with self._lock:
            self._active.pop(request_id, None)

    def get_request_state(self, request_id: str) -> Optional[RequestMigrationState]:
        """Get request state."""
        with self._lock:
            return self._active.get(request_id)

    def get_stats(self) -> dict:
        return {
            "active_tracked": len(self._active),
            "total_migrations": self._migration_count,
            "total_checks": self._check_count,
        }

    def reset(self):
        """Reset the current state."""
        with self._lock:
            self._active.clear()
            self._migration_count = 0
            self._check_count = 0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class LAMLFQScheduler(BaseScheduler):
    """L a m l f q scheduler implementation."""

    DEFAULT_BUCKETS = {
        "short": {"tp_degrees": [1, 2], "max_tokens": 5000},
        "long": {"tp_degrees": [4, 8], "max_tokens": 50000},
    }

    def __init__(
        self,
        buckets: dict[str, dict] | None = None,
        length_thresholds: dict[str, int] | None = None,
        migration_threshold: int = 3000,
        scout_timeout: float = 30.0,
        history_ttl: int = 5,
        load_balance_strategy: str = "least_connections",
        max_queue_length: int = 100,
        enable_fallback: bool = True,
    ):
        super().__init__(name="LA-MLFQ")


        self._buckets = buckets or dict(self.DEFAULT_BUCKETS)


        self._bucket_rules: dict[str, RoutingRule] = {}
        for bname, bcfg in self._buckets.items():
            tp_degrees = bcfg.get("tp_degrees", [1, 2])

            all_other_tps = []
            for other_name, other_cfg in self._buckets.items():
                if other_name != bname:
                    all_other_tps.extend(other_cfg.get("tp_degrees", []))
            self._bucket_rules[bname] = RoutingRule(
                category=bname,
                preferred_tp_degrees=tp_degrees,
                fallback_tp_degrees=all_other_tps,
            )


        self._thresholds = length_thresholds or {
            "short": 5000,
            "medium": 10000,
            "long": 15000,
        }


        if load_balance_strategy == "round_robin":
            self._strategy = LoadBalanceStrategy.ROUND_ROBIN
        else:
            self._strategy = LoadBalanceStrategy.LEAST_CONNECTIONS
        self._max_queue_length = max_queue_length
        self._enable_fallback = enable_fallback
        self._rr_index: dict[int, int] = defaultdict(int)


        self.history_table = HistoryTable(ttl_epochs=history_ttl)
        self.scout_manager = ScoutManager(scout_timeout=scout_timeout)
        self.migration_controller = MigrationController(
            bucket_thresholds={
                bname: bcfg.get("max_tokens", 5000)
                for bname, bcfg in self._buckets.items()
                if bname != "long" and bname != "extra_long"
            },
        )


        self._current_epoch = -1


        self._pending_callbacks: dict[str, list[Callable]] = defaultdict(list)

        logger.info(
            f"LAMLFQScheduler initialized: buckets={list(self._buckets.keys())}, "
            f"migration_threshold={migration_threshold}, "
            f"scout_timeout={scout_timeout}, "
            f"history_ttl={history_ttl}"
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def _categorize_to_bucket(self, input_tokens: int) -> str:
        """Categorize to bucket."""

        short_th = self._thresholds.get("short", 5000)
        if input_tokens <= short_th:
            return "short"
        return "long"

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def schedule(
        self,
        input_tokens: int,
        prompt_id: str = "",
        n_samples: int = 1,
        epoch: int = -1,
    ) -> SchedulingResult:
        """Schedule."""
        if epoch >= 0:
            self._current_epoch = epoch

        with self._lock:
            self._stats.total_requests += 1


        if not prompt_id:
            return self._default_length_route(input_tokens, prompt_id)


        historical_bucket = self.history_table.lookup(
            prompt_id, current_epoch=self._current_epoch
        )
        if historical_bucket:
            logger.debug(
                f"[LA-MLFQ] inter-epoch hit: prompt={prompt_id}, "
                f"bucket={historical_bucket}"
            )
            result = self._route_to_bucket(
                historical_bucket, input_tokens, prompt_id,
                reason="history_hit",
            )
            if result.instance_index >= 0:
                return result



        if n_samples > 1:

            if self.scout_manager.should_scout(prompt_id):

                result = self._route_to_bucket(
                    "short", input_tokens, prompt_id,
                    reason="scout",
                )
                if result.instance_index >= 0:
                    self.scout_manager.register_scout(
                        prompt_id=prompt_id,
                        instance_index=result.instance_index,
                        bucket="short",
                    )

                    self.migration_controller.register_request(
                        request_id=f"{prompt_id}_scout",
                        prompt_id=prompt_id,
                        instance_index=result.instance_index,
                        bucket="short",
                        is_scout=True,
                    )
                    return result

                logger.warning(
                    f"[LA-MLFQ] Short Bucket has no available instance, "
                    f"prompt={prompt_id} using default routing"
                )
                return self._default_length_route(input_tokens, prompt_id)

            else:

                scout_status = self.scout_manager.get_scout_status(prompt_id)
                if scout_status in (ScoutStatus.MIGRATED, ScoutStatus.COMPLETED):

                    target_bucket = self.scout_manager.get_scout_bucket(prompt_id)
                    if target_bucket:
                        return self._route_to_bucket(
                            target_bucket, input_tokens, prompt_id,
                            reason="scout_follow",
                        )


                with self._lock:
                    self._stats.pending_routes += 1
                return SchedulingResult(
                    instance_index=-1,
                    tp_degree=0,
                    category="pending",
                    is_fallback=False,
                    reason="waiting_for_scout",
                    pending=True,
                    prompt_id=prompt_id,
                )


        return self._default_length_route(input_tokens, prompt_id)

    def _route_to_bucket(
        self,
        bucket: str,
        input_tokens: int,
        prompt_id: str = "",
        reason: str = "",
    ) -> SchedulingResult:
        """Route to bucket."""
        rule = self._bucket_rules.get(bucket)
        if rule is None:

            return self._default_length_route(input_tokens, prompt_id)

        with self._lock:
            self._stats.category_counts[bucket] += 1


            selected = self._try_select(rule.preferred_tp_degrees)
            if selected is not None:
                self._stats.preferred_routes += 1
                self._stats.category_tp_counts[bucket][selected.tp_degree] += 1
                selected.inc_active()
                return SchedulingResult(
                    instance_index=selected.index,
                    tp_degree=selected.tp_degree,
                    category=bucket,
                    is_fallback=False,
                    reason=reason,
                    prompt_id=prompt_id,
                )


            if self._enable_fallback and rule.fallback_tp_degrees:
                selected = self._try_select(rule.fallback_tp_degrees)
                if selected is not None:
                    self._stats.fallback_routes += 1
                    self._stats.category_tp_counts[bucket][selected.tp_degree] += 1
                    selected.inc_active()
                    return SchedulingResult(
                        instance_index=selected.index,
                        tp_degree=selected.tp_degree,
                        category=bucket,
                        is_fallback=True,
                        reason=f"{reason}_fallback",
                        prompt_id=prompt_id,
                    )


            all_ready = [h for h in self._instances if h.is_ready]
            if all_ready:
                selected = min(all_ready, key=lambda h: h.active_requests)
                self._stats.fallback_routes += 1
                self._stats.category_tp_counts[bucket][selected.tp_degree] += 1
                selected.inc_active()
                return SchedulingResult(
                    instance_index=selected.index,
                    tp_degree=selected.tp_degree,
                    category=bucket,
                    is_fallback=True,
                    reason="global_fallback",
                    prompt_id=prompt_id,
                )


        with self._lock:
            self._stats.failed_routes += 1
        return SchedulingResult(
            instance_index=-1,
            tp_degree=0,
            category=bucket,
            is_fallback=False,
            reason="no_available_instance",
            prompt_id=prompt_id,
        )

    def _default_length_route(
        self, input_tokens: int, prompt_id: str = ""
    ) -> SchedulingResult:
        """Default length route."""
        bucket = self._categorize_to_bucket(input_tokens)
        return self._route_to_bucket(bucket, input_tokens, prompt_id, reason="default")

    def _try_select(self, tp_preferences: list[int]) -> Optional[InstanceHandle]:
        """Try select."""
        for tp_degree in tp_preferences:
            candidates = [
                h for h in self._instances_by_tp.get(tp_degree, [])
                if h.is_ready
            ]
            if not candidates:
                continue
            if self._max_queue_length > 0:
                candidates = [
                    h for h in candidates
                    if h.active_requests < self._max_queue_length
                ]
            if not candidates:
                continue
            if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
                idx = self._rr_index[tp_degree] % len(candidates)
                self._rr_index[tp_degree] += 1
                return candidates[idx]
            else:
                return min(candidates, key=lambda h: h.active_requests)
        return None

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def on_request_done(
        self,
        instance_index: int,
        prompt_id: str = "",
        final_bucket: str = "",
        output_tokens: int = 0,
    ):
        """On request done."""

        handle = self.get_instance_handle(instance_index)
        if handle:
            with self._lock:
                handle.dec_active()


        if prompt_id and final_bucket:
            self.history_table.update(
                prompt_id, final_bucket, epoch=self._current_epoch
            )


        if prompt_id and self.scout_manager.is_scout(prompt_id, instance_index):
            bucket = final_bucket or self.scout_manager.get_scout_bucket(prompt_id) or "short"
            waiting_requests = self.scout_manager.on_scout_completed(prompt_id, bucket)
            self._process_released_waiting(waiting_requests, bucket)


        request_id = f"{prompt_id}_scout" if prompt_id else ""
        if request_id:
            self.migration_controller.on_request_done(request_id)

    def check_and_migrate(
        self,
        request_id: str,
        generated_tokens: int,
    ) -> MigrationDecision:
        """Check and migrate."""
        decision = self.migration_controller.check_migration(
            request_id, generated_tokens
        )

        if decision.should_migrate:

            self.migration_controller.on_migration_executed(
                request_id, decision.target_bucket
            )
            with self._lock:
                self._stats.migrated_routes += 1


            state = self.migration_controller.get_request_state(request_id)
            if state and state.is_scout:
                waiting = self.scout_manager.on_scout_migrated(
                    state.prompt_id, decision.target_bucket
                )
                self._process_released_waiting(waiting, decision.target_bucket)

            logger.info(
                f"[LA-MLFQ] request migration: {request_id}, "
                f"{decision.current_bucket} -> {decision.target_bucket}, "
                f"generated_tokens={generated_tokens}"
            )

        return decision

    def _process_released_waiting(
        self, waiting_requests: list[WaitingRequest], target_bucket: str
    ):
        """Process released waiting."""
        for wr in waiting_requests:
            result = self._route_to_bucket(
                target_bucket, wr.input_tokens, wr.prompt_id,
                reason="scout_released",
            )

            if wr.future is not None and not wr.future.done():
                wr.future.set_result(result)

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def on_epoch_start(self, epoch: int):
        """On epoch start."""
        self._current_epoch = epoch
        logger.info(
            f"[LA-MLFQ] Epoch {epoch} started, "
            f"history table size={self.history_table.size}"
        )

    def on_epoch_end(self, epoch: int):
        """On epoch end."""
        self.history_table.on_epoch_end(epoch)

        timed_out = self.scout_manager.check_timeouts()
        for pid in timed_out:
            waiting = self.scout_manager.force_release(pid)

            for wr in waiting:
                result = self._default_length_route(wr.input_tokens, wr.prompt_id)
                if wr.future is not None and not wr.future.done():
                    wr.future.set_result(result)
        self.scout_manager.reset()
        self.migration_controller.reset()
        logger.info(
            f"[LA-MLFQ] Epoch {epoch} ended, "
            f"history table size={self.history_table.size}, "
            f"migration statistics={self.migration_controller.get_stats()}"
        )

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    def print_stats(self):
        """Print stats."""
        super().print_stats()
        print(f"\n  [LA-MLFQ extended statistics]")
        print(f"  history table size: {self.history_table.size}")
        print(f"  active scouts: {self.scout_manager.active_scouts}")
        print(f"  waiting requests: {self.scout_manager.total_waiting}")
        mig_stats = self.migration_controller.get_stats()
        print(f"  migration controller: {mig_stats}")

    def reset_stats(self):
        super().reset_stats()

    # ----------------------------------------------------------------

    # ----------------------------------------------------------------

    @classmethod
    def from_config(cls, hetero_config: Any) -> "LAMLFQScheduler":
        """From config."""
        sched = hetero_config.scheduling


        buckets = getattr(sched, "la_mlfq_buckets", None) or cls.DEFAULT_BUCKETS
        migration_threshold = getattr(sched, "la_mlfq_migration_threshold", 3000)
        scout_timeout = getattr(sched, "la_mlfq_scout_timeout", 30.0)
        history_ttl = getattr(sched, "la_mlfq_history_ttl", 5)

        return cls(
            buckets=buckets,
            length_thresholds=(
                sched.length_thresholds
                if isinstance(sched.length_thresholds, dict)
                else {}
            ),
            migration_threshold=migration_threshold,
            scout_timeout=scout_timeout,
            history_ttl=history_ttl,
            load_balance_strategy=sched.load_balance_strategy,
            max_queue_length=sched.max_queue_length,
            enable_fallback=sched.enable_fallback,
        )
