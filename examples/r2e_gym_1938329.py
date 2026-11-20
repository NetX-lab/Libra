"""Train the 100-update elastic R2E example with a fixed heldout cohort."""

from __future__ import annotations

import argparse
import asyncio
import copy
import math
from datetime import timedelta
from pathlib import Path
from random import Random
from statistics import mean
from urllib.parse import urlsplit

import torch.distributed as dist

from RL_Framework import AsyncRLTrainer, load_config
from RL_Framework.engine.rollout_engine import VLLMRolloutEngine
from RL_Framework.infra.elastic.rollout_lifecycle import publish_with_consensus
from RL_Framework.infra.sync.publication_session import attest_nccl_publication, publication_inventory


def split_task_indices(rows):
    by_task = {}
    for index, row in enumerate(rows):
        repo, commit = row.get("repo_name"), row.get("commit_hash")
        if not repo or not commit:
            raise ValueError("R2E tasks must include repo_name and commit_hash")
        by_task.setdefault((repo, commit), index)
    if len(by_task) < 1000:
        raise ValueError("The R2E example requires at least 1,000 distinct repository/commit tasks")
    indices = list(by_task.values())
    Random(42).shuffle(indices)
    return indices[:800], indices[800:820]


class _EvaluationEngine(VLLMRolloutEngine):
    def __init__(self, endpoint, model_path, *, revision):
        url = urlsplit(endpoint if "://" in endpoint else "http://" + endpoint)
        super().__init__(host=url.hostname, port=url.port, model_path=model_path)
        self.revision = revision

    async def generate(self, *args, **kwargs):
        response = await super().generate(*args, **kwargs)
        if response.get("sample_revision") != self.revision:
            raise RuntimeError(
                f"Heldout generation used revision {response.get('sample_revision')}, "
                f"expected {self.revision}"
            )
        return response


async def _heldout_report(workflow, dataset, targets, model_path, revision):
    evaluation = copy.copy(workflow)

    def fixed_seed(row, *, version, rollout_index, turn):
        return workflow.sampling_seed(row, version=0, rollout_index=rollout_index, turn=turn)

    evaluation.sampling_seed = fixed_seed
    engines = [_EvaluationEngine(target["endpoint"], model_path, revision=revision)
               for target in targets]
    scores = []
    try:
        for index, row in enumerate(dataset):
            trajectory = await evaluation.run_episode(
                engines[index % len(engines)], row, version=revision,
            )
            if trajectory.get("sample_revision") != revision:
                raise RuntimeError(f"Heldout episode did not produce samples at revision {revision}")
            reward = float(trajectory["rewards"].reshape(-1)[0])
            if not math.isfinite(reward):
                raise ValueError(f"Heldout task {row['prompt_id']} returned a nonfinite reward")
            scores.append(dict(task_id=row["prompt_id"], reward=reward))
    finally:
        await asyncio.gather(*(engine.close() for engine in engines))
    return dict(version=revision, evaluation_seed=42, task_count=len(scores),
                reward_mean=mean(score["reward"] for score in scores), rows=scores)


class HeldoutTrainer(AsyncRLTrainer):
    def train(self, workflow, dataset=None):
        if self.start_step != 0 or self.initial_model_version != 0:
            raise ValueError("The R2E example starts from optimizer revision zero")
        train, heldout = split_task_indices(dataset)
        self.heldout = dataset.select(heldout)
        self._cleanup_started = False
        try:
            return super().train(workflow, dataset.select(train))
        except BaseException:
            if not self._cleanup_started:
                try:
                    self._cleanup()
                except Exception as exc:
                    print(f"R2E cleanup after training failed: {exc}", flush=True)
            raise

    def _cleanup(self, **kwargs):
        self._cleanup_started = True
        return super()._cleanup(**kwargs)

    def _rank_zero(self, operation):
        packet = [None]
        if self.is_main_process:
            try:
                packet[0] = (operation(), None)
            except Exception as exc:
                packet[0] = (None, f"{type(exc).__name__}: {exc}")
        if dist.is_initialized():
            dist.broadcast_object_list(packet, src=0, group=self.control)
        result, error = packet[0]
        if error:
            raise RuntimeError(error)
        return result

    def setup(self, workflow):
        super().setup(workflow)
        # Evaluation waits with a CPU collective whose deadline exceeds the eval deadline.
        self.control = dist.new_group(backend="gloo", timeout=timedelta(hours=2)) if dist.is_initialized() else None
        self._sync_weights_nccl(0)
        self._evaluate(workflow, 0)

    def _needs_publication(self, revision):
        if self.train_engine.optimizer_revision != revision:
            raise RuntimeError(f"Training has not completed optimizer revision {revision}")
        targets = publication_inventory(self)["targets"]
        if not targets:
            raise RuntimeError(f"No rollout service is available for evaluation at revision {revision}")
        receipt = getattr(self, "last_publication", None)
        if not receipt or receipt.get("optimizer_revision") != revision or receipt.get("targets") != targets:
            return True
        try:
            attest_nccl_publication(self.config, receipt, targets=targets)
        except Exception:
            return True
        return False

    def _run_periodic_evaluation(self, workflow, dataset, step):
        revision = step + 1
        if revision % 20 == 0:
            if self._rank_zero(lambda: self._needs_publication(revision)):
                publish_with_consensus(self, revision)
                if self.train_engine.get_version() != revision:
                    self.train_engine.set_version(revision)
                    if self.dispatcher is not None:
                        self.dispatcher.on_publication(revision)
            self._evaluate(workflow, revision)

    def _evaluate(self, workflow, revision):
        def evaluate():
            receipt = getattr(self, "last_publication", None)
            if (self.train_engine.optimizer_revision != revision or not receipt
                    or receipt.get("optimizer_revision") != revision or not receipt["targets"]):
                raise RuntimeError(f"No verified publication of optimizer revision {revision}")
            directory = Path(self.config.log_dir) / "eval"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"heldout-step-{revision:03d}.json"
            if path.exists():
                raise FileExistsError(path)
            evaluation = _heldout_report(
                workflow, self.heldout, receipt["targets"], self.config.model_path, revision,
            )
            report = asyncio.run(asyncio.wait_for(evaluation, timeout=3600))
            self._write_json_atomic(path, report)
            print(f"[Heldout] revision={revision} reward_mean={report['reward_mean']:.4f}", flush=True)

        self._rank_zero(evaluate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if (config.total_steps != 100 or config.seed != 42 or config.sync_interval != 1
            or config.grpo_objective != "decoupled" or config.weight_sync_mode != "nccl"
            or not config.require_rollout_weight_sync):
        parser.error("Expected the 100-step R2E configuration with seed 42 and NCCL publication after every update")
    from examples.r2e_gym_async_rl import main as train

    try:
        train(config=config, data_path=str(args.tasks), trainer_factory=HeldoutTrainer,
              workflow_env={})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
