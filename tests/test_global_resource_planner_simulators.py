import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.global_resource_planner import (
    GlobalResourcePlanner,
)
from RL_Framework.infra.cost_model.model import (
    RequestInfo,
    RolloutClusterConfig,
    TrainParallelConfig,
)
from RL_Framework.infra.cost_model.simulator_adapters import HybridSimulatorCostModel


class TestGlobalResourcePlannerSimulatorAdapters(unittest.TestCase):
    def _base_config(self) -> AsyncRLConfig:
        cfg = AsyncRLConfig(
            model_path="/tmp/fake-model",
            train_gpus=2,
            rollout_gpus=2,
            train_tp_size=1,
            train_pp_size=1,
            train_dp_size=2,
            micro_batch_size=1,
            batch_size=8,
            n_total_gpus=4,
            max_seq_length=4096,
        )
        cfg.global_resource_planner.enabled = True
        cfg.global_resource_planner.plan_interval = 1
        cfg.global_resource_planner.warmup_steps = 0
        cfg.global_resource_planner.min_history_size = 4
        cfg.global_resource_planner.reconfiguration_cost_s = 0.0
        cfg.global_resource_planner.min_gain_ratio = 0.0
        cfg.global_resource_planner.allowed_train_tp = [1, 2]
        cfg.global_resource_planner.allowed_train_pp = [1]
        cfg.global_resource_planner.allowed_rollout_tp = [1, 2]
        cfg.global_resource_planner.micro_batch_sizes = [1]
        return cfg

    def test_sailor_command_backend_is_used(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sailor = root / "sailor"
            sailor.mkdir()
            script = root / "fake_sailor.py"
            script.write_text(
                """
import json, sys
inp, out = sys.argv[1], sys.argv[2]
payload = json.load(open(inp))
tp = payload["train_config"]["tp"]
json.dump({"t_train": 10.0 / tp, "source": "fake_sailor"}, open(out, "w"))
""".strip(),
                encoding="utf-8",
            )
            cfg = self._base_config()
            cfg.global_resource_planner.train_backend = "sailor"
            cfg.global_resource_planner.sailor_path = str(sailor)
            cfg.global_resource_planner.sailor_train_command = (
                f"{sys.executable} {script} {{input_json}} {{output_json}}"
            )

            evaluator = HybridSimulatorCostModel(cfg)
            seconds, details = evaluator.evaluate_training(
                TrainParallelConfig(tp=2, pp=1, dp=1, b_micro=1),
                B_global=8,
                L=1024,
            )

            self.assertEqual(seconds, 5.0)
            self.assertEqual(details["backend"], "sailor")
            self.assertEqual(details["raw"]["source"], "fake_sailor")

    def test_vidur_command_backend_is_used(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vidur = root / "vidur"
            vidur.mkdir()
            script = root / "fake_vidur.py"
            script.write_text(
                """
import json, sys
trace, out = sys.argv[1], sys.argv[2]
num_requests = sum(1 for _ in open(trace)) - 1
json.dump({"makespan": 7.5, "num_requests": num_requests}, open(out, "w"))
""".strip(),
                encoding="utf-8",
            )
            cfg = self._base_config()
            cfg.global_resource_planner.rollout_backend = "vidur"
            cfg.global_resource_planner.vidur_path = str(vidur)
            cfg.global_resource_planner.vidur_rollout_command = (
                f"{sys.executable} {script} {{trace_csv}} {{output_json}}"
            )

            evaluator = HybridSimulatorCostModel(cfg)
            seconds, details = evaluator.evaluate_rollout(
                RolloutClusterConfig(tp_list=[2]),
                [
                    RequestInfo(prompt_length=128, gen_length=512),
                    RequestInfo(prompt_length=256, gen_length=256),
                ],
            )

            self.assertEqual(seconds, 7.5)
            self.assertEqual(details["backend"], "vidur")
            self.assertEqual(details["raw"]["num_requests"], 2)

    def test_missing_external_backend_falls_back_to_analytic(self):
        cfg = self._base_config()
        cfg.global_resource_planner.train_backend = "sailor"
        cfg.global_resource_planner.rollout_backend = "vidur"
        cfg.global_resource_planner.sailor_path = "/definitely/missing/sailor"
        cfg.global_resource_planner.vidur_path = "/definitely/missing/vidur"
        cfg.global_resource_planner.simulator_allow_fallback = True

        evaluator = HybridSimulatorCostModel(cfg)
        train_seconds, train_details = evaluator.evaluate_training(
            TrainParallelConfig(tp=1, pp=1, dp=2, b_micro=1),
            B_global=8,
            L=1024,
        )
        rollout_seconds, rollout_details = evaluator.evaluate_rollout(
            RolloutClusterConfig(tp_list=[1, 1]),
            [RequestInfo(prompt_length=128, gen_length=128)],
        )

        self.assertGreater(train_seconds, 0)
        self.assertGreaterEqual(rollout_seconds, 0)
        self.assertEqual(train_details["backend"], "analytic")
        self.assertEqual(train_details["requested_backend"], "sailor")
        self.assertEqual(rollout_details["backend"], "analytic")
        self.assertEqual(rollout_details["requested_backend"], "vidur")

    def test_planner_runs_with_hybrid_backends(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sailor = root / "sailor"
            vidur = root / "vidur"
            sailor.mkdir()
            vidur.mkdir()
            sailor_script = root / "fake_sailor.py"
            vidur_script = root / "fake_vidur.py"
            sailor_script.write_text(
                """
import json, sys
payload = json.load(open(sys.argv[1]))
gpus = payload["train_config"]["tp"] * payload["train_config"]["pp"] * payload["train_config"]["dp"]
json.dump({"t_train": 40.0 / gpus}, open(sys.argv[2], "w"))
""".strip(),
                encoding="utf-8",
            )
            vidur_script.write_text(
                """
import json, sys
tp_list = sys.argv[3].split(",") if len(sys.argv) > 3 else ["1"]
gpus = sum(int(x) for x in tp_list if x)
json.dump({"makespan": 80.0 / max(gpus, 1)}, open(sys.argv[2], "w"))
""".strip(),
                encoding="utf-8",
            )

            cfg = self._base_config()
            cfg.global_resource_planner.train_backend = "sailor"
            cfg.global_resource_planner.rollout_backend = "vidur"
            cfg.global_resource_planner.verbose = True
            cfg.global_resource_planner.sailor_path = str(sailor)
            cfg.global_resource_planner.vidur_path = str(vidur)
            cfg.global_resource_planner.sailor_train_command = (
                f"{sys.executable} {sailor_script} {{input_json}} {{output_json}}"
            )
            cfg.global_resource_planner.vidur_rollout_command = (
                f"{sys.executable} {vidur_script} {{trace_csv}} {{output_json}} {{tp_list}}"
            )

            planner = GlobalResourcePlanner.from_config(cfg)
            decision = planner.plan_if_needed(
                step=1,
                config=cfg,
                batch=[
                    {"input_len": 128, "output_len": 512},
                    {"input_len": 128, "output_len": 1024},
                    {"input_len": 256, "output_len": 2048},
                    {"input_len": 256, "output_len": 4096},
                ],
            )

            self.assertIsNotNone(decision.candidate_plan)
            self.assertIn(
                decision.reason,
                {"reconfigure", "same_plan", "gain_below_threshold"},
            )
            evaluated = decision.candidate_plan.metadata.get("evaluated", [])
            self.assertTrue(evaluated)

    def test_online_queue_pressure_triggers_replan_between_intervals(self):
        cfg = self._base_config()
        cfg.global_resource_planner.plan_interval = 100
        cfg.global_resource_planner.runtime_online_replanning = True
        cfg.global_resource_planner.runtime_queue_pressure_threshold = 0.5
        cfg.global_resource_planner.runtime_replan_cooldown_steps = 0

        planner = GlobalResourcePlanner.from_config(cfg)
        for _ in range(4):
            planner.observe_batch([{"input_len": 128, "output_len": 2048}])
        metrics = planner.observe_runtime(
            step=3,
            dispatcher_metrics={
                "pending_inputs": 8,
                "runner_input_queue": 8,
                "runner_output_queue": 0,
                "runner_max_queue_size": 16,
                "staleness_pending_limit": 8,
                "staleness_running": 1,
                "staleness_capacity": 4,
            },
            step_stats={
                "rollout_time": 20.0,
                "train_time": 5.0,
                "max_concurrent_rollouts": 8,
            },
        )

        decision = planner.plan_if_needed(
            step=3,
            config=cfg,
            runtime_metrics=metrics,
        )

        self.assertEqual(decision.trigger, "queue_pressure")
        self.assertIsNotNone(decision.candidate_plan)
        self.assertIn(
            decision.reason,
            {"reconfigure", "same_plan", "gain_below_threshold"},
        )

    def test_rejected_rollout_trigger_is_consumed_after_evaluation(self):
        cfg = self._base_config()
        cfg.global_resource_planner.plan_interval = 100
        cfg.global_resource_planner.runtime_online_replanning = True
        cfg.global_resource_planner.runtime_rejected_rollout_delta_threshold = 1
        cfg.global_resource_planner.runtime_replan_cooldown_steps = 0
        cfg.global_resource_planner.runtime_active_rollout_pressure_threshold = 2.0
        cfg.global_resource_planner.runtime_queue_pressure_threshold = 2.0

        planner = GlobalResourcePlanner.from_config(cfg)
        for _ in range(4):
            planner.observe_batch([{"input_len": 128, "output_len": 2048}])

        first = planner.observe_runtime(
            step=3,
            dispatcher_metrics={
                "runner_max_queue_size": 16,
                "staleness_pending_limit": 8,
                "staleness_running": 0,
                "staleness_rejected": 1,
            },
            step_stats={"max_concurrent_rollouts": 8},
        )
        first_decision = planner.plan_if_needed(
            step=3,
            config=cfg,
            runtime_metrics=first,
        )
        self.assertEqual(first_decision.trigger, "rejected_rollout_delta")

        second = planner.observe_runtime(
            step=4,
            dispatcher_metrics={
                "runner_max_queue_size": 16,
                "staleness_pending_limit": 8,
                "staleness_running": 0,
                "staleness_rejected": 1,
            },
            step_stats={"max_concurrent_rollouts": 8},
        )
        second_decision = planner.plan_if_needed(
            step=4,
            config=cfg,
            runtime_metrics=second,
        )
        self.assertEqual(second_decision.trigger, "interval_skip")

    def test_forced_train_and_rollout_plan_can_express_cluster_swap(self):
        cfg = self._base_config()
        cfg.n_total_gpus = 20
        cfg.train_gpus = 8
        cfg.rollout_gpus = 12
        cfg.train_tp_size = 4
        cfg.train_pp_size = 1
        cfg.train_dp_size = 2
        cfg.global_resource_planner.fixed_train_gpus = 8
        cfg.global_resource_planner.allowed_train_tp = [4]
        cfg.global_resource_planner.allowed_rollout_tp = [4]

        with patch.dict(
            "os.environ",
            {
                "GRP_FORCE_TRAIN_GPUS": "12",
                "GRP_FORCE_ROLLOUT_TP_LIST": "4:4",
            },
            clear=False,
        ):
            planner = GlobalResourcePlanner.from_config(cfg)
            decision = planner.plan_if_needed(
                step=0,
                config=cfg,
                batch=[
                    {"input_len": 128, "output_len": 512},
                    {"input_len": 128, "output_len": 512},
                    {"input_len": 128, "output_len": 512},
                    {"input_len": 128, "output_len": 512},
                ],
            )

        self.assertTrue(decision.should_reconfigure)
        self.assertEqual(decision.candidate_plan.train_gpus, 12)
        self.assertEqual(decision.candidate_plan.rollout_tp_list, [4, 4])


if __name__ == "__main__":
    unittest.main()
