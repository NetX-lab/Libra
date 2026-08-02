import unittest

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.global_resource_planner import GlobalResourcePlanner


class TestForcedRuntimePlanner(unittest.TestCase):
    def test_forced_runtime_plan_survives_optimizer_no_candidate(self):
        config = AsyncRLConfig(model_path="/tmp/test-model")
        config.n_total_gpus = 12
        config.train_gpus = 8
        config.rollout_gpus = 4
        config.batch_size = 8
        config.max_seq_length = 32768
        config.global_resource_planner.enabled = True
        config.global_resource_planner.min_history_size = 1
        config.global_resource_planner.warmup_steps = 0
        config.global_resource_planner.plan_interval = 1
        config.global_resource_planner.fixed_train_gpus = 8
        config.global_resource_planner.allowed_train_tp = [4]
        config.global_resource_planner.allowed_train_pp = [1]
        config.global_resource_planner.allowed_rollout_tp = [2, 4]
        config.global_resource_planner.require_heterogeneous_rollout_tp = True

        config.global_resource_planner.runtime_forced_train_gpus = 4
        config.global_resource_planner.runtime_forced_rollout_tp_list = [4, 2, 2]
        planner = GlobalResourcePlanner.from_config(config)
        history = [{"input_len": 2048, "output_len": 30000}]
        decision = planner.plan_if_needed(1, config, batch=history)

        self.assertTrue(decision.should_reconfigure)
        self.assertEqual(decision.reason, "forced_reconfigure")
        self.assertEqual(decision.candidate_plan.rollout_tp_list, [4, 2, 2])


if __name__ == "__main__":
    unittest.main()
