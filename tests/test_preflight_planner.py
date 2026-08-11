import json
import sys
import tempfile
from pathlib import Path

from RL_Framework.config import AsyncRLConfig
from RL_Framework.infra.cost_model.preflight_planner import (
    PreflightPlanner,
    load_history_jsonl,
    synthetic_history,
)


def _config(tmp_path: Path) -> AsyncRLConfig:
    sailor = tmp_path / "sailor"
    vidur = tmp_path / "vidur"
    sailor.mkdir()
    vidur.mkdir()
    sailor_script = tmp_path / "fake_sailor.py"
    vidur_script = tmp_path / "fake_vidur.py"
    sailor_script.write_text(
        """
import json, sys
payload = json.load(open(sys.argv[1]))
gpus = payload["train_config"]["tp"] * payload["train_config"]["pp"] * payload["train_config"]["dp"]
json.dump({"t_train": 100.0 / max(gpus, 1)}, open(sys.argv[2], "w"))
""".strip(),
        encoding="utf-8",
    )
    vidur_script.write_text(
        """
import json, sys
tp_list = [int(x) for x in sys.argv[3].split(",") if x]
json.dump({"makespan": 300.0 / max(sum(tp_list), 1)}, open(sys.argv[2], "w"))
""".strip(),
        encoding="utf-8",
    )
    cfg = AsyncRLConfig(
        model_path="/tmp/fake-model",
        train_gpus=3,
        rollout_gpus=1,
        train_tp_size=1,
        train_pp_size=1,
        train_dp_size=3,
        micro_batch_size=1,
        batch_size=12,
        n_total_gpus=4,
        max_seq_length=4096,
    )
    cfg.global_resource_planner.enabled = True
    cfg.global_resource_planner.train_backend = "sailor"
    cfg.global_resource_planner.rollout_backend = "vidur"
    cfg.global_resource_planner.sailor_path = str(sailor)
    cfg.global_resource_planner.vidur_path = str(vidur)
    cfg.global_resource_planner.sailor_train_command = (
        f"{sys.executable} {sailor_script} {{input_json}} {{output_json}}"
    )
    cfg.global_resource_planner.vidur_rollout_command = (
        f"{sys.executable} {vidur_script} {{trace_csv}} {{output_json}} {{tp_list}}"
    )
    cfg.global_resource_planner.allowed_train_tp = [1, 2]
    cfg.global_resource_planner.allowed_train_pp = [1]
    cfg.global_resource_planner.allowed_rollout_tp = [1, 2, 4]
    cfg.global_resource_planner.micro_batch_sizes = [1]
    cfg.global_resource_planner.min_history_size = 4
    cfg.global_resource_planner.min_gain_ratio = 0.0
    cfg.global_resource_planner.reconfiguration_cost_s = 0.0
    return cfg


def test_preflight_applies_best_candidate_and_writes_config(tmp_path: Path):
    cfg = _config(tmp_path)
    result = PreflightPlanner(cfg).run(
        synthetic_history(num_requests=8, input_len=128, output_len=1024)
    )

    assert result.decision.candidate_plan is not None
    assert result.applied
    assert result.planned_config.rollout_gpus >= cfg.rollout_gpus

    out_yaml = tmp_path / "planned.yaml"
    out_json = tmp_path / "decision.json"
    result.planned_config.to_yaml(str(out_yaml))
    out_json.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    loaded = AsyncRLConfig.from_yaml(str(out_yaml))
    assert loaded.train_gpus + loaded.rollout_gpus == 4
    assert json.loads(out_json.read_text(encoding="utf-8"))["applied"] is True


def test_grp_startup_ignores_legacy_fixed_train_split(tmp_path: Path):
    cfg = _config(tmp_path)
    cfg.global_resource_planner.initial_allocation_strategy = "grp"
    cfg.global_resource_planner.fixed_train_gpus = 3

    result = PreflightPlanner(cfg).run(
        synthetic_history(num_requests=8, input_len=128, output_len=1024)
    )

    assert result.applied
    assert result.planned_config.train_gpus != 3
    assert result.planned_config.train_gpus + result.planned_config.rollout_gpus == 4
    assert result.metadata["initial_allocation_strategy"] == "grp"


def test_preflight_still_plans_when_runtime_replanning_is_disabled(tmp_path: Path):
    cfg = _config(tmp_path)
    cfg.global_resource_planner.initial_allocation_strategy = "grp"
    cfg.global_resource_planner.runtime_online_replanning = False
    preflight = PreflightPlanner(cfg)

    result = preflight.run(
        synthetic_history(num_requests=8, input_len=128, output_len=1024)
    )

    assert result.applied
    assert result.decision.candidate_plan is not None
    assert preflight.planner.online_replanning is False


def test_history_jsonl_loader(tmp_path: Path):
    path = tmp_path / "history.jsonl"
    path.write_text(
        '{"input_len": 1, "output_len": 2}\n\n{"prompt_length": 3, "gen_length": 4}\n',
        encoding="utf-8",
    )

    records = load_history_jsonl(path)

    assert records == [
        {"input_len": 1, "output_len": 2},
        {"prompt_length": 3, "gen_length": 4},
    ]
