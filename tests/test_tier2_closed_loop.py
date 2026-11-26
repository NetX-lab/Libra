"""CPU-safe unit and dry-run tests for Tier 2 (4-GPU closed loop & Google Colab)."""

import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from repro.tier2_four_gpu_closed_loop.launch_4gpu import (
    COLAB_TIER_PRESETS,
    REPO_ROOT,
    build_run_plan,
    materialize_run_plan,
)
from RL_Framework.config import AsyncRLConfig
from RL_Framework.engine.train_engine import _resolve_train_dtype
from RL_Framework.infra.scheduling.cmlfq_cost_scheduler import CMLFQCostScheduler
from RL_Framework.infra.scheduling.factory import SchedulerFactory
from RL_Framework.infra.scheduling.load_balance import LoadBalanceScheduler


SAMPLE_DATASET = (
    REPO_ROOT / "repro/tier2_four_gpu_closed_loop/sample_data/dapo_math_mini.jsonl"
)
CONFIG_4GPU = REPO_ROOT / "repro/tier2_four_gpu_closed_loop/config_4gpu_qwen3_4b.yaml"
CONFIG_COLAB = REPO_ROOT / "repro/tier2_four_gpu_closed_loop/config_colab_1gpu.yaml"
COLAB_NOTEBOOK = (
    REPO_ROOT / "repro/tier2_four_gpu_closed_loop/colab_quickstart.ipynb"
)
LAUNCH_SCRIPT = REPO_ROOT / "repro/tier2_four_gpu_closed_loop/launch_4gpu.py"


def test_dapo_math_mini_dataset():
    lines = SAMPLE_DATASET.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 64

    prompt_ids: set[str] = set()
    box_re = re.compile(r"\\boxed\s*\{(.*?)\}", re.DOTALL)
    has_grader = importlib.util.find_spec("latex2sympy2") is not None
    if has_grader:
        from RL_Framework.env.dapo_math_reward import dapo_math_reward_fn

    for raw in lines:
        row = json.loads(raw)
        assert row["prompt_id"]
        assert row["prompt_id"] not in prompt_ids
        prompt_ids.add(row["prompt_id"])
        assert row["question"].strip()
        assert row["ground_truth"] == row["answer"]
        assert int(row["ground_truth"]) >= 0
        assert row["data_source"] == "dapo_math_mini"
        completion = f"Let us solve carefully.\nFinal answer: \\boxed{{{row['ground_truth']}}}."
        extracted = box_re.findall(completion)
        assert extracted and extracted[-1] == row["ground_truth"]
        if has_grader:
            assert (
                dapo_math_reward_fn(
                    row["question"],
                    completion,
                    ground_truth=row["ground_truth"],
                )
                == 1.0
            )


def test_static_tier2_and_colab_yaml_configs():
    cfg_4gpu = AsyncRLConfig.from_yaml(str(CONFIG_4GPU))
    assert cfg_4gpu.train_gpus == 1
    assert cfg_4gpu.rollout_gpus == 3
    assert cfg_4gpu.rollout_weight_reload_method == "inplace"
    instances_4gpu = cfg_4gpu.heterogeneous_rollout.instances
    assert [(i.tp, i.port) for i in instances_4gpu] == [(1, 18920), (2, 18921)]
    sched_4gpu = SchedulerFactory.create(
        cfg_4gpu.heterogeneous_rollout.scheduling.scheduler_type,
        cfg_4gpu.heterogeneous_rollout,
    )
    assert isinstance(sched_4gpu, CMLFQCostScheduler)

    cfg_colab = AsyncRLConfig.from_yaml(str(CONFIG_COLAB))
    assert cfg_colab.train_gpus == 1
    assert cfg_colab.rollout_gpus == 1
    assert cfg_colab.rollout_weight_reload_method == "inplace"
    assert len(cfg_colab.heterogeneous_rollout.instances) == 1
    assert cfg_colab.heterogeneous_rollout.instances[0].tp == 1
    sched_colab = SchedulerFactory.create(
        cfg_colab.heterogeneous_rollout.scheduling.scheduler_type,
        cfg_colab.heterogeneous_rollout,
    )
    assert isinstance(sched_colab, LoadBalanceScheduler)


def test_launch_4gpu_dry_run_slurm_free(tmp_path: Path):
    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)

    out_dir = tmp_path / "ab_runs"
    proc = subprocess.run(
        [
            sys.executable,
            str(LAUNCH_SCRIPT),
            "--mode",
            "4gpu",
            "--policy",
            "all",
            "--gpus",
            "0,1,2,3",
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Dry-Run Launch Plan" in proc.stdout

    for expected_dir in ("round_robin_42", "least_connections_42", "cmlfq_cost_42"):
        run_dir = out_dir / expected_dir
        assert (run_dir / "config.yaml").is_file()
        assert (run_dir / "launch_plan.json").is_file()
        loaded = AsyncRLConfig.from_yaml(str(run_dir / "config.yaml"))
        assert loaded.train_gpus == 1
        assert loaded.rollout_gpus == 3

        plan = json.loads((run_dir / "launch_plan.json").read_text(encoding="utf-8"))
        assert plan["train_spec"]["cuda_visible_devices"] == "0"
        assert plan["train_spec"]["env_overrides"] == {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
        assert [s["cuda_visible_devices"] for s in plan["rollout_specs"]] == ["1", "2,3"]
        assert [s["port"] for s in plan["rollout_specs"]] == [18920, 18921]

    offload_plan = build_run_plan(
        mode="4gpu",
        policy="cmlfq_cost",
        gpus=[4, 5, 6, 7],
        output_dir=tmp_path / "offload",
        dataset_path=SAMPLE_DATASET,
        kv_backend="cpu_offload",
    )
    materialize_run_plan(offload_plan)
    assert "LIBRA_KV_STORAGE_PATH" in offload_plan["env_overrides"]
    assert offload_plan["train_spec"]["cuda_visible_devices"] == "4"
    assert [s["cuda_visible_devices"] for s in offload_plan["rollout_specs"]] == [
        "5",
        "6,7",
    ]
    assert "--kv-transfer-config" in offload_plan["rollout_specs"][0]["command"]
    assert "LibraCPUOffloadConnector" in " ".join(
        offload_plan["rollout_specs"][0]["command"]
    )


@pytest.mark.parametrize("tier", list(COLAB_TIER_PRESETS.keys()))
def test_launch_colab_1gpu_tiers_dry_run(tmp_path: Path, tier: str):
    plan = build_run_plan(
        mode="colab-1gpu",
        policy="least_connections",
        gpus=[0],
        output_dir=tmp_path / tier,
        dataset_path=SAMPLE_DATASET,
        colab_tier=tier,
    )
    materialize_run_plan(plan)

    loaded = AsyncRLConfig.from_yaml(plan["config_path"])
    assert loaded.train_gpus == 1
    assert loaded.rollout_gpus == 1
    assert len(plan["rollout_specs"]) == 1
    assert plan["rollout_specs"][0]["cuda_visible_devices"] == "0"
    assert plan["train_spec"]["cuda_visible_devices"] == "0"

    if tier == "t4":
        assert plan["vllm_dtype"] == "half"
        assert plan["fsdp_dtype"] == "float32"
        assert plan["env_overrides"]["FSDP_DTYPE"] == "float32"
        assert plan["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert loaded.heterogeneous_rollout.gpu_memory_utilization == 0.30
    else:
        assert plan["vllm_dtype"] == "bfloat16"
        assert plan["fsdp_dtype"] == "bfloat16"

    if tier == "t4":
        proc = subprocess.run(
            [
                sys.executable,
                str(LAUNCH_SCRIPT),
                "--mode",
                "colab-1gpu",
                "--colab-tier",
                "t4",
                "--output-dir",
                str(tmp_path / "cli_t4"),
                "--dry-run",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        assert "colab-1gpu" in proc.stdout


def test_colab_notebook_structure():
    nb = json.loads(COLAB_NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    cell_types = [c["cell_type"] for c in nb["cells"]]
    assert "markdown" in cell_types
    assert "code" in cell_types

    all_text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "colab.research.google.com/github/NetX-lab/Libra/blob/main/repro/tier2_four_gpu_closed_loop/colab_quickstart.ipynb" in all_text
    assert "repro/tier2_four_gpu_closed_loop/launch_4gpu.py" in all_text
    assert "--mode colab-1gpu" in all_text
    assert "repro/tier1_zero_gpu_replay/run_replay.py" in all_text
    assert "latex2sympy2" in all_text


def test_resolve_train_dtype(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FSDP_DTYPE", "float32")
    assert _resolve_train_dtype() is torch.float32

    monkeypatch.setenv("FSDP_DTYPE", "float16")
    assert _resolve_train_dtype() is torch.float16

    monkeypatch.setenv("FSDP_DTYPE", "bf16")
    assert _resolve_train_dtype() is torch.bfloat16

    monkeypatch.setenv("FSDP_DTYPE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _resolve_train_dtype() is torch.float32

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert _resolve_train_dtype() is torch.bfloat16

    monkeypatch.setenv("FSDP_DTYPE", "invalid_precision")
    with pytest.raises(ValueError, match="FSDP_DTYPE"):
        _resolve_train_dtype()


def test_dapo_math_workflow_populates_cmlfq_cost_distribution(
    monkeypatch: pytest.MonkeyPatch,
):
    if importlib.util.find_spec("latex2sympy2") is None:
        fake_l2s = types.ModuleType("latex2sympy2")
        fake_l2s.latex2sympy = lambda expr: expr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "latex2sympy2", fake_l2s)
    if importlib.util.find_spec("word2number") is None:
        fake_w2n_pkg = types.ModuleType("word2number")
        fake_w2n = types.ModuleType("word2number.w2n")
        fake_w2n.word_to_num = lambda text: int(text)  # type: ignore[attr-defined]
        fake_w2n_pkg.w2n = fake_w2n  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "word2number", fake_w2n_pkg)
        monkeypatch.setitem(sys.modules, "word2number.w2n", fake_w2n)

    from RL_Framework.engine.heterogeneous_engine import HeterogeneousRolloutEngine
    from RL_Framework.workflow.dapo_math import DAPOMathWorkflow

    class _CharTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

        def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
            del skip_special_tokens
            return "".join(chr(tok) for tok in tokens)

        def apply_chat_template(self, messages, **kwargs) -> str:
            del kwargs
            return "\n".join(msg["content"] for msg in messages)

    class _MockVLLM:
        def __init__(self, tokenizer: _CharTokenizer):
            self.base_url = "http://127.0.0.1:0"
            self._tokenizer = tokenizer

        async def generate(self, **kwargs):
            prompt = kwargs.get("prompt", "")
            # Fail Turn 1 ("DAPO-Math Validator" absent) -> succeed Turn 2 after feedback.
            if "DAPO-Math Validator" in prompt:
                text = "Revising the algebra step: $2x = 6$, so \\boxed{3}."
            else:
                text = "Initial attempt: $2x + 7 = 13$ gives \\boxed{999}."
            toks = self._tokenizer.encode(text)
            return {
                "text": text,
                "tokens": toks,
                "logprobs": [-0.1] * len(toks),
                "finish_reason": "stop",
            }

    class _RecordingEngine:
        def __init__(self, inner: HeterogeneousRolloutEngine):
            self._inner = inner
            self.route_infos: list[dict] = []

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        async def generate(self, **kwargs):
            out = await self._inner.generate(**kwargs)
            self.route_infos.append(dict(out.get("_schedule_info", {})))
            return out

    cfg = AsyncRLConfig.from_yaml(str(CONFIG_4GPU))
    cfg.heterogeneous_rollout.scheduling.cmlfq_tree_path = ""
    cfg.heterogeneous_rollout.scheduling.cmlfq_kv_backend = "recompute"

    hetero = HeterogeneousRolloutEngine.from_config(cfg)
    tok = _CharTokenizer()
    hetero.engines = [_MockVLLM(tok), _MockVLLM(tok)]
    recorder = _RecordingEngine(hetero)

    workflow = DAPOMathWorkflow(
        tokenizer=tok,
        max_turns=2,
        max_new_tokens=512,
        max_seq_length=2048,
        max_prompt_tokens=256,
    )
    row = {
        "prompt_id": "dapo_mini_00",
        "question": "Solve for $x$: $2x + 7 = 13$.",
        "ground_truth": "3",
        "answer": "3",
    }

    async def _run_two():
        ep1 = await workflow.run_episode(recorder, row, version=0)
        dist_after_ep1 = hetero.scheduler.prefix_tree.residual_distribution(
            "dapo_mini_00",
            [hetero.scheduler._tool_registry.extract(ep1["tool_returns"][0])],
            hetero.scheduler._bucket_thresholds,
        )
        ep2 = await workflow.run_episode(recorder, row, version=0)
        return ep1, dist_after_ep1, ep2

    ep1, dist_after_ep1, ep2 = asyncio.run(_run_two())
    assert ep1["n_turns"] == 2
    assert ep2["n_turns"] == 2
    assert hetero.scheduler.prefix_tree.remaining_unit == "model_generated_tokens"
    assert dist_after_ep1, "Expected non-empty residual distribution after Episode 1"
    # Episode 2 Turn 2 is the 4th generate() call overall (indices 0,1 = ep1; 2,3 = ep2).
    assert len(recorder.route_infos) == 4
    assert recorder.route_infos[3]["reason"] == "distribution_cost"
    assert recorder.route_infos[3]["estimated_decode_seconds"] > 0.0


def test_rollout_benchmark_topology_and_schedulers(tmp_path: Path):
    from repro.tier2_four_gpu_closed_loop import benchmark_rollout_scheduling as bench

    instances = bench.parse_topology("1,1,1,1,4")
    assert [inst["instance_id"] for inst in instances] == [
        "tp1_0", "tp1_1", "tp1_2", "tp1_3", "tp4_0"
    ]
    assert instances[0]["gpus"] == [0]
    assert instances[4]["gpus"] == [4, 5, 6, 7]
    assert [inst["port"] for inst in instances] == [19001, 19002, 19003, 19004, 19005]

    with pytest.raises(ValueError, match="exactly 2 distinct TP tiers"):
        bench.parse_topology("1,2,4")
    with pytest.raises(ValueError, match="requires 8 GPUs"):
        bench.parse_topology("1,1,1,1,4", gpus=[0, 1, 2, 3])

    buckets, thresholds = bench.make_buckets_and_thresholds(instances)
    assert buckets == {
        "short": {"tp_degrees": [1], "max_tokens": 5000},
        "long": {"tp_degrees": [4], "max_tokens": 24576},
    }
    assert thresholds == {"short": 5000, "medium": 12000, "long": 24576}

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({
            "decode_ms": {},
            "migrations": [
                {
                    "source_tp": 1,
                    "target_tp": 4,
                    "source_host": "127.0.0.1",
                    "target_host": "127.0.0.1",
                    "seq_len": 2048,
                    "recompute_ms": 128.9,
                },
                {
                    "source_tp": 4,
                    "target_tp": 1,
                    "source_host": "127.0.0.1",
                    "target_host": "127.0.0.1",
                    "seq_len": 2048,
                    "recompute_ms": 430.4,
                },
            ],
        }),
        encoding="utf-8",
    )
    cal = {
        "1": {"prefill_scale": 0.6722, "decode_scale": 1.4631},
        "4": {"prefill_scale": 0.7828, "decode_scale": 2.6475},
    }
    for policy in bench.POLICIES_FORWARD:
        eng = bench._make_bare_engine(
            policy,
            instances,
            model_path="Qwen/Qwen3-14B",
            hw_cfg_path=bench.DEFAULT_HW_CFG,
            arch_cfg_path=bench.DEFAULT_ARCH_CFG,
            out_dir=tmp_path,
            cost_profile_path=str(profile_path),
            rollout_calibration=cal,
        )
        if policy == "cmlfq_cost":
            assert isinstance(eng.scheduler, CMLFQCostScheduler)
            assert eng.scheduler.routing_costs.decode_ms == {}
            assert eng.scheduler.routing_costs.rollout_model.calibration == cal
            assert len(eng.scheduler.routing_costs.migrations) == 2


def test_rollout_benchmark_calibration_fit_and_interp():
    from repro.tier2_four_gpu_closed_loop import benchmark_rollout_scheduling as bench

    prefill_specs = [2048, 4096, 8192, 16384]
    decode_specs = [("ctx2048_g1024", 2048, 1024), ("ctx6144_g1024", 6144, 1024)]
    prefill_ms = {
        "1": {2048: 430.4, 4096: 864.2, 8192: 1790.3, 16384: 3970.8},
        "4": {2048: 128.9, 4096: 251.5, 8192: 513.9, 16384: 1128.6},
    }
    raw_decode_ms = {
        "1": {
            "ctx2048_g1024": {"b1_total_ms": 10809.0, "b8_wall_ms": 15940.0},
            "ctx6144_g1024": {"b1_total_ms": 12069.0, "b8_wall_ms": 26381.0},
        },
        "4": {
            "ctx2048_g1024": {"b1_total_ms": 5950.0, "b8_wall_ms": 7485.0},
            "ctx6144_g1024": {"b1_total_ms": 6283.0, "b8_wall_ms": 11185.0},
        },
    }

    assert bench._interp_pf(prefill_ms["1"], 2048) == pytest.approx(430.4)
    assert bench._interp_pf(prefill_ms["1"], 6144) == pytest.approx(0.5 * (864.2 + 1790.3))

    fitted = bench.fit_rollout_calibration(
        prefill_ms,
        raw_decode_ms,
        prefill_specs,
        decode_specs,
        bench.DEFAULT_HW_CFG,
        bench.DEFAULT_ARCH_CFG,
        "Qwen/Qwen3-14B",
    )
    assert fitted["1"]["prefill_scale"] == pytest.approx(0.6722, abs=1e-3)
    assert fitted["1"]["decode_scale"] == pytest.approx(1.4631, abs=1e-3)
    assert fitted["4"]["prefill_scale"] == pytest.approx(0.7828, abs=1e-3)
    assert fitted["4"]["decode_scale"] == pytest.approx(2.6475, abs=1e-3)


def test_rollout_benchmark_workload_and_session_isolation():
    from collections import Counter
    from repro.tier2_four_gpu_closed_loop import benchmark_rollout_scheduling as bench

    class _WordTokenizer:
        def __init__(self):
            self._tok2id: dict[str, int] = {}
            self._id2tok: list[str] = []

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            ids = []
            for tok in text.split():
                if tok not in self._tok2id:
                    self._tok2id[tok] = len(self._id2tok)
                    self._id2tok.append(tok)
                ids.append(self._tok2id[tok])
            return ids

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            del skip_special_tokens
            return " ".join(self._id2tok[i] for i in ids)

        def apply_chat_template(
            self, messages, tokenize: bool = False, add_generation_prompt: bool = True
        ) -> str:
            del tokenize, add_generation_prompt
            return " ".join(msg["content"] for msg in messages)

    tok = _WordTokenizer()
    warmup = bench.build_workload(tok, "W01", is_warmup=True)
    eval_a = bench.build_workload(tok, "E01", is_warmup=False)
    eval_b = bench.build_workload(tok, "E02", is_warmup=False)

    assert len(warmup) == 13
    assert Counter(it["kind"] for it in warmup) == {
        "short": 6,
        "quick_fix": 4,
        "deep_fail": 2,
        "deep_timeout": 1,
    }
    assert len(eval_a) == 60
    assert Counter(it["kind"] for it in eval_a) == {
        "short": 48,
        "quick_fix": 10,
        "deep_fail": 1,
        "deep_timeout": 1,
    }

    expected_turns = {"short": 1, "quick_fix": 2, "deep_fail": 11, "deep_timeout": 11}
    for it_a, it_b in zip(eval_a, eval_b):
        assert it_a["prompt_id"] == it_b["prompt_id"]
        assert it_a["input_tokens0"] == 1024
        assert it_b["input_tokens0"] == 1024
        assert it_a["prompt0"].startswith("[session=E01] ")
        assert it_b["prompt0"].startswith("[session=E02] ")
        assert len(it_a["turns"]) == expected_turns[it_a["kind"]]


def test_rollout_benchmark_dry_run_and_shell_syntax():
    bench_py = (
        REPO_ROOT / "repro/tier2_four_gpu_closed_loop/benchmark_rollout_scheduling.py"
    )
    bench_sh = REPO_ROOT / "repro/tier2_four_gpu_closed_loop/run_rollout_ab.sh"

    proc_py = subprocess.run(
        [sys.executable, str(bench_py), "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(proc_py.stdout)
    assert parsed["dry_run"] is True
    assert list(parsed["policies"].keys()) == [
        "round_robin",
        "least_connections",
        "length_aware",
        "la_mlfq",
        "cmlfq",
        "cmlfq_cost",
    ]

    subprocess.run(["bash", "-n", str(bench_sh)], check=True)
    proc_sh = subprocess.run(
        ["bash", str(bench_sh), "--dry-run", "--python", sys.executable],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[tp1_0] CUDA_VISIBLE_DEVICES=0" in proc_sh.stdout
    assert "[tp4_0] CUDA_VISIBLE_DEVICES=4,5,6,7" in proc_sh.stdout
    assert "--enable-prefix-caching" in proc_sh.stdout
