"""R2E-Gym driven smoke test for C-MLFQ prefix-tree routing."""

import json
from pathlib import Path

from RL_Framework.infra.scheduling.cmlfq_offline_profile import CMLFQOfflineProfiler
from RL_Framework.infra.scheduling.cmlfq_scheduler import CMLFQScheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "data" / "r2e_gym_v1" / "index.jsonl"


def load_first_row() -> dict:
    with INDEX_PATH.open(encoding="utf-8") as source:
        for line in source:
            return json.loads(line)
    raise RuntimeError(f"No rows found in {INDEX_PATH}")


def build_r2e_tool_history(row: dict) -> tuple[str, list[dict]]:
    prompt_id = f"{row['repo_name']}:{row['commit_hash']}"
    task_text = row.get("task_text") or row.get("problem_statement") or row.get("prompt") or ""
    modified_files = row.get("modified_files") or []
    if isinstance(modified_files, str):
        modified_files = json.loads(modified_files)

    first_payload = (
        "R2E before-patch pytest failed. "
        f"repo={row['repo_name']} files={','.join(modified_files[:4])} "
        + task_text[:8000]
    )
    second_payload = "R2E after-patch pytest passed expected tests."
    total_output_tokens = 36000
    raw_trajectory = {
        "prompt_id": prompt_id,
        "total_output_tokens": total_output_tokens,
        "tool_returns": [
            {
                "tool_type": "code_executor",
                "output": first_payload,
                "status": "failure",
                "payload_tokens": max(6000, len(first_payload) // 3),
                "token_position": 1200,
                "remaining_length": total_output_tokens - 1200,
            },
            {
                "tool_type": "code_executor",
                "output": second_payload,
                "status": "success",
                "payload_tokens": len(second_payload) // 3,
                "token_position": 34000,
                "remaining_length": 2000,
            },
        ],
    }
    return prompt_id, [raw_trajectory]


def main():
    row = load_first_row()
    prompt_id, raw_history = build_r2e_tool_history(row)

    profiler = CMLFQOfflineProfiler()
    profile = profiler.profile_from_raw_trajectories(raw_history)

    scheduler = CMLFQScheduler(
        buckets={
            "short": {"tp_degrees": [1], "max_tokens": 5000},
            "medium": {"tp_degrees": [2], "max_tokens": 15000},
            "long": {"tp_degrees": [4], "max_tokens": 50000},
        },
        prefix_tree=profiler.prefix_tree,
    )
    scheduler.register_instance(0, "short_tp1", 1)
    scheduler.register_instance(1, "medium_tp2", 2)
    scheduler.register_instance(2, "long_tp4", 4)

    initial = scheduler.schedule(input_tokens=1024, prompt_id=prompt_id)
    assert initial.category == "short", initial

    first_tool_return = raw_history[0]["tool_returns"][0]
    decision = scheduler.on_tool_return(
        initial.request_id,
        first_tool_return,
        generated_tokens=first_tool_return["token_position"],
    )
    assert decision.should_migrate, decision
    assert decision.target_bucket == "long", decision

    migrated = scheduler.execute_migration(initial.request_id, decision)
    assert migrated.category == "long", migrated
    assert migrated.request_id == initial.request_id

    scheduler.finish_request(
        initial.request_id,
        raw_history[0]["total_output_tokens"],
    )
    stats = scheduler.prefix_tree.get_stats()
    assert stats["total_insertions"] >= 2, stats

    print(
        "R2E_CMLFQ_FLOW_OK "
        f"repo={row['repo_name']} "
        f"nodes={profile['tree_stats']['total_nodes']} "
        f"route={initial.category}->{migrated.category}"
    )


if __name__ == "__main__":
    main()
