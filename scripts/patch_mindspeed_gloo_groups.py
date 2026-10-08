#!/usr/bin/env python3
"""Keep only the Gloo groups needed by Megatron CPU optimizer offload."""
from pathlib import Path
import os


def main():
    path = Path(
        os.environ.get(
            "MEGATRON_PARALLEL_STATE_PATH",
            "/workspace-verl/verl/megatron/core/parallel_state.py",
        )
    )
    source = path.read_text()
    expert_marker = "if create_gloo_process_groups or create_gloo_optimizer_process_groups:\n            group_gloo = create_group("
    if expert_marker in source:
        print(f"MINDSPEED_GLOO_OFFLOAD_GROUPS_PATCHED {path}")
        return

    replacements = (
        (
            "    create_gloo_process_groups: bool = True,\n",
            "    create_gloo_process_groups: bool = True,\n"
            "    create_gloo_optimizer_process_groups: bool = False,\n",
            1,
        ),
        (
            "        if create_gloo_process_groups:\n"
            "            group_with_cp_gloo = create_group(\n",
            "        if create_gloo_process_groups or create_gloo_optimizer_process_groups:\n"
            "            group_with_cp_gloo = create_group(\n",
            1,
        ),
        (
            "                if create_gloo_process_groups:\n"
            "                    intra_partial_data_parallel_group_with_cp_gloo = create_group(\n",
            "                if create_gloo_process_groups or create_gloo_optimizer_process_groups:\n"
            "                    intra_partial_data_parallel_group_with_cp_gloo = create_group(\n",
            1,
        ),
        (
            "        if create_gloo_process_groups:\n"
            "            group_gloo = create_group(\n"
            "                ranks, backend=\"gloo\", group_desc='EXPERT_DATA_PARALLEL_GROUP_GLOO'\n",
            "        if create_gloo_process_groups or create_gloo_optimizer_process_groups:\n"
            "            group_gloo = create_group(\n"
            "                ranks, backend=\"gloo\", group_desc='EXPERT_DATA_PARALLEL_GROUP_GLOO'\n",
            1,
        ),
    )
    for old, new, expected_count in replacements:
        actual_count = source.count(old)
        if actual_count != expected_count:
            raise RuntimeError(
                f"Unexpected Megatron parallel_state.py layout at {path}: "
                f"expected {expected_count} instance(s), found {actual_count}"
            )
        source = source.replace(old, new, expected_count)
    path.write_text(source)
    print(f"MINDSPEED_GLOO_OFFLOAD_GROUPS_PATCHED {path}")


if __name__ == "__main__":
    main()
