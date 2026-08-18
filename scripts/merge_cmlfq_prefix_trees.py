#!/usr/bin/env python3
"""Merge per-rank C-MLFQ tree snapshots into one canonical file."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from RL_Framework.infra.scheduling.cmlfq_prefix_tree import CausalPrefixTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    suffix = output.suffix or ".json"
    stem = output.name[:-len(output.suffix)] if output.suffix else output.name
    rank_paths = sorted(output.parent.glob(f"{stem}.rank_*{suffix}"))
    rank_paths = [path for path in rank_paths if ".step_" not in path.name]
    if not rank_paths:
        print(f"No C-MLFQ rank trees found for {output}")
        return

    merged = CausalPrefixTree()
    loaded_paths: list[str] = []
    for path in rank_paths:
        tree = CausalPrefixTree()
        try:
            tree.load(str(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable C-MLFQ tree {path}: {exc}")
            continue
        merged.merge(tree)
        loaded_paths.append(str(path))

    if not loaded_paths:
        raise RuntimeError("All C-MLFQ rank tree snapshots were unreadable")
    merged.save(str(output))
    manifest_path = output.with_name(f"{stem}.manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": time.time(),
                "canonical_tree": str(output),
                "rank_trees": loaded_paths,
                "stats": merged.get_stats(),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"Merged {len(loaded_paths)} C-MLFQ rank trees into {output}: "
        f"{merged.get_stats()}"
    )


if __name__ == "__main__":
    main()
