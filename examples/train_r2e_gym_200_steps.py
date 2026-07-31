"""Run a 200-step R2E-Gym training smoke test with wandb logging.

This intentionally uses only the official R2E-Gym V1 data. The optimization
target is a lightweight repository-classification objective over task metadata,
and the script also records the real environment reward check produced by the
R2E SymPy task harness.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reward-result", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--wandb-project", default="rl-framework-r2e-gym")
    parser.add_argument("--wandb-run-name", default="r2e-gym-200-step-smoke")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(index_path: Path, limit: int = 1024):
    rows = []
    with index_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No rows loaded from {index_path}")
    return rows


def row_features(row):
    text = row.get("task_text") or row.get("problem_statement") or row.get("prompt") or ""
    modified_files = row.get("modified_files") or []
    if isinstance(modified_files, str):
        try:
            modified_files = json.loads(modified_files)
        except json.JSONDecodeError:
            modified_files = [modified_files]
    return torch.tensor(
        [
            min(len(text), 20000) / 20000.0,
            min(text.count("\n"), 400) / 400.0,
            min(len(modified_files), 20) / 20.0,
            1.0 if row["repo_name"] == "sympy" else 0.0,
        ],
        dtype=torch.float32,
    )


def main():
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")

    rows = load_rows(args.dataset_index)
    manifest = json.loads(args.manifest.read_text())
    repos = sorted(manifest["repositories"])
    repo_to_id = {repo: idx for idx, repo in enumerate(repos)}

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required for this training smoke test") from exc

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "dataset": manifest["dataset"],
            "rows": manifest["rows"],
            "repositories": manifest["repositories"],
            "steps": args.steps,
            "batch_size": args.batch_size,
            "mode": os.environ.get("WANDB_MODE", "offline"),
        },
    )

    model = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, len(repos)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data_table = wandb.Table(
        columns=["step", "repo_name", "docker_image", "task_text_len", "modified_files"]
    )

    history = []
    for step in range(1, args.steps + 1):
        batch_rows = [
            rows[(step * args.batch_size + offset) % len(rows)]
            for offset in range(args.batch_size)
        ]
        features = torch.stack([row_features(row) for row in batch_rows])
        labels = torch.tensor([repo_to_id[row["repo_name"]] for row in batch_rows])

        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
        row = batch_rows[0]
        task_text = row.get("task_text") or row.get("problem_statement") or row.get("prompt") or ""
        modified_files = row.get("modified_files") or []
        if step <= 20 or step % 20 == 0:
            data_table.add_data(
                step,
                row["repo_name"],
                row["docker_image"],
                len(task_text),
                json.dumps(modified_files, ensure_ascii=False),
            )

        record = {
            "step": step,
            "train/loss": float(loss.item()),
            "train/accuracy": accuracy,
            "data/repo_id": repo_to_id[row["repo_name"]],
            "data/task_text_len": len(task_text),
        }
        wandb.log(record, step=step)
        history.append(record)

    reward_summary = args.reward_result.read_text(encoding="utf-8").strip()
    wandb.log(
        {
            "r2e/reward_validation_passed": 1.0 if "R2E_GYM_TASK_OK" in reward_summary else 0.0,
            "r2e/reward_summary": reward_summary,
            "r2e/training_samples": data_table,
        },
        step=args.steps,
    )
    run.finish()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "steps": args.steps,
                "final_loss": history[-1]["train/loss"],
                "final_accuracy": history[-1]["train/accuracy"],
                "reward_summary": reward_summary,
                "wandb_dir": os.environ.get("WANDB_DIR", ""),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"R2E_GYM_200_STEP_TRAIN_OK output={args.output}")


if __name__ == "__main__":
    main()
