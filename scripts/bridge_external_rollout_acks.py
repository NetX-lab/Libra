#!/usr/bin/env python3
"""Bridge physical vLLM reload acknowledgements to legacy logical IDs.

This is a bounded compatibility helper for a training process that was started
before external rollout worker IDs were pinned in ``AsyncRLTrainer``.  It never
starts, stops, or signals a rollout process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--source-ids", nargs="+", required=True)
    parser.add_argument("--target-ids", nargs="+", required=True)
    parser.add_argument("--until-version", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_dir = args.control_dir.resolve()
    request_path = control_dir / "reload_request.json"
    last_bridged = -1
    while True:
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            version = int(request["version"])
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            time.sleep(args.poll_seconds)
            continue

        source_acks = [
            control_dir / f"ack_{source_id}_{version}.json"
            for source_id in args.source_ids
        ]
        source_errors = [
            control_dir / f"error_{source_id}_{version}.json"
            for source_id in args.source_ids
        ]
        if any(path.exists() for path in source_errors):
            error_text = next(
                path.read_text(encoding="utf-8")
                for path in source_errors
                if path.exists()
            )
            for target_id in args.target_ids:
                error_path = control_dir / f"error_{target_id}_{version}.json"
                if not error_path.exists():
                    _atomic_json(
                        error_path,
                        {
                            "instance_id": target_id,
                            "version": version,
                            "error": f"physical rollout reload failed: {error_text}",
                            "bridged_at": time.time(),
                        },
                    )
        elif all(path.exists() for path in source_acks):
            for target_id in args.target_ids:
                ack_path = control_dir / f"ack_{target_id}_{version}.json"
                if not ack_path.exists():
                    _atomic_json(
                        ack_path,
                        {
                            "instance_id": target_id,
                            "version": version,
                            "physical_instance_ids": list(args.source_ids),
                            "bridged_at": time.time(),
                        },
                    )
            if version != last_bridged:
                print(f"bridged rollout reload version={version}", flush=True)
                last_bridged = version
            if version >= args.until_version:
                return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
