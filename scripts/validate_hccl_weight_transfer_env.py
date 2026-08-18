#!/usr/bin/env python3
"""Validate the official vLLM Ascend HCCL weight-transfer runtime."""

from __future__ import annotations

import importlib.metadata
import os


def main() -> int:
    from RL_Framework.infra.sync.hccl_weight_transfer import (
        OfficialHCCLWeightTransfer,
    )

    OfficialHCCLWeightTransfer.validate_runtime()
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerSendWeightsArgs,
        HCCLWeightTransferEngine,
    )

    del HCCLTrainerSendWeightsArgs, HCCLWeightTransferEngine
    versions = {
        package: importlib.metadata.version(package)
        for package in ("torch", "torch-npu", "vllm", "vllm-ascend")
    }
    print("official_hccl_weight_transfer=available")
    for package, version in versions.items():
        print(f"{package}={version}")
    print(f"ascend_home={os.environ.get('ASCEND_HOME_PATH', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
