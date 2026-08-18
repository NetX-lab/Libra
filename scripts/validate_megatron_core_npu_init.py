#!/usr/bin/env python3
"""Load a Qwen checkpoint through Megatron-Core on the active NPU topology."""

from __future__ import annotations

import os

from RL_Framework.engine.megatron_core_train_engine import MegatronCoreTrainEngine


def main() -> None:
    initialize_optimizer = os.environ.get("INITIALIZE_OPTIMIZER", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    engine = MegatronCoreTrainEngine(
        model_path=os.environ["MODEL_PATH"],
        train_tp_size=int(os.environ.get("TRAIN_TP_SIZE", "8")),
        train_pp_size=int(os.environ.get("TRAIN_PP_SIZE", "1")),
        train_cp_size=1,
        train_ep_size=1,
        expert_tensor_parallel_size=1,
        use_distributed_optimizer=True,
        use_transformer_engine=False,
        recompute_num_layers=1,
        streaming_export=False,
        sync_path=os.environ.get("SYNC_PATH", "/tmp/mcore_init_preflight"),
    )
    engine.initialize(
        max_seq_length=int(os.environ.get("MAX_SEQ_LENGTH", "512")),
        initialize_optimizer=initialize_optimizer,
    )
    print(
        "MCORE_NPU_MODEL_INIT_OK "
        f"rank={engine.rank}/{engine.world_size} "
        f"optimizer={initialize_optimizer} "
        f"topology={engine.get_parallel_state()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
