"""Pin vLLM Ray's DP rank-zero placement group to the API-server node.

vLLM 0.25.1 RayExecutorV2 does not reserve an accelerator on its driver by
default. With cross-node data parallelism, DP rank zero can then be placed on
a different host from ``--data-parallel-address``, preventing its HCCL listener
from binding. This narrow runtime patch changes placement only for DP rank 0.
"""

from pathlib import Path


TARGET = Path(
    "/usr/local/python3.12.13/lib/python3.12/site-packages/"
    "vllm/v1/executor/ray_executor_v2.py"
)
ORIGINAL = "initialize_ray_cluster(self.parallel_config, require_gpu_on_driver=False)"
PATCHED = (
    "initialize_ray_cluster(\n"
    "            self.parallel_config,\n"
    "            require_gpu_on_driver=(self.parallel_config.data_parallel_rank == 0),\n"
    "        )"
)


def main() -> None:
    source = TARGET.read_text()
    if PATCHED in source:
        print("VLLM_RAY_DP_ANCHOR_ALREADY_PATCHED")
        return
    if source.count(ORIGINAL) != 1:
        raise RuntimeError(
            f"expected one vLLM Ray placement call, found {source.count(ORIGINAL)}"
        )
    TARGET.write_text(source.replace(ORIGINAL, PATCHED))
    print("VLLM_RAY_DP_ANCHOR_PATCHED")


if __name__ == "__main__":
    main()
