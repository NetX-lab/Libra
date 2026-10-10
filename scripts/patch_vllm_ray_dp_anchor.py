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
RAY_UTILS = TARGET.with_name("ray_utils.py")
ORIGINAL = "initialize_ray_cluster(self.parallel_config, require_gpu_on_driver=False)"
PATCHED = (
    "initialize_ray_cluster(\n"
    "            self.parallel_config,\n"
    "            require_gpu_on_driver=(self.parallel_config.data_parallel_rank == 0),\n"
    "        )"
)


def main() -> None:
    source = TARGET.read_text()
    if PATCHED not in source and source.count(ORIGINAL) != 1:
        raise RuntimeError(
            f"expected one vLLM Ray placement call, found {source.count(ORIGINAL)}"
        )
    if PATCHED not in source:
        TARGET.write_text(source.replace(ORIGINAL, PATCHED))

    # DP engines start concurrently. Reserve a distinct node for each rank so
    # another placement group cannot consume the API node before DP0 starts.
    utils = RAY_UTILS.read_text()
    marker = "        current_node_resource = available_resources_per_node()[current_node_id]\n"
    anchor = (
        marker
        + "        dp_node_ips = [ip for ip in os.environ.get('LIBRA_RAY_DP_NODE_IPS', '').split(',') if ip]\n"
        + "        if dp_node_ips:\n"
        + "            desired_ip = dp_node_ips[parallel_config.data_parallel_rank]\n"
        + "            placement_group_specs[0][f'node:{desired_ip}'] = 0.001\n"
    )
    if anchor not in utils:
        if utils.count(marker) != 1:
            raise RuntimeError("cannot locate Ray placement anchor")
        utils = utils.replace(marker, anchor)
    original_check = "        if require_gpu_on_driver:\n            if current_node_resource.get(device_str, 0) < 1:"
    patched_check = "        if require_gpu_on_driver and not dp_node_ips:\n            if current_node_resource.get(device_str, 0) < 1:"
    if patched_check not in utils:
        if utils.count(original_check) != 1:
            raise RuntimeError("cannot locate Ray driver resource check")
        utils = utils.replace(original_check, patched_check)
    RAY_UTILS.write_text(utils)
    print("VLLM_RAY_DP_ANCHOR_PATCHED")


if __name__ == "__main__":
    main()
