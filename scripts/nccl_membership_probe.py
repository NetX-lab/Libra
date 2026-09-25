"""Exercise the production weight transport with one sender and eight receivers."""
import argparse
import json
import os
import socket
import torch
from RL_Framework.infra.sync.nccl_weight_sync import NcclReloadSpec, send_weights, receive_weights

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True)
parser.add_argument("--port", type=int, default=29731)
args = parser.parse_args()
rank = int(os.environ["SLURM_PROCID"])
size = int(os.environ["SLURM_NTASKS"])
torch.cuda.set_device(0)
spec = NcclReloadSpec(args.host, args.port, size, rank, "cuda:0", timeout_s=120, chunk_bytes=4096)
expected = torch.arange(32768, device="cuda:0", dtype=torch.float32)
# Reuse the communicator for two generations, as in real weight refreshes.
for version in range(2):
    if rank == 0:
        send_weights([("probe", expected + version)], spec)
    else:
        received = list(receive_weights(spec))
        assert len(received) == 1 and received[0][0] == "probe"
        torch.testing.assert_close(received[0][1], expected + version, rtol=0, atol=0)
    torch.cuda.synchronize()
    print(json.dumps(dict(rank=rank, world_size=size, host=socket.gethostname(), version=version, verified=True)), flush=True)
# Keep the sender TCPStore alive until every receiver consumed the end marker.
from RL_Framework.infra.sync.nccl_weight_sync import _communicator
_communicator(spec).group.barrier()
# This standalone probe owns no persistent service. Avoid interpreter teardown
# racing vLLM's cached TCPStore/NCCL background threads after verification.
import sys
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
