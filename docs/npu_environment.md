# Ascend NPU Environment

For the newer Megatron-Core/Bridge baseline, run
`scripts/setup_ascend_megatron_validated_env.sh`. Its direct dependency pins
are recorded in `requirements-ascend-megatron-validated.lock.txt` and
`requirements-vllm-ascend-validated.lock.txt`.

This is the first-pass dependency split for running Libra on Ascend NPUs. Use
it on NPU hosts instead of the CUDA-oriented `requirements.txt`.

The initial target inspected on 2026-07-31 is:

- Ascend 910B3 with 8 devices
- CANN 8.5.2 under `/usr/local/Ascend`
- aarch64 Linux
- Python 3.9.9

## Create the environment

```bash
cd /root/RL_Framework_npu
bash scripts/setup_npu_env.sh
source .venv-npu/bin/activate
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
```

Megatron is optional during the first NPU bring-up because the current code
pins `megatron-core==0.14.0`, while the inspected host's default Python 3.9
only exposes older Megatron wheels from PyPI. Enable it after providing a
compatible Python or wheelhouse:

```bash
INSTALL_MEGATRON_NPU=1 bash scripts/setup_npu_env.sh
```

## vLLM Ascend rollout environment

Use a separate Python 3.11 environment for vLLM. Do not reuse `.venv-npu`; the
training environment stays on the stable PyTorch NPU line, while upstream vLLM
requires a Python 3.11 aarch64 wheel.

Validated rollout stack on the inspected host:

- Python 3.11.13
- `torch==2.7.1`
- `torch-npu==2.7.1`
- `vllm-ascend==0.11.0`
- upstream `vllm==0.11.0` installed with `--no-deps`
- CANN 8.5.2 plus NNAL/ATB 8.5.2

NNAL/ATB is required for real vLLM Ascend serving because it provides
`libatb.so`. Install it with the same Python 3.11 environment active so the
`torch_atb` wheel is installed into the vLLM environment:

```bash
source /root/vllm_ascend_env/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/tmp/Ascend-cann-nnal_8.5.2_linux-aarch64.run \
  --install --quiet --install-for-all --force --whitelist=all --torch_atb
```

Start a single-card vLLM server on a different NPU from training:

```bash
cd /root/RL_Framework_npu
VENV_DIR=/root/vllm_ascend_env \
ASCEND_DEVICES=1 \
MAX_MODEL_LEN=512 \
bash scripts/run_vllm_ascend_server.sh /data/qianzhirong/models/Qwen3-0.6B
```

Then run the real rollout smoke test on NPU 0:

```bash
cd /root/RL_Framework_npu
source .venv-npu/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHONPATH=/root \
ASCEND_RT_VISIBLE_DEVICES=0 \
DEVICE_BACKEND=npu \
WANDB_MODE=disabled \
TOKENIZERS_PARALLELISM=false \
python examples/gsm8k_async_rl.py --config configs/gsm8k_npu_vllm_smoke.yaml
```

The 2026-07-31 smoke run completed rollout collection, logprob recomputation,
and one GRPO optimizer step.

## Dependency differences from the CUDA environment

- `torch`, `torchaudio`, and `torchvision` are pinned to the NPU-compatible
  2.7.1 family.
- `torch-npu==2.7.1.post2` is added for the CANN 8.5.x baseline.
- `vllm-ascend==0.11.0` is isolated in
  `requirements-vllm-ascend-npu.txt` for rollout-path validation.
- CUDA-only packages are removed from the NPU baseline: `xformers`, `pynvml`,
  `nvidia-ml-py`, `grouped_gemm`, and the base install's Megatron extension
  stack.

## Next adaptation points

After dependency installation succeeds, continue replacing CUDA runtime
assumptions in code paths that call `torch.cuda`, use NCCL directly, or launch
upstream vLLM. The remaining candidates include `infra/sync/weight_sync.py`,
`scripts/nccl_allgather_preflight.py`, and the Slurm launchers that export
`CUDA_VISIBLE_DEVICES`.
