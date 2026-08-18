# Ascend NPU Environment

This is the first-pass dependency split for running Libra on Ascend NPUs. Use
it on NPU hosts instead of the CUDA-oriented `requirements.txt`.

Configure the following values for your own target environment:

- an Ascend NPU model and device count supported by your deployment
- a CANN toolkit path exported as `ASCEND_SET_ENV`
- a Linux host and Python version supported by the selected packages

## Create the environment

```bash
cd ./RL_Framework_npu
bash scripts/setup_npu_env.sh
source .venv-npu/bin/activate
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
```

Megatron is optional during the first NPU bring-up because the current code
pins `megatron-core==0.14.0`. Enable it after providing a compatible Python
runtime or wheelhouse:

```bash
INSTALL_MEGATRON_NPU=1 bash scripts/setup_npu_env.sh
```

## vLLM Ascend rollout environment

The original restart-based rollout path remains available as a rollback
profile and does not provide direct HCCL weight transfer. Select mutually
compatible Python, PyTorch, torch-npu, vLLM Ascend, CANN, and NNAL/ATB
versions for your deployment.

NNAL/ATB is required for real vLLM Ascend serving because it provides
`libatb.so`. Install it with the selected vLLM environment active so the
`torch_atb` wheel is installed into that environment:

```bash
source ./venv/npu/bin/activate
source ${ASCEND_SET_ENV:?set ASCEND_SET_ENV}
./installer/Ascend-cann-nnal.run \
  --install --quiet --install-for-all --force --whitelist=all --torch_atb
```

Start a single-card vLLM server on a different NPU from training:

```bash
cd ./RL_Framework_npu
VENV_DIR=./venv/npu \
ASCEND_DEVICES=1 \
MAX_MODEL_LEN=512 \
bash scripts/run_vllm_ascend_server.sh ./models/Qwen3-0.6B
```

Then run the real rollout smoke test on NPU 0:

```bash
cd ./RL_Framework_npu
source .venv-npu/bin/activate
source ${ASCEND_SET_ENV:?set ASCEND_SET_ENV}
PYTHONPATH="$(dirname "$PWD")" \
ASCEND_RT_VISIBLE_DEVICES=0 \
DEVICE_BACKEND=npu \
WANDB_MODE=disabled \
TOKENIZERS_PARALLELISM=false \
python examples/gsm8k_async_rl.py --config configs/gsm8k_npu_vllm_smoke.yaml
```

Verify rollout collection, logprob recomputation, and an optimizer step in
your environment before starting a larger run.

## Official HCCL in-place weight refresh

Use a fresh environment with mutually compatible CANN, Python, PyTorch,
torch-npu, vLLM, and vLLM Ascend versions.

Create the upgraded environment with:

```bash
NPU_STACK_PROFILE=hccl \
INSTALL_MEGATRON_NPU=1 \
bash scripts/setup_npu_env.sh
```

The trainer process imports the official vLLM Ascend sender, so the training
and rollout environments must expose the same HCCL weight-transfer API and a
compatible CANN runtime. Set `weight_sync_mode` and
`rollout_weight_sync_mode` to `hccl`, and select
`rollout_weight_reload_method: inplace`. The vLLM server must run with
`VLLM_SERVER_DEV_MODE=1`, `VLLM_ASCEND_ENABLE_NZ=0`, and a weight-transfer
backend of `nccl`; the Ascend plugin maps that upstream backend name to its
HCCL implementation.

The communicator is persistent. If GRP changes the physical rollout worker or
TP topology, restart the HCCL-enabled rollout group before the next transfer.

## Dependency differences from the CUDA environment

- `torch`, `torchaudio`, `torchvision`, and `torch-npu` must be selected as a
  mutually compatible NPU stack.
- The vLLM Ascend dependencies are isolated in
  `requirements-vllm-ascend-npu.txt` for rollout-path validation.
- CUDA-only packages are removed from the NPU baseline: `xformers`, `pynvml`,
  `nvidia-ml-py`, `grouped_gemm`, and the base install's Megatron extension
  stack.

## Next adaptation points

The legacy `NCCLWeightSync` class remains for the CUDA/FSDP path. NPU direct
refresh uses `infra/sync/hccl_weight_transfer.py` and must not call that class.
The NPU branch keeps only SSH/`torchrun` launchers; GPU-only launchers are not
part of this branch.
