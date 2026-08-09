# Validated Ascend Megatron-Core environment

This file freezes the first real hardware baseline that completed Qwen3 weight
conversion and a Megatron-Core forward pass on an Ascend NPU.  It is separate
from later 14B/48-card results so an experimental upgrade cannot silently change
the known-good base.

## Validation record

- Date: 2026-08-05 (Asia/Hong_Kong)
- Host used for the preflight: `npu-node-a`, aarch64
- OS: Huawei Cloud EulerOS 2.0
- Kernel: `5.10.0-136.12.0.86.r1526_92.hce2.aarch64`
- Accelerator: Ascend 910B3, 64 GiB HBM
- CANN toolkit: `8.5.2`, inner version `V100R001C25SPC005B220`
- `npu-smi`: `25.5.1`
- Python: `3.10.20`
- Environment: `/opt/libra/envs/rl_mindspeed`
- Base site packages: `/opt/libra/envs/rl_framework_py310`

The exact direct package pins are in
`requirements-ascend-megatron-validated.lock.txt`.  The important compatibility
pair is the PyTorch `2.7.1` distribution (runtime version string `2.7.1+cpu`)
plus torch-npu `2.7.1.post2`; Megatron-Core is `0.14.0` and Megatron Bridge is
`0.2.0rc6`.

The live environment is also captured as a secret-free snapshot. It includes
the full transitive `pip freeze` for both training and vLLM rollout Python,
package metadata, OS/kernel/CANN/NPU fingerprints, a source hash manifest, and
the small whitelist of runtime variables required by this stack. It explicitly
does not dump the process environment, SSH configuration, passwords, or tokens.

```bash
cd /opt/libra/RL_Framework_NPU
SNAPSHOT_NAME=ascend_megatron_validated \
  bash scripts/capture_ascend_megatron_environment.sh
```

The canonical validated snapshot is stored on shared storage at
`/opt/libra/environment_snapshots/ascend_megatron_validated`, so
all six selected nodes can read it. Verify its contents from that directory
with `sha256sum -c MANIFEST.sha256`.

The rollout environment is intentionally separate: Python `3.11.13`, PyTorch
`2.7.1` (`2.7.1+cpu` at runtime), torch-npu `2.7.1`, vLLM `0.11.0`, and
vLLM-Ascend `0.11.0`. Its complete 152-package freeze is stored as
`rollout.pip-freeze.txt`; the training environment's 130-package freeze is
`training.pip-freeze.txt`. Direct rollout pins are also kept in
`requirements-vllm-ascend-validated.lock.txt`.

The rollout virtual environment currently lives at `/opt/libra/envs/vllm_ascend`.
It was created by `uv`, so copying only that directory is insufficient: its
`bin/python3.11` is an absolute symlink into the uv-managed CPython runtime.
Clone all three runtime layers to another homogeneous host:

```bash
rsync -a /opt/libra/envs/vllm_ascend/ HOST:/opt/libra/envs/vllm_ascend/
rsync -a /opt/libra/uv/python/cpython-3.11.13-linux-aarch64-gnu/ \
  HOST:/opt/libra/uv/python/cpython-3.11.13-linux-aarch64-gnu/
rsync -a /usr/local/Ascend/nnal/ HOST:/usr/local/Ascend/nnal/
```

The NNAL copy must remain on the same CANN `8.5.2` line. The serving launcher
activates the vLLM environment, then sources the CANN and NNAL/ATB `set_env.sh`
files before starting vLLM. A copied host is accepted only after all configured
`/health` endpoints respond and a real `/v1/completions` request succeeds.

## Ascend integration used

The validated path uses torch-npu and the HCCL distributed backend. A
project-local, process-scoped compatibility layer rewrites MCore 0.14's embedded
CUDA device/type literals after import; it does not modify installed packages.
The compatibility code additionally:

1. maps MCore's direct CUDA RNG helper access to `torch.npu`;
2. uses the local PyTorch SDPA attention layer instead of Transformer Engine;
3. disables CUDA-only grouped GEMM, fused softmax, fused RoPE, and bias/dropout
   JIT fusion;
4. allocates engine tensors explicitly on `npu:<local_rank>`.

These source files are part of the environment contract:

- `engine/megatron_core_train_engine.py`
  SHA256 `c2ad2c6d9756808e80ea96d5762f80e70419ac2328c70236bd7c15f71861cd60`
- `engine/megatron_core_attention.py`
  SHA256 `aa96958b470bccfc824d182d606d7f74c009089c277015c8344f08afce6b0f8c`
- `engine/megatron_npu_compat.py`
  SHA256 `b613cf598a2e1d14493dabe9de666efab2804d2083eea2c70621e1013c63934c`
- `examples/test_megatron_core_npu_runtime.py`
  SHA256 `93f8a79e0424c1627aa31a8ea10ab384d48973fcc5c14c98f2a53b0c4389d67b`

## Recreate and activate

The setup script intentionally overlays an existing Ascend PyTorch environment
with `--system-site-packages`.  This preserves Huawei's aarch64 torch-npu wheel
instead of allowing a generic package resolver to replace it.

```bash
cd /opt/libra/RL_Framework_NPU
BASE_PYTHON=/opt/libra/envs/rl_framework_py310/bin/python \
VENV_DIR=/opt/libra/envs/rl_mindspeed \
bash scripts/setup_ascend_megatron_validated_env.sh

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/root
export DEVICE_BACKEND=npu
export DIST_BACKEND=hccl
export ASCEND_RT_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=/opt/libra/torch_extensions
export MCORE_MOE_GROUPED_GEMM=0
export TOKENIZERS_PARALLELISM=false
```

## Re-run the hardware proof

Use an idle NPU.  The preflight loads the Qwen3-0.6B checkpoint through Bridge,
constructs the MCore model, runs a real NPU forward pass, and checks the logits
are finite.

```bash
/opt/libra/envs/rl_mindspeed/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=1 \
  examples/test_megatron_core_npu_runtime.py \
  --model /opt/libra/models/Qwen3-0.6B \
  --max-seq-length 128
```

Expected terminal marker:

```text
MCORE_NPU_PREFLIGHT_OK device=npu:0 shape=(1, 9, 151936) dtype=torch.bfloat16
```

The captured server log is
`/opt/libra/runs/mcore_npu_preflight_0600m.log`.

The stronger training proof initializes MCore's distributed optimizer and runs
logprob recomputation, GRPO backward, gradient finalization, and one real
optimizer update:

```bash
/opt/libra/envs/rl_mindspeed/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=1 \
  examples/test_megatron_core_npu_runtime.py \
  --model /opt/libra/models/Qwen3-0.6B \
  --max-seq-length 128 --training-step
```

Expected marker: `MCORE_NPU_TRAIN_STEP_OK`. The validated run reported a finite
loss of `-0.5` and gradient norm `148.24038696289062`; its log is
`/opt/libra/runs/mcore_npu_train_step_0600m.log`.

## Versioned but not active in this baseline

Official Ascend sources were also checked out for the alternative MindSpeed
0.12 compatibility line.  They are preserved under
`/opt/libra/vendor/mindspeed`:

- MindSpeed branch `26.0.0_core_r0.12.1`, commit `3508f2f`
- MindSpeed-LLM tag `v26.0.0`, commit `57c3329`
- Megatron-LM tag `core_v0.12.1`, commit `a845aa7`

They are not imported by the passing baseline because Megatron Bridge
`0.2.0rc6` requires Megatron-Core `>=0.14,<0.16`.  Mixing that Bridge with the
MindSpeed 0.12 MCore fork fails at import time.  Keep the two lines isolated.
