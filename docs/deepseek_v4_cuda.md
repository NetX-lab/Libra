# DeepSeek V4 Flash training on CUDA

The opt-in `megatron_model_family: deepseek_v4` selects NVIDIA's native
DeepSeek V4 Bridge and Megatron hybrid model. Libra retains its GRPO loss,
policy logprob recomputation and distributed training checkpoint path. Both
training and recomputation use Megatron's non-interleaved pipeline scheduler.
The default Megatron backend keeps its existing PP=1 and dependency behavior.

## Runtime and checkpoint

Use an isolated CUDA training environment. The integration targets
[Megatron Bridge eea53e4](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/eea53e431a8b0b4e9c4f47a76784c6f3200c9a1f)
and its
[matching Megatron-Core commit](https://github.com/NVIDIA/Megatron-LM/tree/43a2ef6d35a0188a4f0601a5ce5e569d2c011562).
Build the CUDA environment from the Bridge checkout with its pinned Megatron-LM
submodule, following the upstream container setup. That checkout requires Python
3.12, a recent Transformers release with `DeepseekV4ForCausalLM`, and a CUDA
build of Transformer Engine. Install Libra into that environment with
`python -m pip install --no-deps -e .`. Use the Bridge environment's torch,
Transformers and Megatron versions. Libra's Qwen Megatron install script pins
older versions for a separate stack.

Make a complete local Hugging Face snapshot of
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
available at the same path on all training nodes. Megatron Bridge reads the
official packed checkpoint and loads BF16 training weights. The initial profile
uses the native attention and MoE layout with MTP disabled, unfused mHC,
and selective recomputation. Attention TP and expert TP are fixed at 1.
CP1 retains the unfused DSA path. CP>1 uses Megatron-Core's contiguous
packed attention with the cuDNN DSA backend and requires a compatible
cuDNN Frontend and FlashMLA sparse forward in the CUDA runtime. Older
Hopper FlashMLA with `topk_length` supports this path when Libra's indexer
loss is disabled. Requesting indexer LSE requires the extended FlashMLA API.

## Context parallelism

Set `train_cp_size` in the YAML and set `train_dp_size: 0` to derive DP
from the number of training GPUs. The resulting topology must satisfy
`train_gpus = TP × PP × CP × DP` and EP must divide `DP × CP`.
The global batch must still divide into complete GRPO groups and
microbatches on each logical DP replica. Choose CP at process startup.

For example, with 48 training GPUs, PP3 and EP8, CP1, CP2, CP4 and CP8
give DP16, DP8, DP4 and DP2 respectively. At PP4, EP4 fits the
12 ranks per pipeline stage. CP1, CP2, CP3, CP4, CP6 and CP12 then
give DP12, DP6, DP4, DP3, DP2 and DP1 respectively. These are topology
examples. Activation memory and expert communication locality depend on
the actual placement and workload.

Each CP group handles the same logical trajectories. The packed
attention shards the padded sequence, preserving document boundaries and
global next-token targets across rank boundaries. Libra gathers scalar
logprobs before returning them to the trainer. GRPO uses global action
counts to normalize the local loss contributions and reports each
trajectory once. Additional right padding aligns shards and gives
Megatron enough rows for the attention boundary exchange.

## 32×H20 training gate

`configs/deepseek_v4_flash_h20_32_train_gate.yaml` uses all 32 H20s for
training across four 8-GPU nodes (`TP1 / PP4 / EP8 / DP8`), with CPU optimizer
offload and 1024-token synthetic trajectories. Confirm the GPUs are free and
there is enough host RAM for the offloaded optimizer state. The topology
follows the upstream 32-GPU Hopper recipe and requires validation on H20.

From the repo root on each node, set the same reachable rank-zero address
and shared checkpoint/output paths. Set `NODE_RANK` to 0, 1, 2 or 3.

```bash
torchrun --nnodes=4 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  scripts/validate_deepseek_v4_cuda.py \
  --config configs/deepseek_v4_flash_h20_32_train_gate.yaml \
  --model "$MODEL_PATH" --output "$RUN_OUTPUT"
```

Success requires 32 per-rank receipts and
`LIBRA_DEEPSEEK_V4_CUDA_GATE_PASSED ranks=32`. The gate checks finite
recomputed logprobs and loss, one optimizer revision, a change in sampled
parameters, and GPU memory peaks.
Pass `--cp-size 2` or `--cp-size 4` to the same gate to derive a
matching DP degree. Use a fresh output directory for each run. The gate
also checks that CP peers reconstruct identical logprobs and that the
optimizer reports the configured global sample count.

The gate measures a training update. A separate rollout allocation or a
validated colocated runtime is needed for generation during this run. Before
using this path for a full Libra efficiency benchmark, validate HF export,
rollout weight publication and consumption, and elastic training with the
chosen backend and parallel layout.

## Compact replica memory

For a compact BF16 replica, the distributed optimizer can allocate its new
parameter and gradient buffers while all original parameter storage is still
resident. The patch below moves independent, non-NCCL-registered gradient
allocation after the parameters are remapped. Apply it to a separate checkout
of the compatible Megatron-Core source used by each training rank.

```bash
git -C "$MCORE_PATH" apply --check "$LIBRA_PATH/patches/mcore-defer-ddp-grad-allocation.patch"
git -C "$MCORE_PATH" apply "$LIBRA_PATH/patches/mcore-defer-ddp-grad-allocation.patch"
```

CUDA V4 still requires a static topology. Enabling EHP additionally requires
cross-host launch for the complete PP/EP replica, matching expert gradient
lanes, and full rollout weight publication.
