#!/usr/bin/env bash
# Source inside automodelwire deepseek-v4-dspark:v30-fixed (do not run on host).
export PYTHONPATH="${LIBRA_SOURCE_PARENT:?Set LIBRA_SOURCE_PARENT to the directory containing RL_Framework}:/workspace-verl:/workspace-verl/verl:/workspace-verl/MindSpeed-LLM:/workspace-verl/MindSpeed:/workspace-verl/Megatron-LM:/workspace-verl/mbridge:${PYTHONPATH:-}"
if [[ -n "${LIBRA_CANN_TOOLKIT_ROOT:-}" ]]; then
    toolkit_env="${LIBRA_CANN_TOOLKIT_ROOT}/set_env.sh"
    if [[ ! -f "$toolkit_env" ]]; then
        toolkit_env="${LIBRA_CANN_TOOLKIT_ROOT}/ascend-toolkit/latest/set_env.sh"
    fi
    if [[ ! -f "$toolkit_env" ]]; then
        toolkit_env="${LIBRA_CANN_TOOLKIT_ROOT}/cann/set_env.sh"
    fi
    if [[ -f "$toolkit_env" ]]; then
        # Runtime libraries are kept isolated to this container environment.
        source "$toolkit_env"
        # CANN's set_env.sh resolves ASCEND_HOME_PATH to the versioned install
        # directory. Keep that value: LIBRA_CANN_TOOLKIT_ROOT may instead be
        # its parent directory containing `cann/` or `ascend-toolkit/latest/`.
        if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
            export ASCEND_HOME_PATH="$(cd "$(dirname "$toolkit_env")" && pwd)"
        fi
    else
        echo "CANN toolkit set_env.sh not found under $LIBRA_CANN_TOOLKIT_ROOT" >&2
        return 2
    fi
fi
if [[ -n "${LIBRA_CANN_OPS_ROOT:-}" ]]; then
    ops_root="$LIBRA_CANN_OPS_ROOT"
    if [[ ! -d "$ops_root/opp" && -d "$ops_root/cann/opp" ]]; then
        ops_root="$ops_root/cann"
    fi
    # When Toolkit and ops are installed under separate prefixes, AICPU
    # operator registration data lives with ops/opp rather than Toolkit.
    export ASCEND_AICPU_PATH="$ops_root"
    export LD_LIBRARY_PATH="${ops_root}/aarch64-linux/lib64:${ops_root}/lib64:${ops_root}/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64:${LD_LIBRARY_PATH:-}"
    if [[ -d "${ops_root}/python/site-packages" ]]; then
        export PYTHONPATH="${ops_root}/python/site-packages:${PYTHONPATH}"
    fi
    if [[ -d "${ops_root}/opp/built-in/op_impl/ai_core/tbe" ]]; then
        export PYTHONPATH="${ops_root}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"
    fi
    export ASCEND_OPP_PATH="${ops_root}/opp"
fi
# The CANN 9.1 toolkit payload available on the cluster is missing
# libcann_kb.so. DeepSeek's transfer_to_npu path initializes TBE and looks up
# that library beneath ASCEND_HOME_PATH, so point only that lookup at the
# complete 9.1 beta3 runtime shipped in the automodelwire base image. Keep the
# 9.1 final toolkit and ops paths above as the active compiler/operator paths.
if [[ -n "${LIBRA_CANN_KB_ROOT:-}" ]]; then
    [[ -f "${LIBRA_CANN_KB_ROOT}/aarch64-linux/lib64/libcann_kb.so" ]] || {
        echo "libcann_kb.so not found under $LIBRA_CANN_KB_ROOT" >&2
        return 2
    }
    export ASCEND_HOME_PATH="$LIBRA_CANN_KB_ROOT"
    export ASCEND_TOOLKIT_HOME="$LIBRA_CANN_KB_ROOT"
fi
# The automodelwire image registers DeepSeek's transformer operators from a
# separate vendor tree. Keep this registration visible when ASCEND_OPP_PATH is
# redirected to a separately installed CANN ops package; otherwise GE/TBE can
# fail during ACL precision-mode initialization before the first NPU op.
if [[ -z "${ASCEND_CUSTOM_OPP_PATH:-}" && -d /usr/local/Ascend/vendors/custom_transformer ]]; then
    export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/vendors/custom_transformer
fi
export ACCELERATOR_BACKEND=npu
export DIST_BACKEND=hccl
export PYTHONUNBUFFERED=1
export HCCL_BUFFSIZE=200
export VLLM_DSA_INDEXER_MODE=int8
# Megatron validates this even when MindSpeed redirects CUDA APIs to NPU.
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Megatron creates auxiliary Gloo process groups; bind them to the cluster NIC,
# since compute hostnames resolve to loopback inside the container image.
export GLOO_SOCKET_IFNAME=enp67s0f5
export HCCL_SOCKET_IFNAME=enp67s0f5
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export TORCH_DISTRIBUTED_TIMEOUT=3600
export DEVICE_BACKEND=npu
export MCORE_MOE_GROUPED_GEMM=0
export TOKENIZERS_PARALLELISM=false
