#!/usr/bin/env bash
# Capture a secret-free, reproducible fingerprint of the validated Ascend stack.
set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
train_python="${TRAIN_PYTHON:-/opt/libra/envs/rl_mindspeed/bin/python}"
rollout_python="${ROLLOUT_PYTHON:-/opt/libra/envs/vllm_ascend/bin/python}"
snapshot_root="${SNAPSHOT_ROOT:-/opt/libra/environment_snapshots}"
snapshot_name="${SNAPSHOT_NAME:-ascend_megatron_$(date +%Y%m%d_%H%M%S)}"
snapshot_dir="${snapshot_root}/${snapshot_name}"

[[ -x "$train_python" ]] || { echo "Missing training Python: $train_python" >&2; exit 1; }
mkdir -p "$snapshot_dir"

capture_python_env() {
    local python_bin="$1"
    local prefix="$2"
    "$python_bin" -m pip freeze --all >"${snapshot_dir}/${prefix}.pip-freeze.txt"
    "$python_bin" -m pip list --format=json >"${snapshot_dir}/${prefix}.pip-list.json"
    "$python_bin" - <<'PY' >"${snapshot_dir}/${prefix}.runtime.json"
import importlib.metadata as md
import json
import platform
import sys

result = {
    "executable": sys.executable,
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {},
}
for name in (
    "torch", "torch-npu", "megatron-core", "megatron-bridge", "transformers",
    "tokenizers", "datasets", "accelerate", "vllm", "vllm-ascend",
):
    try:
        result["packages"][name] = md.version(name)
    except md.PackageNotFoundError:
        pass
try:
    import torch
    result["torch_runtime"] = torch.__version__
    result["npu_available"] = bool(getattr(torch, "npu", None) and torch.npu.is_available())
except Exception as exc:
    result["torch_import_error"] = repr(exc)
print(json.dumps(result, indent=2, sort_keys=True))
PY
}

capture_python_env "$train_python" training
if [[ -x "$rollout_python" ]]; then
    capture_python_env "$rollout_python" rollout
fi

{
    echo "captured_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -srmo)"
    [[ -r /etc/os-release ]] && sed -n 's/^\(NAME\|VERSION\|ID\|VERSION_ID\)=/os_\1=/p' /etc/os-release
    echo "train_python=$train_python"
    echo "rollout_python=$rollout_python"
    echo "project_dir=$project_dir"
} >"${snapshot_dir}/system.txt"

if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info >"${snapshot_dir}/npu-smi-info.txt" 2>&1 || true
    npu-smi info -t board -i 0 >"${snapshot_dir}/npu-smi-board0.txt" 2>&1 || true
fi

for version_file in \
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info \
    /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
    /usr/local/Ascend/driver/version.info; do
    if [[ -r "$version_file" ]]; then
        target="${snapshot_dir}/$(echo "$version_file" | tr '/' '_')"
        cp -a "$version_file" "$target"
    fi
done

source_files=(
    requirements-ascend-megatron-validated.lock.txt
    requirements-vllm-ascend-validated.lock.txt
    scripts/setup_ascend_megatron_validated_env.sh
    scripts/capture_ascend_megatron_environment.sh
    engine/megatron_core_train_engine.py
    engine/megatron_core_attention.py
    engine/megatron_npu_compat.py
    examples/test_megatron_core_npu_runtime.py
)
: >"${snapshot_dir}/project-files.sha256"
for relative_path in "${source_files[@]}"; do
    if [[ -f "${project_dir}/${relative_path}" ]]; then
        sha256sum "${project_dir}/${relative_path}" >>"${snapshot_dir}/project-files.sha256"
    fi
done

cp -a "${project_dir}/requirements-ascend-megatron-validated.lock.txt" "$snapshot_dir/"
cp -a "${project_dir}/requirements-vllm-ascend-validated.lock.txt" "$snapshot_dir/"
cp -a "${project_dir}/scripts/setup_ascend_megatron_validated_env.sh" "$snapshot_dir/"
if [[ -f "${project_dir}/docs/ascend_megatron_validated_environment.md" ]]; then
    cp -a "${project_dir}/docs/ascend_megatron_validated_environment.md" "$snapshot_dir/"
fi

# Record only operational variables needed to reproduce the run. Never dump env.
{
    printf 'DEVICE_BACKEND=%s\n' "${DEVICE_BACKEND:-npu}"
    printf 'DIST_BACKEND=%s\n' "${DIST_BACKEND:-hccl}"
    printf 'CANN_ENV=%s\n' "${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
    printf 'TORCH_EXTENSIONS_DIR=%s\n' "${TORCH_EXTENSIONS_DIR:-/opt/libra/torch_extensions}"
    printf 'MCORE_MOE_GROUPED_GEMM=%s\n' "${MCORE_MOE_GROUPED_GEMM:-0}"
    printf 'TOKENIZERS_PARALLELISM=%s\n' "${TOKENIZERS_PARALLELISM:-false}"
} >"${snapshot_dir}/runtime.env"

{
    printf 'rollout_python_realpath=%s\n' "$(readlink -f "$rollout_python" 2>/dev/null || true)"
    if [[ -d /usr/local/Ascend/nnal ]]; then
        du -sh /usr/local/Ascend/nnal
        find /usr/local/Ascend/nnal -maxdepth 5 \
            \( -name set_env.sh -o -name libatb.so -o -name 'torch_atb-*.whl' \) \
            -print | sort
        find /usr/local/Ascend/nnal -maxdepth 5 \
            \( -name libatb.so -o -name 'torch_atb-*.whl' \) \
            -type f -print0 | sort -z | xargs -0 -r sha256sum
    fi
} >"${snapshot_dir}/rollout-system-dependencies.txt"

manifest_tmp="$(mktemp)"
(
    cd "$snapshot_dir"
    find . -maxdepth 1 -type f ! -name MANIFEST.sha256 -print0 \
        | sort -z \
        | xargs -0 sha256sum >"$manifest_tmp"
)
mv "$manifest_tmp" "${snapshot_dir}/MANIFEST.sha256"
printf '%s\n' "$snapshot_dir"
