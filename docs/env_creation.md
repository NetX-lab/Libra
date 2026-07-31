# Installation and Offline Slurm Environment

This guide covers both connected installation and clusters whose compute nodes
cannot access the internet. Commands assume the repository root is the current
directory.

## Compatibility baseline

The released dependency set targets:

| Component | Supported or pinned value |
| --- | --- |
| Operating system | Linux x86_64 |
| Python | 3.10, 3.11, or 3.12 |
| Accelerator | NVIDIA GPU |
| PyTorch | 2.7.0 |
| vLLM | 0.9.2 |
| xFormers | 0.0.30 |
| Transformers | `>=4.51.1,<5.0.0` |
| Megatron-Core | 0.14.0 |
| Megatron Bridge | 0.2.0rc6 |
| grouped-GEMM | 1.1.4 |
| Reference CUDA runtime | CUDA 12.6 |

The NVIDIA driver must support the CUDA runtime used by the installed PyTorch
and vLLM builds. A local CUDA toolkit and `nvcc` are not required for compatible
binary wheels, but they are required for vLLM or grouped-GEMM source builds.
Wheel and source artifacts must match the cluster's operating system, CPU
architecture, Python ABI, CUDA version, GPU architecture, and glibc.

## Connected installation

```bash
git clone https://github.com/NetX-lab/Libra.git
cd Libra
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -c "import RL_Framework; print(RL_Framework.__version__)"
```

The default `megatron_core` backend additionally requires Megatron Bridge and
grouped-GEMM:

```bash
bash scripts/install_megatron_core.sh
```

The installer uses the active virtual environment. Alternatively set
`VENV_PATH=/path/to/venv/bin/activate`. When `vendor/megatron` is absent, the
installer downloads the pinned public packages.

## Prepare an offline wheelhouse

Use a connected Linux host that matches the compute nodes as closely as
possible. In particular, use the same Python minor version and CUDA/glibc
baseline.

```bash
python3 -m venv wheelhouse-env
source wheelhouse-env/bin/activate
python -m pip install --upgrade pip

mkdir -p wheelhouse
python -m pip download --dest wheelhouse \
  setuptools wheel ninja pybind11
python -m pip download --dest wheelhouse -r requirements.txt
python -m pip download --dest wheelhouse -r requirements-megatron.txt
python -m pip download --dest wheelhouse \
  "megatron-bridge==0.2.0rc6"
```

Inspect the result before transfer:

```bash
python -m pip install --dry-run --ignore-installed \
  --no-index --find-links="$PWD/wheelhouse" \
  -r requirements.txt
python -m pip install --dry-run --ignore-installed \
  --no-index --find-links="$PWD/wheelhouse" \
  -r requirements-megatron.txt
```

`pip download` can produce source archives for packages without compatible
wheels. Those packages still need their build-system dependencies and compiler
toolchain on the cluster.

### vLLM source-build option

The normal requirements file permits a compatible vLLM wheel. To prepare the
vLLM source archive instead:

```bash
python -m pip download --dest wheelhouse --no-binary=vllm \
  "vllm==0.9.2"
```

An offline vLLM build requires a compatible CUDA toolkit and `nvcc`, a supported
host compiler, CMake, Ninja, Python development headers, enough temporary disk
space, and all build-isolation dependencies in the wheelhouse. Prefer building
a reusable wheel on a matching connected builder:

```bash
python -m pip wheel --wheel-dir wheelhouse --no-binary=vllm \
  "vllm==0.9.2"
```

### Megatron-Core offline artifacts

For the repository installer, copy the following to `vendor/megatron` or point
the corresponding environment variables at equivalent files:

```text
megatron_core-0.14.0.tar.gz
megatron_bridge-0.2.0rc6-py3-none-any.whl
grouped_gemm-1.1.4-<python-tag>-<abi-tag>-<platform>.whl
  or grouped_gemm-1.1.4-with-cutlass.tar.gz
antlr4-python3-runtime-4.7.2.tar.gz
pybind11 and the small dependency wheels used by requirements-megatron.txt
```

The grouped-GEMM wheel must match the active Python ABI—for example `cp310`,
`cp311`, or `cp312`. The installer detects a wheel matching the current
interpreter rather than assuming Python 3.12.

## Verify Python on a compute node

Login and compute nodes may expose different Python installations. A virtual
environment normally contains links to its base interpreter, so a venv created
on a login node can fail when that interpreter or standard library is absent
from compute nodes.

```bash
srun --nodes=1 --ntasks=1 python3 --version
srun --nodes=1 --ntasks=1 python3 -c \
  "import platform, sys; print(sys.executable); print(platform.platform())"
```

Prefer a site-provided module, container, or shared Python installation.
Copying a system Python manually is not a portable installation method because
the interpreter can depend on libraries outside its standard-library
directory.

## Create the shared environment

Load the same site modules that every Slurm job will load, then create the venv
on shared storage:

```bash
module load python/3.11 cuda/12.6   # adapt to the site
python3 -m venv "$HOME/libra-env"
source "$HOME/libra-env/bin/activate"
python -m pip install --upgrade pip setuptools wheel
```

Copy the repository and wheelhouse to shared storage, then install without
network access:

```bash
export PROJECT_DIR=/shared/path/Libra
export WHEELHOUSE=/shared/path/wheelhouse

python -m pip install \
  --no-index \
  --find-links="$WHEELHOUSE" \
  -r "$PROJECT_DIR/requirements.txt"
python -m pip install \
  --no-deps \
  --no-build-isolation \
  -e "$PROJECT_DIR"
```

For the default Megatron-Core backend:

```bash
export MCORE_WHEELHOUSE="$PROJECT_DIR/vendor/megatron"
bash "$PROJECT_DIR/scripts/install_megatron_core.sh"
```

## Validate on a compute node

Run the commands from [the compute-node checklist](env_creation_compute_node.md)
inside an allocation. At minimum, validate:

- the `RL_Framework` editable package;
- Torch CUDA visibility and a real GPU tensor allocation;
- Transformers, vLLM, xFormers, PyArrow, and math-reward dependencies;
- Megatron-Core, Megatron Bridge, and grouped-GEMM when using the default
  backend;
- NCCL collectives across the exact nodes and network interfaces used by the
  training job.

Use the same module loads, activation path, `PROJECT_DIR`, and environment
exports in every Slurm launcher.

## Common failures

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| `No module named RL_Framework` | Repository was not installed | Run `python -m pip install -e "$PROJECT_DIR"` |
| `No module named encodings` | Base Python or standard library is unavailable on the compute node | Recreate the venv from a shared/site Python |
| `not a supported wheel on this platform` | Python ABI or platform tag mismatch | Rebuild/download for the compute-node interpreter |
| Missing `libcuda.so` or CUDA symbols | Driver/runtime mismatch | Align the driver, PyTorch, vLLM, and CUDA toolkit |
| vLLM build isolation tries the network | Incomplete wheelhouse | Pre-build the vLLM wheel or add every build dependency |
| grouped-GEMM build fails | Missing CUDA/CUTLASS/compiler or wrong GPU arch | Use a matching wheel or set the documented build toolchain |
