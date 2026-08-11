# Offline Python Environment for a Slurm Cluster

Use this procedure when compute nodes cannot access the internet. The download host must match the cluster's operating system, CPU architecture, Python ABI, CUDA version, and glibc as closely as possible. Packages such as PyTorch, vLLM, and xFormers include compiled extensions and are not portable across arbitrary platforms.

## 1. Download dependencies on a connected machine

```bash
mkdir my_offline_packages
cd my_offline_packages
python3 -m pip download -r /path/to/requirements.txt -d .
```

This downloads the requested packages and their recursive dependencies without installing them. Copy `my_offline_packages/` and `requirements.txt` to a shared filesystem visible from the cluster.

For a different target platform, use pip's `--platform`, `--python-version`, `--implementation`, and `--abi` options. Source-only packages still require their build dependencies on the cluster.

## 2. Verify Python on a compute node

Login and compute nodes may expose different Python installations. A virtual environment normally contains symbolic links to its base interpreter, so a venv created on a login node can fail if the same interpreter and standard library are missing on compute nodes.

Run a short allocation to verify the interpreter:

```bash
srun --nodes=1 --ntasks=1 python3 --version
srun --nodes=1 --ntasks=1 python3 -c "import sys; print(sys.executable); print(sys.prefix)"
```

Prefer a site-provided module, container, or shared Python installation. Copying a system Python manually should be a last resort because the interpreter may also depend on shared libraries outside the Python standard library.

If the cluster's Python installation is self-contained and redistribution is permitted, place both the interpreter and its matching standard library on shared storage before creating the venv:

```bash
mkdir -p "$HOME/python310/bin" "$HOME/python310/lib"
cp /usr/local/bin/python3.10 "$HOME/python310/bin/"
cp -R /usr/local/lib/python3.10 "$HOME/python310/lib/"
"$HOME/python310/bin/python3.10" --version
```

## 3. Create the virtual environment on shared storage

```bash
"$HOME/python310/bin/python3.10" -m venv "$HOME/rl_framework_env"
source "$HOME/rl_framework_env/bin/activate"
```

If the cluster provides Python through an environment module, load that module before creating and activating the venv in every Slurm job.

## 4. Install without network access

```bash
python3 -m pip install \
  --no-index \
  --find-links=/path/to/my_offline_packages \
  -r requirements.txt
```

If your cluster cannot use the published vLLM wheel, build vLLM from source in
the target environment. An offline source build also needs a CUDA compiler,
CMake, Ninja, build-system wheels, and the source archives of nested
dependencies.

## 5. Validate from a compute node

```bash
srun --nodes=1 --ntasks=1 bash -lc '
  source "$HOME/rl_framework_env/bin/activate"
  which python3
  python3 -c "import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)"
'
```

Use the same activation path in every Slurm launcher. Also verify CUDA visibility and a minimal tensor allocation before starting a multi-node run.
