# Compute-Node Environment Checklist

The complete offline installation procedure is in [env_creation.md](env_creation.md). Before launching Libra, validate the environment from an actual compute-node allocation rather than only on the login node.

```bash
srun --nodes=1 --ntasks=1 bash -lc '
  source "$HOME/rl_framework_env/bin/activate"
  set -e
  echo "Python: $(which python3)"
  python3 --version
  python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  python3 -c "import transformers, vllm; print(transformers.__version__, vllm.__version__)"
'
```

Check the following before submitting a long job:

- The base Python interpreter and standard library are visible on every node.
- The virtual environment is stored on a shared filesystem or reproduced consistently per node.
- The CUDA runtime and driver are compatible with the installed PyTorch and vLLM builds.
- `python3`, not an unrelated system `python`, is used consistently by the launchers.
- The activation path in the Slurm script matches the environment where dependencies were installed.
- Model, dataset, checkpoint, and log paths are visible from every participating node.
- `PYTHONPATH` contains the parent directory of `RL_Framework`.

If a venv created on the login node reports `No module named 'encodings'` on a compute node, its base interpreter or standard library path is unavailable there. Recreate the venv from a Python installation visible to the compute nodes.
