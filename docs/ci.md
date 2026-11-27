# Continuous integration and merge policy

Pushes to any branch and pull requests targeting `main` run `.github/workflows/ci.yml`.
The workflow can also be started manually from GitHub Actions.

The CPU regression suite runs on Python 3.10 and 3.12 with PyTorch 2.7.0 CPU.
It covers configuration, asynchronous execution, reload control, C-MLFQ and
cost-aware scheduling, Global Resource Planner and preflight simulation,
Elastic Hybrid Pool and runtime boundary integration, serving protocol, GRPO
grouping, Megatron-Core adapter checks, and R2E-Gym workflow checks.
`scripts/test_cpu_ci.sh` runs the entire consolidated test suite under `tests/`.

To reproduce in a clean Python 3.10–3.12 virtual environment:

```bash
python -m pip install -r requirements-ci.txt
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
bash scripts/test_cpu_ci.sh
```

JUnit reports are uploaded for 14 days. Jobs have timeouts and superseded runs
are cancelled. `CI required` succeeds only when every matrix job passes;
failed, cancelled, or skipped dependencies do not satisfy the gate.
No path filters are used, so documentation-only PRs also receive the required check.
PR workflows use read-only permissions, hosted runners, and no cluster credentials.

The `main` branch protection requires a PR and the GitHub Actions `CI required`
check on an up-to-date branch, including administrators. Force pushes and deletion
are disabled. No reviewer approval is required by this initial policy; a maintainer
still merges manually after checks pass. GitHub branch protection is repository
configuration, separate from this workflow file; changing YAML alone does not
activate it. This setup does not publish packages or deploy training services.

CUDA kernel, vLLM connector, multi-GPU KV migration, and full RL end-to-end
validation remain GPU cluster runs. CPU CI does not certify GPU correctness or
performance.
