# R2E-Gym patch execution

The default R2E-Gym workflow generates an issue and scores its text. It does not execute a patch. Set `R2E_PATCH_EXECUTION=1` to switch the same workflow to unified-diff generation and container test reward. The model must return a `git diff` style patch. Each turn runs tests before and after applying the patch in a fresh task checkout. Reward is 1 only when the baseline fails and the patched tests pass; failed execution is returned as tool feedback for a possible revision.

The public dataset index contains `repo_name`, `commit_hash`, and `docker_image`, but no repository checkout or test command. Supply these explicitly before enabling patch mode:

```sh
export R2E_PATCH_EXECUTION=1
export R2E_REPO_ROOT=/absolute/path/to/task-repositories
export R2E_TEST_COMMAND_JSON='["pytest", "-q"]'
export R2E_PATCH_TIMEOUT=300
```

`R2E_REPO_ROOT/<repo_name>` must be a local Git repository containing the requested commit. `R2E_TEST_COMMAND_JSON` is an argument array run inside the dataset's Docker image with the isolated checkout mounted at `/workspace`. Use an image and test command appropriate to your R2E-Gym task; the dataset's `expected_output_json` is not itself an executable test specification. Docker must be available on each rollout worker. The harness disables container networking and drops Linux capabilities. Never point patch mode at a privileged Docker daemon when processing untrusted model output.

For R2E-Gym `.sif` task images containing `/testbed` and `/r2e_tests`, use Singularity:

```sh
export R2E_PATCH_BACKEND=singularity
export R2E_SIF_ROOT=/absolute/path/to/sif-images
export R2E_TEST_COMMAND_JSON='["bash", "/testbed/run_tests.sh"]'
```

The entrypoint resolves each image as `<repo_name>_<first-eight-commit-chars>.sif` under `R2E_SIF_ROOT`. Every rollout worker must have Singularity and access to these images.
