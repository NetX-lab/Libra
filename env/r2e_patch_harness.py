"""Execute a unified diff against an R2E-Gym repository in Docker."""
from __future__ import annotations

import subprocess
import tempfile
import shlex
import shutil
from pathlib import Path
from typing import Any, Sequence


class R2EPatchExecutionHarness:
    requires_repo_path = True
    def __init__(self, timeout: int = 300):
        self.timeout = max(1, int(timeout))

    def execute(
        self,
        patch: str,
        workspace: str | Path,
        test_command: Sequence[str],
        docker_image: str,
        commit_hash: str,
    ) -> dict[str, Any]:
        """Apply a patch to an isolated commit checkout, then run tests in Docker.

        The caller must supply a trusted image and command. Model output is only
        used as patch data; it never becomes a shell command.
        """
        source = Path(workspace).resolve()
        if not source.is_dir() or not (source / ".git").exists():
            raise ValueError(f"workspace is not a git checkout: {source}")
        if not patch.strip() or not docker_image or not commit_hash or not test_command:
            raise ValueError("patch, docker_image, commit_hash and test_command are required")
        with tempfile.TemporaryDirectory(prefix="r2e-patch-") as td:
            checkout = Path(td) / "repo"
            clone = subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--", str(source), str(checkout)],
                                   capture_output=True, text=True, timeout=self.timeout)
            if clone.returncode:
                raise RuntimeError(f"cannot clone task repository: {clone.stderr}")
            reset = subprocess.run(["git", "checkout", "--detach", commit_hash], cwd=checkout,
                                   capture_output=True, text=True, timeout=self.timeout)
            if reset.returncode:
                raise ValueError(f"commit is unavailable in workspace: {commit_hash}")
            patch_file = Path(td) / "candidate.patch"
            patch_file.write_text(patch, encoding="utf-8")
            command = ["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL",
                       "-v", f"{checkout}:/workspace", "-w", "/workspace", docker_image,
                       *test_command]
            try:
                baseline = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError("R2E baseline test timed out")
            check = subprocess.run(["git", "apply", "--check", "--", str(patch_file)], cwd=checkout,
                                   capture_output=True, text=True, timeout=self.timeout)
            if check.returncode:
                return {"patch_applied": False, "tests_passed": False,
                        "returncode": check.returncode, "stdout": check.stdout, "stderr": check.stderr}
            apply = subprocess.run(["git", "apply", "--", str(patch_file)], cwd=checkout,
                                   capture_output=True, text=True, timeout=self.timeout)
            if apply.returncode:
                return {"patch_applied": False, "tests_passed": False,
                        "returncode": apply.returncode, "stdout": apply.stdout, "stderr": apply.stderr}
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                return {"patch_applied": True, "tests_passed": False, "returncode": None,
                        "stdout": "", "stderr": "test timeout"}
            return {"patch_applied": True, "baseline_failed": baseline.returncode != 0,
                    "tests_passed": baseline.returncode != 0 and result.returncode == 0,
                    "returncode": result.returncode, "stdout": result.stdout[-10000:],
                    "stderr": result.stderr[-10000:]}


class R2ESingularityPatchHarness:
    """Execute patches in R2E-Gym SIF images containing /testbed."""

    requires_repo_path = False

    def __init__(self, timeout: int = 300):
        self.timeout = max(1, int(timeout))

    def execute(
        self,
        patch: str,
        workspace: str | Path | None,
        test_command: Sequence[str],
        docker_image: str,
        commit_hash: str,
    ) -> dict[str, Any]:
        del workspace, commit_hash
        image = Path(docker_image).resolve()
        if not image.is_file() or image.suffix != ".sif":
            raise ValueError(f"R2E SIF image is unavailable: {image}")
        if not patch.strip() or not test_command:
            raise ValueError("patch and test_command are required")
        if shutil.which("singularity") is None:
            raise RuntimeError("Singularity is unavailable on this worker")
        with tempfile.TemporaryDirectory(prefix="r2e-sif-") as td:
            sandbox = Path(td) / "sandbox"
            build = subprocess.run(
                ["singularity", "build", "--sandbox", str(sandbox), str(image)],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if build.returncode:
                raise RuntimeError(f"cannot build R2E sandbox: {build.stderr[-3000:]}")
            testbed = sandbox / "testbed"
            if not (testbed / ".git").exists():
                raise ValueError("R2E image does not contain a /testbed git checkout")
            patch_path = testbed / ".r2e_candidate.patch"
            patch_path.write_text(patch, encoding="utf-8")
            tests_link = testbed / "r2e_tests"
            if not tests_link.exists() and (sandbox / "r2e_tests").exists():
                tests_link.symlink_to("/r2e_tests")
            prefix = ["singularity", "exec", "--writable", "--no-home", str(sandbox),
                      "bash", "-lc"]
            setup = "export GIT_CONFIG_GLOBAL=/tmp/r2e_gitconfig; git config --global --add safe.directory /testbed; cd /testbed && "
            command = setup + shlex.join(test_command)
            try:
                baseline = subprocess.run(prefix + [command], capture_output=True,
                                          text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError("R2E baseline test timed out")
            for action in ("git apply --check .r2e_candidate.patch", "git apply .r2e_candidate.patch"):
                result = subprocess.run(prefix + [setup + action], capture_output=True,
                                        text=True, timeout=self.timeout)
                if result.returncode:
                    return {"patch_applied": False, "tests_passed": False,
                            "returncode": result.returncode, "stdout": result.stdout[-10000:],
                            "stderr": result.stderr[-10000:]}
            try:
                result = subprocess.run(prefix + [command], capture_output=True,
                                        text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                return {"patch_applied": True, "tests_passed": False,
                        "returncode": None, "stdout": "", "stderr": "test timeout"}
            return {"patch_applied": True, "baseline_failed": baseline.returncode != 0,
                    "tests_passed": baseline.returncode != 0 and result.returncode == 0,
                    "returncode": result.returncode, "stdout": result.stdout[-10000:],
                    "stderr": result.stderr[-10000:]}
