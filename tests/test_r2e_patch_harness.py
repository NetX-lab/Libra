import subprocess
from pathlib import Path

from env.r2e_patch_harness import R2EPatchExecutionHarness


def test_patch_harness_uses_isolated_checkout_and_docker(tmp_path: Path, monkeypatch):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "value.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "value.py").write_text("VALUE = 2\n")
    patch = subprocess.check_output(["git", "diff", "--", "value.py"], cwd=repo, text=True)
    subprocess.run(["git", "checkout", "--", "value.py"], cwd=repo, check=True)
    real_run = subprocess.run
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        if command[0] == "docker":
            calls += 1
            mount = command[command.index("-v") + 1].split(":/workspace")[0]
            expected = "VALUE = 1\n" if calls == 1 else "VALUE = 2\n"
            assert (Path(mount) / "value.py").read_text() == expected
            return subprocess.CompletedProcess(command, 1 if calls == 1 else 0, "", "")
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    result = R2EPatchExecutionHarness().execute(patch, repo, ["pytest", "-q"], "example:test", commit)
    assert result["patch_applied"] and result["baseline_failed"] and result["tests_passed"]
    assert (repo / "value.py").read_text() == "VALUE = 1\n"
