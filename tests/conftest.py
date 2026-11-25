"""Pytest bootstrap to keep RL_Framework importable across environments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def pytest_configure() -> None:
    """Register RL_Framework from repo root if editableFinder is hidden/missing."""
    if importlib.util.find_spec("RL_Framework") is None:
        repo_root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "RL_Framework",
            repo_root / "__init__.py",
            submodule_search_locations=[str(repo_root)],
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules["RL_Framework"] = module
            spec.loader.exec_module(module)
