"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal git-initialized temporary directory."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, check=True,
    )
    return tmp_path
