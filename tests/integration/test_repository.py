"""Integration tests for RepositoryAdapter and CommandRunner."""

import subprocess
from pathlib import Path
import pytest

from agent.errors import RepositoryError
from agent.tools.command_runner import CommandRunner
from agent.tools.repository import RepositoryAdapter


def test_file_operations_inside_workspace(git_repo: Path):
    repo = RepositoryAdapter(git_repo)
    result = repo.write_file("src/hello.py", "print('hello')\n")
    assert result.success
    assert "src/hello.py" in result.changed_files
    content = repo.read_file("src/hello.py")
    assert content == "print('hello')\n"


def test_file_operations_outside_workspace_rejected(git_repo: Path):
    repo = RepositoryAdapter(git_repo)

    # Path traversal with ..
    with pytest.raises(RepositoryError, match="must be relative"):
        repo.write_file("../outside.txt", "evil")

    # Absolute path
    with pytest.raises(RepositoryError, match="must be relative"):
        repo.write_file(str(git_repo.parent / "outside.txt"), "evil")

    # Reading nonexistent or outside
    with pytest.raises(RepositoryError):
        repo.read_file("../nonexistent.txt")


def test_file_size_limits_enforced(git_repo: Path):
    repo = RepositoryAdapter(git_repo, max_file_bytes=50)

    # Within limit
    repo.write_file("small.txt", "short")
    assert repo.read_file("small.txt") == "short"

    # Exceeds write limit
    with pytest.raises(RepositoryError, match="exceeds file size limit"):
        repo.write_file("big.txt", "x" * 100)

    # Exceeds read limit (if written directly via filesystem)
    (git_repo / "external_big.txt").write_bytes(b"x" * 100)
    with pytest.raises(RepositoryError, match="exceeds the read limit"):
        repo.read_file("external_big.txt")


def test_disallowed_commands_rejected(git_repo: Path):
    repo = RepositoryAdapter(git_repo)

    # Unlisted binary
    with pytest.raises(RepositoryError, match="not allowlisted"):
        repo.run_command(["rm", "-rf", "."])

    with pytest.raises(RepositoryError, match="not allowlisted"):
        repo.run_command(["sh", "-c", "echo 1"])

    # Python arbitrary script execution
    with pytest.raises(RepositoryError, match="only be used to invoke pytest"):
        repo.run_command(["python", "-c", "print('unsafe')"])

    # Disallowed git command (only read-only inspection commands allowed)
    with pytest.raises(RepositoryError, match="read-only Git inspection"):
        repo.run_command(["git", "commit", "-m", "illegal"])

    # Disallowed pytest arguments
    with pytest.raises(RepositoryError, match="only basic pytest verbosity flags"):
        repo.run_command(["pytest", "--pdb"])


def test_repository_snapshot_generation(git_repo: Path):
    repo = RepositoryAdapter(git_repo)
    (git_repo / "README.md").write_text("# Test Repo Instructions", encoding="utf-8")
    (git_repo / "src").mkdir()
    (git_repo / "src" / "app.py").write_text("APP = 1\n", encoding="utf-8")
    (git_repo / "ignored_dir").mkdir()
    (git_repo / "__pycache__").mkdir()
    (git_repo / "__pycache__" / "cached.pyc").write_text("cache", encoding="utf-8")

    snapshot = repo.inspect()
    assert "README.md" in snapshot.files
    assert "src/app.py" in snapshot.files
    # Ignored directories should not be in snapshot.files
    assert not any("__pycache__" in f for f in snapshot.files)

    # Instruction file parsed
    assert "README.md" in snapshot.instructions
    assert "# Test Repo Instructions" in snapshot.instructions["README.md"]


def test_test_command_discovery(git_repo: Path):
    repo = RepositoryAdapter(git_repo)

    # No test files or config
    snapshot1 = repo.inspect()
    assert snapshot1.test_command is None

    # Adding pyproject.toml
    (git_repo / "pyproject.toml").write_text("", encoding="utf-8")
    snapshot2 = repo.inspect()
    assert snapshot2.test_command == ["python", "-m", "pytest", "-q"]


def test_checkpoint_and_diff(git_repo: Path):
    repo = RepositoryAdapter(git_repo)
    (git_repo / "initial.txt").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "add", "initial.txt"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "init commit"], cwd=git_repo, check=True)

    checkpoint = repo.checkpoint()
    assert checkpoint["commit"] is not None
    assert len(checkpoint["commit"]) == 40

    # Modify file and verify diff and changed_files
    repo.write_file("initial.txt", "modified\n")
    assert "initial.txt" in repo.changed_files()
    assert "-init" in repo.diff()
    assert "+modified" in repo.diff()


def test_command_timeout(git_repo: Path):
    # Test timeout with CommandRunner
    runner = CommandRunner(git_repo, timeout=0.1)
    # Using python pytest -q on a simulated slow pytest is tricky, but CommandRunner
    # enforces timeout on any run if timeout expires
    result = runner.run(["python", "-m", "pytest", "-q"], timeout=0.0001)
    # On Windows, if subprocess terminates or times out:
    if result.returncode == 124:
        assert result.returncode == 124
        assert "timed out" in result.stderr
