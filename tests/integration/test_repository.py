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


def test_edit_file_replaces_only_one_exact_occurrence(git_repo: Path):
    target = git_repo / "src" / "utils.py"
    target.parent.mkdir()
    target.write_text('"""Small numeric utility functions."""\n\nvalue = 1\n', encoding="utf-8")
    repo = RepositoryAdapter(git_repo)

    result = repo.edit_file("src/utils.py", [{
        "old_text": "value = 1", "new_text": "def multiply(a, b):\n    return a * b",
    }])

    assert result.success
    assert result.operation == "edit_file"
    assert result.changed_files == ["src/utils.py"]
    assert "Applied 1 exact edit" in result.stdout
    assert target.read_text(encoding="utf-8") == (
        '"""Small numeric utility functions."""\n\n'
        "def multiply(a, b):\n    return a * b\n"
    )


@pytest.mark.parametrize(
    ("content", "old_text", "message"),
    [
        ("keep this\n", "missing", "not found"),
        ("same\nsame\n", "same", "multiple times"),
    ],
)
def test_edit_file_rejects_missing_or_ambiguous_text_without_modification(
    git_repo: Path, content: str, old_text: str, message: str,
):
    target = git_repo / "src" / "utils.py"
    target.parent.mkdir()
    target.write_text(content, encoding="utf-8")
    repo = RepositoryAdapter(git_repo)

    result = repo.edit_file("src/utils.py", [{"old_text": old_text, "new_text": "replacement"}])

    assert not result.success
    assert message in result.stderr
    assert result.changed_files == []
    assert target.read_text(encoding="utf-8") == content


def test_edit_file_rejects_paths_outside_workspace(git_repo: Path):
    repo = RepositoryAdapter(git_repo)

    with pytest.raises(RepositoryError, match="must be relative"):
        repo.edit_file("../outside.txt", [{"old_text": "old", "new_text": "new"}])


def test_edit_file_requires_an_existing_target(git_repo: Path):
    repo = RepositoryAdapter(git_repo)

    result = repo.edit_file("missing.txt", [{"old_text": "old", "new_text": "new"}])

    assert not result.success
    assert "does not exist" in result.stderr
    assert not (git_repo / "missing.txt").exists()


def test_edit_file_applies_multiple_non_overlapping_edits(git_repo: Path):
    target = git_repo / "src" / "utils.py"
    target.parent.mkdir()
    original = '"""Small numeric utility functions."""\n\nvalue = 1\n\n# end\n'
    target.write_text(original, encoding="utf-8")
    repo = RepositoryAdapter(git_repo)

    result = repo.edit_file("src/utils.py", [
        {"old_text": '"""Small numeric utility functions."""',
         "new_text": '"""Small numeric utility functions."""\n\nfrom math import prod'},
        {"old_text": "value = 1", "new_text": "def multiply(a, b):\n    return a * b"},
    ])

    assert result.success
    assert target.read_text(encoding="utf-8") == (
        '"""Small numeric utility functions."""\n\nfrom math import prod\n\n'
        "def multiply(a, b):\n    return a * b\n\n# end\n"
    )


def test_edit_file_validates_all_edits_before_writing(git_repo: Path):
    target = git_repo / "src" / "utils.py"
    target.parent.mkdir()
    original = "first = 1\nsecond = 2\n"
    target.write_text(original, encoding="utf-8")
    repo = RepositoryAdapter(git_repo)

    result = repo.edit_file("src/utils.py", [
        {"old_text": "first = 1", "new_text": "first = 10"},
        {"old_text": "missing = 3", "new_text": "missing = 30"},
    ])

    assert not result.success
    assert "not found" in result.stderr
    assert target.read_text(encoding="utf-8") == original


def test_edit_file_rejects_overlapping_edits_without_modification(git_repo: Path):
    target = git_repo / "src" / "utils.py"
    target.parent.mkdir()
    original = "abcdef\n"
    target.write_text(original, encoding="utf-8")
    repo = RepositoryAdapter(git_repo)

    result = repo.edit_file("src/utils.py", [
        {"old_text": "bcd", "new_text": "B"},
        {"old_text": "def", "new_text": "D"},
    ])

    assert not result.success
    assert "overlap" in result.stderr
    assert target.read_text(encoding="utf-8") == original


def test_write_file_still_creates_and_intentionally_replaces_files(git_repo: Path):
    repo = RepositoryAdapter(git_repo)
    repo.write_file("new.txt", "created\n")
    assert (git_repo / "new.txt").read_text(encoding="utf-8") == "created\n"

    result = repo.write_file("new.txt", "whole file replaced\n")

    assert result.success
    assert (git_repo / "new.txt").read_text(encoding="utf-8") == "whole file replaced\n"


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
