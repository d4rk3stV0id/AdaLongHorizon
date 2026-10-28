from __future__ import annotations

import os
from pathlib import Path

from ..errors import RepositoryError
from ..models import RepositorySnapshot, ToolResult
from .command_runner import CommandRunner


class RepositoryAdapter:
    """Bounded filesystem and Git operations behind a small replaceable interface."""

    INSTRUCTION_NAMES = ("AGENTS.md", "README.md", "CONTRIBUTING.md")
    IGNORED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".agent-state", "node_modules"}

    def __init__(self, root: str | Path, max_file_bytes: int = 100_000,
                 max_snapshot_files: int = 500, timeout: float = 30.0):
        candidate = Path(root).resolve(strict=True)
        if not candidate.is_dir():
            raise RepositoryError("workspace must be a directory")
        self.root = candidate
        self.max_file_bytes = max_file_bytes
        self.max_snapshot_files = max_snapshot_files
        self.runner = CommandRunner(candidate, timeout)

    def _path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise RepositoryError("path must be relative to the workspace")
        candidate = self.root / path
        if any(part.is_symlink() for part in [candidate, *candidate.parents] if part != self.root.parent):
            raise RepositoryError("symlink paths are not permitted")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root):
            raise RepositoryError("path escapes the workspace")
        return resolved

    def read_file(self, relative: str) -> str:
        path = self._path(relative)
        if not path.is_file() or path.stat().st_size > self.max_file_bytes:
            raise RepositoryError("file is missing or exceeds the read limit")
        return path.read_text(encoding="utf-8")

    def write_file(self, relative: str, content: str) -> ToolResult:
        path = self._path(relative)
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise RepositoryError("write exceeds file size limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        return ToolResult("write_file", True, changed_files=[path.relative_to(self.root).as_posix()])

    def edit_file(self, relative: str, edits: list[dict[str, str]]) -> ToolResult:
        """Atomically replace exact, unique, non-overlapping spans in one file."""
        path = self._path(relative)
        if not path.is_file():
            return ToolResult("edit_file", False, stderr="target file does not exist")
        if not isinstance(edits, list) or not edits:
            return ToolResult("edit_file", False, stderr="edits must be a non-empty list")
        if path.stat().st_size > self.max_file_bytes:
            return ToolResult("edit_file", False, stderr="target file exceeds the read limit")

        content = path.read_text(encoding="utf-8")
        ranges: list[tuple[int, int, str]] = []
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return ToolResult("edit_file", False, stderr=f"edit {index} must be an object")
            old_text = edit.get("old_text")
            new_text = edit.get("new_text")
            if not isinstance(old_text, str) or not old_text:
                return ToolResult("edit_file", False, stderr=f"edit {index} old_text must be a non-empty string")
            if not isinstance(new_text, str):
                return ToolResult("edit_file", False, stderr=f"edit {index} new_text must be a string")
            first = content.find(old_text)
            if first < 0:
                return ToolResult("edit_file", False, stderr=f"edit {index} old_text was not found in target file")
            if content.find(old_text, first + 1) >= 0:
                return ToolResult("edit_file", False, stderr=f"edit {index} old_text was found multiple times in target file")
            ranges.append((first, first + len(old_text), new_text))

        ordered_ranges = sorted(ranges, key=lambda item: item[0])
        for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
            if current[0] < previous[1]:
                return ToolResult("edit_file", False, stderr="requested edit ranges overlap")

        updated = content
        for start, end, new_text in sorted(ranges, key=lambda item: item[0], reverse=True):
            updated = updated[:start] + new_text + updated[end:]
        encoded = updated.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            return ToolResult("edit_file", False, stderr="edit exceeds file size limit")
        path.write_bytes(encoded)
        relative_path = path.relative_to(self.root).as_posix()
        return ToolResult(
            "edit_file", True,
            stdout=f"Applied {len(edits)} exact edit(s) to {relative_path}",
            changed_files=[relative_path],
        )

    def run_command(self, args: list[str], timeout: float | None = None) -> ToolResult:
        before = set(self.changed_files())
        proc = self.runner.run(args, timeout)
        changed = sorted(set(self.changed_files()) - before)
        return ToolResult("run_command", proc.returncode == 0, proc.stdout, proc.stderr,
                          proc.returncode, changed, proc.returncode == 124)

    def git(self, args: list[str]) -> str:
        result = self.run_command(["git", *args])
        if not result.success:
            return ""
        return result.stdout.strip()

    def changed_files(self) -> list[str]:
        result = self.run_command_raw(["git", "status", "--porcelain", "--untracked-files=all"])
        if result is None or result.returncode:
            return []
        files = []
        for line in result.stdout.splitlines():
            name = line[3:].split(" -> ")[-1]
            files.append(name.replace("\\", "/"))
        return sorted(set(files))

    def run_command_raw(self, args: list[str]):
        try:
            return self.runner.run(args)
        except RepositoryError:
            return None

    def inspect(self) -> RepositorySnapshot:
        status = self.git(["status", "--short", "--branch"])
        commit = self.git(["rev-parse", "HEAD"]) or None
        files: list[str] = []
        instructions: dict[str, str] = {}
        for current, dirs, names in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in self.IGNORED and not (Path(current) / d).is_symlink())
            for name in sorted(names):
                path = Path(current) / name
                rel = path.relative_to(self.root).as_posix()
                if len(files) >= self.max_snapshot_files:
                    break
                files.append(rel)
                if name in self.INSTRUCTION_NAMES and path.stat().st_size <= self.max_file_bytes:
                    instructions[rel] = path.read_text(encoding="utf-8", errors="replace")
        test_command = self.discover_test_command(files)
        return RepositorySnapshot(str(self.root), files, instructions, status, commit, status, test_command)

    @staticmethod
    def discover_test_command(files: list[str]) -> list[str] | None:
        if any(Path(p).name.startswith("test_") and p.endswith(".py") for p in files):
            return ["python", "-m", "pytest", "-q"]
        if "pyproject.toml" in files or "pytest.ini" in files:
            return ["python", "-m", "pytest", "-q"]
        return None

    def diff(self) -> str:
        return self.git(["diff", "--"])

    def checkpoint(self) -> dict[str, str | None]:
        return {"commit": self.git(["rev-parse", "HEAD"]) or None,
                "status": self.git(["status", "--porcelain", "--untracked-files=all"])}
