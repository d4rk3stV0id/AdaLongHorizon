from __future__ import annotations

import subprocess
from pathlib import Path

from ..errors import RepositoryError


class CommandRunner:
    """Run only explicitly permitted argument-array commands in the workspace."""

    ALLOWED = {"pytest", "python", "py", "git"}

    def __init__(self, workspace: Path, timeout: float = 30.0):
        self.workspace = workspace.resolve(strict=True)
        self.timeout = timeout

    def run(self, args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        if not args or args[0] not in self.ALLOWED:
            raise RepositoryError("command is not allowlisted")
        if any("\x00" in item for item in args):
            raise RepositoryError("invalid command argument")
        if args[0] == "git" and args[1:] not in (
            ["status", "--short", "--branch"],
            ["status", "--porcelain", "--untracked-files=all"],
            ["diff", "--"],
            ["rev-parse", "HEAD"],
        ):
            raise RepositoryError("only fixed read-only Git inspection commands are permitted")
        if args[0] in {"python", "py"} and args[1:] not in (["-m", "pytest"], ["-m", "pytest", "-q"]):
            raise RepositoryError("Python may only be used to invoke pytest")
        if args[0] == "pytest" and any(arg not in {"-q", "-v"} for arg in args[1:]):
            raise RepositoryError("only basic pytest verbosity flags are permitted")
        try:
            return subprocess.run(args, cwd=self.workspace, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True,
                                  timeout=timeout or self.timeout, check=False, shell=False)
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(args, 124, exc.stdout or "", exc.stderr or "command timed out")
