"""CLI entry point for running agent demonstrations and tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from .config import load_config
from .models import IssueSpec
from .orchestrator import Orchestrator


def demo(task: str = "multiply") -> int:
    with tempfile.TemporaryDirectory(prefix="ada-agent-demo-") as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\n"
            "testpaths = ['tests']\n"
            "pythonpath = ['.']\n"
            "addopts = '-p no:cacheprovider --basetemp=.pytest_tmp'\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "demo@test.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Demo"], cwd=root, check=True)

        if task == "multiply":
            issue = IssueSpec(
                title="Add multiply function",
                description="Add a utility function multiply(a, b) and tests",
                acceptance_criteria=["function returns product of a and b", "pytest passes"],
            )
        else:
            issue = IssueSpec(
                title="Normalize whitespace",
                description="Add a function that collapses repeated whitespace.",
                acceptance_criteria=["function returns trimmed single-space text", "pytest passes"],
            )

        orchestrator = Orchestrator(load_config(root))
        state, observations = orchestrator.run(issue)

        events_file = root / ".agent-state" / "events.jsonl"
        events = (
            [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if events_file.exists()
            else []
        )

        output = {
            "task": task,
            "status": state.status.value,
            "plan": [s.step_id for s in state.plan.steps] if state.plan else [],
            "completed_steps": state.completed_steps,
            "files_in_workspace": [
                p.relative_to(root).as_posix()
                for p in root.rglob("*.py")
                if not any(part.startswith(".") for part in p.parts)
            ],
            "event_transitions": [e["event_type"] for e in events],
            "observations": [
                {
                    "operation": r.operation,
                    "success": r.success,
                    "exit_code": r.exit_code,
                    "stdout": r.stdout.strip(),
                    "stderr": r.stderr.strip(),
                }
                for r in observations.results
            ],
            "notes": observations.notes,
        }
        print(json.dumps(output, indent=2))
        return 0 if state.status.value == "COMPLETED" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Budget-aware software engineering prototype")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run deterministic end-to-end demonstration")
    demo_parser.add_argument(
        "--task",
        choices=["multiply", "normalize"],
        default="multiply",
        help="Demo task to execute (default: multiply)",
    )

    args = parser.parse_args()
    if args.command == "demo":
        return demo(task=args.task)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
