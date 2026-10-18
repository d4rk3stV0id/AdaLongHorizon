"""Integration tests for the Orchestrator end-to-end task execution."""

import json
from pathlib import Path
import subprocess

from agent.llm.base import LLMProvider
from agent.llm.provider import DeterministicProvider
from agent.models import (
    AgentConfig,
    BudgetLedger,
    IssueSpec,
    Lifecycle,
    StepStatus,
)
from agent.orchestrator import Orchestrator


class FailureMockProvider(LLMProvider):
    """Provider that writes a failing test."""

    def propose_action(self, issue, step, snapshot, state):
        if "test" in step.objective.lower():
            return {
                "type": "write_file",
                "path": "tests/test_text_tools.py",
                "content": "def test_fail():\n    assert False\n",
            }
        return {
            "type": "write_file",
            "path": "src/text_tools.py",
            "content": "def normalize_whitespace(s): return s\n",
        }


def test_orchestrator_full_task_run_completed(git_repo: Path):
    """Verify that a standard issue runs through all phases to COMPLETED."""
    (git_repo / "pyproject.toml").write_text("", encoding="utf-8")
    config = AgentConfig(workspace=str(git_repo))
    orchestrator = Orchestrator(config, DeterministicProvider())

    issue = IssueSpec(
        title="Add normalize_whitespace",
        description="Add a function to normalize whitespace in strings and tests",
        acceptance_criteria=["function exists", "pytest passes"],
    )

    state, obs = orchestrator.run(issue)

    assert state.status == Lifecycle.COMPLETED
    assert state.plan is not None
    assert len(state.plan.steps) == 2
    assert state.completed_steps == ["implementation", "test"]

    # Verify files created on disk
    assert (git_repo / "src" / "text_tools.py").exists()
    assert (git_repo / "tests" / "test_text_tools.py").exists()

    # Verify state persistence on disk
    state_file = git_repo / ".agent-state" / f"{state.task_id}.json"
    assert state_file.exists()
    saved_state = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved_state["status"] == "COMPLETED"

    # Verify event logger
    events_file = git_repo / ".agent-state" / "events.jsonl"
    assert events_file.exists()
    event_types = [
        json.loads(line)["event_type"]
        for line in events_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_sequence = [
        "task_created",
        "repository_inspected",
        "plan_created",
        "step_executed",
        "step_executed",
        "task_finished",
    ]
    assert event_types == expected_sequence


def test_orchestrator_budget_exhaustion_terminates_cleanly(git_repo: Path):
    """Verify that budget exhaustion cleanly transitions to BUDGET_EXHAUSTED."""
    (git_repo / "pyproject.toml").write_text("", encoding="utf-8")
    config = AgentConfig(workspace=str(git_repo))
    orchestrator = Orchestrator(config, DeterministicProvider())

    issue = IssueSpec(
        title="Add function",
        description="Add a function and tests",
        acceptance_criteria=["function exists"],
    )

    # Exhaust budget before any LLM calls are allowed
    budget = BudgetLedger(max_llm_calls=0)
    state, obs = orchestrator.run(issue, budget=budget)

    assert state.status == Lifecycle.BUDGET_EXHAUSTED
    assert any("budget exhausted" in note.lower() for note in obs.notes)

    # Event logger records task_finished with BUDGET_EXHAUSTED
    events_file = git_repo / ".agent-state" / "events.jsonl"
    last_event = json.loads(events_file.read_text(encoding="utf-8").splitlines()[-1])
    assert last_event["event_type"] == "task_finished"
    assert last_event["payload"]["status"] == "BUDGET_EXHAUSTED"


def test_orchestrator_test_failure_transitions_to_blocked(git_repo: Path):
    """Verify that when verification fails, orchestrator marks task BLOCKED."""
    (git_repo / "pyproject.toml").write_text("", encoding="utf-8")
    config = AgentConfig(workspace=str(git_repo))
    orchestrator = Orchestrator(config, FailureMockProvider())

    issue = IssueSpec(
        title="Add function",
        description="Add a function and tests",
        acceptance_criteria=["function exists", "pytest passes"],
    )

    state, obs = orchestrator.run(issue)

    assert state.status == Lifecycle.BLOCKED
    assert state.budget.consecutive_failures == 1


def test_orchestrator_multiply_task_completed(git_repo: Path):
    """Verify multiply(a, b) task runs through all phases to COMPLETED and tests pass."""
    (git_repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\npythonpath = ['.']\n",
        encoding="utf-8",
    )
    (git_repo / "src").mkdir()
    (git_repo / "tests").mkdir()
    (git_repo / "src" / "__init__.py").write_text("", encoding="utf-8")
    (git_repo / "tests" / "__init__.py").write_text("", encoding="utf-8")

    config = AgentConfig(workspace=str(git_repo))
    orchestrator = Orchestrator(config, DeterministicProvider())

    issue = IssueSpec(
        title="Add multiply function",
        description="Add a utility function multiply(a, b) and tests",
        acceptance_criteria=["function exists", "pytest passes"],
    )

    state, obs = orchestrator.run(issue)

    assert state.status == Lifecycle.COMPLETED
    assert state.completed_steps == ["implementation", "test"]

    # Verify created files exist and contain multiply implementation & tests
    utils_file = git_repo / "src" / "utils.py"
    test_file = git_repo / "tests" / "test_utils.py"
    assert utils_file.exists()
    assert test_file.exists()
    assert "def multiply(a" in utils_file.read_text(encoding="utf-8")
    assert "assert multiply(3, 4) == 12" in test_file.read_text(encoding="utf-8")

    # Verify tests pass when run directly
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp=.pytest_tmp"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Verify git status shows modified/untracked files
    repo = orchestrator.repository
    assert "src/utils.py" in repo.changed_files()
    assert "tests/test_utils.py" in repo.changed_files()

