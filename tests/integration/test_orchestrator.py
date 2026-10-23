"""Integration tests for the Orchestrator end-to-end task execution."""

import json
from pathlib import Path
import subprocess
from unittest import mock

from agent.llm.base import LLMProvider
from agent.llm.provider import DeterministicProvider
from agent.llm.base import LLMResponse
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
    assert state.transient["verification_result"]["status"] == "verified"
    assert state.transient["verification_result"]["verified"] is True

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
    assert state.transient["verification_result"]["status"] == "rejected"


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
    assert state.transient["verification_result"]["verified"] is True

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


def test_orchestrator_blocks_when_independent_verifier_rejects_repository_state(git_repo: Path):
    (git_repo / "src").mkdir()
    (git_repo / "tests").mkdir()
    (git_repo / "src" / "app.py").write_text("original = True\n", encoding="utf-8")
    (git_repo / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    (git_repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\npythonpath = ['.']\n",
        encoding="utf-8",
    )

    class WriteProvider:
        def propose_action(self, issue, step, snapshot, state):
            return {"type": "write_file", "path": "src/app.py", "content": "value = 1\n"}

    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo)), WriteProvider())
    write_file = orchestrator.repository.write_file

    def write_incorrect_state(path, content):
        result = write_file(path, content)
        (git_repo / path).write_text("value = 2\n", encoding="utf-8")
        return result

    with mock.patch.object(orchestrator.repository, "write_file", side_effect=write_incorrect_state):
        state, observations = orchestrator.run(IssueSpec(
            "Set a value", "Set the value in src/app.py", ["the value is written"],
        ))

    assert observations.results[-1].operation == "run_command"
    assert observations.results[-1].success
    assert state.status == Lifecycle.BLOCKED
    assert state.transient["verification_result"]["status"] == "rejected"
    assert "repository content" in " ".join(state.transient["verification_result"]["failures"])


class StructuredPlanAndActionProvider:
    def __init__(self):
        self.plan_calls = 0
        self.action_calls = 0

    def propose_plan(self, issue, snapshot, state, budget, response_schema):
        self.plan_calls += 1
        return LLMResponse(
            content={"steps": [
                {
                    "step_id": "implementation", "objective": "Implement multiply(a, b)",
                    "dependencies": [], "expected_files": ["src/utils.py"],
                    "action_type": "write_file", "acceptance_checks": ["product is returned"],
                    "risk_level": "low", "estimated_cost": 1,
                },
                {
                    "step_id": "test", "objective": "Add unit tests for multiply(a, b)",
                    "dependencies": ["implementation"], "expected_files": ["tests/test_utils.py"],
                    "action_type": "write_file", "acceptance_checks": ["pytest passes"],
                    "risk_level": "low", "estimated_cost": 1,
                },
            ]},
            prompt_tokens=11, output_tokens=17, total_tokens=28,
            duration_ms=125.0, model="mock-qwen", provider="mock-ollama", usage_available=True,
        )

    def propose_action(self, issue, step, snapshot, state):
        self.action_calls += 1
        if step.step_id == "implementation":
            action = {
                "type": "write_file", "path": "src/utils.py",
                "content": "def multiply(a, b):\n    return a * b\n",
            }
            prompt_tokens, output_tokens = 5, 7
        else:
            action = {
                "type": "write_file", "path": "tests/test_utils.py",
                "content": (
                    "from src.utils import multiply\n\n\n"
                    "def test_multiply():\n"
                    "    assert multiply(3, 4) == 12\n"
                    "    assert multiply(-2, 3) == -6\n"
                ),
            }
            prompt_tokens, output_tokens = 6, 8
        return LLMResponse(
            content=action, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens, duration_ms=75.0,
            model="mock-qwen", provider="mock-ollama", usage_available=True,
        )


def test_orchestrator_uses_provider_for_plan_and_records_actual_usage(git_repo: Path):
    (git_repo / "src").mkdir()
    (git_repo / "src" / "__init__.py").write_text("", encoding="utf-8")
    (git_repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\npythonpath = ['.']\n",
        encoding="utf-8",
    )
    provider = StructuredPlanAndActionProvider()
    config = AgentConfig(workspace=str(git_repo), provider="ollama")
    orchestrator = Orchestrator(config, provider)
    issue = IssueSpec(
        "Add multiply", "Add multiply(a, b) and a unit test",
        ["returns the product of its arguments", "pytest passes"],
    )

    state, observations = orchestrator.run(issue)

    assert state.status == Lifecycle.COMPLETED, state.transient.get("error")
    assert orchestrator.planner.provider is provider
    assert provider.plan_calls == 1
    assert provider.action_calls == 2
    assert state.budget.llm_calls == 3
    assert state.budget.actual_input_tokens == 22
    assert state.budget.actual_output_tokens == 32
    assert state.budget.actual_tokens == 54
    assert state.budget.estimated_tokens == 0
    assert [item["purpose"] for item in state.transient["llm_usage"]] == [
        "planning", "action", "action",
    ]
    assert observations.results[-1].success is True
    assert state.transient["verification_result"]["status"] == "verified"

    events = [json.loads(line) for line in (
        git_repo / ".agent-state" / "events.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    plan_event = next(event for event in events if event["event_type"] == "plan_created")
    assert plan_event["payload"]["llm_usage"]["purpose"] == "planning"
    assert plan_event["payload"]["llm_usage"]["plan_valid"] is True
    step_events = [event for event in events if event["event_type"] == "step_executed"]
    assert all(event["payload"]["llm_usage"]["purpose"] == "action" for event in step_events)


def test_ollama_planning_failure_does_not_fall_back_to_deterministic(git_repo: Path):
    from agent.errors import LLMConnectionError

    class OfflineProvider:
        action_calls = 0

        def propose_plan(self, issue, snapshot, state, budget, response_schema):
            raise LLMConnectionError("Ollama is offline")

        def propose_action(self, issue, step, snapshot, state):
            self.action_calls += 1
            return {"type": "finish", "message": "unexpected fallback"}

    provider = OfflineProvider()
    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo), provider="ollama"), provider)
    state, _ = orchestrator.run(IssueSpec("Add multiply", "Add a multiply function"))

    assert state.status == Lifecycle.FAILED
    assert state.plan is None
    assert "Ollama is offline" in state.transient["error"]
    assert provider.action_calls == 0
    assert state.transient["last_llm_usage"]["success"] is False

