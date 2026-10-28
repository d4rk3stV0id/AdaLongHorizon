"""Unit tests for Planner, Action, and Executor."""

import json
from pathlib import Path
from unittest import mock
import pytest

from agent.errors import BudgetExceeded, LLMConnectionError
from agent.executor import Action, Executor
from agent.llm.base import LLMProvider, LLMResponse
from agent.llm.ollama import OllamaProvider
from agent.llm.provider import DeterministicProvider
from agent.models import (
    BudgetLedger,
    IssueSpec,
    ObservationBundle,
    Plan,
    PlanStep,
    RepositorySnapshot,
    StepStatus,
    TaskState,
    ToolResult,
)
from agent.planner import Planner, PlanValidationError
from agent.tools.repository import RepositoryAdapter


class MockProvider(LLMProvider):
    def __init__(self, action_to_return: dict):
        self.action_to_return = action_to_return
        self.received_context = None

    def propose_action(self, issue, step, snapshot, state):
        self.received_context = (issue, step, snapshot, state)
        return self.action_to_return


def test_planner_creates_function_plan():
    planner = Planner()
    issue = IssueSpec("Add normalize", "Please add a function to normalize strings", ["function works"])
    snapshot = RepositorySnapshot("root", [], {}, "", None, "", None)
    state = TaskState(issue)

    plan = planner.create_plan(issue, snapshot, state)
    assert len(plan.steps) == 2
    assert plan.steps[0].step_id == "implementation"
    assert plan.steps[0].objective == "Add the requested function"
    assert plan.steps[1].step_id == "test"
    assert plan.steps[1].dependencies == ["implementation"]
    assert plan.validate() == []


def test_planner_creates_generic_plan():
    planner = Planner()
    issue = IssueSpec("Fix typo", "Correct spelling in documentation")
    snapshot = RepositorySnapshot("root", ["docs/guide.md", "src/core.py"], {}, "", None, "", None)
    state = TaskState(issue)

    plan = planner.create_plan(issue, snapshot, state)
    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == "change"
    assert plan.steps[0].dependencies == []
    assert plan.validate() == []


def test_action_validation():
    # Valid write_file
    act1 = Action.validate({"type": "write_file", "path": "src/app.py", "content": "x = 1"})
    assert act1.type == "write_file"
    assert act1.path == "src/app.py"
    assert act1.content == "x = 1"

    # Valid finish
    act2 = Action.validate({"type": "finish", "message": "All done"})
    assert act2.type == "finish"
    assert act2.message == "All done"

    # Invalid action type
    with pytest.raises(ValueError, match="unknown action type"):
        Action.validate({"type": "delete_file", "path": "src/app.py"})

    # write_file missing path
    with pytest.raises(ValueError, match="requires a non-empty 'path'"):
        Action.validate({"type": "write_file", "path": "", "content": "x"})

    # write_file missing content
    with pytest.raises(ValueError, match="requires a string 'content'"):
        Action.validate({"type": "write_file", "path": "a.py", "content": 123})

    edit = Action.validate({
        "type": "edit_file", "path": "src/app.py",
        "edits": [{"old_text": "before", "new_text": "after"}],
    })
    assert edit.type == "edit_file"
    assert edit.path == "src/app.py"
    assert edit.edits == [{"old_text": "before", "new_text": "after"}]
    with pytest.raises(ValueError, match="requires a non-empty 'old_text'"):
        Action.validate({"type": "edit_file", "path": "a.py", "edits": [{"old_text": "", "new_text": "x"}]})
    with pytest.raises(ValueError, match="requires a string 'new_text'"):
        Action.validate({"type": "edit_file", "path": "a.py", "edits": [{"old_text": "x", "new_text": 1}]})
    with pytest.raises(ValueError, match="requires a non-empty 'edits' list"):
        Action.validate({"type": "edit_file", "path": "a.py", "edits": []})
    with pytest.raises(ValueError, match="'edits' must be a list"):
        Action.validate({"type": "edit_file", "path": "a.py", "edits": "bad"})


def test_step_eligibility_resolution():
    issue = IssueSpec("Test", "Test desc")
    state = TaskState(issue)
    step1 = PlanStep(step_id="step1", objective="First")
    step2 = PlanStep(step_id="step2", objective="Second", dependencies=["step1"])
    step3 = PlanStep(step_id="step3", objective="Third", dependencies=["step2"])
    state.plan = Plan(issue.issue_id, [step1, step2, step3])

    repo = None  # Not needed for eligibility
    executor = Executor(repo, DeterministicProvider())

    # Initially step1 is eligible
    eligible = executor.eligible_step(state)
    assert eligible is not None
    assert eligible.step_id == "step1"

    # After step1 is completed
    step1.status = StepStatus.COMPLETED
    state.completed_steps.append("step1")
    eligible = executor.eligible_step(state)
    assert eligible is not None
    assert eligible.step_id == "step2"

    # After step2 is completed
    step2.status = StepStatus.COMPLETED
    state.completed_steps.append("step2")
    eligible = executor.eligible_step(state)
    assert eligible is not None
    assert eligible.step_id == "step3"

    # After all are completed
    step3.status = StepStatus.COMPLETED
    state.completed_steps.append("step3")
    assert executor.eligible_step(state) is None


def test_executor_context_bounding(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    repo = RepositoryAdapter(tmp_path)

    # Create oversized inputs
    many_files = [f"file_{i}.py" for i in range(200)]
    huge_instructions = {f"INST_{i}.md": "x" * 12000 for i in range(12)}
    snapshot = RepositorySnapshot(
        root=str(tmp_path),
        files=many_files,
        instructions=huge_instructions,
        git_status="status" * 500,
        baseline_commit="abc",
        baseline_status="status" * 500,
        test_command=None,
    )

    issue = IssueSpec("Long title " * 100, "Long desc " * 1000, ["Criterion " * 100] * 30)
    state = TaskState(issue, budget=BudgetLedger(max_token_estimate=100000))
    state.plan = Plan(issue.issue_id, [PlanStep("step1", "Do something", expected_files=["src/out.py"])])

    mock_prov = MockProvider({"type": "write_file", "path": "src/out.py", "content": "# out"})
    executor = Executor(repo, mock_prov)
    bundle = ObservationBundle(state.task_id)

    executed = executor.execute_next(state, snapshot, bundle)
    assert executed is True

    # Check bounded context passed to provider
    bound_issue, bound_step, bound_snap, _ = mock_prov.received_context
    assert len(bound_snap.files) == 100
    assert len(bound_snap.instructions) <= 8
    for v in bound_snap.instructions.values():
        assert len(v) <= 8000
    assert len(bound_snap.git_status) <= 2000
    assert len(bound_issue.title) <= 500
    assert len(bound_issue.description) <= 4000
    assert len(bound_issue.acceptance_criteria) <= 20


def test_executor_includes_only_existing_expected_file_contents(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "utils.py").write_text('"""Keep me."""\n', encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("secret = True\n", encoding="utf-8")
    repo = RepositoryAdapter(tmp_path)
    issue = IssueSpec("Add multiply", "Add multiply(a, b).")
    state = TaskState(issue, budget=BudgetLedger(max_token_estimate=100000))
    state.plan = Plan(issue.issue_id, [PlanStep("step1", "Add multiply", expected_files=["src/utils.py", "new.py"])])
    provider = MockProvider({"type": "finish", "message": "done"})

    Executor(repo, provider).execute_next(state, repo.inspect(), ObservationBundle(state.task_id))

    snapshot = provider.received_context[2]
    assert snapshot.target_file_contents == {"src/utils.py": '"""Keep me."""\n'}
    assert "unrelated.py" not in snapshot.target_file_contents


def test_executor_edit_file_preserves_source_and_existing_test_content(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    source = '"""Small numeric utility functions."""\n'
    test_docstring = '"""Tests for numeric utilities."""'
    existing_test = "def test_existing_behavior():\n    assert 2 + 2 == 4"
    existing_tests = test_docstring + "\n\n" + existing_test + "\n"
    (tmp_path / "src" / "utils.py").write_text(source, encoding="utf-8")
    (tmp_path / "tests" / "test_utils.py").write_text(existing_tests, encoding="utf-8")
    repo = RepositoryAdapter(tmp_path)
    issue = IssueSpec("Add multiply", "Add multiply(a, b) and a unit test.")
    state = TaskState(issue, budget=BudgetLedger(max_token_estimate=100000))
    state.plan = Plan(issue.issue_id, [PlanStep(
        "step1", "Add multiply", expected_files=["src/utils.py"], action_type="edit_file",
    )])
    action = {
        "type": "edit_file", "path": "src/utils.py", "edits": [{
            "old_text": source,
            "new_text": source + "\ndef multiply(a, b):\n    return a * b\n",
        }],
    }
    observations = ObservationBundle(state.task_id)

    assert Executor(repo, MockProvider(action)).execute_next(state, repo.inspect(), observations)
    assert (tmp_path / "src" / "utils.py").read_text(encoding="utf-8") == (
        '"""Small numeric utility functions."""\n\n'
        "def multiply(a, b):\n    return a * b\n"
    )
    assert '"""Small numeric utility functions."""' in (tmp_path / "src" / "utils.py").read_text(encoding="utf-8")

    test_action = {
        "type": "edit_file", "path": "tests/test_utils.py", "edits": [
            {
                "old_text": test_docstring,
                "new_text": test_docstring + "\n\nfrom src.utils import multiply",
            },
            {
                "old_text": existing_test,
                "new_text": existing_test + "\n\n\ndef test_multiply():\n    assert multiply(3, 4) == 12",
            },
        ],
    }
    state.plan.steps[0].status = StepStatus.PENDING
    state.completed_steps.clear()
    state.plan.steps[0].expected_files = ["tests/test_utils.py"]
    observations = ObservationBundle(state.task_id)
    assert Executor(repo, MockProvider(test_action)).execute_next(state, repo.inspect(), observations)
    result_test_content = (tmp_path / "tests" / "test_utils.py").read_text(encoding="utf-8")
    assert '"""Tests for numeric utilities."""' in result_test_content
    assert "from src.utils import multiply" in result_test_content
    assert "def test_existing_behavior():" in result_test_content
    assert "def test_multiply():" in result_test_content


def test_executor_rejects_action_that_differs_from_planned_action_type(tmp_path: Path):
    target = tmp_path / "src.py"
    target.write_text("before\n", encoding="utf-8")
    repo = RepositoryAdapter(tmp_path)
    issue = IssueSpec("Edit", "Make a localized edit")
    step = PlanStep("step1", "Edit src.py", expected_files=["src.py"], action_type="write_file")
    state = TaskState(issue, plan=Plan(issue.issue_id, [step]))
    observations = ObservationBundle(state.task_id)
    provider = MockProvider({
        "type": "edit_file", "path": "src.py",
        "edits": [{"old_text": "before", "new_text": "after"}],
    })

    with pytest.raises(ValueError, match="does not match planned action_type"):
        Executor(repo, provider).execute_next(state, repo.inspect(), observations)

    assert target.read_text(encoding="utf-8") == "before\n"
    assert observations.results == []


def test_executor_records_failed_ambiguous_edit_without_modifying_file(tmp_path: Path):
    target = tmp_path / "src.py"
    target.write_text("same\nsame\n", encoding="utf-8")
    repo = RepositoryAdapter(tmp_path)
    issue = IssueSpec("Edit", "Replace a unique substring")
    state = TaskState(issue, budget=BudgetLedger(max_token_estimate=100000))
    state.plan = Plan(issue.issue_id, [PlanStep(
        "step1", "Edit", expected_files=["src.py"], action_type="edit_file",
    )])
    observations = ObservationBundle(state.task_id)
    action = {"type": "edit_file", "path": "src.py", "edits": [{
        "old_text": "same", "new_text": "changed",
    }]}

    assert Executor(repo, MockProvider(action)).execute_next(state, repo.inspect(), observations)

    assert target.read_text(encoding="utf-8") == "same\nsame\n"
    assert state.plan.steps[0].status == StepStatus.FAILED
    assert observations.results[-1].operation == "edit_file"
    assert not observations.results[-1].success
    assert "multiple times" in observations.results[-1].stderr


def test_executor_finish_action_skips_step(tmp_path: Path):
    repo = RepositoryAdapter(tmp_path)
    state = TaskState(IssueSpec("Finish test", "Desc"))
    step = PlanStep("step1", "Objective")
    state.plan = Plan(state.issue.issue_id, [step])
    snapshot = repo.inspect()

    mock_prov = MockProvider({"type": "finish", "message": "Nothing to do"})
    executor = Executor(repo, mock_prov)
    bundle = ObservationBundle(state.task_id)

    res = executor.execute_next(state, snapshot, bundle)
    assert res is False
    assert step.status == StepStatus.SKIPPED
    assert state.transient.get("provider_message") == "Nothing to do"


def test_executor_acceptance_check_evaluates_expected_files(tmp_path: Path):
    repo = RepositoryAdapter(tmp_path)
    state = TaskState(IssueSpec("Acceptance test", "Desc"))
    # Expects out.py, but provider writes other.py
    step = PlanStep("step1", "Objective", expected_files=["out.py"])
    state.plan = Plan(state.issue.issue_id, [step])
    snapshot = repo.inspect()

    mock_prov = MockProvider({"type": "write_file", "path": "wrong.py", "content": "# wrong"})
    executor = Executor(repo, mock_prov)
    bundle = ObservationBundle(state.task_id)

    res = executor.execute_next(state, snapshot, bundle)
    assert res is True
    # Step failed acceptance check because wrong file was written
    assert step.status == StepStatus.FAILED
    assert "step1" not in state.completed_steps


def _mock_ollama_http(payload):
    response = mock.MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    context = mock.MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = None
    return context


def test_executor_consumes_ollama_response_validates_and_executes(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    repository = RepositoryAdapter(tmp_path)
    issue = IssueSpec(
        "Add multiply", "Add multiply(a, b) to src/utils.py", ["returns the product"],
    )
    step = PlanStep("implementation", "Write multiply(a, b)", expected_files=["src/utils.py"])
    state = TaskState(issue, plan=Plan(issue.issue_id, [step]))
    provider = OllamaProvider(model="qwen3:8b")
    ollama_response = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": json.dumps({
            "type": "write_file",
            "path": "src/utils.py",
            "content": "def multiply(a, b):\n    return a * b\n",
        })},
        "prompt_eval_count": 42,
        "eval_count": 15,
        "total_duration": 800_000_000,
        "done": True,
        "done_reason": "stop",
    }
    with mock.patch("urllib.request.urlopen", return_value=_mock_ollama_http(ollama_response)):
        executed = Executor(repository, provider).execute_next(
            state, repository.inspect(), ObservationBundle(state.task_id),
        )

    assert executed is True
    assert step.status == StepStatus.COMPLETED
    assert (tmp_path / "src" / "utils.py").read_text(encoding="utf-8") == (
        "def multiply(a, b):\n    return a * b\n"
    )
    assert state.transient["last_action"]["type"] == "write_file"
    assert state.transient["last_llm_response"] == {
        "provider": "ollama", "model": "qwen3:8b", "prompt_tokens": 42,
        "output_tokens": 15, "total_tokens": 57, "duration_ms": 800.0,
        "done": True, "done_reason": "stop", "thinking_present": False,
        "usage_available": True,
    }
    assert state.budget.llm_calls == 1
    assert state.budget.actual_tokens == 57
    assert state.budget.estimated_tokens == 0


def test_executor_rejects_invalid_ollama_action(tmp_path: Path):
    repository = RepositoryAdapter(tmp_path)
    issue = IssueSpec("Bad action", "A task")
    step = PlanStep("step", "Make a change")
    state = TaskState(issue, plan=Plan(issue.issue_id, [step]))
    provider = OllamaProvider()
    invalid = LLMResponse(content={"type": "delete_file", "path": "x"})
    with mock.patch.object(provider, "propose_action", return_value=invalid):
        with pytest.raises(ValueError, match="unknown action type"):
            Executor(repository, provider).execute_next(state, repository.inspect(), ObservationBundle(state.task_id))


def test_executor_propagates_ollama_provider_failure(tmp_path: Path):
    repository = RepositoryAdapter(tmp_path)
    issue = IssueSpec("Provider failure", "A task")
    step = PlanStep("step", "Make a change")
    state = TaskState(issue, plan=Plan(issue.issue_id, [step]))
    provider = OllamaProvider()
    with mock.patch.object(provider, "propose_action", side_effect=LLMConnectionError("offline")):
        with pytest.raises(LLMConnectionError, match="offline"):
            Executor(repository, provider).execute_next(state, repository.inspect(), ObservationBundle(state.task_id))


def test_executor_estimates_only_when_usage_metadata_is_unavailable(tmp_path: Path):
    repository = RepositoryAdapter(tmp_path)
    issue = IssueSpec("Estimate usage", "A task")
    step = PlanStep("step", "Make a change")
    state = TaskState(issue, plan=Plan(issue.issue_id, [step]))
    provider = OllamaProvider()
    response = LLMResponse(content={
        "type": "write_file", "path": "out.py", "content": "value = 1\n",
    })
    with mock.patch.object(provider, "propose_action", return_value=response):
        assert Executor(repository, provider).execute_next(
            state, repository.inspect(), ObservationBundle(state.task_id),
        ) is True
    assert state.budget.actual_tokens == 0
    assert state.budget.estimated_tokens > 0
    assert state.transient["last_llm_usage"]["usage_available"] is False


def test_executor_budget_rejection_prevents_ollama_call(tmp_path: Path):
    repository = RepositoryAdapter(tmp_path)
    issue = IssueSpec("Budget", "A task")
    step = PlanStep("step", "Make a change")
    state = TaskState(issue, plan=Plan(issue.issue_id, [step]), budget=BudgetLedger(max_llm_calls=0))
    provider = OllamaProvider()
    with mock.patch.object(provider, "propose_action") as propose:
        with pytest.raises(BudgetExceeded, match="llm budget exhausted"):
            Executor(repository, provider).execute_next(state, repository.inspect(), ObservationBundle(state.task_id))
    propose.assert_not_called()
