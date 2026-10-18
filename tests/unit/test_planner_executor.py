"""Unit tests for Planner, Action, and Executor."""

from pathlib import Path
import pytest

from agent.executor import Action, Executor
from agent.llm.base import LLMProvider
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
