from pathlib import Path

from agent.manager import BoundedExecutionContract
from agent.models import AgentConfig, ExecutionRecord, Plan, PlanStep, TaskState, ToolResult
from agent.replanning import ReplanDecisionKind, ReplanningController
from agent.tools.repository import RepositoryAdapter
from agent.verifier import AssessmentStatus, Verifier


def _contract():
    return BoundedExecutionContract("task", 1, "issue", "add value", ["R1"], ["value exists"], [], [])


def _record(repo, *, timed_out=False, content="value = 1\n"):
    result = repo.write_file("out.py", content)
    result.timed_out = timed_out
    return ExecutionRecord("step", ["out.py"], {"type": "write_file", "path": "out.py", "content": content}, None, result)


def test_intermediate_assessment_is_read_only_and_not_durable(tmp_path: Path):
    repo = RepositoryAdapter(tmp_path)
    record = _record(repo)
    assessment = Verifier().assess_execution_record(repo, record)

    assert assessment.status == AssessmentStatus.ACHIEVED
    assert (tmp_path / "out.py").read_text(encoding="utf-8") == "value = 1\n"


def test_ambiguous_timeout_with_achieved_postcondition_does_not_retry(tmp_path: Path):
    repo = RepositoryAdapter(tmp_path)
    record = _record(repo, timed_out=True)
    verifier = Verifier()
    controller = ReplanningController(AgentConfig(workspace=str(tmp_path)), verifier)
    assessment = verifier.assess_execution_record(repo, record)
    outcome = controller.classify(record, assessment)

    assert assessment.status == AssessmentStatus.ACHIEVED
    assert controller.decide(outcome, assessment) == ReplanDecisionKind.CONTINUE


def test_timeout_with_absent_postcondition_retries_once_when_safe(tmp_path: Path):
    repo = RepositoryAdapter(tmp_path)
    record = ExecutionRecord("step", ["out.py"], {"type": "write_file", "path": "out.py", "content": "value = 1\n"}, None,
                             ToolResult("write_file", False, stderr="command timed out", timed_out=True))
    verifier = Verifier()
    controller = ReplanningController(AgentConfig(workspace=str(tmp_path)), verifier)
    assessment = verifier.assess_execution_record(repo, record)
    outcome = controller.classify(record, assessment)

    assert assessment.status == AssessmentStatus.NOT_ACHIEVED
    assert controller.decide(outcome, assessment) == ReplanDecisionKind.RETRY
    assert controller.decide(outcome, assessment) == ReplanDecisionKind.REPLAN


def test_candidate_rejects_completed_or_equivalent_strategy(tmp_path: Path):
    repo = RepositoryAdapter(tmp_path)
    snapshot = repo.inspect()
    contract = _contract()
    old = Plan("issue", [PlanStep("done", "done")])
    old.steps[0].status = old.steps[0].status.COMPLETED
    state = TaskState(issue=contract.to_issue({"title": "x", "description": "x"}), plan=old)
    controller = ReplanningController(AgentConfig(workspace=str(tmp_path)), Verifier())
    candidate = Plan("issue", [PlanStep("done", "done")])

    assert "candidate unnecessarily reintroduces completed subgoals" in controller.validate_candidate(candidate, state, contract, snapshot)

    candidate = Plan("issue", [PlanStep("other", "other", expected_files=["out.py"])])
    controller._seen.add(controller.plan_fingerprint(candidate, contract, snapshot))
    assert "candidate repeats an equivalent strategy in an unchanged environment" in controller.validate_candidate(candidate, state, contract, snapshot)
