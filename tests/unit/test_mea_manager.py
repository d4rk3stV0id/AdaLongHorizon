from agent.manager import ControlDecisionKind, MEAManager
from agent.models import BudgetLedger, IssueSpec
from agent.state.progress import RoundTrace, ProgressStatus, RequirementStatus, create_progress_state


def _progress():
    return create_progress_state(
        "task-1", IssueSpec("Add utilities", "Implement two small utilities", ["first exists", "second exists"]),
    )


def test_manager_emits_execute_with_a_bounded_contract_for_pending_work():
    progress = _progress()
    progress.facts = [f"fact-{index}" for index in range(30)]
    progress.artifacts = [f"src/file-{index}.py" for index in range(30)]

    decision = MEAManager().decide(progress, BudgetLedger(), max_rounds=4)

    assert decision.kind == ControlDecisionKind.EXECUTE
    assert decision.contract is not None
    assert decision.contract.requirement_ids == ["R1"]
    assert decision.contract.acceptance_criteria == ["first exists"]
    assert len(decision.contract.verified_facts) == 20
    assert len(decision.contract.verified_artifacts) == 20
    assert "verifier checks" in decision.contract.expected_verification_condition


def test_manager_returns_done_without_a_contract_when_all_requirements_are_verified():
    progress = _progress()
    for requirement in progress.requirements:
        requirement.status = RequirementStatus.COMPLETED
    progress.status = ProgressStatus.COMPLETED

    decision = MEAManager().decide(progress, BudgetLedger(), max_rounds=4)

    assert decision.kind == ControlDecisionKind.DONE
    assert decision.contract is None


def test_manager_returns_blocked_when_no_pending_requirement_is_eligible():
    progress = _progress()
    for requirement in progress.requirements:
        requirement.status = RequirementStatus.BLOCKED

    decision = MEAManager().decide(progress, BudgetLedger(), max_rounds=4)

    assert decision.kind == ControlDecisionKind.BLOCKED
    assert "no eligible pending" in decision.reason


def test_manager_returns_blocked_for_round_or_budget_boundaries():
    progress = _progress()
    progress.rounds.append(RoundTrace("task-1", 1))

    round_limited = MEAManager().decide(progress, BudgetLedger(), max_rounds=1)
    budget_limited = MEAManager().decide(_progress(), BudgetLedger(max_llm_calls=0), max_rounds=4)

    assert round_limited.kind == ControlDecisionKind.BLOCKED
    assert "round limit" in round_limited.reason
    assert budget_limited.kind == ControlDecisionKind.BLOCKED
    assert "budget boundary" in budget_limited.reason
