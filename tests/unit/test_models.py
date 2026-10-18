"""Unit tests for domain models, lifecycle state machine, and budget ledger."""

from datetime import datetime, timedelta, timezone
import pytest

from agent.errors import BudgetExceeded
from agent.models import (
    ALLOWED_TRANSITIONS,
    BudgetLedger,
    IssueSpec,
    Lifecycle,
    ObservationBundle,
    Plan,
    PlanStep,
    RunEvent,
    StepStatus,
    TaskState,
    ToolResult,
)


def test_valid_lifecycle_transitions():
    """Verify standard happy-path and branching transitions."""
    state = TaskState(IssueSpec("Test issue", "Description"))
    assert state.status == Lifecycle.CREATED

    # Normal progression
    state.transition(Lifecycle.INSPECTING)
    assert state.status == Lifecycle.INSPECTING

    state.transition(Lifecycle.PLANNING)
    assert state.status == Lifecycle.PLANNING

    state.transition(Lifecycle.EXECUTING)
    assert state.status == Lifecycle.EXECUTING

    state.transition(Lifecycle.VERIFYING)
    assert state.status == Lifecycle.VERIFYING

    state.transition(Lifecycle.COMPLETED)
    assert state.status == Lifecycle.COMPLETED


def test_recovering_and_blocked_transitions():
    """Verify recovery and blocked paths through lifecycle."""
    state = TaskState(IssueSpec("Recovery test", "Description"))
    state.transition(Lifecycle.INSPECTING)
    state.transition(Lifecycle.PLANNING)
    state.transition(Lifecycle.EXECUTING)

    # Transition to RECOVERING from EXECUTING
    state.transition(Lifecycle.RECOVERING)
    assert state.status == Lifecycle.RECOVERING

    # From RECOVERING back to PLANNING
    state.transition(Lifecycle.PLANNING)
    state.transition(Lifecycle.EXECUTING)
    state.transition(Lifecycle.VERIFYING)

    # From VERIFYING to BLOCKED
    state.transition(Lifecycle.BLOCKED)
    assert state.status == Lifecycle.BLOCKED


def test_invalid_lifecycle_transitions():
    """Verify that illegal transitions raise ValueError."""
    state = TaskState(IssueSpec("Invalid test", "Description"))

    # Direct to COMPLETED from CREATED is disallowed
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        state.transition(Lifecycle.COMPLETED)

    # Direct to EXECUTING from CREATED is disallowed
    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        state.transition(Lifecycle.EXECUTING)

    # Once COMPLETED, no transitions allowed
    state.transition(Lifecycle.INSPECTING)
    state.transition(Lifecycle.PLANNING)
    state.transition(Lifecycle.EXECUTING)
    state.transition(Lifecycle.VERIFYING)
    state.transition(Lifecycle.COMPLETED)

    for target in Lifecycle:
        with pytest.raises(ValueError, match="invalid lifecycle transition"):
            state.transition(target)


def test_terminal_states_have_no_outbound_transitions():
    """COMPLETED, BLOCKED, FAILED, and BUDGET_EXHAUSTED must be sinks."""
    for terminal in (Lifecycle.COMPLETED, Lifecycle.BLOCKED, Lifecycle.FAILED, Lifecycle.BUDGET_EXHAUSTED):
        assert len(ALLOWED_TRANSITIONS[terminal]) == 0


def test_budget_llm_calls_limit():
    ledger = BudgetLedger(max_llm_calls=2)
    ledger.record("llm", 1)
    ledger.record("llm", 1)
    assert ledger.llm_calls == 2
    with pytest.raises(BudgetExceeded, match="llm budget exhausted"):
        ledger.record("llm", 1)


def test_budget_tool_actions_limit():
    ledger = BudgetLedger(max_tool_actions=1)
    ledger.record("tool", 1)
    assert ledger.tool_actions == 1
    with pytest.raises(BudgetExceeded, match="tool budget exhausted"):
        ledger.record("tool", 1)


def test_budget_retry_limit():
    ledger = BudgetLedger(max_retries=2)
    ledger.record("retry", 1)
    ledger.record("retry", 1)
    assert ledger.retries == 2
    with pytest.raises(BudgetExceeded, match="retry budget exhausted"):
        ledger.record("retry", 1)


def test_budget_token_limit():
    ledger = BudgetLedger(max_token_estimate=500)
    ledger.record("tokens", 400)
    assert ledger.estimated_tokens == 400
    with pytest.raises(BudgetExceeded, match="tokens budget exhausted"):
        ledger.record("tokens", 150)


def test_budget_duration_limit():
    past = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
    ledger = BudgetLedger(max_task_duration_seconds=500.0, started_at=past)
    with pytest.raises(BudgetExceeded, match="task duration budget exhausted"):
        ledger.check("llm")


def test_budget_cost_limit():
    ledger = BudgetLedger(max_cost_estimate=5.0)
    ledger.record_cost(3.5)
    assert ledger.estimated_cost == 3.5

    with pytest.raises(BudgetExceeded, match="estimated cost budget exhausted"):
        ledger.record_cost(2.0)

    with pytest.raises(ValueError, match="cost cannot be negative"):
        ledger.record_cost(-1.0)


def test_budget_consecutive_identical_failures():
    ledger = BudgetLedger(max_consecutive_identical_failures=2)

    # First failure
    ledger.record_failure("error_type_a")
    assert ledger.consecutive_failures == 1
    assert ledger.last_failure_signature == "error_type_a"

    # Second identical failure
    ledger.record_failure("error_type_a")
    assert ledger.consecutive_failures == 2

    # Third identical failure exceeds limit
    with pytest.raises(BudgetExceeded, match="consecutive identical failure"):
        ledger.record_failure("error_type_a")

    # Different failure resets counter to 1
    ledger.consecutive_failures = 1
    ledger.record_failure("error_type_b")
    assert ledger.consecutive_failures == 1
    assert ledger.last_failure_signature == "error_type_b"

    # Clearing failures
    ledger.clear_failures()
    assert ledger.consecutive_failures == 0
    assert ledger.last_failure_signature is None


def test_budget_unknown_kind():
    ledger = BudgetLedger()
    with pytest.raises(ValueError, match="unknown budget kind"):
        ledger.check("magic")


def test_plan_structural_validation():
    # Valid plan
    step1 = PlanStep(step_id="step1", objective="Do step 1")
    step2 = PlanStep(step_id="step2", objective="Do step 2", dependencies=["step1"])
    plan = Plan(issue_id="issue-1", steps=[step1, step2])
    assert plan.validate() == []

    # Empty plan
    empty_plan = Plan(issue_id="issue-1", steps=[])
    errors = empty_plan.validate()
    assert any("no steps" in e for e in errors)

    # Duplicate step_ids
    dup_plan = Plan(
        issue_id="issue-1",
        steps=[PlanStep("s1", "first"), PlanStep("s1", "duplicate")],
    )
    assert any("duplicate" in e for e in dup_plan.validate())

    # Missing step_id or objective
    missing_plan = Plan(
        issue_id="issue-1",
        steps=[PlanStep("", "objective"), PlanStep("s2", "")],
    )
    assert any("missing step_id or objective" in e for e in missing_plan.validate())

    # Unknown dependency
    bad_dep_plan = Plan(
        issue_id="issue-1",
        steps=[PlanStep("s1", "objective", dependencies=["nonexistent"])],
    )
    assert any("depends on unknown step" in e for e in bad_dep_plan.validate())
