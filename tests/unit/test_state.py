"""Unit tests for task state persistence and event logging."""

import json
from pathlib import Path

from agent.models import (
    BudgetLedger,
    IssueSpec,
    Lifecycle,
    Plan,
    PlanStep,
    RunEvent,
    StepStatus,
    TaskState,
)
from agent.state.store import StateStore
from agent.telemetry.logger import EventLogger


def test_state_round_trip_with_plan_and_budget(tmp_path: Path):
    """Test full persistence round-trip of TaskState with Plan and BudgetLedger."""
    issue = IssueSpec(
        title="Add feature",
        description="Detailed description",
        acceptance_criteria=["criterion 1", "criterion 2"],
    )
    budget = BudgetLedger(
        max_llm_calls=10,
        max_tool_actions=20,
        llm_calls=3,
        tool_actions=5,
        estimated_cost=0.05,
    )
    plan = Plan(
        issue_id=issue.issue_id,
        steps=[
            PlanStep(
                step_id="step1",
                objective="Create helper",
                expected_files=["src/helper.py"],
                status=StepStatus.COMPLETED,
                attempts=1,
                evidence_refs=["write:src/helper.py"],
            ),
            PlanStep(
                step_id="step2",
                objective="Add tests",
                dependencies=["step1"],
                expected_files=["tests/test_helper.py"],
                status=StepStatus.PENDING,
            ),
        ],
    )
    state = TaskState(
        issue=issue,
        status=Lifecycle.EXECUTING,
        plan=plan,
        completed_steps=["step1"],
        transient={"checkpoint_hash": "abc1234"},
        budget=budget,
    )

    store = StateStore(tmp_path / "state")
    saved_path = store.save(state)
    assert saved_path.exists()

    restored = store.load(state.task_id)

    # Verify task state
    assert restored.task_id == state.task_id
    assert restored.status == Lifecycle.EXECUTING
    assert restored.issue.title == "Add feature"
    assert restored.issue.acceptance_criteria == ["criterion 1", "criterion 2"]
    assert restored.completed_steps == ["step1"]
    assert restored.transient == {"checkpoint_hash": "abc1234"}

    # Verify budget ledger
    assert restored.budget.llm_calls == 3
    assert restored.budget.tool_actions == 5
    assert restored.budget.estimated_cost == 0.05

    # Verify plan and steps
    assert restored.plan is not None
    assert len(restored.plan.steps) == 2
    assert restored.plan.steps[0].step_id == "step1"
    assert restored.plan.steps[0].objective == "Create helper"
    assert restored.plan.steps[0].status == StepStatus.COMPLETED
    assert restored.plan.steps[0].evidence_refs == ["write:src/helper.py"]
    assert restored.plan.steps[1].step_id == "step2"
    assert restored.plan.steps[1].dependencies == ["step1"]
    assert restored.plan.steps[1].status == StepStatus.PENDING


def test_event_logger_append_and_read_all(tmp_path: Path):
    """Test append-only event logging, JSONL formatting, and read_all helper."""
    log_file = tmp_path / "events.jsonl"
    logger = EventLogger(log_file)

    assert logger.read_all() == []

    event1 = RunEvent(task_id="task-1", event_type="task_created", payload={"user": "test"})
    event2 = RunEvent(task_id="task-1", event_type="step_executed", payload={"step": "step1", "success": True})

    logger.append(event1)
    logger.append(event2)

    events = logger.read_all()
    assert len(events) == 2
    assert events[0]["task_id"] == "task-1"
    assert events[0]["event_type"] == "task_created"
    assert events[0]["payload"] == {"user": "test"}
    assert "timestamp" in events[0]

    assert events[1]["task_id"] == "task-1"
    assert events[1]["event_type"] == "step_executed"
    assert events[1]["payload"] == {"step": "step1", "success": True}

    # Verify raw file format: valid JSON lines with newlines
    raw_lines = [line.strip() for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(raw_lines) == 2
    for line in raw_lines:
        parsed = json.loads(line)
        assert "task_id" in parsed
        assert "event_type" in parsed
