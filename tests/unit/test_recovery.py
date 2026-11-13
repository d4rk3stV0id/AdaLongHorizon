"""M5 persistence and reconciliation tests without durable requirement updates.

Covers all M5-E scenarios:
  1.  Interruption before intent persistence (integration — see test_recovery_integration.py)
  2.  Interruption after intent persistence, before mutation
  3.  Interruption during mutation (ambiguous outcome recorded)
  4.  Successful mutation with missing journal outcome
  5.  Interruption after outcome but before checkpoint
  6.  Corrupt latest checkpoint with valid previous checkpoint
  7.  Corrupt all checkpoints — no valid fallback
  8.  Truncated and malformed journal records
  9.  Journal replay after checkpoint fallback
  10. Relevant external artifact changes detected
  11. Unrelated repository changes do not block recovery
  12. Replanning-active recovery — replanning_state restored from checkpoint
  13. Budget preservation through recovery
  14. Multiple reconciliation entries appended without corrupting sequence
  15. Preservation of previously-verified step progress (completed_steps)
  16. Recovery returns state requiring normal verification (no requirement auto-complete)
"""

from pathlib import Path

import pytest

from agent.models import (
    BudgetLedger, IssueSpec, JournalIntent, JournalOutcome, Lifecycle,
    Plan, PlanStep, RecoveryCheckpoint, StepStatus, TaskState,
)
from agent.recovery import RecoveryBlocked, reconcile_recovery
from agent.state.store import StateStore
from agent.tools.repository import RepositoryAdapter
from agent.verifier import Verifier


def _state(task_id="task", extra_steps=None):
    issue = IssueSpec("write", "write value", ["value exists"], issue_id="issue")
    steps = [PlanStep("step", "write", expected_files=["out.py"])]
    if extra_steps:
        steps.extend(extra_steps)
    return TaskState(issue, plan=Plan("issue", steps), task_id=task_id)


def _checkpoint(state, journal_position=0, replanning_state=None):
    return RecoveryCheckpoint(
        1, state.task_id, "repo", journal_position, state.to_dict(),
        {"commit": None, "status": ""},
        replanning_state=replanning_state or {},
    )


def test_corrupt_latest_checkpoint_falls_back_to_previous(tmp_path: Path):
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state))
    state.plan.version = 2
    store.save_recovery_checkpoint(_checkpoint(state))
    store._checkpoint_path("task").write_text("not-json", encoding="utf-8")

    loaded, fallback = store.load_recovery_checkpoint("task", "repo")

    assert fallback is True
    assert loaded.runtime_state["plan"]["version"] == 1


def test_no_valid_checkpoint_blocks_recovery(tmp_path: Path):
    store = StateStore(tmp_path)
    store._checkpoint_path("task").write_text("bad", encoding="utf-8")

    with pytest.raises(ValueError, match="no valid recovery checkpoint"):
        store.load_recovery_checkpoint("task", "repo")


def test_truncated_final_journal_tail_keeps_valid_prefix(tmp_path: Path):
    store = StateStore(tmp_path)
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", "plan", 1, "step", {}, [], None))
    store.journal_path("task").open("ab").write(b'{"sequence": 2')

    records = store.read_journal("task", "repo")

    assert [item["sequence"] for item in records] == [1]


def test_sequence_or_identity_break_blocks_journal(tmp_path: Path):
    store = StateStore(tmp_path)
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", "plan", 1, "step", {}, [], None))
    store.append_journal("task", JournalOutcome(2, "a", "task", "wrong", "succeeded", None))

    with pytest.raises(ValueError, match="integrity"):
        store.read_journal("task", "repo")


def test_checkpoint_replays_only_suffix_but_validates_entire_journal(tmp_path: Path):
    repo, store = RepositoryAdapter(tmp_path), StateStore(tmp_path)
    state = _state(extra_steps=[PlanStep("step2", "write second", expected_files=["out2.py"])])
    plan_id = state.plan.plan_id
    # The checkpoint already accounts for the completed prefix action.
    state.plan.steps[0].status = StepStatus.COMPLETED
    state.completed_steps = ["step"]
    store.save_recovery_checkpoint(_checkpoint(state, 2))
    prefix = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a1", "task", "repo", plan_id, 1, "step", prefix, ["out.py"], None))
    store.append_journal("task", JournalOutcome(2, "a1", "task", "repo", "succeeded", {"success": True}))
    suffix = {"type": "write_file", "path": "out2.py", "content": "value = 2\n"}
    store.append_journal("task", JournalIntent(3, "a2", "task", "repo", plan_id, 1, "step2", suffix, ["out2.py"], None))
    repo.write_file("out2.py", "value = 2\n")

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.transient["recovery"]["replayed_entries"] == 1
    assert recovered.transient["recovery"]["journal_entries"] == 3
    assert recovered.completed_steps == ["step", "step2"]

    # Even a record in the checkpointed prefix is integrity checked.
    path = store.journal_path("task")
    content = path.read_text(encoding="utf-8").replace('"sequence": 1', '"sequence": 7', 1)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(RecoveryBlocked, match="journal/checkpoint position conflict"):
        reconcile_recovery(store, Verifier(), repo, "task", "repo")


def test_fallback_checkpoint_replays_from_its_own_position(tmp_path: Path):
    repo, store = RepositoryAdapter(tmp_path), StateStore(tmp_path)
    state = _state(extra_steps=[PlanStep("step2", "write second", expected_files=["out2.py"])])
    plan_id = state.plan.plan_id
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    state.plan.version = 2
    store.save_recovery_checkpoint(_checkpoint(state, 2))
    store._checkpoint_path("task").write_text("broken", encoding="utf-8")
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a1", "task", "repo", plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")
    store.append_journal("task", JournalOutcome(2, "a1", "task", "repo", "succeeded", {"success": True}))
    second = {"type": "write_file", "path": "out2.py", "content": "value = 2\n"}
    store.append_journal("task", JournalIntent(3, "a2", "task", "repo", plan_id, 1, "step2", second, ["out2.py"], None))
    repo.write_file("out2.py", "value = 2\n")

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.transient["recovery"]["used_fallback"] is True
    assert recovered.transient["recovery"]["replayed_entries"] == 3
    assert recovered.completed_steps == ["step", "step2"]


def test_recovery_retry_after_outcome_and_checkpoint_write_interruptions(tmp_path: Path):
    from agent.models import JournalOutcome
    from agent.recovery import checkpoint_runtime

    repo, store = RepositoryAdapter(tmp_path), StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")

    append = store.append_journal
    failed = [False]
    def fail_once(task_id, record):
        if isinstance(record, JournalOutcome) and not failed[0]:
            failed[0] = True
            raise OSError("interrupted while persisting reconciliation outcome")
        return append(task_id, record)
    store.append_journal = fail_once
    with pytest.raises(OSError, match="reconciliation outcome"):
        reconcile_recovery(store, Verifier(), repo, "task", "repo")
    store.append_journal = append

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")
    assert recovered.completed_steps == ["step"]
    # Simulate interruption while compacting recovered knowledge into a checkpoint.
    atomic = store._atomic_json
    store._atomic_json = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("checkpoint interrupted"))
    with pytest.raises(OSError, match="checkpoint interrupted"):
        checkpoint_runtime(store, recovered, "repo", {}, len(store.read_journal("task", "repo")))
    store._atomic_json = atomic

    recovered_again = reconcile_recovery(store, Verifier(), repo, "task", "repo")
    assert recovered_again.completed_steps == ["step"]
    checkpoint_runtime(store, recovered_again, "repo", {}, len(store.read_journal("task", "repo")))
    assert store.load_recovery_checkpoint("task", "repo")[0].journal_position == 2


def test_recovery_reconciles_achieved_intent_without_durable_completion(tmp_path: Path):
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.plan.steps[0].status.value == "COMPLETED"
    assert recovered.transient["recovery"]["journal_entries"] == 1
    assert len(store.read_journal("task", "repo")) == 2


def test_recovery_replays_successful_outcome_after_older_checkpoint(tmp_path: Path):
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")
    store.append_journal("task", JournalOutcome(2, "a", "task", "repo", "succeeded", {"success": True}))

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.plan.steps[0].status.value == "COMPLETED"
    assert recovered.completed_steps == ["step"]


def test_indeterminate_inflight_action_blocks(tmp_path: Path):
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "edit_file", "path": "out.py", "edits": [{"old_text": "x", "new_text": "y"}]}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))

    with pytest.raises(RecoveryBlocked, match="precondition"):
        reconcile_recovery(store, Verifier(), repo, "task", "repo")


# ---------------------------------------------------------------------------
# M5-E Scenario 2: Interruption after intent persistence, before mutation
# The file does NOT exist yet; step must be reset to PENDING, not COMPLETED.
# ---------------------------------------------------------------------------

def test_scenario2_intent_recorded_mutation_not_applied(tmp_path: Path):
    """After intent but before write_file: file absent → step reset to PENDING."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    # File NOT written — interruption before the mutation

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.plan.steps[0].status == StepStatus.PENDING
    assert "step" not in recovered.completed_steps


# ---------------------------------------------------------------------------
# M5-E Scenario 3: Interruption during mutation — ambiguous outcome recorded
# The file IS present (mutation partially or fully applied) but outcome="ambiguous".
# Recovery should detect ACHIEVED and write a "reconciled" outcome.
# ---------------------------------------------------------------------------

def test_scenario3_ambiguous_outcome_file_present_reconciles(tmp_path: Path):
    """Ambiguous outcome + file present → reconcile as ACHIEVED, step COMPLETED."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    store.append_journal("task", JournalOutcome(2, "a", "task", "repo", "ambiguous", None))
    repo.write_file("out.py", "value = 1\n")  # File is present

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.plan.steps[0].status == StepStatus.COMPLETED
    assert "step" in recovered.completed_steps
    # A reconciled outcome should have been appended
    final_journal = store.read_journal("task", "repo")
    assert any(r.get("status") == "reconciled" for r in final_journal)


def test_scenario3_ambiguous_outcome_file_absent_step_pending(tmp_path: Path):
    """Ambiguous outcome + file absent → write_file step reset to PENDING for retry."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    store.append_journal("task", JournalOutcome(2, "a", "task", "repo", "ambiguous", None))
    # File NOT present

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.plan.steps[0].status == StepStatus.PENDING
    assert "step" not in recovered.completed_steps


# ---------------------------------------------------------------------------
# M5-E Scenario 4: Successful mutation with missing journal outcome
# (covered by test_recovery_reconciles_achieved_intent_without_durable_completion)
# ---------------------------------------------------------------------------

def test_scenario4_successful_mutation_no_outcome_in_journal(tmp_path: Path):
    """No outcome record + file present → reconcile appends outcome, marks COMPLETED."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 42\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 42\n")

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.plan.steps[0].status == StepStatus.COMPLETED
    # Reconciled outcome must be durable
    journal_after = store.read_journal("task", "repo")
    reconciled = [r for r in journal_after if r.get("kind") == "outcome" and r.get("status") == "reconciled"]
    assert len(reconciled) == 1


# ---------------------------------------------------------------------------
# M5-E Scenario 5: Interruption after outcome but before checkpoint
# A new checkpoint has NOT been written; the old checkpoint precedes the
# intent+outcome pair. Full replay must still reconstruct correct state.
# ---------------------------------------------------------------------------

def test_scenario5_outcome_recorded_no_checkpoint_written(tmp_path: Path):
    """Old checkpoint + post-checkpoint intent+outcome pair → replay succeeds."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    # Checkpoint written before the action
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 5\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 5\n")
    store.append_journal("task", JournalOutcome(2, "a", "task", "repo", "succeeded", {"success": True}))
    # No checkpoint saved after the outcome

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.plan.steps[0].status == StepStatus.COMPLETED
    assert "step" in recovered.completed_steps


# ---------------------------------------------------------------------------
# M5-E Scenario 6: Corrupt latest checkpoint with valid previous
# (covered by test_corrupt_latest_checkpoint_falls_back_to_previous)
# ---------------------------------------------------------------------------

def test_scenario6_corrupt_latest_valid_previous(tmp_path: Path):
    """Corrupt latest checkpoint falls back to previous; replay reconciles correctly."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))   # becomes 'previous'
    state.plan.version = 2
    store.save_recovery_checkpoint(_checkpoint(state, 0))   # becomes 'latest' — corrupt it
    store._checkpoint_path("task").write_text("CORRUPT", encoding="utf-8")

    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")
    store.append_journal("task", JournalOutcome(2, "a", "task", "repo", "succeeded", {"success": True}))

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.transient["recovery"]["used_fallback"] is True
    # The previous checkpoint was for plan version 1; replay must still find the step
    assert recovered.plan.steps[0].status == StepStatus.COMPLETED


# ---------------------------------------------------------------------------
# M5-E Scenario 7: Corrupt all checkpoints — no valid fallback
# (covered by test_no_valid_checkpoint_blocks_recovery)
# ---------------------------------------------------------------------------

def test_scenario7_no_valid_checkpoint_raises(tmp_path: Path):
    store = StateStore(tmp_path)
    store._checkpoint_path("task").write_text("{}", encoding="utf-8")
    store._checkpoint_path("task", previous=True).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="no valid recovery checkpoint"):
        store.load_recovery_checkpoint("task", "repo")


# ---------------------------------------------------------------------------
# M5-E Scenario 8: Truncated and malformed journal records
# ---------------------------------------------------------------------------

def test_scenario8_truncated_tail_keeps_valid_prefix(tmp_path: Path):
    """Truncated final record is discarded; valid prefix is accepted."""
    store = StateStore(tmp_path)
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", "plan", 1, "step", {}, [], None))
    # Append a half-written record (no closing brace, no newline)
    store.journal_path("task").open("ab").write(b'{"sequence": 2, "kind": "outcome"')

    records = store.read_journal("task", "repo")

    assert len(records) == 1
    assert records[0]["sequence"] == 1


def test_scenario8_malformed_non_final_record_raises(tmp_path: Path):
    """Malformed non-final record raises ValueError immediately."""
    store = StateStore(tmp_path)
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", "plan", 1, "step", {}, [], None))
    # Inject a corrupt line followed by a valid record so it is non-final
    path = store.journal_path("task")
    with path.open("ab") as f:
        f.write(b"not-json\n")
        import json
        good = {"sequence": 3, "kind": "outcome", "action_id": "b", "task_id": "task",
                "repository_id": "repo", "status": "succeeded", "tool_result": None,
                "timestamp": "2024-01-01T00:00:00+00:00"}
        f.write((json.dumps(good) + "\n").encode())

    with pytest.raises(ValueError, match="malformed"):
        store.read_journal("task", "repo")


def test_scenario8_identity_mismatch_raises(tmp_path: Path):
    """Wrong repository_id in a record raises integrity error."""
    store = StateStore(tmp_path)
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", "plan", 1, "step", {}, [], None))
    store.append_journal("task", JournalOutcome(2, "a", "task", "wrong_repo", "succeeded", None))

    with pytest.raises(ValueError, match="integrity"):
        store.read_journal("task", "repo")


# ---------------------------------------------------------------------------
# M5-E Scenario 9: Journal replay after checkpoint fallback
# (covered by test_recovery_replays_successful_outcome_after_older_checkpoint
# and test_scenario5 above; this adds a multi-action case)
# ---------------------------------------------------------------------------

def test_scenario9_multi_action_journal_replay_after_fallback(tmp_path: Path):
    """Two actions journaled; fallback checkpoint; both replayed correctly."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state(extra_steps=[PlanStep("step2", "write second", expected_files=["out2.py"])])
    plan_id = state.plan.plan_id

    # Save checkpoint before any actions
    store.save_recovery_checkpoint(_checkpoint(state, 0))

    # First action — written and confirmed
    action1 = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a1", "task", "repo", plan_id, 1, "step", action1, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")
    store.append_journal("task", JournalOutcome(2, "a1", "task", "repo", "succeeded", {"success": True}))

    # Second action — intent written but no outcome (crash before outcome)
    action2 = {"type": "write_file", "path": "out2.py", "content": "value = 2\n"}
    store.append_journal("task", JournalIntent(3, "a2", "task", "repo", plan_id, 1, "step2", action2, ["out2.py"], None))
    repo.write_file("out2.py", "value = 2\n")
    # Outcome NOT recorded — crash here

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    steps = {s.step_id: s.status for s in recovered.plan.steps}
    assert steps["step"] == StepStatus.COMPLETED
    assert steps["step2"] == StepStatus.COMPLETED


# ---------------------------------------------------------------------------
# M5-E Scenario 10: Relevant external artifact change detected
# File content changed externally after journaled action — reconcile should block.
# ---------------------------------------------------------------------------

def test_scenario10_external_artifact_change_blocks_reconciliation(tmp_path: Path):
    """File changed externally after a 'succeeded' outcome → reconciliation blocked."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")
    store.append_journal("task", JournalOutcome(2, "a", "task", "repo", "succeeded", {"success": True}))
    # Externally mutate the file after the successful outcome was recorded
    (tmp_path / "out.py").write_text("value = 999\n", encoding="utf-8")

    with pytest.raises(RecoveryBlocked, match="postcondition"):
        reconcile_recovery(store, Verifier(), repo, "task", "repo")


# ---------------------------------------------------------------------------
# M5-E Scenario 11: Unrelated repository changes do NOT block recovery
# A file not referenced by any journal intent is modified externally.
# ---------------------------------------------------------------------------

def test_scenario11_unrelated_repository_change_does_not_block(tmp_path: Path):
    """External change to an unrelated file does not trigger RecoveryBlocked."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")
    store.append_journal("task", JournalOutcome(2, "a", "task", "repo", "succeeded", {"success": True}))
    # Modify an unrelated file
    (tmp_path / "unrelated.py").write_text("x = 1\n", encoding="utf-8")

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.plan.steps[0].status == StepStatus.COMPLETED


# ---------------------------------------------------------------------------
# M5-E Scenario 12: Replanning-active recovery — replanning_state restored
# ---------------------------------------------------------------------------

def test_scenario12_replanning_state_restored_from_checkpoint(tmp_path: Path):
    """Checkpoint with replanning_state has outcomes/history restored into transient."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    fake_history = [{"from_plan_id": "old-plan", "result": "activated"}]
    fake_outcomes = [{"kind": "operation_blocked", "signature": "sig"}]
    cp = _checkpoint(state, 0, replanning_state={"history": fake_history, "outcomes": fake_outcomes})
    store.save_recovery_checkpoint(cp)

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.transient.get("replan_history") == fake_history
    assert recovered.transient.get("execution_outcomes") == fake_outcomes
    assert recovered.transient["recovery"]["replanning_state_restored"] is True


def test_scenario12_empty_replanning_state_no_error(tmp_path: Path):
    """Missing replanning_state in checkpoint does not cause errors."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.transient["recovery"]["replanning_state_restored"] is False


# ---------------------------------------------------------------------------
# M5-E Scenario 13: Budget preservation through recovery
# ---------------------------------------------------------------------------

def test_scenario13_budget_counters_preserved_through_recovery(tmp_path: Path):
    """Budget from checkpoint runtime_state is faithfully restored."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    state.budget.llm_calls = 3
    state.budget.tool_actions = 5
    state.budget.retries = 1
    store.save_recovery_checkpoint(_checkpoint(state, 0))

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert recovered.budget.llm_calls == 3
    assert recovered.budget.tool_actions == 5
    assert recovered.budget.retries == 1


# ---------------------------------------------------------------------------
# M5-E Scenario 14: Multiple reconciliation entries appended without corrupting sequence
# ---------------------------------------------------------------------------

def test_scenario14_multiple_reconcile_outcomes_sequence_integrity(tmp_path: Path):
    """Two in-flight intents without outcomes; both reconciled; journal remains valid."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state(extra_steps=[PlanStep("step2", "write second", expected_files=["out2.py"])])
    plan_id = state.plan.plan_id

    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action1 = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    action2 = {"type": "write_file", "path": "out2.py", "content": "value = 2\n"}
    store.append_journal("task", JournalIntent(1, "a1", "task", "repo", plan_id, 1, "step", action1, ["out.py"], None))
    store.append_journal("task", JournalIntent(2, "a2", "task", "repo", plan_id, 1, "step2", action2, ["out2.py"], None))
    repo.write_file("out.py", "value = 1\n")
    repo.write_file("out2.py", "value = 2\n")

    reconcile_recovery(store, Verifier(), repo, "task", "repo")

    # Journal must remain structurally valid after recovery appended two outcomes
    final = store.read_journal("task", "repo")
    sequences = [r["sequence"] for r in final]
    assert sequences == list(range(1, len(final) + 1)), f"sequence gap: {sequences}"


# ---------------------------------------------------------------------------
# M5-E Scenario 15: Preservation of previously verified M3 step progress
# (completed_steps from before the checkpoint must survive recovery)
# ---------------------------------------------------------------------------

def test_scenario15_already_completed_steps_preserved(tmp_path: Path):
    """Steps completed before checkpoint are not rolled back by recovery."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state(extra_steps=[PlanStep("step2", "second write", expected_files=["out2.py"])])
    plan_id = state.plan.plan_id

    # Mark step 1 as already completed in the checkpoint state
    state.plan.steps[0].status = StepStatus.COMPLETED
    state.completed_steps = ["step"]

    store.save_recovery_checkpoint(_checkpoint(state, 0))
    # Only an intent for step2 (pending action after the checkpoint)
    action2 = {"type": "write_file", "path": "out2.py", "content": "value = 2\n"}
    store.append_journal("task", JournalIntent(1, "a2", "task", "repo", plan_id, 1, "step2", action2, ["out2.py"], None))
    # step2 not yet written — crash before mutation

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    assert "step" in recovered.completed_steps
    assert recovered.plan.steps[0].status == StepStatus.COMPLETED
    assert recovered.plan.steps[1].status == StepStatus.PENDING


# ---------------------------------------------------------------------------
# M5-E Scenario 16: Recovery does NOT auto-complete requirements
# (reconcile_recovery must not modify PersistentTaskState)
# ---------------------------------------------------------------------------

def test_scenario16_recovery_does_not_establish_requirement_completion(tmp_path: Path):
    """Step marked COMPLETED by reconciliation ≠ durable requirement completion.
    PersistentTaskState is never touched; only TaskState step statuses change.
    """
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(1, "a", "task", "repo", state.plan.plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    # Step is COMPLETED by reconciliation (structural state) ...
    assert recovered.plan.steps[0].status == StepStatus.COMPLETED
    # ... but no progress file has been written — no PersistentTaskState exists.
    progress_files = list(tmp_path.glob("*.progress.json"))
    assert not progress_files, "Recovery must not create or modify PersistentTaskState"


# ---------------------------------------------------------------------------
# Atomic checkpoint write properties
# ---------------------------------------------------------------------------

def test_atomic_checkpoint_no_partial_file_on_failure(tmp_path: Path):
    """If atomic write fails, no partial checkpoint file is left behind."""
    import os
    store = StateStore(tmp_path)
    state = _state()
    path = store._checkpoint_path("task")

    original_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("simulated rename failure")

    import unittest.mock as mock
    with mock.patch("os.replace", side_effect=failing_replace):
        with pytest.raises(OSError):
            store.save_recovery_checkpoint(_checkpoint(state))

    # Main checkpoint file must not exist (partial write cleaned up)
    assert not path.exists()
    # Temp files must be cleaned up
    tmp_files = list(tmp_path.glob(".task.recovery*.tmp"))
    assert not tmp_files, f"Temp files leaked: {tmp_files}"


def test_atomic_checkpoint_previous_preserved_on_failure(tmp_path: Path):
    """If second checkpoint write fails, the first checkpoint is not lost."""
    import os
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state))  # First — succeeds

    state.plan.version = 2
    original_replace = os.replace
    call_count = [0]

    def failing_on_second(src, dst):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise OSError("simulated rename failure on second write")
        return original_replace(src, dst)

    import unittest.mock as mock
    with mock.patch("os.replace", side_effect=failing_on_second):
        with pytest.raises(OSError):
            store.save_recovery_checkpoint(_checkpoint(state))

    # First checkpoint must still be loadable
    loaded, fallback = store.load_recovery_checkpoint("task", "repo")
    assert loaded.runtime_state["plan"]["version"] == 1


# ---------------------------------------------------------------------------
# Schema and identity validation
# ---------------------------------------------------------------------------

def test_checkpoint_wrong_schema_version_rejected(tmp_path: Path):
    """Checkpoint with unsupported schema_version cannot be loaded."""
    store = StateStore(tmp_path)
    state = _state()
    bad = RecoveryCheckpoint(99, "task", "repo", 0, state.to_dict(), {})
    # Write directly bypassing validation
    import json
    store._checkpoint_path("task").write_text(json.dumps(bad.__dict__), encoding="utf-8")

    with pytest.raises(ValueError, match="no valid recovery checkpoint"):
        store.load_recovery_checkpoint("task", "repo")


def test_checkpoint_identity_mismatch_rejected(tmp_path: Path):
    """Checkpoint with wrong task_id or repository_id is rejected."""
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state))

    with pytest.raises(ValueError, match="no valid recovery checkpoint"):
        store.load_recovery_checkpoint("task", "WRONG_REPO")


def test_journal_sequence_must_start_at_one(tmp_path: Path):
    """append_journal rejects a record with sequence < 1."""
    store = StateStore(tmp_path)
    with pytest.raises(ValueError, match="sequence"):
        store.append_journal("task", JournalIntent(0, "a", "task", "repo", "plan", 1, "step", {}, [], None))


def test_journal_task_id_must_match(tmp_path: Path):
    """append_journal rejects a record with mismatched task_id."""
    store = StateStore(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        store.append_journal("task", JournalIntent(1, "a", "WRONG", "repo", "plan", 1, "step", {}, [], None))


def test_empty_journal_returns_empty_list(tmp_path: Path):
    """read_journal returns [] when no journal file exists."""
    store = StateStore(tmp_path)
    assert store.read_journal("task", "repo") == []


# ---------------------------------------------------------------------------
# Cross-plan provenance validation and legitimate M4 evolution (D1)
# ---------------------------------------------------------------------------

def test_unrelated_plan_intent_blocks_recovery(tmp_path: Path):
    """Journal intent referencing an unknown plan_id with no provenance blocks recovery."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    plan_id = state.plan.plan_id
    store.save_recovery_checkpoint(_checkpoint(state, 0))

    # Intent from an unrelated/unknown plan_id
    old_action = {"type": "write_file", "path": "old.py", "content": "x = 1\n"}
    store.append_journal("task", JournalIntent(1, "old-action", "task", "repo", "unrelated-plan-id", 1, "step", old_action, ["old.py"], None))

    with pytest.raises(RecoveryBlocked, match="unknown plan_id"):
        reconcile_recovery(store, Verifier(), repo, "task", "repo")


def test_legitimate_ancestor_plan_intent_supported_during_recovery(tmp_path: Path):
    """Journal intent from a legitimate ancestor plan (in replan_history) is validated and supported."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    plan_id = state.plan.plan_id
    state.plan.version = 2
    state.plan.parent_plan_id = "ancestor-plan-id"
    history = [{"from_plan_id": "ancestor-plan-id", "from_version": 1, "result": "activated", "to_plan_id": plan_id, "to_version": 2}]
    store.save_recovery_checkpoint(_checkpoint(state, 0, replanning_state={"history": history}))

    # Action 1 from the legitimate ancestor plan (was executed and succeeded)
    action1 = {"type": "write_file", "path": "ancestor.py", "content": "x = 1\n"}
    store.append_journal("task", JournalIntent(1, "a1", "task", "repo", "ancestor-plan-id", 1, "step_anc", action1, ["ancestor.py"], None))
    repo.write_file("ancestor.py", "x = 1\n")
    store.append_journal("task", JournalOutcome(2, "a1", "task", "repo", "succeeded", {"success": True}, plan_id="ancestor-plan-id", plan_version=1))

    # Action 2 from the current active plan
    action2 = {"type": "write_file", "path": "out.py", "content": "value = 1\n"}
    store.append_journal("task", JournalIntent(3, "a2", "task", "repo", plan_id, 2, "step", action2, ["out.py"], None))
    repo.write_file("out.py", "value = 1\n")
    store.append_journal("task", JournalOutcome(4, "a2", "task", "repo", "succeeded", {"success": True}, plan_id=plan_id, plan_version=2))

    recovered = reconcile_recovery(store, Verifier(), repo, "task", "repo")

    # Current step is completed and both actions' files exist
    assert recovered.plan.steps[0].status == StepStatus.COMPLETED
    assert (tmp_path / "ancestor.py").exists()
    assert (tmp_path / "out.py").exists()


def test_orphan_outcome_blocks_recovery(tmp_path: Path):
    """Journal outcome with no preceding intent blocks recovery."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    store.append_journal("task", JournalOutcome(1, "orphan-action", "task", "repo", "succeeded", None))

    with pytest.raises(RecoveryBlocked, match="orphan"):
        reconcile_recovery(store, Verifier(), repo, "task", "repo")


def test_mismatched_outcome_plan_blocks_recovery(tmp_path: Path):
    """Journal outcome whose plan_id mismatches its intent blocks recovery."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    plan_id = state.plan.plan_id
    store.save_recovery_checkpoint(_checkpoint(state, 0))
    action = {"type": "write_file", "path": "out.py", "content": "val = 1\n"}
    store.append_journal("task", JournalIntent(1, "act-1", "task", "repo", plan_id, 1, "step", action, ["out.py"], None))
    store.append_journal("task", JournalOutcome(2, "act-1", "task", "repo", "succeeded", None, plan_id="different-plan"))

    with pytest.raises(RecoveryBlocked, match="plan_id mismatch"):
        reconcile_recovery(store, Verifier(), repo, "task", "repo")


def test_multiple_interruptions_during_recovery_is_idempotent(tmp_path: Path):
    """Recovery interrupted after writing a reconciled outcome can run again safely."""
    repo = RepositoryAdapter(tmp_path)
    store = StateStore(tmp_path)
    state = _state()
    plan_id = state.plan.plan_id
    store.save_recovery_checkpoint(_checkpoint(state, 0))

    action = {"type": "write_file", "path": "out.py", "content": "val = 1\n"}
    store.append_journal("task", JournalIntent(1, "act-1", "task", "repo", plan_id, 1, "step", action, ["out.py"], None))
    repo.write_file("out.py", "val = 1\n")
    # No outcome recorded in journal yet (in-flight crash)

    # First recovery run reconciles the action and appends a "reconciled" outcome
    recovered1 = reconcile_recovery(store, Verifier(), repo, "task", "repo")
    assert recovered1.plan.steps[0].status == StepStatus.COMPLETED

    # Second recovery run (simulating a crash right after the first recovery run)
    recovered2 = reconcile_recovery(store, Verifier(), repo, "task", "repo")
    assert recovered2.plan.steps[0].status == StepStatus.COMPLETED
    assert "step" in recovered2.completed_steps


def test_d3_malformed_runtime_state_in_checkpoint_fails_validation(tmp_path: Path):
    """Checkpoint with malformed runtime state causes explicit failure."""
    store = StateStore(tmp_path)
    malformed_cp = RecoveryCheckpoint(
        1, "task", "repo", 0,
        {"status": "NONEXISTENT_LIFECYCLE", "issue": {}},
        {},
    )
    with pytest.raises(ValueError, match="malformed runtime state"):
        store.save_recovery_checkpoint(malformed_cp)
