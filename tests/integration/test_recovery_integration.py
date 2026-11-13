"""M5 integration checks at the executor/orchestrator durability boundary.

Covers the boundary between the Orchestrator, Executor, StateStore and
repository for the M5 checkpointing/journaling/recovery lifecycle.

Scenarios:
  1.  Intent persistence failure prevents mutation (original test)
  2.  Interruption after intent, before mutation — recovery resets step to PENDING
  3.  Ambiguous outcome (crash during mutation) — recovery reconciles correctly
  5.  Outcome recorded but no post-action checkpoint — recover() replay succeeds
  12. Orchestrator.recover() restores replanning state so M4 controller can resume
  13. Budget counters are preserved through Orchestrator.recover()
  15. Already-completed steps in checkpoint survive recovery
  16. Orchestrator.recover() does not update PersistentTaskState (no auto-completion)
"""

from pathlib import Path
from unittest import mock

from agent.llm.provider import DeterministicProvider
from agent.models import (
    AgentConfig, BudgetLedger, IssueSpec, Lifecycle, Plan, PlanStep,
    RecoveryCheckpoint, StepStatus, TaskState,
)
from agent.orchestrator import Orchestrator
from agent.recovery import reconcile_recovery, repository_identity
from agent.state.store import StateStore
from agent.tools.repository import RepositoryAdapter
from agent.verifier import Verifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_orchestrator(git_repo: Path):
    (git_repo / "pyproject.toml").write_text("", encoding="utf-8")
    return Orchestrator(AgentConfig(workspace=str(git_repo)), DeterministicProvider())


def _write_checkpoint(store: StateStore, state: TaskState, repo_id: str, journal_pos: int = 0):
    store.save_recovery_checkpoint(RecoveryCheckpoint(
        1, state.task_id, repo_id, journal_pos,
        state.to_dict(), {"commit": None, "status": ""},
    ))


# ---------------------------------------------------------------------------
# Scenario 1: Intent persistence failure prevents mutation
# ---------------------------------------------------------------------------

def test_intent_persistence_failure_prevents_mutation(git_repo: Path):
    orchestrator = _minimal_orchestrator(git_repo)
    original_append = orchestrator.store.append_journal

    def fail_intent(task_id, record):
        if record.kind == "intent":
            raise OSError("journal unavailable")
        return original_append(task_id, record)

    orchestrator.store.append_journal = fail_intent
    state, _ = orchestrator.run(IssueSpec(
        "Add function", "Add a function and tests", ["function exists"],
    ))

    assert not (git_repo / "src" / "text_tools.py").exists()
    assert "journal unavailable" in state.transient.get("error", "")


# ---------------------------------------------------------------------------
# Scenario 2: Interruption after intent, before mutation — recover() resets step
# ---------------------------------------------------------------------------

def test_scenario2_recover_resets_step_when_mutation_never_applied(git_repo: Path):
    """Intent journaled but write_file never reached; recover() sets step PENDING."""
    from agent.models import IssueSpec, Plan, PlanStep, TaskState
    from agent.recovery import RecoveryJournal, checkpoint_runtime, repository_identity
    from agent.models import JournalIntent

    state_dir = git_repo / ".agent-state"
    store = StateStore(state_dir)
    repo = RepositoryAdapter(git_repo)
    snapshot = repo.inspect()
    repo_id = repository_identity(snapshot.root, snapshot.baseline_commit)

    issue = IssueSpec("test", "test issue", ["out exists"], issue_id="issue-1")
    plan = Plan("issue-1", [PlanStep("s1", "write out.py", expected_files=["out.py"])])
    state = TaskState(issue, plan=plan, task_id="task-s2")

    # Write checkpoint then journal an intent but NOT the mutation
    _write_checkpoint(store, state, repo_id, 0)
    store.append_journal("task-s2", JournalIntent(
        1, "act-1", "task-s2", repo_id, plan.plan_id, 1, "s1",
        {"type": "write_file", "path": "out.py", "content": "x = 1\n"},
        ["out.py"], None,
    ))
    # out.py never written

    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo)), DeterministicProvider())
    recovered = orchestrator.recover("task-s2")

    assert recovered.plan.steps[0].status == StepStatus.PENDING
    assert "s1" not in recovered.completed_steps


def test_resume_holds_ownership_across_nested_recover_call(git_repo: Path):
    """resume() owns the task while its public recover() call re-enters safely."""
    from agent.models import ObservationBundle

    orchestrator = _minimal_orchestrator(git_repo)
    snapshot = orchestrator.repository.inspect()
    repo_id = repository_identity(snapshot.root, snapshot.baseline_commit)
    state = TaskState(
        IssueSpec("resume", "resume", ["done"], issue_id="resume-issue"),
        plan=Plan("resume-issue", [PlanStep("s1", "write", expected_files=["out.py"])]),
        task_id="resume-task",
    )
    _write_checkpoint(orchestrator.store, state, repo_id)
    original_recover = orchestrator.recover
    observed = []

    def observed_recover(task_id):
        observed.append(task_id)
        return original_recover(task_id)

    orchestrator.recover = observed_recover
    orchestrator._execute_and_verify = lambda recovered, *args, **kwargs: (recovered, ObservationBundle(recovered.task_id))

    resumed, _ = orchestrator.resume("resume-task")

    assert observed == ["resume-task"]
    assert resumed.task_id == "resume-task"


# ---------------------------------------------------------------------------
# Scenario 3: Ambiguous outcome (crash during mutation) — recover() reconciles
# ---------------------------------------------------------------------------

def test_scenario3_recover_reconciles_ambiguous_outcome_file_present(git_repo: Path):
    """Ambiguous outcome + file present: recover() marks step COMPLETED via reconcile."""
    from agent.models import IssueSpec, Plan, PlanStep, TaskState, JournalIntent, JournalOutcome
    from agent.recovery import repository_identity

    state_dir = git_repo / ".agent-state"
    store = StateStore(state_dir)
    repo = RepositoryAdapter(git_repo)
    snapshot = repo.inspect()
    repo_id = repository_identity(snapshot.root, snapshot.baseline_commit)

    issue = IssueSpec("test", "test issue", ["out exists"], issue_id="issue-3")
    plan = Plan("issue-3", [PlanStep("s1", "write out.py", expected_files=["out.py"])])
    state = TaskState(issue, plan=plan, task_id="task-s3")

    _write_checkpoint(store, state, repo_id, 0)
    store.append_journal("task-s3", JournalIntent(
        1, "act-1", "task-s3", repo_id, plan.plan_id, 1, "s1",
        {"type": "write_file", "path": "out.py", "content": "x = 1\n"},
        ["out.py"], None,
    ))
    store.append_journal("task-s3", JournalOutcome(2, "act-1", "task-s3", repo_id, "ambiguous", None))
    repo.write_file("out.py", "x = 1\n")  # File IS present (mutation completed)

    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo)), DeterministicProvider())
    recovered = orchestrator.recover("task-s3")

    assert recovered.plan.steps[0].status == StepStatus.COMPLETED
    assert "s1" in recovered.completed_steps
    # A reconciled outcome must be durable in the journal
    final_journal = store.read_journal("task-s3", repo_id)
    assert any(r.get("status") == "reconciled" for r in final_journal)


# ---------------------------------------------------------------------------
# Scenario 5: Outcome recorded but no post-action checkpoint — recover() replay
# ---------------------------------------------------------------------------

def test_scenario5_recover_replays_post_checkpoint_outcome(git_repo: Path):
    """Checkpoint saved before action; outcome recorded; no second checkpoint.
    recover() must replay the outcome and mark the step COMPLETED."""
    from agent.models import IssueSpec, Plan, PlanStep, TaskState, JournalIntent, JournalOutcome
    from agent.recovery import repository_identity

    state_dir = git_repo / ".agent-state"
    store = StateStore(state_dir)
    repo = RepositoryAdapter(git_repo)
    snapshot = repo.inspect()
    repo_id = repository_identity(snapshot.root, snapshot.baseline_commit)

    issue = IssueSpec("test", "test issue", ["out exists"], issue_id="issue-5")
    plan = Plan("issue-5", [PlanStep("s1", "write out.py", expected_files=["out.py"])])
    state = TaskState(issue, plan=plan, task_id="task-s5")

    _write_checkpoint(store, state, repo_id, 0)  # Checkpoint BEFORE the action
    store.append_journal("task-s5", JournalIntent(
        1, "act-1", "task-s5", repo_id, plan.plan_id, 1, "s1",
        {"type": "write_file", "path": "out.py", "content": "x = 1\n"},
        ["out.py"], None,
    ))
    repo.write_file("out.py", "x = 1\n")
    store.append_journal("task-s5", JournalOutcome(2, "act-1", "task-s5", repo_id, "succeeded", {"success": True}))
    # No checkpoint written after the outcome

    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo)), DeterministicProvider())
    recovered = orchestrator.recover("task-s5")

    assert recovered.plan.steps[0].status == StepStatus.COMPLETED
    assert "s1" in recovered.completed_steps


# ---------------------------------------------------------------------------
# Scenario 12: recover() restores replanning_state into transient
# ---------------------------------------------------------------------------

def test_scenario12_recover_restores_replanning_state(git_repo: Path):
    """Checkpoint with replanning_state: recover() restores history/outcomes."""
    from agent.models import IssueSpec, Plan, PlanStep, TaskState
    from agent.recovery import repository_identity

    state_dir = git_repo / ".agent-state"
    store = StateStore(state_dir)
    repo = RepositoryAdapter(git_repo)
    snapshot = repo.inspect()
    repo_id = repository_identity(snapshot.root, snapshot.baseline_commit)

    issue = IssueSpec("test", "test", [], issue_id="issue-12")
    plan = Plan("issue-12", [PlanStep("s1", "write", expected_files=["out.py"])])
    state = TaskState(issue, plan=plan, task_id="task-s12")

    fake_history = [{"from_plan_id": "old", "result": "activated", "reason": "blocked"}]
    fake_outcomes = [{"kind": "operation_blocked", "signature": "sig1"}]
    store.save_recovery_checkpoint(RecoveryCheckpoint(
        1, "task-s12", repo_id, 0, state.to_dict(), {},
        replanning_state={"history": fake_history, "outcomes": fake_outcomes},
    ))

    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo)), DeterministicProvider())
    recovered = orchestrator.recover("task-s12")

    assert recovered.transient.get("replan_history") == fake_history
    assert recovered.transient.get("execution_outcomes") == fake_outcomes
    assert recovered.transient["recovery"]["replanning_state_restored"] is True


# ---------------------------------------------------------------------------
# Scenario 13: Budget counters preserved through recover()
# ---------------------------------------------------------------------------

def test_scenario13_recover_preserves_budget_counters(git_repo: Path):
    """Budget from checkpoint runtime_state faithfully restored via recover()."""
    from agent.models import IssueSpec, Plan, PlanStep, TaskState
    from agent.recovery import repository_identity

    state_dir = git_repo / ".agent-state"
    store = StateStore(state_dir)
    repo = RepositoryAdapter(git_repo)
    snapshot = repo.inspect()
    repo_id = repository_identity(snapshot.root, snapshot.baseline_commit)

    issue = IssueSpec("test", "test", [], issue_id="issue-13")
    plan = Plan("issue-13", [PlanStep("s1", "write", expected_files=["out.py"])])
    state = TaskState(issue, plan=plan, task_id="task-s13")
    state.budget.llm_calls = 4
    state.budget.tool_actions = 6
    state.budget.retries = 2

    _write_checkpoint(store, state, repo_id, 0)

    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo)), DeterministicProvider())
    recovered = orchestrator.recover("task-s13")

    assert recovered.budget.llm_calls == 4
    assert recovered.budget.tool_actions == 6
    assert recovered.budget.retries == 2


# ---------------------------------------------------------------------------
# Scenario 15: Already-completed steps in checkpoint survive recover()
# ---------------------------------------------------------------------------

def test_scenario15_completed_steps_survive_recover(git_repo: Path):
    """Steps marked COMPLETED before the checkpoint are not rolled back."""
    from agent.models import IssueSpec, Plan, PlanStep, TaskState, StepStatus
    from agent.recovery import repository_identity

    state_dir = git_repo / ".agent-state"
    store = StateStore(state_dir)
    repo = RepositoryAdapter(git_repo)
    snapshot = repo.inspect()
    repo_id = repository_identity(snapshot.root, snapshot.baseline_commit)

    issue = IssueSpec("test", "test", [], issue_id="issue-15")
    plan = Plan("issue-15", [
        PlanStep("s1", "write first", expected_files=["a.py"]),
        PlanStep("s2", "write second", expected_files=["b.py"], dependencies=["s1"]),
    ])
    state = TaskState(issue, plan=plan, task_id="task-s15")
    state.plan.steps[0].status = StepStatus.COMPLETED
    state.completed_steps = ["s1"]
    repo.write_file("a.py", "a = 1\n")  # file exists so postcondition is consistent

    _write_checkpoint(store, state, repo_id, 0)

    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo)), DeterministicProvider())
    recovered = orchestrator.recover("task-s15")

    assert "s1" in recovered.completed_steps
    assert recovered.plan.steps[0].status == StepStatus.COMPLETED
    assert recovered.plan.steps[1].status == StepStatus.PENDING


# ---------------------------------------------------------------------------
# Scenario 16: recover() does NOT update PersistentTaskState
# ---------------------------------------------------------------------------

def test_scenario16_recover_does_not_modify_persistent_task_state(git_repo: Path):
    """Orchestrator.recover() must not write or update any PersistentTaskState."""
    from agent.models import IssueSpec, Plan, PlanStep, TaskState, JournalIntent
    from agent.recovery import repository_identity
    from agent.state.progress import create_progress_state

    state_dir = git_repo / ".agent-state"
    store = StateStore(state_dir)
    repo = RepositoryAdapter(git_repo)
    snapshot = repo.inspect()
    repo_id = repository_identity(snapshot.root, snapshot.baseline_commit)

    issue = IssueSpec("test", "test", ["out exists"], issue_id="issue-16")
    plan = Plan("issue-16", [PlanStep("s1", "write out.py", expected_files=["out.py"])])
    state = TaskState(issue, plan=plan, task_id="task-s16")

    _write_checkpoint(store, state, repo_id, 0)
    store.append_journal("task-s16", JournalIntent(
        1, "act-1", "task-s16", repo_id, plan.plan_id, 1, "s1",
        {"type": "write_file", "path": "out.py", "content": "x = 1\n"},
        ["out.py"], None,
    ))
    repo.write_file("out.py", "x = 1\n")

    # List progress files before
    before = set((git_repo / ".agent-state").glob("*.progress.json"))

    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo)), DeterministicProvider())
    recovered = orchestrator.recover("task-s16")

    after = set((git_repo / ".agent-state").glob("*.progress.json"))
    assert after == before, "recover() must not create or modify PersistentTaskState files"
    assert recovered.plan.steps[0].status == StepStatus.COMPLETED


# ---------------------------------------------------------------------------
# End-to-end: checkpoint written during run, then recovered via recover()
# ---------------------------------------------------------------------------

def test_e2e_run_produces_checkpoint_recoverable_by_recover(git_repo: Path):
    """A full orchestrator.run() produces a checkpoint that recover() can load."""
    orchestrator = _minimal_orchestrator(git_repo)
    issue = IssueSpec("Add normalize_whitespace",
                      "Add a function to normalize whitespace in strings and tests",
                      ["function exists", "pytest passes"])

    state, _ = orchestrator.run(issue)
    task_id = state.task_id

    # Checkpoint must exist on disk after a successful run
    state_dir = git_repo / ".agent-state"
    checkpoint_file = next(state_dir.glob(f"{task_id}.recovery.json"), None)
    assert checkpoint_file is not None, "run() must produce a recovery checkpoint"

    # Journal must exist on disk after a successful run
    journal_file = next(state_dir.glob(f"{task_id}.journal.jsonl"), None)
    assert journal_file is not None, "run() must produce a journal file"

    # recover() must load a consistent state without raising
    store = StateStore(state_dir)
    repo = RepositoryAdapter(git_repo)
    snapshot = repo.inspect()
    repo_id = repository_identity(snapshot.root, snapshot.baseline_commit)
    recovered = reconcile_recovery(store, Verifier(), repo, task_id, repo_id)

    assert recovered.task_id == task_id


# ---------------------------------------------------------------------------
# Section 4: End-to-end multi-round interrupted execution, recovery & resumption
# ---------------------------------------------------------------------------

def test_end_to_end_interrupted_run_resumes_and_verifies_durable_progress(git_repo: Path):
    """Demonstrates complete M5 long-horizon recovery:
    1. Multi-step task starts execution across rounds.
    2. Round 1 completes and durable progress (R1) is verified.
    3. Round 2 begins; mutation happens but process is interrupted before completion.
    4. Durable state shows R1 is completed, R2 is pending (recovery alone never completes R2).
    5. orchestrator.resume() recovers checkpoint, reconciles in-flight mutation.
    6. Execution resumes through normal M3/M4 lifecycle.
    7. Final independent verification runs.
    8. Durable completion of R2 is updated ONLY through the verifier-driven update mechanism.
    """
    from agent.manager import BoundedExecutionContract
    from agent.state.progress import create_progress_state, RequirementStatus

    (git_repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\npythonpath = ['.']\n",
        encoding="utf-8",
    )
    (git_repo / "src").mkdir()
    (git_repo / "tests").mkdir()
    (git_repo / "src" / "__init__.py").write_text("", encoding="utf-8")
    (git_repo / "tests" / "__init__.py").write_text("", encoding="utf-8")

    orchestrator = Orchestrator(AgentConfig(workspace=str(git_repo)), DeterministicProvider())

    issue = IssueSpec(
        "Add normalize_whitespace",
        "Add a function to normalize whitespace in strings and tests",
        ["function exists", "pytest passes"],
    )
    progress = create_progress_state("task-e2e", issue)
    orchestrator.store.save_progress(progress)

    # --- Step 1 & 2: Round 1 executes and verifies R1 ---
    decision_1 = orchestrator.manager.decide(progress, BudgetLedger(), max_rounds=12)
    contract_1 = decision_1.contract
    assert contract_1 is not None

    state_1, obs_1 = orchestrator.run(
        contract_1.to_issue(progress.issue),
        persistent_state=progress,
        requirement_checks={"R1": ["execution_record_0_repository_state"]},
        round_id=1,
        contract=contract_1,
    )
    assert state_1.status == Lifecycle.COMPLETED
    assert progress.requirements[0].status == RequirementStatus.COMPLETED
    assert progress.requirements[1].status == RequirementStatus.PENDING

    # Verify R1 was saved to disk
    progress_loaded = orchestrator.store.load_progress("task-e2e")
    assert progress_loaded.requirements[0].status == RequirementStatus.COMPLETED
    assert progress_loaded.requirements[1].status == RequirementStatus.PENDING

    # --- Step 3: Round 2 starts, mutates repo, but is interrupted before completion ---
    decision_2 = orchestrator.manager.decide(progress, BudgetLedger(), max_rounds=12)
    contract_2 = decision_2.contract
    assert contract_2 is not None

    # Intercept tool call in round 2: let the file be written, then crash
    original_write = orchestrator.repository.write_file
    crash_injected = False
    round_2_task_id = None

    def interrupted_write(path, content):
        nonlocal crash_injected
        result = original_write(path, content)
        if "test_text_tools" in path and not crash_injected:
            crash_injected = True
            raise KeyboardInterrupt("Simulated sudden power loss / interruption after write")
        return result

    with mock.patch.object(orchestrator.repository, "write_file", side_effect=interrupted_write):
        try:
            orchestrator.run(
                contract_2.to_issue(progress.issue),
                persistent_state=progress,
                requirement_checks={"R2": ["relevant_tests_passed"]},
                round_id=2,
                contract=contract_2,
            )
        except KeyboardInterrupt:
            pass

    # Find the interrupted Round 2 task_id from state store
    state_dir = git_repo / ".agent-state"
    checkpoints = [
        p.name.replace(".recovery.json", "")
        for p in state_dir.glob("*.recovery.json")
        if not p.name.endswith(".previous.json") and p.name != f"{state_1.task_id}.recovery.json"
    ]
    assert len(checkpoints) == 1, f"Expected 1 round 2 checkpoint, found: {checkpoints}"
    round_2_task_id = checkpoints[0]

    # --- Step 4: Verify durable state before recovery: R1 intact, R2 still PENDING ---
    progress_before_resume = orchestrator.store.load_progress("task-e2e")
    assert progress_before_resume.requirements[0].status == RequirementStatus.COMPLETED
    assert progress_before_resume.requirements[1].status == RequirementStatus.PENDING

    # --- Step 5, 6, 7: Resume execution of round 2 via orchestrator.resume ---
    resumed_state, resumed_obs = orchestrator.resume(
        round_2_task_id,
        persistent_state=progress,
        requirement_checks={"R2": ["relevant_tests_passed"]},
        round_id=2,
        contract=contract_2,
    )

    # --- Step 8 & 9: Verification ran and durable completion updated only via verifier ---
    assert resumed_state.status == Lifecycle.COMPLETED, f"Failed with: {resumed_state.transient.get('error')}, notes: {resumed_obs.notes}"
    assert resumed_obs.audit_result is not None
    assert resumed_obs.audit_result.verified is True

    # Check durable state on disk
    progress_final = orchestrator.store.load_progress("task-e2e")
    assert progress_final.requirements[0].status == RequirementStatus.COMPLETED
    assert progress_final.requirements[1].status == RequirementStatus.COMPLETED
    assert progress_final.status.value == "completed" or all(r.status == RequirementStatus.COMPLETED for r in progress_final.requirements)
