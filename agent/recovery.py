"""M5 single-writer recovery coordination around the existing runtime loop."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256

from .models import (
    ExecutionRecord, JournalIntent, JournalOutcome, RecoveryCheckpoint,
    ReplanHistoryEntry, StepStatus, TaskState, ToolResult,
)
from .state.store import StateStore
from .verifier import AssessmentStatus, Verifier


class RecoveryBlocked(RuntimeError):
    """Raised when recorded knowledge cannot be safely reconciled."""


class RecoveryJournal:
    """Single-process journal adapter used by Executor before file mutations."""

    def __init__(self, store: StateStore, task_id: str, repository_id: str):
        self.store, self.task_id, self.repository_id = store, task_id, repository_id
        self.sequence = len(store.read_journal(task_id, repository_id))
        # A valid empty journal is still part of a newly checkpointed task's state.
        store.journal_path(task_id).touch(exist_ok=True)

    def intent(self, action_id, state, step, action, before_content) -> None:
        self.sequence += 1
        plan = state.plan
        if plan is None:
            raise RecoveryBlocked("cannot journal an action without an active plan")
        self.store.append_journal(self.task_id, JournalIntent(
            self.sequence, action_id, self.task_id, self.repository_id, plan.plan_id,
            plan.version, step.step_id, action, list(step.expected_files), before_content,
        ))

    def outcome(self, action_id, state, status, result, step_id: str | None = None) -> None:
        self.sequence += 1
        payload = asdict(result) if result is not None else None
        plan = state.plan
        self.store.append_journal(self.task_id, JournalOutcome(
            self.sequence, action_id, self.task_id, self.repository_id, status, payload,
            plan_id=plan.plan_id if plan else None,
            plan_version=plan.version if plan else None,
            step_id=step_id,
        ))


def repository_identity(root: str, baseline_commit: str | None) -> str:
    """Stable enough for one local repository; checkpoint data is never portable."""
    return sha256(f"{root}|{baseline_commit or ''}".encode("utf-8")).hexdigest()


def checkpoint_runtime(store: StateStore, state: TaskState, repository_id: str,
                       observation: dict, journal_position: int, persistent_task_id: str | None = None) -> None:
    store.save_recovery_checkpoint(RecoveryCheckpoint(
        StoreVersion, state.task_id, repository_id, journal_position, state.to_dict(), observation,
        replanning_state={
            "outcomes": state.transient.get("execution_outcomes", []),
            "history": state.transient.get("replan_history", []),
        }, persistent_task_id=persistent_task_id,
    ))


StoreVersion = StateStore.CHECKPOINT_VERSION


def reconcile_recovery(store: StateStore, verifier: Verifier, repository, task_id: str,
                       repository_id: str) -> TaskState:
    """Recover runtime knowledge and reconcile intents lacking an outcome. No completion updates occur.

    Design invariants:
    - PersistentTaskState is NOT modified here; requirement completion authority remains with the verifier.
    - Journal intent plan_id and version are validated against legitimate plan execution history.
    - Historical actions across legitimate M4 plan revisions are validated and supported.
    - Replanning state (outcomes/history) is restored from checkpoint so M4 controller can resume.
    - Any ambiguous in-flight action blocks recovery; the caller must not proceed without explicit resolution.
    """
    checkpoint, used_fallback = store.load_recovery_checkpoint(task_id, repository_id)
    state = TaskState.from_dict(checkpoint.runtime_state)

    # D2 Fix: restore replanning state so M4 ReplanningController can resume correctly.
    replanning_state = checkpoint.replanning_state or {}
    if replanning_state.get("outcomes"):
        state.transient.setdefault("execution_outcomes", replanning_state["outcomes"])
    if replanning_state.get("history"):
        state.transient.setdefault("replan_history", replanning_state["history"])

    # read_journal validates every record, including the checkpoint prefix. Only
    # the suffix represents events newer than this snapshot.
    try:
        journal = store.validate_journal_prefix(task_id, repository_id, checkpoint.journal_position)
    except ValueError as exc:
        raise RecoveryBlocked(f"journal/checkpoint position conflict: {exc}") from exc

    # D1 Fix: determine legitimate plan history for provenance validation.
    legitimate_plans: dict[str, int] = {}
    if state.plan is not None:
        legitimate_plans[state.plan.plan_id] = state.plan.version
        if state.plan.parent_plan_id:
            legitimate_plans[state.plan.parent_plan_id] = max(1, state.plan.version - 1)
    for entry in state.transient.get("replan_history", []):
        if isinstance(entry, dict):
            if entry.get("from_plan_id"):
                legitimate_plans[entry["from_plan_id"]] = entry.get("from_version", 1)
            if entry.get("to_plan_id"):
                legitimate_plans[entry["to_plan_id"]] = entry.get("to_version", 1)
        elif isinstance(entry, ReplanHistoryEntry):
            legitimate_plans[entry.from_plan_id] = entry.from_version
            if entry.to_plan_id:
                legitimate_plans[entry.to_plan_id] = entry.to_version or 1

    intents: dict[str, dict] = {}
    outcomes: dict[str, dict] = {}

    for item in journal:
        action_id = item["action_id"]
        if item["kind"] == "intent":
            if action_id in intents:
                raise RecoveryBlocked(f"duplicate journal intent action_id: {action_id}")
            plan_id = item.get("plan_id")
            plan_version = item.get("plan_version")
            step_id = item.get("step_id")
            if legitimate_plans:
                if plan_id not in legitimate_plans:
                    raise RecoveryBlocked(f"journal intent references unknown plan_id: {plan_id}")
                expected_v = legitimate_plans[plan_id]
                if plan_version is not None and plan_version != expected_v:
                    raise RecoveryBlocked(f"journal intent plan_version {plan_version} does not match expected {expected_v}")
            if state.plan is not None and plan_id == state.plan.plan_id:
                known_step_ids = {s.step_id for s in state.plan.steps}
                if step_id not in known_step_ids:
                    raise RecoveryBlocked(f"journal intent references unknown step_id '{step_id}' in active plan")
            intents[action_id] = item
        else:
            if action_id not in intents:
                raise RecoveryBlocked(f"orphan journal outcome without prior intent: {action_id}")
            if action_id in outcomes:
                raise RecoveryBlocked(f"duplicate journal outcome for action_id: {action_id}")
            intent = intents[action_id]
            if item.get("plan_id") and intent.get("plan_id") and item["plan_id"] != intent["plan_id"]:
                raise RecoveryBlocked(f"journal outcome plan_id mismatch for action {action_id}")
            if item.get("plan_version") and intent.get("plan_version") and item["plan_version"] != intent["plan_version"]:
                raise RecoveryBlocked(f"journal outcome plan_version mismatch for action {action_id}")
            outcomes[action_id] = item

    recovered_records: list[ExecutionRecord] = []
    active_plan_id = state.plan.plan_id if state.plan is not None else None

    for action_id, intent in intents.items():
        outcome = outcomes.get(action_id)
        is_active_plan = (active_plan_id is None) or (intent.get("plan_id") == active_plan_id)
        step = next((s for s in (state.plan.steps if state.plan else []) if s.step_id == intent["step_id"]), None) if is_active_plan else None

        result = ToolResult(intent["action"].get("type", "unknown"), False)
        record = ExecutionRecord(intent["step_id"], intent["expected_files"], intent["action"],
                                 intent.get("before_content"), result)
        assessment = verifier.assess_execution_record(repository, record)

        if outcome is not None:
            if outcome["status"] in {"succeeded", "reconciled"}:
                if assessment.status != AssessmentStatus.ACHIEVED:
                    raise RecoveryBlocked(f"recorded successful action no longer matches postcondition: {action_id}")
                if step is not None:
                    step.status = StepStatus.COMPLETED
                    if step.step_id not in state.completed_steps:
                        state.completed_steps.append(step.step_id)
                recovered_records.append(ExecutionRecord(
                    intent["step_id"], intent["expected_files"], intent["action"],
                    intent.get("before_content"),
                    ToolResult(intent["action"].get("type", "unknown"), True, changed_files=list(intent.get("expected_files", []))),
                    active=True,
                ))
                continue
            if outcome["status"] == "failed":
                if step is not None:
                    step.status = StepStatus.PENDING
                continue
            # "ambiguous" outcome — check postcondition

        if not is_active_plan:
            raise RecoveryBlocked(f"unresolved in-flight action in ancestor plan {intent.get('plan_id')}")

        if assessment.status == AssessmentStatus.ACHIEVED:
            # The mutation happened; record reconciliation but do not declare any requirement complete.
            sequence = len(store.read_journal(task_id, repository_id)) + 1
            store.append_journal(task_id, JournalOutcome(
                sequence, action_id, task_id, repository_id, "reconciled", None,
                plan_id=intent.get("plan_id"),
                plan_version=intent.get("plan_version"),
                step_id=intent.get("step_id"),
            ))
            if step is not None:
                step.status = StepStatus.COMPLETED
                if step.step_id not in state.completed_steps:
                    state.completed_steps.append(step.step_id)
            recovered_records.append(ExecutionRecord(
                intent["step_id"], intent["expected_files"], intent["action"],
                intent.get("before_content"),
                ToolResult(intent["action"].get("type", "unknown"), True, changed_files=list(intent.get("expected_files", []))),
                active=True,
            ))
        elif assessment.status == AssessmentStatus.NOT_ACHIEVED:
            action = intent["action"]
            if action.get("type") == "edit_file":
                before = intent.get("before_content")
                if not isinstance(before, str):
                    raise RecoveryBlocked(f"edit precondition is unavailable for in-flight action {action_id}")
                try:
                    current = repository.read_file(action.get("path", ""))
                except Exception as exc:
                    raise RecoveryBlocked(f"edit precondition cannot be rechecked for {action_id}") from exc
                if current != before:
                    raise RecoveryBlocked(f"edit precondition changed for in-flight action {action_id}")
            if step is not None:
                step.status = StepStatus.PENDING
        else:
            raise RecoveryBlocked(f"ambiguous in-flight action {action_id}")

    state.transient["recovered_execution_records"] = [asdict(r) for r in recovered_records]
    state.transient["recovery"] = {
        "used_fallback": used_fallback,
        "journal_entries": checkpoint.journal_position + len(journal),
        "replayed_entries": len(journal),
        "replanning_state_restored": bool(replanning_state),
    }
    return state
