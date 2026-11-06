"""Deterministic M4 retry and replanning policy for one runtime round."""

from __future__ import annotations

from dataclasses import asdict
from enum import StrEnum
from hashlib import sha256
import json

from .manager import BoundedExecutionContract
from .models import (
    AgentConfig, ExecutionOutcome, ExecutionOutcomeKind, ExecutionRecord,
    InvalidationReason, Plan, ReplanHistoryEntry, RepositorySnapshot, StepStatus, TaskState,
)
from .verifier import AssessmentStatus, IntermediateAssessment, Verifier


class ReplanDecisionKind(StrEnum):
    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    BLOCK = "block"


class ReplanningController:
    """Owns policy only; it cannot update persistent requirement state."""

    def __init__(self, config: AgentConfig, verifier: Verifier):
        self.config = config
        self.verifier = verifier
        self.outcomes: list[ExecutionOutcome] = []
        self.history: list[ReplanHistoryEntry] = []
        self._replans = 0
        self._retries: dict[str, int] = {}
        self._seen: set[str] = set()

    def classify(self, record: ExecutionRecord | None, assessment: IntermediateAssessment | None = None,
                 error: Exception | None = None) -> ExecutionOutcome:
        if error is not None:
            detail = f"{type(error).__name__}: {error}"
            lower = detail.lower()
            if "permission" in lower or "budget" in lower:
                reason = InvalidationReason.PERMISSION_DENIED if "permission" in lower else InvalidationReason.BUDGET_EXHAUSTED
                kind = ExecutionOutcomeKind.TERMINAL_BLOCKER
            elif "timeout" in lower or "connection" in lower:
                reason, kind = InvalidationReason.TRANSIENT_TIMEOUT, ExecutionOutcomeKind.RECOVERABLE_FAILURE
            else:
                reason, kind = InvalidationReason.UNAVAILABLE_OPERATION, ExecutionOutcomeKind.OPERATION_BLOCKED
            return self._outcome(None, kind, reason, False, detail)
        if record is None:
            return self._outcome(None, ExecutionOutcomeKind.TERMINAL_BLOCKER, InvalidationReason.UNAVAILABLE_OPERATION, False, "no execution record")
        result = record.tool_result
        detail = result.stderr or result.stdout or "tool attempt completed"
        if result.success and assessment and assessment.status == AssessmentStatus.ACHIEVED:
            return self._outcome(record.step_id, ExecutionOutcomeKind.PROGRESS, None, False, detail)
        if result.timed_out:
            return self._outcome(record.step_id, ExecutionOutcomeKind.RECOVERABLE_FAILURE,
                                 InvalidationReason.TRANSIENT_TIMEOUT, True, detail)
        lower = detail.lower()
        if "permission" in lower:
            return self._outcome(record.step_id, ExecutionOutcomeKind.TERMINAL_BLOCKER,
                                 InvalidationReason.PERMISSION_DENIED, False, detail)
        if "does not exist" in lower or "not found" in lower or "missing" in lower:
            return self._outcome(record.step_id, ExecutionOutcomeKind.OPERATION_BLOCKED,
                                 InvalidationReason.MISSING_RESOURCE, False, detail)
        if assessment and assessment.status == AssessmentStatus.NOT_ACHIEVED:
            return self._outcome(record.step_id, ExecutionOutcomeKind.NO_PROGRESS,
                                 InvalidationReason.REPEATED_NO_PROGRESS, False, detail)
        return self._outcome(record.step_id, ExecutionOutcomeKind.OPERATION_BLOCKED,
                             InvalidationReason.UNAVAILABLE_OPERATION, False, detail)

    def decide(self, outcome: ExecutionOutcome, assessment: IntermediateAssessment | None) -> ReplanDecisionKind:
        self.outcomes.append(outcome)
        if outcome.kind == ExecutionOutcomeKind.PROGRESS:
            return ReplanDecisionKind.CONTINUE
        if outcome.kind == ExecutionOutcomeKind.TERMINAL_BLOCKER:
            return ReplanDecisionKind.BLOCK
        # Ambiguous transient outcomes are retried only after an assessment proves absent postcondition.
        if outcome.retryable and assessment and assessment.status == AssessmentStatus.NOT_ACHIEVED:
            used = self._retries.get(outcome.signature, 0)
            if used < self.config.max_safe_retries:
                self._retries[outcome.signature] = used + 1
                return ReplanDecisionKind.RETRY
        equivalents = sum(item.signature == outcome.signature for item in self.outcomes)
        if outcome.kind == ExecutionOutcomeKind.NO_PROGRESS and equivalents < self.config.no_progress_threshold:
            return ReplanDecisionKind.RETRY if self._retries.get(outcome.signature, 0) < self.config.max_safe_retries else ReplanDecisionKind.REPLAN
        return ReplanDecisionKind.REPLAN if self._replans < self.config.max_replans else ReplanDecisionKind.BLOCK

    def context(self, state: TaskState, contract: BoundedExecutionContract, snapshot: RepositorySnapshot) -> dict:
        remaining = [asdict(step) for step in (state.plan.steps if state.plan else [])
                     if step.status not in {StepStatus.COMPLETED, StepStatus.SKIPPED}]
        return {
            "contract": contract.planner_context(),
            "selected_requirement_ids": list(contract.requirement_ids),
            "objective": contract.immediate_objective,
            "remaining_subgoals": remaining[:20],
            "recent_outcomes": [asdict(item) for item in self.outcomes[-8:]],
            "environment": self.environment_fingerprint(snapshot, state.plan),
            "remaining_budget": {"llm": state.budget.max_llm_calls - state.budget.llm_calls,
                                 "tool": state.budget.max_tool_actions - state.budget.tool_actions},
        }

    def validate_candidate(self, candidate: Plan, state: TaskState, contract: BoundedExecutionContract,
                           snapshot: RepositorySnapshot) -> list[str]:
        errors = candidate.validate()
        if candidate.issue_id != contract.issue_id:
            errors.append("candidate plan has a different issue")
        if candidate.requirement_ids and candidate.requirement_ids != contract.requirement_ids:
            errors.append("candidate plan targets requirements outside the current contract")
        if not candidate.steps:
            errors.append("candidate has no remaining subgoals")
        completed = {step.step_id for step in (state.plan.steps if state.plan else []) if step.status == StepStatus.COMPLETED}
        if completed & {step.step_id for step in candidate.steps}:
            errors.append("candidate unnecessarily reintroduces completed subgoals")
        if state.budget.llm_calls + len(candidate.steps) > state.budget.max_llm_calls:
            errors.append("candidate exceeds remaining LLM budget")
        fingerprint = self.plan_fingerprint(candidate, contract, snapshot)
        if fingerprint in self._seen:
            errors.append("candidate repeats an equivalent strategy in an unchanged environment")
        return errors

    def activate(self, old: Plan, candidate: Plan, contract: BoundedExecutionContract,
                 snapshot: RepositorySnapshot, outcome: ExecutionOutcome) -> None:
        candidate.parent_plan_id = old.plan_id
        candidate.version = old.version + 1
        candidate.requirement_ids = list(contract.requirement_ids)
        candidate.revision_reason = outcome.reason.value if outcome.reason else outcome.kind.value
        self._seen.add(self.plan_fingerprint(candidate, contract, snapshot))
        self._replans += 1
        self.history.append(ReplanHistoryEntry(old.plan_id, old.version, outcome.signature, "activated",
                                               candidate.revision_reason, candidate.plan_id, candidate.version))

    def reject(self, old: Plan, outcome: ExecutionOutcome, reason: str) -> None:
        self.history.append(ReplanHistoryEntry(old.plan_id, old.version, outcome.signature, "rejected", reason))

    @staticmethod
    def environment_fingerprint(snapshot: RepositorySnapshot, plan: Plan | None) -> str:
        paths = sorted({path for step in (plan.steps if plan else []) for path in step.expected_files})[:20]
        relevant = {path: snapshot.target_file_contents.get(path, "<not-captured>") for path in paths}
        value = json.dumps({"files": paths, "contents": relevant, "test": snapshot.test_command}, sort_keys=True)
        return sha256(value.encode()).hexdigest()

    @classmethod
    def plan_fingerprint(cls, plan: Plan, contract: BoundedExecutionContract, snapshot: RepositorySnapshot) -> str:
        steps = sorted((step.objective.strip().lower(), tuple(sorted(step.dependencies)),
                        step.action_type, tuple(sorted(step.expected_files)), tuple(sorted(step.completion_conditions)))
                       for step in plan.steps)
        value = json.dumps({"requirements": contract.requirement_ids, "criteria": contract.acceptance_criteria,
                            "steps": steps, "environment": cls.environment_fingerprint(snapshot, plan)}, sort_keys=True)
        return sha256(value.encode()).hexdigest()

    @staticmethod
    def _outcome(step_id: str | None, kind: ExecutionOutcomeKind, reason: InvalidationReason | None,
                 retryable: bool, detail: str) -> ExecutionOutcome:
        signature = f"{step_id or '-'}:{kind.value}:{reason.value if reason else '-'}:{detail[:160].lower()}"
        return ExecutionOutcome(step_id, kind, reason, signature, retryable, detail)
