"""Bounded executor: selects eligible steps, calls the LLM adapter,
validates responses, performs allowed operations, and captures results.

The executor does NOT certify completion (SDD §5.3).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .errors import BudgetExceeded
from .llm.base import LLMProvider
from .models import (
    ObservationBundle, PlanStep, RepositorySnapshot, StepStatus, TaskState, ToolResult,
)
from .tools.repository import RepositoryAdapter


# ---------------------------------------------------------------------------
# Action schema
# ---------------------------------------------------------------------------

ALLOWED_ACTION_TYPES = {"write_file", "finish"}


@dataclass
class Action:
    type: str
    path: str | None = None
    content: str | None = None
    message: str | None = None

    @classmethod
    def validate(cls, value: dict[str, object]) -> "Action":
        """Parse and validate an action dict from the provider.

        Raises ``ValueError`` on malformed input.
        """
        kind = value.get("type")
        if kind not in ALLOWED_ACTION_TYPES:
            raise ValueError(f"unknown action type: {kind}")
        if kind == "write_file":
            if not isinstance(value.get("path"), str) or not value["path"]:
                raise ValueError("write_file action requires a non-empty 'path'")
            if not isinstance(value.get("content"), str):
                raise ValueError("write_file action requires a string 'content'")
            return cls(kind, value["path"], value["content"])
        if kind == "finish":
            return cls(kind, message=str(value.get("message", "")))
        raise ValueError("provider returned an invalid action schema")


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    def __init__(self, repository: RepositoryAdapter, provider: LLMProvider):
        self.repository = repository
        self.provider = provider

    def eligible_step(self, state: TaskState) -> PlanStep | None:
        """Return the next step whose dependencies are all completed."""
        if state.plan is None:
            return None
        for step in state.plan.steps:
            if step.status == StepStatus.PENDING and all(
                dep in state.completed_steps for dep in step.dependencies
            ):
                return step
        return None

    def execute_next(
        self, state: TaskState, snapshot: RepositorySnapshot, observations: ObservationBundle,
    ) -> bool:
        """Execute the next eligible step.  Returns True if a step was executed."""
        step = self.eligible_step(state)
        if step is None:
            return False

        # Mark in-progress
        step.status = StepStatus.IN_PROGRESS
        step.attempts += 1

        # Budget: LLM call
        state.budget.record("llm")

        # Build bounded context — truncate large inputs
        context = replace(
            snapshot,
            files=snapshot.files[:100],
            instructions={k: v[:8000] for k, v in list(snapshot.instructions.items())[:8]},
            git_status=snapshot.git_status[:2000],
            baseline_status=snapshot.baseline_status[:2000],
        )
        issue = replace(
            state.issue,
            title=state.issue.title[:500],
            description=state.issue.description[:4000],
            acceptance_criteria=[item[:500] for item in state.issue.acceptance_criteria[:20]],
        )

        # Token estimate
        estimate = len(repr((issue, step, context.files, context.instructions))) // 4
        state.budget.record("tokens", max(1, estimate))

        # Call provider
        action = Action.validate(self.provider.propose_action(issue, step, context, state))

        if action.type == "finish":
            state.transient["provider_message"] = action.message
            step.status = StepStatus.SKIPPED
            return False

        # Tool action
        state.budget.record("tool")
        result = self.repository.write_file(action.path or "", action.content or "")
        observations.results.append(result)

        # Evaluate basic acceptance checks
        checks_passed = self._evaluate_acceptance(step, result)

        if checks_passed:
            step.status = StepStatus.COMPLETED
            step.evidence_refs.append(f"write:{action.path}")
            state.completed_steps.append(step.step_id)
        else:
            step.status = StepStatus.FAILED

        state.transient["last_step"] = step.step_id
        return True

    @staticmethod
    def _evaluate_acceptance(step: PlanStep, result: ToolResult) -> bool:
        """Check basic step acceptance criteria against the tool result.

        For Phase 3 this is limited to checking that the write succeeded
        and the expected files were produced.  Full acceptance evaluation
        (test results, syntax checks, etc.) belongs to Phase 4.
        """
        if not result.success:
            return False
        # If expected_files are listed, check the write produced one of them
        if step.expected_files and result.changed_files:
            produced = {f.replace("\\", "/") for f in result.changed_files}
            expected = {f.replace("\\", "/") for f in step.expected_files}
            if not produced & expected:
                return False
        return True
