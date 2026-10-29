"""Deterministic Manage control decisions for the bounded M3 MEA loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .errors import BudgetExceeded
from .models import BudgetLedger, IssueSpec
from .state.progress import PersistentTaskState, ProgressStatus, RequirementStatus


class ControlDecisionKind(StrEnum):
    EXECUTE = "EXECUTE"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class BoundedExecutionContract:
    """The durable, bounded input permitted to enter one execution round."""

    task_id: str
    round_id: int
    issue_id: str
    immediate_objective: str
    requirement_ids: list[str]
    acceptance_criteria: list[str]
    verified_facts: list[str]
    verified_artifacts: list[str]
    constraints: list[str] = field(default_factory=lambda: [
        "Mutate only through the repository adapter.",
        "Completion requires independent verifier evidence and passing round tests.",
    ])
    expected_verification_condition: str = (
        "Mapped verifier checks and the required round test command must pass."
    )

    def to_issue(self, original_issue: dict[str, object]) -> IssueSpec:
        """Construct the narrow issue passed to the fresh planner/executor context."""
        return IssueSpec(
            title=str(original_issue["title"]),
            description=(
                f"{original_issue['description']}\n"
                f"Current round requirement ({self.requirement_ids[0]}): "
                f"{self.immediate_objective}"
            ),
            acceptance_criteria=list(self.acceptance_criteria),
            issue_id=self.issue_id,
        )

    def planner_context(self) -> dict[str, object]:
        return {
            "pending_requirements": [
                {"requirement_id": requirement_id, "description": criterion}
                for requirement_id, criterion in zip(self.requirement_ids, self.acceptance_criteria)
            ],
            "verified_facts": list(self.verified_facts),
            "artifacts": list(self.verified_artifacts),
            "constraints": list(self.constraints),
            "expected_verification_condition": self.expected_verification_condition,
        }


@dataclass(frozen=True)
class ControlDecision:
    kind: ControlDecisionKind
    reason: str
    contract: BoundedExecutionContract | None = None


class MEAManager:
    """Select one verified-state-backed unit of work or an explicit terminal."""

    def decide(
        self,
        progress: PersistentTaskState,
        budget: BudgetLedger,
        *,
        max_rounds: int,
    ) -> ControlDecision:
        if progress.status == ProgressStatus.COMPLETED or all(
            item.status == RequirementStatus.COMPLETED for item in progress.requirements
        ):
            return ControlDecision(ControlDecisionKind.DONE, "all requirements are verified complete")
        if progress.status == ProgressStatus.BLOCKED:
            return ControlDecision(ControlDecisionKind.BLOCKED, "persistent task is already blocked")
        if len(progress.rounds) >= max_rounds:
            return ControlDecision(
                ControlDecisionKind.BLOCKED,
                f"configured round limit reached ({len(progress.rounds)}/{max_rounds})",
            )
        try:
            # Every supported round needs at least one proposed action and one
            # test/verification tool operation.  Finer plan feasibility stays
            # with Planner because it knows the number of plan steps.
            budget.check("llm", 1)
            budget.check("tool", 1)
            budget.check("tokens", 0)
        except BudgetExceeded as exc:
            return ControlDecision(ControlDecisionKind.BLOCKED, f"budget boundary reached: {exc}")

        requirement = next(
            (item for item in progress.requirements if item.status == RequirementStatus.PENDING), None,
        )
        if requirement is None:
            return ControlDecision(ControlDecisionKind.BLOCKED, "no eligible pending requirement exists")

        contract = BoundedExecutionContract(
            task_id=progress.task_id,
            round_id=len(progress.rounds) + 1,
            issue_id=str(progress.issue["issue_id"]),
            immediate_objective=requirement.description[:1000],
            requirement_ids=[requirement.requirement_id],
            acceptance_criteria=[requirement.description[:1000]],
            verified_facts=[fact[:1000] for fact in progress.facts[:20]],
            verified_artifacts=[artifact[:500] for artifact in progress.artifacts[:20]],
        )
        return ControlDecision(
            ControlDecisionKind.EXECUTE,
            f"selected pending requirement {requirement.requirement_id}",
            contract,
        )
