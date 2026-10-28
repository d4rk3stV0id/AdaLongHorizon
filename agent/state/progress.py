"""Durable, verifier-grounded task progress, separate from round execution state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from ..models import IssueSpec, ToolResult
from ..verifier import VerificationResult


_ISSUE_FIELDS = {"title", "description", "acceptance_criteria", "issue_id"}
_REQUIREMENT_FIELDS = {"requirement_id", "description", "status"}
_ROUND_TRACE_REQUIRED_FIELDS = {
    "task_id", "round_id", "llm_calls", "input_tokens", "output_tokens",
    "total_tokens", "tool_calls", "planning_calls", "verification_calls",
    "execution_time_seconds", "verification_result", "final_outcome",
}
_ROUND_TRACE_OPTIONAL_FIELDS = {
    "provider_latency_ms", "planning_result", "pending_requirements",
    "pending_requirement_ids", "newly_verified_requirements",
}


class RequirementStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class ProgressStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class TaskRequirement:
    requirement_id: str
    description: str
    status: RequirementStatus = RequirementStatus.PENDING


@dataclass
class RoundTrace:
    """Observed round metrics; token values are measured or unavailable (`None`)."""

    task_id: str
    round_id: int
    llm_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tool_calls: int | None = None
    planning_calls: int | None = None
    verification_calls: int | None = None
    execution_time_seconds: float | None = None
    provider_latency_ms: float | None = None
    planning_result: str | None = None
    verification_result: str | None = None
    pending_requirements: int | None = None
    pending_requirement_ids: list[str] = field(default_factory=list)
    newly_verified_requirements: int | None = None
    final_outcome: str | None = None


@dataclass(frozen=True)
class ProviderUsageSummary:
    llm_calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    provider_latency_ms: float | None


@dataclass
class RunTrace:
    """Run summary; optional measurements remain unavailable instead of estimated."""

    task_id: str
    rounds_attempted: int
    final_outcome: str
    run_id: str | None = None
    elapsed_seconds: float | None = None
    llm_calls: int | None = None
    tool_actions: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_latency_ms: float | None = None
    verification_result: str | None = None
    rounds: list[RoundTrace] = field(default_factory=list)

    @classmethod
    def from_rounds(
        cls,
        task_id: str,
        rounds: list[RoundTrace],
        final_outcome: str,
        elapsed_seconds: float,
        verification_result: str | None,
        run_id: str | None = None,
    ) -> "RunTrace":
        """Aggregate measured round values without filling gaps with estimates."""
        llm_calls = sum(item.llm_calls for item in rounds) if all(item.llm_calls is not None for item in rounds) else None
        tool_actions = sum(item.tool_calls for item in rounds) if all(item.tool_calls is not None for item in rounds) else None
        active_rounds = [item for item in rounds if item.llm_calls is not None and item.llm_calls > 0]

        def measured_total(attribute: str) -> int | float | None:
            if llm_calls is None or not active_rounds:
                return None
            values = [getattr(item, attribute) for item in active_rounds]
            if any(value is None for value in values):
                return None
            return sum(values)

        return cls(
            task_id=task_id,
            rounds_attempted=len(rounds),
            final_outcome=final_outcome,
            run_id=run_id,
            elapsed_seconds=elapsed_seconds,
            llm_calls=llm_calls,
            tool_actions=tool_actions,
            input_tokens=measured_total("input_tokens"),
            output_tokens=measured_total("output_tokens"),
            total_tokens=measured_total("total_tokens"),
            provider_latency_ms=measured_total("provider_latency_ms"),
            verification_result=verification_result,
            rounds=list(rounds),
        )


def summarize_provider_usage(
    usage_records: list[dict[str, Any]], *, llm_provider_enabled: bool,
) -> ProviderUsageSummary:
    """Summarize provider observations; estimated tokens never enter measured totals."""
    calls = usage_records if llm_provider_enabled else []

    def reported_total(
        field_name: str, *, token_metric: bool = False,
    ) -> int | float | None:
        if not calls:
            return None
        if token_metric and any(record.get("usage_available") is not True for record in calls):
            return None
        values = [record.get(field_name) for record in calls]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (token_metric and not isinstance(value, int))
            or value < 0
            for value in values
        ):
            return None
        return sum(values)

    return ProviderUsageSummary(
        llm_calls=len(calls),
        input_tokens=reported_total("input_tokens", token_metric=True),
        output_tokens=reported_total("output_tokens", token_metric=True),
        total_tokens=reported_total("total_tokens", token_metric=True),
        provider_latency_ms=reported_total("latency_ms"),
    )


@dataclass
class PersistentTaskState:
    task_id: str
    issue: dict[str, Any]
    requirements: list[TaskRequirement]
    artifacts: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    status: ProgressStatus = ProgressStatus.IN_PROGRESS
    rounds: list[RoundTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "issue": self.issue,
            "requirements": [
                {**asdict(item), "status": item.status.value} for item in self.requirements
            ],
            "artifacts": list(self.artifacts), "facts": list(self.facts),
            "status": self.status.value,
            "rounds": [asdict(item) for item in self.rounds],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PersistentTaskState":
        if not isinstance(value, dict) or set(value) != {
            "task_id", "issue", "requirements", "artifacts", "facts", "status", "rounds",
        }:
            raise ValueError("malformed persistent task state")
        try:
            _validate_issue(value["issue"])
            _validate_string_list(value["artifacts"])
            _validate_string_list(value["facts"])
            requirements = [_requirement_from_dict(item) for item in value["requirements"]]
            rounds = [_round_trace_from_dict(item) for item in value["rounds"]]
            state = cls(
                value["task_id"], value["issue"], requirements,
                list(value["artifacts"]), list(value["facts"]),
                ProgressStatus(value["status"]), rounds,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed persistent task state") from exc
        if not isinstance(state.task_id, str) or not state.task_id.strip() or not isinstance(value["requirements"], list):
            raise ValueError("malformed persistent task state")
        if not requirements:
            raise ValueError("persistent task state requires at least one requirement")
        if len({item.requirement_id for item in requirements}) != len(requirements):
            raise ValueError("duplicate requirement IDs")
        return state


class StateUpdateError(ValueError):
    """Raised when verification evidence cannot support a safe atomic update."""


def create_progress_state(task_id: str, issue: IssueSpec) -> PersistentTaskState:
    requirement_descriptions = list(issue.acceptance_criteria) or [issue.description]
    return PersistentTaskState(
        task_id=task_id,
        issue={"title": issue.title, "description": issue.description,
               "acceptance_criteria": list(issue.acceptance_criteria), "issue_id": issue.issue_id},
        requirements=[TaskRequirement(f"R{i + 1}", text) for i, text in enumerate(requirement_descriptions)],
    )


def update_from_verification(
    state: PersistentTaskState,
    verification: VerificationResult,
    requirement_checks: dict[str, list[str]],
    test_result: ToolResult,
    round_id: int,
) -> None:
    """Atomically complete only requirements with explicit passing evidence and passing tests."""
    if not isinstance(state, PersistentTaskState):
        raise StateUpdateError("state must be persistent task state")
    if not isinstance(verification, VerificationResult):
        raise StateUpdateError("state updates require a VerificationResult")
    if verification.status.value != "verified":
        return
    if not isinstance(test_result, ToolResult) or test_result.operation != "run_command" or not test_result.success or test_result.exit_code != 0:
        return
    if not isinstance(requirement_checks, dict) or isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 1:
        raise StateUpdateError("invalid state update inputs")
    if any(not isinstance(item.name, str) or not item.name or not isinstance(item.passed, bool) for item in verification.checks):
        raise StateUpdateError("verification checks are malformed")
    checks = {item.name: item for item in verification.checks}
    by_id = {item.requirement_id: item for item in state.requirements}
    staged: list[TaskRequirement] = []
    for requirement_id, check_ids in requirement_checks.items():
        if requirement_id not in by_id or not isinstance(check_ids, list) or not check_ids:
            raise StateUpdateError(f"invalid requirement evidence mapping: {requirement_id}")
        if any(not isinstance(check_id, str) or check_id not in checks for check_id in check_ids):
            raise StateUpdateError(f"unknown verifier check mapped to {requirement_id}")
        if all(checks[check_id].passed for check_id in check_ids):
            staged.append(by_id[requirement_id])
    artifacts = list(state.artifacts)
    facts = list(state.facts)
    for requirement in staged:
        for check_id in requirement_checks[requirement.requirement_id]:
            check = checks[check_id]
            path = check.evidence.get("path") if isinstance(check.evidence, dict) else None
            if isinstance(path, str) and path and path not in artifacts:
                artifacts.append(path)
            fact = f"{requirement.requirement_id}: {check.detail}"
            if fact not in facts:
                facts.append(fact)
    for requirement in staged:
        requirement.status = RequirementStatus.COMPLETED
    state.artifacts = artifacts
    state.facts = facts
    state.status = (
        ProgressStatus.COMPLETED
        if state.requirements and all(item.status == RequirementStatus.COMPLETED for item in state.requirements)
        else ProgressStatus.IN_PROGRESS
    )
    state.rounds.append(RoundTrace(
        state.task_id, round_id, verification_result=verification.status.value,
        final_outcome=state.status.value,
    ))


def _validate_issue(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _ISSUE_FIELDS:
        raise ValueError("malformed issue")
    if not all(isinstance(value[name], str) and value[name].strip() for name in ("title", "description", "issue_id")):
        raise ValueError("malformed issue")
    _validate_string_list(value["acceptance_criteria"])


def _validate_string_list(value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("expected a list of strings")


def _requirement_from_dict(value: Any) -> TaskRequirement:
    if not isinstance(value, dict) or set(value) != _REQUIREMENT_FIELDS:
        raise ValueError("malformed requirement")
    if not all(isinstance(value[name], str) and value[name].strip() for name in ("requirement_id", "description")):
        raise ValueError("malformed requirement")
    return TaskRequirement(value["requirement_id"], value["description"], RequirementStatus(value["status"]))


def _round_trace_from_dict(value: Any) -> RoundTrace:
    if (
        not isinstance(value, dict)
        or not _ROUND_TRACE_REQUIRED_FIELDS <= set(value)
        or set(value) - (_ROUND_TRACE_REQUIRED_FIELDS | _ROUND_TRACE_OPTIONAL_FIELDS)
    ):
        raise ValueError("malformed round trace")
    defaults = asdict(RoundTrace(value["task_id"], value["round_id"]))
    value = {**defaults, **value}
    if not isinstance(value["task_id"], str) or not value["task_id"].strip():
        raise ValueError("malformed round trace")
    if isinstance(value["round_id"], bool) or not isinstance(value["round_id"], int) or value["round_id"] < 1:
        raise ValueError("malformed round trace")
    for name in ("llm_calls", "input_tokens", "output_tokens", "total_tokens", "tool_calls", "planning_calls", "verification_calls"):
        item = value[name]
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
            raise ValueError("malformed round trace")
    for name in ("execution_time_seconds", "provider_latency_ms"):
        duration = value[name]
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0):
            raise ValueError("malformed round trace")
    for name in ("pending_requirements", "newly_verified_requirements"):
        count = value[name]
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise ValueError("malformed round trace")
    if not isinstance(value["pending_requirement_ids"], list) or not all(
        isinstance(item, str) and item.strip() for item in value["pending_requirement_ids"]
    ):
        raise ValueError("malformed round trace")
    if any(value[name] is not None and not isinstance(value[name], str) for name in ("planning_result", "verification_result", "final_outcome")):
        raise ValueError("malformed round trace")
    return RoundTrace(**value)
