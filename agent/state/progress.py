"""Durable, verifier-grounded task progress, separate from round execution state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from ..models import IssueSpec, ToolResult
from ..verifier import VerificationResult


_ISSUE_FIELDS = {"title", "description", "acceptance_criteria", "issue_id"}
_REQUIREMENT_FIELDS = {"requirement_id", "description", "status"}
_ROUND_TRACE_FIELDS = {
    "task_id", "round_id", "llm_calls", "input_tokens", "output_tokens",
    "total_tokens", "tool_calls", "planning_calls", "verification_calls",
    "execution_time_seconds", "verification_result", "final_outcome",
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
    verification_result: str | None = None
    final_outcome: str | None = None


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
    if not isinstance(value, dict) or set(value) != _ROUND_TRACE_FIELDS:
        raise ValueError("malformed round trace")
    if not isinstance(value["task_id"], str) or not value["task_id"].strip():
        raise ValueError("malformed round trace")
    if isinstance(value["round_id"], bool) or not isinstance(value["round_id"], int) or value["round_id"] < 1:
        raise ValueError("malformed round trace")
    for name in ("llm_calls", "input_tokens", "output_tokens", "total_tokens", "tool_calls", "planning_calls", "verification_calls"):
        item = value[name]
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
            raise ValueError("malformed round trace")
    duration = value["execution_time_seconds"]
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0):
        raise ValueError("malformed round trace")
    if any(value[name] is not None and not isinstance(value[name], str) for name in ("verification_result", "final_outcome")):
        raise ValueError("malformed round trace")
    return RoundTrace(**value)
