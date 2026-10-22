"""Typed records shared by the agent's replaceable components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class Lifecycle(StrEnum):
    CREATED = "CREATED"
    INSPECTING = "INSPECTING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


ALLOWED_TRANSITIONS: dict[Lifecycle, set[Lifecycle]] = {
    Lifecycle.CREATED: {Lifecycle.INSPECTING, Lifecycle.FAILED},
    Lifecycle.INSPECTING: {Lifecycle.PLANNING, Lifecycle.BLOCKED, Lifecycle.FAILED, Lifecycle.BUDGET_EXHAUSTED},
    Lifecycle.PLANNING: {Lifecycle.EXECUTING, Lifecycle.BLOCKED, Lifecycle.FAILED, Lifecycle.BUDGET_EXHAUSTED},
    Lifecycle.EXECUTING: {Lifecycle.VERIFYING, Lifecycle.RECOVERING, Lifecycle.BLOCKED, Lifecycle.FAILED, Lifecycle.BUDGET_EXHAUSTED},
    Lifecycle.VERIFYING: {Lifecycle.COMPLETED, Lifecycle.RECOVERING, Lifecycle.BLOCKED, Lifecycle.FAILED, Lifecycle.BUDGET_EXHAUSTED},
    Lifecycle.RECOVERING: {Lifecycle.PLANNING, Lifecycle.EXECUTING, Lifecycle.BLOCKED, Lifecycle.FAILED, Lifecycle.BUDGET_EXHAUSTED},
    Lifecycle.COMPLETED: set(),
    Lifecycle.BLOCKED: set(),
    Lifecycle.FAILED: set(),
    Lifecycle.BUDGET_EXHAUSTED: set(),
}

# ---------------------------------------------------------------------------
# Step status
# ---------------------------------------------------------------------------

class StepStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------

@dataclass
class IssueSpec:
    title: str
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    issue_id: str = field(default_factory=lambda: str(uuid4()))


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in an execution plan.

    Field names match the SDD §5.3 suggested PlanStep schema:
    step_id, objective, dependencies, expected_files, action_type,
    acceptance_checks, risk_level, estimated_cost, status, attempts,
    evidence_refs.

    The legacy field ``description`` is kept as an alias write-path
    but ``objective`` is the canonical field.
    """
    step_id: str
    objective: str
    dependencies: list[str] = field(default_factory=list)
    expected_files: list[str] = field(default_factory=list)
    action_type: str = "write_file"
    acceptance_checks: list[str] = field(default_factory=list)
    risk_level: str = "low"
    estimated_cost: int = 1
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    evidence_refs: list[str] = field(default_factory=list)

    # Convenience backward-compat alias
    @property
    def description(self) -> str:
        return self.objective


@dataclass
class Plan:
    issue_id: str
    steps: list[PlanStep]
    created_at: str = field(default_factory=utc_now)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        if not self.steps:
            errors.append("plan has no steps")
        ids = {s.step_id for s in self.steps}
        if len(ids) != len(self.steps):
            errors.append("duplicate step_id values")
        for step in self.steps:
            if not step.step_id or not step.objective:
                errors.append(f"step missing step_id or objective")
            for dep in step.dependencies:
                if dep not in ids:
                    errors.append(f"step '{step.step_id}' depends on unknown step '{dep}'")
        return errors


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass
class BudgetLedger:
    max_llm_calls: int = 12
    max_tool_actions: int = 24
    max_retries: int = 3
    max_consecutive_identical_failures: int = 2
    command_timeout_seconds: float = 30.0
    max_task_duration_seconds: float = 900.0
    max_token_estimate: int = 20000
    max_cost_estimate: float | None = None
    llm_calls: int = 0
    tool_actions: int = 0
    retries: int = 0
    estimated_tokens: int = 0
    estimated_cost: float = 0.0
    consecutive_failures: int = 0
    last_failure_signature: str | None = None
    started_at: str = field(default_factory=utc_now)

    def check(self, kind: str, amount: int = 1) -> None:
        from .errors import BudgetExceeded

        limits = {
            "llm": (self.llm_calls, self.max_llm_calls),
            "tool": (self.tool_actions, self.max_tool_actions),
            "retry": (self.retries, self.max_retries),
            "tokens": (self.estimated_tokens, self.max_token_estimate),
        }
        if kind not in limits:
            raise ValueError(f"unknown budget kind: {kind}")
        used, limit = limits[kind]
        if used + amount > limit:
            raise BudgetExceeded(f"{kind} budget exhausted ({used}/{limit})")
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(self.started_at)).total_seconds()
        if elapsed > self.max_task_duration_seconds:
            raise BudgetExceeded("task duration budget exhausted")

    def record(self, kind: str, amount: int = 1) -> None:
        self.check(kind, amount)
        attr = {"llm": "llm_calls", "tool": "tool_actions", "retry": "retries", "tokens": "estimated_tokens"}[kind]
        setattr(self, attr, getattr(self, attr) + amount)

    def record_cost(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("cost cannot be negative")
        if self.max_cost_estimate is not None and self.estimated_cost + amount > self.max_cost_estimate:
            from .errors import BudgetExceeded
            raise BudgetExceeded("estimated cost budget exhausted")
        self.estimated_cost += amount

    def record_failure(self, signature: str) -> None:
        self.consecutive_failures = self.consecutive_failures + 1 if signature == self.last_failure_signature else 1
        self.last_failure_signature = signature
        if self.consecutive_failures > self.max_consecutive_identical_failures:
            from .errors import BudgetExceeded
            raise BudgetExceeded("consecutive identical failure budget exhausted")

    def clear_failures(self) -> None:
        self.consecutive_failures = 0
        self.last_failure_signature = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BudgetLedger":
        return cls(**value)


# ---------------------------------------------------------------------------
# Task state
# ---------------------------------------------------------------------------

@dataclass
class TaskState:
    issue: IssueSpec
    status: Lifecycle = Lifecycle.CREATED
    plan: Plan | None = None
    completed_steps: list[str] = field(default_factory=list)
    transient: dict[str, Any] = field(default_factory=dict)
    budget: BudgetLedger = field(default_factory=BudgetLedger)
    task_id: str = field(default_factory=lambda: str(uuid4()))
    updated_at: str = field(default_factory=utc_now)

    def transition(self, target: Lifecycle) -> None:
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid lifecycle transition: {self.status} -> {target}")
        self.status = target
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["issue"]["issue_id"] = self.issue.issue_id
        if self.plan:
            plan_dict = asdict(self.plan)
            for i, step in enumerate(self.plan.steps):
                plan_dict["steps"][i]["status"] = step.status.value
            value["plan"] = plan_dict
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskState":
        data = dict(value)
        data["issue"] = IssueSpec(**data["issue"])
        data["status"] = Lifecycle(data["status"])
        data["budget"] = BudgetLedger.from_dict(data["budget"])
        if data.get("plan"):
            plan = dict(data["plan"])
            steps = []
            for step_data in plan["steps"]:
                step_data = dict(step_data)
                step_data["status"] = StepStatus(step_data.get("status", "PENDING"))
                steps.append(PlanStep(**step_data))
            plan["steps"] = steps
            data["plan"] = Plan(**plan)
        return cls(**data)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class RunEvent:
    task_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Tool results and observations
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    operation: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    changed_files: list[str] = field(default_factory=list)
    timed_out: bool = False


@dataclass
class ObservationBundle:
    task_id: str
    results: list[ToolResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Repository snapshot
# ---------------------------------------------------------------------------

@dataclass
class RepositorySnapshot:
    root: str
    files: list[str]
    instructions: dict[str, str]
    git_status: str
    baseline_commit: str | None
    baseline_status: str
    test_command: list[str] | None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    workspace: str
    state_dir: str = ".agent-state"
    max_file_bytes: int = 100_000
    max_snapshot_files: int = 500
    command_timeout_seconds: float = 30.0
    provider: str = "deterministic"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
