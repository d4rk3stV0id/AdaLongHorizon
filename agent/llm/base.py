from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import BudgetLedger, IssueSpec, Plan, PlanStep, RepositorySnapshot, TaskState


@dataclass
class LLMResponse:
    """Carries structured model output and inference usage telemetry."""
    content: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    model: str = ""
    provider: str = ""
    done: bool | None = None
    done_reason: str | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None
    thinking: str | None = None
    usage_available: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        return self.content.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.content[key]

    def __contains__(self, key: str) -> bool:
        return key in self.content


class LLMProvider(Protocol):
    def propose_action(
        self, issue: IssueSpec, step: PlanStep, snapshot: RepositorySnapshot, state: TaskState
    ) -> LLMResponse | dict[str, object]: ...

    def propose_plan(
        self,
        issue: IssueSpec,
        snapshot: RepositorySnapshot,
        state: TaskState,
        budget: BudgetLedger,
        response_schema: dict[str, Any],
    ) -> LLMResponse | dict[str, object] | Plan: ...
