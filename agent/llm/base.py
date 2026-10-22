from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import IssueSpec, PlanStep, RepositorySnapshot, TaskState


@dataclass
class LLMResponse:
    """Carries structured model output and inference usage telemetry."""
    content: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    model: str = ""

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
