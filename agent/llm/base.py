from __future__ import annotations

from typing import Protocol

from ..models import IssueSpec, PlanStep, RepositorySnapshot, TaskState


class LLMProvider(Protocol):
    def propose_action(self, issue: IssueSpec, step: PlanStep, snapshot: RepositorySnapshot,
                       state: TaskState) -> dict[str, object]: ...
