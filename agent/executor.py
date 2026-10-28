"""Bounded executor: selects eligible steps, calls the LLM adapter,
validates responses, performs allowed operations, and captures results.

The executor does NOT certify completion (SDD §5.3).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

from .errors import BudgetExceeded, RepositoryError
from .llm.base import LLMProvider, LLMResponse
from .models import (
    ExecutionRecord, ObservationBundle, PLAN_ACTION_TYPES, PlanStep,
    RepositorySnapshot, StepStatus, TaskState, ToolResult,
)
from .tools.repository import RepositoryAdapter


# ---------------------------------------------------------------------------
# Action schema
# ---------------------------------------------------------------------------

ALLOWED_ACTION_TYPES = PLAN_ACTION_TYPES | {"finish"}


@dataclass
class Action:
    type: str
    path: str | None = None
    content: str | None = None
    message: str | None = None
    edits: list[dict[str, str]] | None = None

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
        if kind == "edit_file":
            if not isinstance(value.get("path"), str) or not value["path"]:
                raise ValueError("edit_file action requires a non-empty 'path'")
            edits = value.get("edits")
            if not isinstance(edits, list):
                raise ValueError("edit_file action 'edits' must be a list")
            if not edits:
                raise ValueError("edit_file action requires a non-empty 'edits' list")
            normalized_edits: list[dict[str, str]] = []
            for index, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    raise ValueError(f"edit_file edit {index} must be an object")
                if not isinstance(edit.get("old_text"), str) or not edit["old_text"]:
                    raise ValueError(f"edit_file edit {index} requires a non-empty 'old_text'")
                if not isinstance(edit.get("new_text"), str):
                    raise ValueError(f"edit_file edit {index} requires a string 'new_text'")
                normalized_edits.append({"old_text": edit["old_text"], "new_text": edit["new_text"]})
            return cls(kind, path=value["path"], edits=normalized_edits)
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
        state.budget.check("llm")

        # Build bounded context — truncate large inputs
        context = replace(
            snapshot,
            files=snapshot.files[:100],
            instructions={k: v[:8000] for k, v in list(snapshot.instructions.items())[:8]},
            git_status=snapshot.git_status[:2000],
            baseline_status=snapshot.baseline_status[:2000],
        )
        target_file_contents: dict[str, str] = {}
        remaining_chars = 8000
        for path in step.expected_files[:10]:
            if remaining_chars <= 0:
                break
            try:
                contents = self.repository.read_file(path)
            except RepositoryError:
                # Missing/new files have no current contents to provide.
                continue
            if len(contents) > remaining_chars:
                marker = "\n[content truncated for prompt size]"
                target_file_contents[path] = contents[:max(0, remaining_chars - len(marker))] + marker
                remaining_chars = 0
            else:
                target_file_contents[path] = contents
                remaining_chars -= len(contents)
        context = replace(context, target_file_contents=target_file_contents)
        issue = replace(
            state.issue,
            title=state.issue.title[:500],
            description=state.issue.description[:4000],
            acceptance_criteria=[item[:500] for item in state.issue.acceptance_criteria[:20]],
        )

        # Token estimate
        estimate = len(repr((issue, step, context.files, context.instructions,
                             context.target_file_contents))) // 4
        estimate = max(1, estimate)
        state.budget.check("tokens", estimate)
        state.budget.record("llm")
        state.budget.record("tokens", max(1, estimate))

        # Call provider
        try:
            provider_response = self.provider.propose_action(issue, step, context, state)
        except Exception as exc:
            usage = {
                "purpose": "action", "success": False,
                "provider": type(self.provider).__name__,
                "estimated_tokens": estimate,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._record_usage(state, usage)
            raise
        if isinstance(provider_response, LLMResponse):
            usage = {
                "purpose": "action", "success": True,
                "provider": provider_response.provider,
                "model": provider_response.model,
                "usage_available": provider_response.usage_available,
                "latency_ms": provider_response.duration_ms,
                "done": provider_response.done,
                "done_reason": provider_response.done_reason,
                "thinking_present": bool(provider_response.thinking),
            }
            if provider_response.usage_available:
                usage.update({
                    "input_tokens": provider_response.prompt_tokens,
                    "output_tokens": provider_response.output_tokens,
                    "total_tokens": provider_response.total_tokens,
                })
            else:
                output_estimate = max(1, len(json.dumps(provider_response.content, default=str)) // 4)
                usage["estimated_tokens"] = estimate + output_estimate
            self._record_usage(state, usage)
            state.transient["last_llm_response"] = {
                "provider": provider_response.provider,
                "model": provider_response.model,
                "prompt_tokens": provider_response.prompt_tokens if provider_response.usage_available else None,
                "output_tokens": provider_response.output_tokens if provider_response.usage_available else None,
                "total_tokens": provider_response.total_tokens if provider_response.usage_available else None,
                "duration_ms": provider_response.duration_ms,
                "done": provider_response.done,
                "done_reason": provider_response.done_reason,
                "thinking_present": bool(provider_response.thinking),
                "usage_available": provider_response.usage_available,
            }
            if provider_response.usage_available:
                state.budget.record_actual_usage(
                    provider_response.prompt_tokens, provider_response.output_tokens,
                    replace_estimated=estimate,
                )
            else:
                state.budget.record_estimated_tokens(output_estimate)
            action_value = provider_response.content
        else:
            action_value = provider_response
            output_estimate = max(1, len(json.dumps(provider_response, default=str)) // 4)
            self._record_usage(state, {
                "purpose": "action", "success": True,
                "provider": type(self.provider).__name__, "usage_available": False,
                "estimated_tokens": estimate + output_estimate,
            })
            state.budget.record_estimated_tokens(output_estimate)
        action = Action.validate(action_value)
        if action.type != "finish" and action.type != step.action_type:
            raise ValueError(
                f"action type '{action.type}' does not match planned action_type '{step.action_type}'",
            )
        state.transient["last_action"] = asdict(action)
        state.transient.setdefault("generated_actions", []).append(asdict(action))

        if action.type == "finish":
            state.transient["provider_message"] = action.message
            step.status = StepStatus.SKIPPED
            return False

        # Tool action
        state.budget.record("tool")
        before_content = None
        if action.type == "edit_file":
            try:
                before_content = self.repository.read_file(action.path or "")
            except RepositoryError:
                pass
            result = self.repository.edit_file(action.path or "", action.edits or [])
        else:
            result = self.repository.write_file(action.path or "", action.content or "")
        observations.results.append(result)
        observations.execution_records.append(ExecutionRecord(
            step_id=step.step_id,
            expected_files=list(step.expected_files),
            action=asdict(action),
            before_content=before_content,
            tool_result=result,
        ))

        # Evaluate basic acceptance checks
        checks_passed = self._evaluate_acceptance(step, result)

        if checks_passed:
            step.status = StepStatus.COMPLETED
            step.evidence_refs.append(f"{action.type}:{action.path}")
            state.completed_steps.append(step.step_id)
        else:
            step.status = StepStatus.FAILED

        state.transient["last_step"] = step.step_id
        return True

    @staticmethod
    def _record_usage(state: TaskState, usage: dict) -> None:
        state.transient.setdefault("llm_usage", []).append(usage)
        state.transient["last_llm_usage"] = usage

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
