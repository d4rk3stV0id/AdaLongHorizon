"""Deterministic and provider-backed planners with shared domain validation."""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, fields, is_dataclass, replace
from enum import Enum
from typing import Any, get_args, get_origin, get_type_hints

from .llm.base import LLMProvider, LLMResponse
from .models import (
    BudgetLedger, IssueSpec, PLAN_ACTION_TYPES, Plan, PlanStep, RepositorySnapshot,
    StepStatus, TaskState,
)


class PlanValidationError(ValueError):
    """Raised when a generated plan fails structural or budget validation."""


def _schema_for_type(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin is list:
        (item_type,) = get_args(annotation)
        return {"type": "array", "items": _schema_for_type(item_type)}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [item.value for item in annotation]}
    primitives = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if annotation in primitives:
        return {"type": primitives[annotation]}
    if is_dataclass(annotation):
        return _dataclass_schema(annotation)
    raise TypeError(f"No JSON Schema mapping for {annotation!r}")


def _dataclass_schema(model: type, excluded: set[str] | None = None) -> dict[str, Any]:
    """Build a small JSON Schema adapter from the actual dataclass fields."""
    excluded = excluded or set()
    hints = get_type_hints(model)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in fields(model):
        if field.name in excluded:
            continue
        properties[field.name] = _schema_for_type(hints[field.name])
        if field.default is MISSING and field.default_factory is MISSING:
            required.append(field.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def plan_response_schema() -> dict[str, Any]:
    """Return the proposal subset of Plan/PlanStep, leaving runtime fields local."""
    step_schema = _dataclass_schema(
        PlanStep, excluded={"status", "attempts", "evidence_refs"},
    )
    # Keep these fields aligned with the domain constraints applied by Plan.validate.
    step_schema["properties"]["action_type"] = {
        "type": "string", "enum": sorted(PLAN_ACTION_TYPES),
    }
    step_schema["properties"]["risk_level"] = {
        "type": "string", "enum": ["low", "medium", "high"],
    }
    step_schema["required"] = list(step_schema["properties"])

    plan_schema = _dataclass_schema(Plan, excluded={"issue_id", "created_at"})
    plan_schema["properties"]["steps"]["items"] = step_schema
    plan_schema["required"] = ["steps"]
    return plan_schema


class Planner:
    """Use an injected provider when present; otherwise preserve the deterministic baseline."""

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider

    def create_plan(
        self,
        issue: IssueSpec,
        snapshot: RepositorySnapshot,
        state: TaskState,
        budget: BudgetLedger | None = None,
    ) -> Plan:
        if self.provider is None:
            return self.create_deterministic_plan(issue, snapshot, state, budget)
        return self._create_provider_plan(issue, snapshot, state, budget or state.budget)

    def create_deterministic_plan(
        self,
        issue: IssueSpec,
        snapshot: RepositorySnapshot,
        state: TaskState,
        budget: BudgetLedger | None = None,
    ) -> Plan:
        """Produce the existing deterministic plan used by baseline runs."""
        criteria = " ".join(issue.acceptance_criteria).lower()
        desc = issue.description.lower()

        if "function" in desc or "function" in criteria:
            impl_file = "src/utils.py" if "multiply" in desc else "src/text_tools.py"
            test_file = "tests/test_utils.py" if "multiply" in desc else "tests/test_text_tools.py"
            steps = [
                PlanStep(
                    step_id="implementation",
                    objective="Add the requested function",
                    expected_files=[impl_file],
                    action_type="write_file",
                    acceptance_checks=["function exists and is importable"],
                    risk_level="low",
                    estimated_cost=2,
                ),
                PlanStep(
                    step_id="test",
                    objective="Add a corresponding test",
                    dependencies=["implementation"],
                    expected_files=[test_file],
                    action_type="write_file",
                    acceptance_checks=["pytest passes"],
                    risk_level="low",
                    estimated_cost=1,
                ),
            ]
        else:
            files = [f for f in snapshot.files if f.endswith(".py")][:1]
            steps = [
                PlanStep(
                    step_id="change",
                    objective=f"Implement issue: {issue.title}",
                    expected_files=files,
                    acceptance_checks=issue.acceptance_criteria or ["review resulting diff"],
                    risk_level="medium",
                    estimated_cost=2,
                ),
            ]

        plan = Plan(issue.issue_id, steps)
        errors = plan.validate()
        if errors:
            raise PlanValidationError(f"Generated deterministic plan is invalid: {'; '.join(errors)}")
        return plan

    def _create_provider_plan(
        self,
        issue: IssueSpec,
        snapshot: RepositorySnapshot,
        state: TaskState,
        budget: BudgetLedger,
    ) -> Plan:
        bounded_snapshot = replace(
            snapshot,
            files=snapshot.files[:100],
            instructions={k: v[:8000] for k, v in list(snapshot.instructions.items())[:8]},
            git_status=snapshot.git_status[:2000],
            baseline_status=snapshot.baseline_status[:2000],
        )
        bounded_issue = replace(
            issue,
            title=issue.title[:500],
            description=issue.description[:4000],
            acceptance_criteria=[item[:500] for item in issue.acceptance_criteria[:20]],
        )
        schema = plan_response_schema()
        estimate = max(1, len(json.dumps({
            "issue": bounded_issue.__dict__,
            "snapshot": {
                "files": bounded_snapshot.files,
                "instructions": bounded_snapshot.instructions,
                "test_command": bounded_snapshot.test_command,
                "git_status": bounded_snapshot.git_status,
            },
            "state": {"status": state.status.value, "completed_steps": state.completed_steps},
            "schema": schema,
        }, default=str)) // 4)
        budget.check("llm")
        budget.check("tokens", estimate)
        budget.record("llm")
        budget.record("tokens", estimate)

        try:
            response = self.provider.propose_plan(
                bounded_issue, bounded_snapshot, state, budget, schema,
            )
        except Exception as exc:
            self._append_usage(state, {
                "purpose": "planning", "success": False,
                "estimated_tokens": estimate, "error": f"{type(exc).__name__}: {exc}",
            })
            raise

        usage: dict[str, Any] = {
            "purpose": "planning", "success": True,
            "provider": getattr(response, "provider", type(self.provider).__name__),
            "model": getattr(response, "model", ""),
        }
        if isinstance(response, LLMResponse):
            usage["latency_ms"] = response.duration_ms
            usage["usage_available"] = response.usage_available
            if response.usage_available:
                usage.update({
                    "input_tokens": response.prompt_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                })
                self._append_usage(state, usage)
                budget.record_actual_usage(
                    response.prompt_tokens, response.output_tokens, replace_estimated=estimate,
                )
            else:
                output_estimate = max(1, len(json.dumps(response.content, default=str)) // 4)
                usage["estimated_tokens"] = estimate + output_estimate
                self._append_usage(state, usage)
                budget.record_estimated_tokens(output_estimate)
        elif isinstance(response, Plan):
            output_estimate = max(1, len(json.dumps(asdict(response), default=str)) // 4)
            usage.update({"usage_available": False, "estimated_tokens": estimate + output_estimate})
            self._append_usage(state, usage)
            budget.record_estimated_tokens(output_estimate)
        elif isinstance(response, dict):
            output_estimate = max(1, len(json.dumps(response, default=str)) // 4)
            usage.update({"usage_available": False, "estimated_tokens": estimate + output_estimate})
            self._append_usage(state, usage)
            budget.record_estimated_tokens(output_estimate)
        else:
            self._append_usage(state, {**usage, "success": False, "error": "unsupported provider response type"})
            raise PlanValidationError("Provider returned an unsupported plan response type")

        try:
            plan = self._parse_plan_response(issue, response)
            errors = plan.validate()
            if errors:
                raise PlanValidationError("Invalid generated plan: " + "; ".join(errors))
            self._validate_budget_feasibility(plan, budget)
            if state.transient.get("last_llm_usage", {}).get("purpose") == "planning":
                state.transient["last_llm_usage"]["plan_valid"] = True
        except PlanValidationError as exc:
            self._mark_plan_invalid(state, str(exc))
            raise
        return plan

    @staticmethod
    def _parse_plan_response(issue: IssueSpec, response: LLMResponse | dict | Plan) -> Plan:
        if isinstance(response, Plan):
            if response.issue_id != issue.issue_id:
                raise PlanValidationError("Provider returned a plan for a different issue")
            if any(step.status != StepStatus.PENDING or step.attempts != 0 or step.evidence_refs
                   for step in response.steps):
                raise PlanValidationError("Provider plan contains execution state; new plan steps must be pending")
            return response
        content = response.content if isinstance(response, LLMResponse) else response
        if not isinstance(content, dict):
            raise PlanValidationError("Plan response must be a JSON object")
        if set(content) != {"steps"}:
            missing = {"steps"} - set(content)
            extra = set(content) - {"steps"}
            raise PlanValidationError(f"Plan response fields invalid (missing={sorted(missing)}, extra={sorted(extra)})")
        raw_steps = content["steps"]
        if not isinstance(raw_steps, list):
            raise PlanValidationError("Plan response 'steps' must be a list")
        required = set(plan_response_schema()["properties"]["steps"]["items"]["properties"])
        steps: list[PlanStep] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict):
                raise PlanValidationError(f"Plan step {index} must be an object")
            if set(raw_step) != required:
                missing = required - set(raw_step)
                extra = set(raw_step) - required
                raise PlanValidationError(
                    f"Plan step {index} fields invalid (missing={sorted(missing)}, extra={sorted(extra)})",
                )
            for name in ("step_id", "objective", "action_type", "risk_level"):
                if not isinstance(raw_step[name], str):
                    raise PlanValidationError(f"Plan step {index} field '{name}' must be a string")
            for name in ("dependencies", "expected_files", "acceptance_checks"):
                if (not isinstance(raw_step[name], list)
                        or not all(isinstance(value, str) for value in raw_step[name])):
                    raise PlanValidationError(f"Plan step {index} field '{name}' must be a list of strings")
            cost = raw_step["estimated_cost"]
            if isinstance(cost, bool) or not isinstance(cost, int):
                raise PlanValidationError(f"Plan step {index} field 'estimated_cost' must be an integer")
            steps.append(PlanStep(**raw_step))
        return Plan(issue.issue_id, steps)

    @staticmethod
    def _validate_budget_feasibility(plan: Plan, budget: BudgetLedger) -> None:
        # The planning call is already counted. Reserve one future LLM call per
        # step and one tool action for the orchestrator's final pytest run.
        needed_llm = len(plan.steps)
        needed_tools = len(plan.steps) + 1
        if budget.llm_calls + needed_llm > budget.max_llm_calls:
            raise PlanValidationError(
                f"Plan needs {needed_llm} additional LLM calls, but only "
                f"{budget.max_llm_calls - budget.llm_calls} remain",
            )
        if budget.tool_actions + needed_tools > budget.max_tool_actions:
            raise PlanValidationError(
                f"Plan needs {needed_tools} additional tool actions including final tests, but only "
                f"{budget.max_tool_actions - budget.tool_actions} remain",
            )

    @staticmethod
    def _append_usage(state: TaskState, usage: dict[str, Any]) -> None:
        state.transient.setdefault("llm_usage", []).append(usage)
        state.transient["last_llm_usage"] = usage

    @staticmethod
    def _mark_plan_invalid(state: TaskState, error: str) -> None:
        usage = state.transient.get("last_llm_usage")
        if usage and usage.get("purpose") == "planning":
            usage["plan_valid"] = False
            usage["validation_error"] = error
