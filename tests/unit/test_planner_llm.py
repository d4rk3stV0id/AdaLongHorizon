"""Provider-backed Planner validation tests; no Ollama service required."""

import pytest

from agent.errors import LLMConnectionError
from agent.llm.base import LLMResponse
from agent.models import BudgetLedger, IssueSpec, Plan, PlanStep, RepositorySnapshot, TaskState
from agent.planner import PlanValidationError, Planner, plan_response_schema


def _step(step_id="implementation", objective="Implement the requested change", dependencies=None):
    return {
        "step_id": step_id,
        "objective": objective,
        "dependencies": dependencies or [],
        "expected_files": ["src/utils.py"],
        "action_type": "write_file",
        "acceptance_checks": ["pytest passes"],
        "risk_level": "low",
        "estimated_cost": 1,
    }


class StubPlanProvider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def propose_plan(self, issue, snapshot, state, budget, response_schema):
        self.calls.append((issue, snapshot, state, budget, response_schema))
        if self.error:
            raise self.error
        return self.response


def _response(content):
    return LLMResponse(
        content=content,
        prompt_tokens=12,
        output_tokens=18,
        total_tokens=30,
        duration_ms=250.0,
        model="test-model",
        provider="test",
        usage_available=True,
    )


def _inputs(budget=None):
    issue = IssueSpec("Add multiply", "Add a multiply function", ["returns the product"])
    snapshot = RepositorySnapshot(
        ".", ["src/__init__.py", "tests/test_utils.py"],
        {"AGENTS.md": "Keep changes small."}, "clean", "abc", "clean", ["python", "-m", "pytest", "-q"],
    )
    state = TaskState(issue, budget=budget or BudgetLedger())
    return issue, snapshot, state


def test_valid_structured_plan_becomes_valid_plan_and_records_actual_usage():
    issue, snapshot, state = _inputs()
    provider = StubPlanProvider(_response({"steps": [_step()]}))
    plan = Planner(provider).create_plan(issue, snapshot, state)

    assert isinstance(plan, Plan)
    assert plan.issue_id == issue.issue_id
    assert plan.validate() == []
    assert plan.steps[0].status.value == "PENDING"
    assert plan.steps[0].attempts == 0
    assert plan.steps[0].evidence_refs == []
    assert state.budget.llm_calls == 1
    assert state.budget.actual_input_tokens == 12
    assert state.budget.actual_output_tokens == 18
    assert state.budget.actual_tokens == 30
    assert state.budget.estimated_tokens == 0
    assert state.transient["last_llm_usage"]["purpose"] == "planning"
    assert provider.calls[0][1].files == snapshot.files
    schema = provider.calls[0][4]
    assert set(schema["properties"]) == {"steps"}
    assert "status" not in schema["properties"]["steps"]["items"]["properties"]


def test_plan_without_usage_metadata_remains_estimated():
    issue, snapshot, state = _inputs()
    response = LLMResponse(content={"steps": [_step()]}, model="provider-without-usage")
    plan = Planner(StubPlanProvider(response)).create_plan(issue, snapshot, state)
    assert plan.validate() == []
    assert state.budget.actual_tokens == 0
    assert state.budget.estimated_tokens > 0
    assert state.transient["last_llm_usage"]["usage_available"] is False


def test_malformed_plan_object_is_rejected():
    issue, snapshot, state = _inputs()
    provider = StubPlanProvider(_response(["not", "an", "object"]))
    with pytest.raises(PlanValidationError, match="JSON object"):
        Planner(provider).create_plan(issue, snapshot, state)


def test_plan_missing_required_step_field_is_rejected():
    issue, snapshot, state = _inputs()
    invalid = _step()
    del invalid["expected_files"]
    with pytest.raises(PlanValidationError, match="missing=.*expected_files"):
        Planner(StubPlanProvider(_response({"steps": [invalid]}))).create_plan(issue, snapshot, state)


def test_duplicate_step_ids_are_rejected():
    issue, snapshot, state = _inputs()
    provider = StubPlanProvider(_response({"steps": [
        _step("same", "First"), _step("same", "Second"),
    ]}))
    with pytest.raises(PlanValidationError, match="duplicate step_id"):
        Planner(provider).create_plan(issue, snapshot, state)


@pytest.mark.parametrize("steps, message", [
    ([_step("one", "One", ["missing"])], "unknown step"),
    ([_step("one", "One", ["two"]), _step("two", "Two", ["one"])], "cycle"),
])
def test_invalid_dependencies_and_cycles_are_rejected(steps, message):
    issue, snapshot, state = _inputs()
    with pytest.raises(PlanValidationError, match=message):
        Planner(StubPlanProvider(_response({"steps": steps}))).create_plan(issue, snapshot, state)


def test_unsupported_action_and_unsafe_paths_are_rejected():
    issue, snapshot, state = _inputs()
    invalid = _step()
    invalid["action_type"] = "delete_file"
    invalid["expected_files"] = ["../outside.py"]
    with pytest.raises(PlanValidationError, match="unsupported action_type|unsafe expected file"):
        Planner(StubPlanProvider(_response({"steps": [invalid]}))).create_plan(issue, snapshot, state)


def test_budget_infeasible_plan_is_rejected_after_planning_call():
    budget = BudgetLedger(max_llm_calls=1)
    issue, snapshot, state = _inputs(budget)
    provider = StubPlanProvider(_response({"steps": [_step()]}))
    with pytest.raises(PlanValidationError, match="additional LLM calls"):
        Planner(provider).create_plan(issue, snapshot, state)
    assert budget.llm_calls == 1


def test_provider_failure_is_visible_and_does_not_fall_back():
    issue, snapshot, state = _inputs()
    provider = StubPlanProvider(error=LLMConnectionError("offline"))
    with pytest.raises(LLMConnectionError, match="offline"):
        Planner(provider).create_plan(issue, snapshot, state)
    assert len(provider.calls) == 1
    assert state.transient["last_llm_usage"]["success"] is False


def test_deterministic_planner_baseline_remains_available():
    issue, snapshot, state = _inputs()
    plan = Planner().create_plan(issue, snapshot, state)
    assert [step.step_id for step in plan.steps] == ["implementation", "test"]
    assert plan.validate() == []


def test_schema_adapter_tracks_plan_step_domain_fields():
    schema = plan_response_schema()
    step_schema = schema["properties"]["steps"]["items"]
    assert set(step_schema["properties"]) == {
        "step_id", "objective", "dependencies", "expected_files", "action_type",
        "acceptance_checks", "risk_level", "estimated_cost",
    }
    assert set(step_schema["properties"]["action_type"]["enum"]) == {"write_file", "edit_file"}
