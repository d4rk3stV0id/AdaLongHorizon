"""Provider-independent planner that produces structured, validated plans."""

from __future__ import annotations

from .models import (
    BudgetLedger, IssueSpec, Plan, PlanStep, RepositorySnapshot, TaskState,
)


class PlanValidationError(ValueError):
    """Raised when a generated plan fails structural validation."""


class Planner:
    """Produce a small explicit plan; the deterministic prototype targets Python changes."""

    def create_plan(
        self,
        issue: IssueSpec,
        snapshot: RepositorySnapshot,
        state: TaskState,
        budget: BudgetLedger | None = None,
    ) -> Plan:
        """Generate and validate a plan for the given issue.

        The plan is structurally validated before being returned.
        Malformed plans raise ``PlanValidationError``.
        """
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
            raise PlanValidationError(f"Generated plan is invalid: {'; '.join(errors)}")
        return plan
