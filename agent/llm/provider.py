"""A deterministic provider for an end-to-end demonstration without credentials."""

from __future__ import annotations

from ..models import IssueSpec, PlanStep, RepositorySnapshot, TaskState


class DeterministicProvider:
    """Returns hard-coded actions for the normalize_whitespace demo task.

    Also supports the multiply(a, b) demonstration task.
    """

    def propose_action(self, issue: IssueSpec, step: PlanStep, snapshot: RepositorySnapshot,
                       state: TaskState) -> dict[str, object]:
        objective = step.objective.lower()

        # multiply demo — used by the demonstration task
        if "multiply" in issue.description.lower() or "multiply" in objective:
            if "test" in objective:
                return {
                    "type": "write_file",
                    "path": "tests/test_utils.py",
                    "content": (
                        'from src.utils import multiply\n\n\n'
                        'def test_multiply():\n'
                        '    assert multiply(3, 4) == 12\n\n\n'
                        'def test_multiply_zero():\n'
                        '    assert multiply(0, 5) == 0\n\n\n'
                        'def test_multiply_negative():\n'
                        '    assert multiply(-2, 3) == -6\n'
                    ),
                }
            return {
                "type": "write_file",
                "path": "src/utils.py",
                "content": (
                    'def multiply(a: int | float, b: int | float) -> int | float:\n'
                    '    """Return the product of a and b."""\n'
                    '    return a * b\n'
                ),
            }

        # normalize_whitespace demo
        if "function" in objective and "test" not in objective:
            return {
                "type": "write_file",
                "path": "src/text_tools.py",
                "content": (
                    'def normalize_whitespace(value: str) -> str:\n'
                    '    """Collapse runs of whitespace and trim the ends."""\n'
                    '    return " ".join(value.split())\n'
                ),
            }
        if "test" in objective and "function" not in objective:
            return {
                "type": "write_file",
                "path": "tests/test_text_tools.py",
                "content": (
                    'from src.text_tools import normalize_whitespace\n\n\n'
                    'def test_normalize_whitespace():\n'
                    '    assert normalize_whitespace("  Ada   Lovelace ") == "Ada Lovelace"\n'
                ),
            }

        return {"type": "finish", "message": "No deterministic action is defined for this step."}
