"""Deterministic, read-only checks of executor results against repository state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .executor import Action
from .errors import RepositoryError
from .models import ExecutionRecord, PLAN_ACTION_TYPES, ToolResult
from .tools.repository import RepositoryAdapter


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    status: VerificationStatus
    checks: list[VerificationCheck] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "verified": self.verified,
            "checks": [asdict(check) for check in self.checks],
            "failures": list(self.failures),
            "evidence": list(self.evidence),
        }


class Verifier:
    """Compare the observed repository with validated actions and test results."""

    def verify(
        self,
        repository: RepositoryAdapter,
        execution_records: list[ExecutionRecord],
        test_result: ToolResult | None,
    ) -> VerificationResult:
        checks: list[VerificationCheck] = []
        failures: list[str] = []
        evidence: list[dict[str, Any]] = []

        def record_check(name: str, passed: bool, detail: str, **facts: Any) -> None:
            check = VerificationCheck(name, passed, detail, facts)
            checks.append(check)
            evidence.append({"check": name, **facts})
            if not passed:
                failures.append(f"{name}: {detail}")

        valid_records = isinstance(execution_records, list) and bool(execution_records)
        record_check(
            "execution_records_present", valid_records,
            "at least one executed file mutation is required" if not valid_records else "execution records were provided",
            count=len(execution_records) if isinstance(execution_records, list) else None,
        )

        if isinstance(execution_records, list):
            for index, record in enumerate(execution_records):
                if not isinstance(record, ExecutionRecord):
                    record_check(
                        f"execution_record_{index}_valid", False,
                        "execution record is malformed", record_type=type(record).__name__,
                    )
                    continue

                try:
                    action = Action.validate(record.action)
                except (AttributeError, TypeError, ValueError) as exc:
                    record_check(
                        f"execution_record_{index}_action_valid", False,
                        f"action is malformed or unsupported: {exc}", step_id=record.step_id,
                    )
                    continue

                supported = action.type in PLAN_ACTION_TYPES
                record_check(
                    f"execution_record_{index}_action_supported", supported,
                    "action is not a supported file mutation" if not supported else "action is supported",
                    action_type=action.type,
                    step_id=record.step_id,
                )
                if not supported:
                    continue

                result = record.tool_result
                result_valid = isinstance(result, ToolResult)
                result_matches = result_valid and result.operation == action.type
                record_check(
                    f"execution_record_{index}_tool_result_matches", result_matches,
                    "tool result is missing or reports a different operation" if not result_matches else "tool result matches the action",
                    operation=result.operation if result_valid else None,
                    action_type=action.type,
                )
                execution_succeeded = result_valid and result.success
                record_check(
                    f"execution_record_{index}_execution_succeeded", execution_succeeded,
                    "executor reported that the repository operation failed" if not execution_succeeded else "executor reported operation success",
                    success=result.success if result_valid else None,
                    stderr=result.stderr[-500:] if result_valid else None,
                )

                expected_files_valid = (
                    isinstance(record.expected_files, list)
                    and all(isinstance(path, str) and path.strip() for path in record.expected_files)
                )
                expected_files = {
                    path.replace("\\", "/") for path in record.expected_files
                } if expected_files_valid else set()
                path = action.path or ""
                path_matches = expected_files_valid and (
                    not expected_files or path.replace("\\", "/") in expected_files
                )
                record_check(
                    f"execution_record_{index}_expected_path", path_matches,
                    "action target is not listed among the step's expected files" if not path_matches else "action target is expected",
                    path=path,
                    expected_files=sorted(expected_files),
                )

                expected_content: str | None = None
                expected_content_valid = True
                if action.type == "write_file":
                    expected_content = action.content
                elif not isinstance(record.before_content, str):
                    expected_content_valid = False
                    detail = "the pre-edit file contents were not captured"
                    record_check(
                        f"execution_record_{index}_edit_precondition", False, detail, path=path,
                    )
                else:
                    try:
                        expected_content = self._apply_edits(record.before_content, action.edits or [])
                    except ValueError as exc:
                        expected_content_valid = False
                        record_check(
                            f"execution_record_{index}_edit_precondition", False,
                            f"edit cannot be independently evaluated: {exc}", path=path,
                        )
                    else:
                        record_check(
                            f"execution_record_{index}_edit_precondition", True,
                            "all exact edits apply to the captured pre-edit contents", path=path,
                            edit_count=len(action.edits or []),
                        )

                try:
                    observed_content = repository.read_file(path)
                except (RepositoryError, OSError, UnicodeError) as exc:
                    record_check(
                        f"execution_record_{index}_repository_state", False,
                        f"target file is missing or unreadable: {exc}", path=path,
                    )
                else:
                    matches = expected_content_valid and observed_content == expected_content
                    record_check(
                        f"execution_record_{index}_repository_state", matches,
                        "repository content differs from the expected action result" if not matches else "repository content matches the expected action result",
                        path=path,
                        observed_bytes=len(observed_content.encode("utf-8")),
                        expected_bytes=(len(expected_content.encode("utf-8")) if expected_content is not None else None),
                    )

        tests_valid = isinstance(test_result, ToolResult) and test_result.operation == "run_command"
        tests_passed = tests_valid and test_result.success and test_result.exit_code == 0
        record_check(
            "relevant_tests_passed", tests_passed,
            "test command result is missing, malformed, or unsuccessful" if not tests_passed else "test command succeeded",
            operation=test_result.operation if isinstance(test_result, ToolResult) else None,
            exit_code=test_result.exit_code if isinstance(test_result, ToolResult) else None,
            stdout=(test_result.stdout[-1000:] if isinstance(test_result, ToolResult) else None),
            stderr=(test_result.stderr[-1000:] if isinstance(test_result, ToolResult) else None),
        )

        status = VerificationStatus.REJECTED if failures else VerificationStatus.VERIFIED
        return VerificationResult(status, checks, failures, evidence)

    @staticmethod
    def _apply_edits(before_content: str, edits: list[dict[str, str]]) -> str:
        """Independently derive the file contents required by exact edits."""
        ranges: list[tuple[int, int, str]] = []
        for index, edit in enumerate(edits):
            old_text = edit["old_text"]
            new_text = edit["new_text"]
            first = before_content.find(old_text)
            if first < 0:
                raise ValueError(f"edit {index} old_text was not found")
            if before_content.find(old_text, first + 1) >= 0:
                raise ValueError(f"edit {index} old_text was not unique")
            ranges.append((first, first + len(old_text), new_text))

        ordered_ranges = sorted(ranges, key=lambda item: item[0])
        for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError("edit ranges overlap")

        expected = before_content
        for start, end, replacement in sorted(ranges, key=lambda item: item[0], reverse=True):
            expected = expected[:start] + replacement + expected[end:]
        return expected
