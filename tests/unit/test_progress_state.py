import json

import pytest

from agent.models import IssueSpec, ToolResult
from agent.state.progress import (
    PersistentTaskState,
    ProgressStatus,
    RequirementStatus,
    StateUpdateError,
    create_progress_state,
    update_from_verification,
)
from agent.state.store import StateStore
from agent.verifier import VerificationCheck, VerificationResult, VerificationStatus


def test_initial_requirements_are_pending_and_task_is_in_progress():
    progress = create_progress_state(
        "task-1", IssueSpec("Add X", "Do X", ["X exists", "tests cover X"]),
    )

    assert [requirement.status for requirement in progress.requirements] == [
        RequirementStatus.PENDING, RequirementStatus.PENDING,
    ]
    assert progress.status == ProgressStatus.IN_PROGRESS


def test_task_without_acceptance_criteria_has_one_stable_requirement():
    progress = create_progress_state("task-1", IssueSpec("Add X", "Do X"))

    assert [(item.requirement_id, item.description) for item in progress.requirements] == [("R1", "Do X")]


def test_only_requirement_with_mapped_passed_check_and_round_tests_completes():
    progress = create_progress_state(
        "task-1", IssueSpec("Add X", "Do X", ["X exists", "Y exists"]),
    )
    verification = VerificationResult(
        VerificationStatus.VERIFIED,
        [VerificationCheck("repository_state", True, "matches")],
    )

    update_from_verification(
        progress, verification, {"R1": ["repository_state"]},
        ToolResult("run_command", True, exit_code=0), round_id=1,
    )

    assert [item.status for item in progress.requirements] == [
        RequirementStatus.COMPLETED, RequirementStatus.PENDING,
    ]
    assert progress.status == ProgressStatus.IN_PROGRESS


def test_verified_update_records_mapped_facts_and_artifacts_only():
    progress = create_progress_state("task-1", IssueSpec("Add X", "Do X", ["X exists", "Y exists"]))
    verification = VerificationResult(
        VerificationStatus.VERIFIED,
        [
            VerificationCheck("x_exists", True, "src/x.py exists", {"path": "src/x.py"}),
            VerificationCheck("y_exists", True, "src/y.py exists", {"path": "src/y.py"}),
        ],
    )

    update_from_verification(
        progress, verification, {"R1": ["x_exists"]},
        ToolResult("run_command", True, exit_code=0), round_id=1,
    )

    assert progress.artifacts == ["src/x.py"]
    assert progress.facts == ["R1: src/x.py exists"]
    assert progress.requirements[1].status == RequirementStatus.PENDING


@pytest.mark.parametrize(
    "verification,test_result,mapping",
    [
        (VerificationResult(VerificationStatus.REJECTED, [VerificationCheck("repository_state", False, "missing")]), ToolResult("run_command", True, exit_code=0), {"R1": ["repository_state"]}),
        (VerificationResult(VerificationStatus.VERIFIED, [VerificationCheck("repository_state", True, "matches")]), ToolResult("run_command", False, exit_code=1), {"R1": ["repository_state"]}),
        (VerificationResult(VerificationStatus.VERIFIED, [VerificationCheck("repository_state", True, "matches")]), ToolResult("run_command", True, exit_code=0), {}),
    ],
)
def test_rejected_or_failed_or_unmapped_evidence_does_not_complete(verification, test_result, mapping):
    progress = create_progress_state("task-1", IssueSpec("Add X", "Do X", ["X exists"]))

    update_from_verification(progress, verification, mapping, test_result, round_id=1)

    assert progress.requirements[0].status == RequirementStatus.PENDING
    assert progress.status == ProgressStatus.IN_PROGRESS


def test_unknown_check_mapping_rejects_entire_state_update_atomically():
    progress = create_progress_state(
        "task-1", IssueSpec("Add X", "Do X", ["X exists", "Y exists"]),
    )
    verification = VerificationResult(
        VerificationStatus.VERIFIED,
        [VerificationCheck("repository_state", True, "matches")],
    )

    with pytest.raises(StateUpdateError):
        update_from_verification(
            progress, verification,
            {"R1": ["repository_state"], "R2": ["not_a_check"]},
            ToolResult("run_command", True, exit_code=0), round_id=1,
        )

    assert all(item.status == RequirementStatus.PENDING for item in progress.requirements)


def test_all_verified_requirements_complete_task_and_survive_json_round_trip(tmp_path):
    progress = create_progress_state("task-1", IssueSpec("Add X", "Do X", ["X exists"]))
    verification = VerificationResult(
        VerificationStatus.VERIFIED,
        [VerificationCheck("repository_state", True, "matches")],
    )
    update_from_verification(
        progress, verification, {"R1": ["repository_state"]},
        ToolResult("run_command", True, exit_code=0), round_id=1,
    )
    store = StateStore(tmp_path)
    store.save_progress(progress)

    restored = store.load_progress("task-1")

    assert restored == progress
    assert restored.status == ProgressStatus.COMPLETED


def test_corrupt_persistent_progress_is_rejected(tmp_path):
    (tmp_path / "task-1.progress.json").write_text("{broken", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        StateStore(tmp_path).load_progress("task-1")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(status="unknown"),
        lambda value: value["requirements"][0].update(status="unknown"),
        lambda value: value["artifacts"].append(7),
        lambda value: value["issue"].pop("title"),
        lambda value: value["requirements"][0].update(extra="not allowed"),
    ],
)
def test_structurally_invalid_progress_json_is_rejected(mutate):
    progress = create_progress_state("task-1", IssueSpec("Add X", "Do X", ["X exists"]))
    value = progress.to_dict()
    mutate(value)

    with pytest.raises(ValueError):
        PersistentTaskState.from_dict(value)


def test_executor_tool_success_cannot_be_used_as_verified_state_evidence():
    progress = create_progress_state("task-1", IssueSpec("Add X", "Do X", ["X exists"]))

    with pytest.raises(StateUpdateError):
        update_from_verification(
            progress, ToolResult("write_file", True), {"R1": ["x_exists"]},
            ToolResult("run_command", True, exit_code=0), round_id=1,
        )

    assert progress.requirements[0].status == RequirementStatus.PENDING
