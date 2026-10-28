import json

import pytest

from agent.models import IssueSpec, ToolResult
from agent.state.progress import (
    PersistentTaskState,
    ProgressStatus,
    RequirementStatus,
    RoundTrace,
    RunTrace,
    StateUpdateError,
    create_progress_state,
    summarize_provider_usage,
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


def test_existing_persistent_round_trace_without_new_optional_fields_still_loads():
    progress = create_progress_state("task-1", IssueSpec("Add X", "Do X", ["X exists"]))
    value = progress.to_dict()
    value["rounds"] = [{
        "task_id": "task-1", "round_id": 1, "llm_calls": 2,
        "input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
        "tool_calls": 2, "planning_calls": 1, "verification_calls": 1,
        "execution_time_seconds": 1.5, "verification_result": "verified",
        "final_outcome": "in_progress",
    }]

    restored = PersistentTaskState.from_dict(value)

    assert restored.rounds[0].round_id == 1
    assert restored.rounds[0].provider_latency_ms is None
    assert restored.rounds[0].pending_requirement_ids == []


def test_executor_tool_success_cannot_be_used_as_verified_state_evidence():
    progress = create_progress_state("task-1", IssueSpec("Add X", "Do X", ["X exists"]))

    with pytest.raises(StateUpdateError):
        update_from_verification(
            progress, ToolResult("write_file", True), {"R1": ["x_exists"]},
            ToolResult("run_command", True, exit_code=0), round_id=1,
        )

    assert progress.requirements[0].status == RequirementStatus.PENDING


def test_round_and_run_trace_models_capture_observed_counts_and_leave_missing_values_unavailable():
    round_trace = RoundTrace(
        task_id="task-1", round_id=1, llm_calls=3, tool_calls=2,
        input_tokens=1493, output_tokens=411, total_tokens=1904,
        provider_latency_ms=1250.5, planning_result="accepted",
        verification_result="verified", pending_requirements=2,
        pending_requirement_ids=["R1", "R2"],
        newly_verified_requirements=1,
        execution_time_seconds=3.2, final_outcome="in_progress",
    )
    run_trace = RunTrace(
        task_id="task-1", rounds_attempted=1, final_outcome="in_progress",
        llm_calls=3, tool_actions=2, input_tokens=1493, output_tokens=411,
        total_tokens=1904, provider_latency_ms=1250.5,
        verification_result="verified", rounds=[round_trace],
    )

    assert round_trace.newly_verified_requirements == 1
    assert round_trace.pending_requirements == 2
    assert round_trace.pending_requirement_ids == ["R1", "R2"]
    assert run_trace.rounds[0].round_id == 1
    assert run_trace.input_tokens == 1493

    persistent = create_progress_state("task-1", IssueSpec("Add X", "Do X", ["X exists"]))
    persistent.rounds.append(round_trace)
    restored = PersistentTaskState.from_dict(persistent.to_dict())
    assert restored.rounds == [round_trace]

    unavailable = RoundTrace("task-2", 1)
    unavailable_run = RunTrace("task-2", 1, "completed")
    assert unavailable.input_tokens is None
    assert unavailable.output_tokens is None
    assert unavailable.provider_latency_ms is None
    assert unavailable_run.input_tokens is None
    assert unavailable_run.output_tokens is None
    assert unavailable_run.total_tokens is None


def test_provider_usage_summary_uses_reported_counts_tokens_and_latency():
    summary = summarize_provider_usage(
        [
            {"usage_available": True, "input_tokens": 10, "output_tokens": 3, "total_tokens": 13, "latency_ms": 120.5},
            {"usage_available": True, "input_tokens": 8, "output_tokens": 2, "total_tokens": 10, "latency_ms": 80.0},
        ],
        llm_provider_enabled=True,
    )

    assert summary.llm_calls == 2
    assert summary.input_tokens == 18
    assert summary.output_tokens == 5
    assert summary.total_tokens == 23
    assert summary.provider_latency_ms == 200.5


def test_provider_usage_summary_does_not_convert_estimated_tokens_to_measured_usage():
    summary = summarize_provider_usage(
        [{"usage_available": False, "estimated_tokens": 409, "latency_ms": 52.5}],
        llm_provider_enabled=True,
    )

    assert summary.llm_calls == 1
    assert summary.input_tokens is None
    assert summary.output_tokens is None
    assert summary.total_tokens is None
    assert summary.provider_latency_ms == 52.5


def test_deterministic_provider_is_not_reported_as_an_llm_call():
    summary = summarize_provider_usage(
        [{"provider": "DeterministicProvider", "estimated_tokens": 100}],
        llm_provider_enabled=False,
    )

    assert summary.llm_calls == 0
    assert summary.input_tokens is None


def test_run_trace_aggregates_rounds_and_marks_partial_usage_unavailable():
    first = RoundTrace(
        "task-1", 1, llm_calls=2, tool_calls=4, input_tokens=10,
        output_tokens=3, total_tokens=13, provider_latency_ms=120.0,
    )
    second = RoundTrace(
        "task-1", 2, llm_calls=1, tool_calls=3, input_tokens=8,
        output_tokens=2, total_tokens=10, provider_latency_ms=80.0,
    )

    run = RunTrace.from_rounds("task-1", [first, second], "completed", 4.5, "verified")

    assert run.rounds_attempted == 2
    assert run.llm_calls == 3
    assert run.tool_actions == 7
    assert run.input_tokens == 18
    assert run.output_tokens == 5
    assert run.total_tokens == 23
    assert run.provider_latency_ms == 200.0
    assert run.verification_result == "verified"

    incomplete = RoundTrace("task-1", 2, llm_calls=1, input_tokens=None, output_tokens=None, total_tokens=None)
    partial = RunTrace.from_rounds("task-1", [first, incomplete], "blocked", 3.0, "rejected")
    assert partial.llm_calls == 3
    assert partial.input_tokens is None
    assert partial.output_tokens is None
    assert partial.total_tokens is None
