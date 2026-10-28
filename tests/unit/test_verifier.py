"""Behavior tests for independent, read-only repository verification."""

from pathlib import Path


def _api():
    from agent.models import ExecutionRecord, ToolResult
    from agent.tools.repository import RepositoryAdapter
    from agent.verifier import VerificationStatus, Verifier

    return ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier


def _test_success(ToolResult):
    return ToolResult("run_command", True, exit_code=0)


def _record(ExecutionRecord, ToolResult, action, *, expected_files=None, before_content=None,
            success=True, operation=None):
    return ExecutionRecord(
        step_id="step-1",
        expected_files=expected_files if expected_files is not None else [action.get("path", "src.py")],
        action=action,
        before_content=before_content,
        tool_result=ToolResult(operation or action.get("type", "write_file"), success),
    )


def _assert_rejected(result, VerificationStatus):
    assert result.status == VerificationStatus.REJECTED
    assert result.failures
    assert not result.verified


def test_verifier_accepts_written_file_when_repository_and_tests_match(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    action = {"type": "write_file", "path": "src.py", "content": "value = 3\n"}
    tool_result = repo.write_file("src.py", action["content"])
    record = ExecutionRecord("step-1", ["src.py"], action, None, tool_result)

    result = Verifier().verify(repo, [record], _test_success(ToolResult))

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified
    assert all(check.passed for check in result.checks)


def test_verifier_accepts_exact_multi_edit_and_does_not_modify_repository(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    before = '"""module docs"""\n\nvalue = 1\n'
    path = tmp_path / "src.py"
    path.write_text(before, encoding="utf-8")
    action = {
        "type": "edit_file", "path": "src.py", "edits": [
            {"old_text": '"""module docs"""', "new_text": '"""module docs"""\n\nfrom math import prod'},
            {"old_text": "value = 1", "new_text": "value = prod([2, 3])"},
        ],
    }
    tool_result = repo.edit_file("src.py", action["edits"])
    expected = '"""module docs"""\n\nfrom math import prod\n\nvalue = prod([2, 3])\n'
    assert path.read_text(encoding="utf-8") == expected
    record = ExecutionRecord("step-1", ["src.py"], action, before, tool_result)

    result = Verifier().verify(repo, [record], _test_success(ToolResult))

    assert result.status == VerificationStatus.VERIFIED
    assert path.read_text(encoding="utf-8") == expected


def test_verifier_rejects_executor_success_when_expected_file_is_absent(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    action = {"type": "write_file", "path": "src.py", "content": "value = 3\n"}
    record = _record(ExecutionRecord, ToolResult, action, success=True)

    result = Verifier().verify(repo, [record], _test_success(ToolResult))

    _assert_rejected(result, VerificationStatus)


def test_verifier_rejects_action_for_unexpected_file(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    action = {"type": "write_file", "path": "src.py", "content": "value = 3\n"}
    repo.write_file("src.py", action["content"])
    record = _record(ExecutionRecord, ToolResult, action, expected_files=["other.py"])

    result = Verifier().verify(repo, [record], _test_success(ToolResult))

    _assert_rejected(result, VerificationStatus)


def test_verifier_rejects_repository_content_that_differs_from_action(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    repo.write_file("src.py", "value = 4\n")
    action = {"type": "write_file", "path": "src.py", "content": "value = 3\n"}
    record = _record(ExecutionRecord, ToolResult, action)

    result = Verifier().verify(repo, [record], _test_success(ToolResult))

    _assert_rejected(result, VerificationStatus)


def test_verifier_rejects_failed_relevant_test_command(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    action = {"type": "write_file", "path": "src.py", "content": "value = 3\n"}
    repo.write_file("src.py", action["content"])
    record = _record(ExecutionRecord, ToolResult, action)

    result = Verifier().verify(
        repo, [record], ToolResult("run_command", False, stderr="test failed", exit_code=1),
    )

    _assert_rejected(result, VerificationStatus)


def test_verifier_rejects_edit_when_observed_state_does_not_match_expected_result(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    before = "value = 1\n"
    (tmp_path / "src.py").write_text(before, encoding="utf-8")
    action = {
        "type": "edit_file", "path": "src.py",
        "edits": [{"old_text": "value = 1", "new_text": "value = 2"}],
    }
    record = _record(ExecutionRecord, ToolResult, action, before_content=before, success=True)

    result = Verifier().verify(repo, [record], _test_success(ToolResult))

    _assert_rejected(result, VerificationStatus)


def test_verifier_rejects_incomplete_or_malformed_input(tmp_path: Path):
    _, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    verifier = Verifier()

    _assert_rejected(verifier.verify(repo, None, _test_success(ToolResult)), VerificationStatus)
    _assert_rejected(verifier.verify(repo, [], _test_success(ToolResult)), VerificationStatus)
    _assert_rejected(verifier.verify(repo, [], None), VerificationStatus)
    _assert_rejected(verifier.verify(repo, [object()], _test_success(ToolResult)), VerificationStatus)


def test_verifier_rejects_unsupported_action_type(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    action = {"type": "delete_file", "path": "src.py"}
    record = _record(ExecutionRecord, ToolResult, action, operation="delete_file")

    result = Verifier().verify(repo, [record], _test_success(ToolResult))

    _assert_rejected(result, VerificationStatus)


def test_verifier_does_not_convert_executor_failure_to_success(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    action = {"type": "write_file", "path": "src.py", "content": "value = 3\n"}
    repo.write_file("src.py", action["content"])
    record = _record(ExecutionRecord, ToolResult, action, success=False)

    result = Verifier().verify(repo, [record], _test_success(ToolResult))

    _assert_rejected(result, VerificationStatus)


def test_verifier_rejects_executor_result_for_wrong_operation(tmp_path: Path):
    ExecutionRecord, RepositoryAdapter, ToolResult, VerificationStatus, Verifier = _api()
    repo = RepositoryAdapter(tmp_path)
    action = {"type": "write_file", "path": "src.py", "content": "value = 3\n"}
    repo.write_file("src.py", action["content"])
    record = _record(ExecutionRecord, ToolResult, action, operation="edit_file")

    result = Verifier().verify(repo, [record], _test_success(ToolResult))

    _assert_rejected(result, VerificationStatus)
