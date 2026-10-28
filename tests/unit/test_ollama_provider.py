"""Unit tests for OllamaProvider with mocked HTTP layer."""

import io
import json
from unittest import mock
import urllib.error
import pytest

from agent.errors import LLMConnectionError, LLMResponseError
from agent.llm.ollama import OllamaProvider
from agent.models import IssueSpec, PlanStep, RepositorySnapshot, TaskState


def _make_http_response(status_code: int, payload: dict) -> mock.MagicMock:
    """Create a mock HTTP response context manager."""
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.status = status_code
    mock_cm = mock.MagicMock()
    mock_cm.__enter__.return_value = resp
    mock_cm.__exit__.return_value = None
    return mock_cm


def test_successful_structured_response_and_metadata_extraction():
    """Verify JSON parsing and token/duration metadata extraction from valid Ollama response."""
    raw_payload = {
        "model": "qwen3:8b",
        "created_at": "2026-10-19T23:00:00Z",
        "message": {
            "role": "assistant",
            "content": json.dumps({"type": "write_file", "path": "src/utils.py", "content": "x = 1"}),
        },
        "total_duration": 450_000_000,  # 450 ms in nanoseconds
        "prompt_eval_duration": 100_000_000,
        "eval_duration": 300_000_000,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 55,
        "eval_count": 30,
    }

    provider = OllamaProvider(host="http://localhost:11434", model="qwen3:8b")

    with mock.patch("urllib.request.urlopen", return_value=_make_http_response(200, raw_payload)):
        response = provider.chat([{"role": "user", "content": "test"}])

    # Structured content
    assert response.content == {"type": "write_file", "path": "src/utils.py", "content": "x = 1"}
    assert response.get("type") == "write_file"
    assert response["path"] == "src/utils.py"
    assert "content" in response

    # Telemetry
    assert response.model == "qwen3:8b"
    assert response.prompt_tokens == 55
    assert response.output_tokens == 30
    assert response.total_tokens == 85
    assert response.duration_ms == 450.0
    assert response.provider == "ollama"
    assert response.done is True
    assert response.done_reason == "stop"
    assert response.prompt_eval_duration_ms == 100.0
    assert response.eval_duration_ms == 300.0
    assert response.usage_available is True


def test_markdown_codeblock_stripped():
    """Verify that JSON wrapped in markdown codeblocks is parsed cleanly."""
    fenced = '```json\n{"type": "finish", "message": "Done!"}\n```'
    raw_payload = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": fenced},
        "prompt_eval_count": 10,
        "eval_count": 5,
        "total_duration": 100_000_000,
    }

    provider = OllamaProvider()
    with mock.patch("urllib.request.urlopen", return_value=_make_http_response(200, raw_payload)):
        response = provider.chat([{"role": "user", "content": "test"}])

    assert response.content == {"type": "finish", "message": "Done!"}
    assert response.get("type") == "finish"


def test_fallback_to_thinking_trace_if_content_empty():
    """Verify that if content is empty, valid JSON embedded in thinking is extracted."""
    raw_payload = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": 'Thinking about it... {\n  "type": "write_file",\n  "path": "test.py",\n  "content": ""\n}',
        },
        "prompt_eval_count": 20,
        "eval_count": 15,
        "total_duration": 200_000_000,
    }

    provider = OllamaProvider()
    with mock.patch("urllib.request.urlopen", return_value=_make_http_response(200, raw_payload)):
        response = provider.chat([{"role": "user", "content": "test"}])

    assert response.content["type"] == "write_file"
    assert response.content["path"] == "test.py"


def test_malformed_json_response_raises():
    """Verify LLMResponseError is raised when model returns unparseable JSON."""
    raw_payload = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": "Here is your JSON: {unclosed quote"},
    }

    provider = OllamaProvider()
    with mock.patch("urllib.request.urlopen", return_value=_make_http_response(200, raw_payload)):
        with pytest.raises(LLMResponseError, match="invalid JSON"):
            provider.chat([{"role": "user", "content": "test"}])


def test_non_dict_json_response_raises():
    """Verify LLMResponseError is raised when model returns a list or primitive instead of an object."""
    raw_payload = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": "[1, 2, 3]"},
    }

    provider = OllamaProvider()
    with mock.patch("urllib.request.urlopen", return_value=_make_http_response(200, raw_payload)):
        with pytest.raises(LLMResponseError, match="Expected JSON object"):
            provider.chat([{"role": "user", "content": "test"}])


def test_empty_response_raises():
    """Verify LLMResponseError is raised when model returns an empty string."""
    raw_payload = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": ""},
    }

    provider = OllamaProvider()
    with mock.patch("urllib.request.urlopen", return_value=_make_http_response(200, raw_payload)):
        with pytest.raises(LLMResponseError, match="empty response"):
            provider.chat([{"role": "user", "content": "test"}])


@pytest.mark.parametrize("raw", [[], None, "not an object"])
def test_non_object_api_response_raises_cleanly(raw):
    provider = OllamaProvider()
    with pytest.raises(LLMResponseError, match="JSON object"):
        provider._parse_chat_response(raw)


def test_missing_model_in_api_response_raises_cleanly():
    provider = OllamaProvider()
    with pytest.raises(LLMResponseError, match="missing a valid model"):
        provider._parse_chat_response({"message": {"content": '{"ok": true}'}})


def test_invalid_model_in_api_response_raises_cleanly():
    provider = OllamaProvider()
    with pytest.raises(LLMResponseError, match="missing a valid model"):
        provider._parse_chat_response({"model": 8, "message": {"content": '{"ok": true}'}})


def test_invalid_usage_metadata_raises_cleanly():
    provider = OllamaProvider()
    raw = {
        "model": "qwen3:8b",
        "message": {"content": '{"ok": true}'},
        "prompt_eval_count": "many",
    }
    with pytest.raises(LLMResponseError, match="invalid prompt_eval_count"):
        provider._parse_chat_response(raw)


def test_absent_usage_counts_are_marked_unavailable():
    provider = OllamaProvider()
    response = provider._parse_chat_response({
        "model": "qwen3:8b", "message": {"content": '{"ok": true}'},
    })
    assert response.usage_available is False


def test_malformed_api_json_raises_cleanly():
    response = mock.MagicMock()
    response.read.return_value = b"not json"
    context = mock.MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = None
    provider = OllamaProvider()
    with mock.patch("urllib.request.urlopen", return_value=context):
        with pytest.raises(LLMResponseError, match="malformed JSON"):
            provider.chat([{"role": "user", "content": "test"}])


def test_http_error_handling():
    """Verify LLMResponseError is raised when Ollama returns HTTP 500 or 404."""
    http_err = urllib.error.HTTPError(
        url="http://localhost:11434/api/chat",
        code=500,
        msg="Internal Server Error",
        hdrs=None,
        fp=io.BytesIO(b'{"error": "model crashed"}'),
    )

    provider = OllamaProvider()
    with mock.patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(LLMResponseError, match="Ollama HTTP 500"):
            provider.chat([{"role": "user", "content": "test"}])


def test_connection_failure_handling():
    """Verify LLMConnectionError is raised when Ollama daemon is not reachable."""
    url_err = urllib.error.URLError("Connection refused")

    provider = OllamaProvider()
    with mock.patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(LLMConnectionError, match="Failed to connect to Ollama"):
            provider.chat([{"role": "user", "content": "test"}])


def test_timeout_handling():
    """Verify LLMConnectionError is raised when request times out."""
    provider = OllamaProvider()
    with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(LLMConnectionError, match="Failed to connect to Ollama"):
            provider.chat([{"role": "user", "content": "test"}])


def test_propose_action_request_formatting():
    """Verify propose_action formats issue, step, and repository snapshot into chat request."""
    provider = OllamaProvider(model="qwen3:8b")
    issue = IssueSpec("Add multiply", "Add a multiply function", ["returns product"])
    step = PlanStep("step-1", "Implement function", expected_files=["src/math.py"])
    snapshot = RepositorySnapshot(
        "root", ["src/init.py", "unrelated.py"], {}, "", None, "", None,
        {"src/math.py": '"""Math helpers."""\n\nvalue = 1\n'},
    )
    state = TaskState(issue)

    sent_body = None

    def fake_urlopen(req, timeout):
        nonlocal sent_body
        sent_body = json.loads(req.data.decode("utf-8"))
        return _make_http_response(
            200,
            {
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": '{"type": "finish", "message": "done"}'},
                "prompt_eval_count": 100,
                "eval_count": 20,
                "total_duration": 150_000_000,
            },
        )

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        response = provider.propose_action(issue, step, snapshot, state)

    assert sent_body is not None
    assert sent_body["model"] == "qwen3:8b"
    assert sent_body["format"] == "json"
    assert sent_body["think"] is False
    assert sent_body["think"] is False
    assert len(sent_body["messages"]) == 2
    assert "Add multiply" in sent_body["messages"][1]["content"]
    assert "src/math.py" in sent_body["messages"][1]["content"]
    assert "Acceptance Criteria" in sent_body["messages"][1]["content"]
    assert "write_file" in sent_body["messages"][1]["content"]
    assert "edit_file" in sent_body["messages"][1]["content"]
    assert '"edits": [{"old_text":' in sent_body["messages"][0]["content"]
    assert "exact, unique text" in sent_body["messages"][0]["content"]
    assert "exactly once" in sent_body["messages"][1]["content"]
    assert "write_file only for new files or explicitly requested full replacement" in sent_body["messages"][1]["content"]
    assert "Tests must import every symbol they use; add a separate edit for any missing import" in sent_body["messages"][1]["content"]
    assert "multiple exact replacements in the same file" in sent_body["messages"][1]["content"]
    assert "Keep edits small and preserve unrelated content" in sent_body["messages"][1]["content"]
    assert "Math helpers." in sent_body["messages"][1]["content"]
    assert "secret = True" not in sent_body["messages"][1]["content"]
    assert response.content["type"] == "finish"


def test_thinking_text_is_exposed_separately_from_content():
    raw_payload = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": '{"type":"finish","message":"done"}',
            "thinking": "Reasoning is provided in a separate field.",
        },
        "prompt_eval_count": 3,
        "eval_count": 20,
        "total_duration": 100_000_000,
    }
    response = OllamaProvider()._parse_chat_response(raw_payload)
    assert response.content == {"type": "finish", "message": "done"}
    assert response.thinking == "Reasoning is provided in a separate field."
    assert response.output_tokens == 20


def test_propose_plan_uses_json_schema_and_bounded_generation_settings():
    provider = OllamaProvider(model="qwen3:8b")
    issue = IssueSpec("Add multiply", "Add multiply(a, b)", ["returns product"])
    snapshot = RepositorySnapshot(
        ".", ["src/utils.py", "tests/test_utils.py"], {}, "clean", None, "clean", ["pytest", "-q"],
    )
    state = TaskState(issue)
    schema = {
        "type": "object", "properties": {"steps": {"type": "array", "items": {"type": "object"}}},
        "required": ["steps"], "additionalProperties": False,
    }
    raw = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": '{"steps": []}'},
        "prompt_eval_count": 40, "eval_count": 5, "total_duration": 100_000_000,
    }
    request_body = None

    def fake_urlopen(req, timeout):
        nonlocal request_body
        request_body = json.loads(req.data.decode("utf-8"))
        return _make_http_response(200, raw)

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        response = provider.propose_plan(issue, snapshot, state, state.budget, schema)

    assert request_body["format"] == schema
    assert request_body["think"] is False
    assert request_body["options"] == {"temperature": 0.0, "num_predict": 1536}
    prompt = request_body["messages"][1]["content"]
    assert "Add multiply(a, b)" in prompt
    assert "returns product" in prompt
    assert "src/utils.py" in prompt
    assert response.content == {"steps": []}
