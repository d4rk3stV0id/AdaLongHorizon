"""Ollama provider for local LLM inference via HTTP API."""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from typing import Any

from ..errors import LLMConnectionError, LLMResponseError
from ..models import BudgetLedger, IssueSpec, PlanStep, RepositorySnapshot, TaskState
from .base import LLMResponse


class OllamaProvider:
    """Connects to a locally running Ollama instance (default: http://localhost:11434).

    Produces structured JSON responses and extracts real token usage metadata.
    Does NOT enforce action/plan domain validity—that is left to the validation layer.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        timeout: float = 120.0,
        think: bool = False,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.think = think

    def chat(
        self,
        messages: list[dict[str, str]],
        format: str | dict[str, Any] = "json",
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a chat request to Ollama and parse structured JSON output and telemetry."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "format": format,
            "stream": False,
            "think": self.think,
            "options": options or {"temperature": 0.0},
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_bytes = resp.read()
                try:
                    raw_response = json.loads(raw_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise LLMResponseError(f"Ollama returned malformed JSON: {exc}") from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMResponseError(f"Ollama HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self.host}: {exc}"
            ) from exc

        return self._parse_chat_response(raw_response)

    def _parse_chat_response(self, raw: dict[str, Any]) -> LLMResponse:
        """Extract structured JSON and token metadata from raw Ollama response."""
        if not isinstance(raw, dict):
            raise LLMResponseError("Ollama response must be a JSON object")
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            raise LLMResponseError("Ollama response is missing a valid model name")
        message = raw.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("Ollama response is missing a valid message object")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise LLMResponseError("Ollama message content must be a string")
        content = content.strip()

        # If content is empty but model emitted a thinking trace, check if JSON was emitted there
        if not content and message.get("thinking"):
            thinking = message["thinking"]
            json_match = re.search(r"\{[\s\S]*\}", thinking)
            if json_match:
                content = json_match.group(0).strip()

        if not content:
            raise LLMResponseError("Model returned an empty response")

        # Strip markdown fences if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Model returned invalid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise LLMResponseError(
                f"Expected JSON object from model, got {type(parsed).__name__}"
            )

        def usage_count(name: str) -> int:
            value = raw.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LLMResponseError(f"Ollama response has invalid {name}")
            return value

        prompt_tokens = usage_count("prompt_eval_count")
        output_tokens = usage_count("eval_count")
        total_duration_ns = usage_count("total_duration")
        prompt_duration_ns = usage_count("prompt_eval_duration")
        eval_duration_ns = usage_count("eval_duration")
        done = raw.get("done")
        if done is not None and not isinstance(done, bool):
            raise LLMResponseError("Ollama response has invalid done status")
        duration_ms = round(total_duration_ns / 1_000_000, 2)

        return LLMResponse(
            content=parsed,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            duration_ms=duration_ms,
            model=model,
            provider="ollama",
            done=done,
            done_reason=raw.get("done_reason") if isinstance(raw.get("done_reason"), str) else None,
            prompt_eval_duration_ms=round(prompt_duration_ns / 1_000_000, 2),
            eval_duration_ms=round(eval_duration_ns / 1_000_000, 2),
            thinking=message.get("thinking") if isinstance(message.get("thinking"), str) else None,
            usage_available="prompt_eval_count" in raw and "eval_count" in raw,
        )

    def propose_action(
        self,
        issue: IssueSpec,
        step: PlanStep,
        snapshot: RepositorySnapshot,
        state: TaskState,
    ) -> LLMResponse:
        """Propose an action for the current step formatted as structured JSON."""
        system_prompt = (
            "You are a software engineering assistant. "
            "Respond only with one JSON object representing the next action. "
            "Use exactly one of the action shapes below; do not add other fields.\n"
            "Action schemas allowed:\n"
            "1. Write a file (new file or intentional full replacement):\n"
            '   {"type": "write_file", "path": "<relative_path>", "content": "<file_content>"}\n'
            "2. Edit one existing file with one or more exact, unique text replacements:\n"
            '   {"type": "edit_file", "path": "<relative_path>", "edits": [{"old_text": "<exact unique text>", "new_text": "<replacement>"}]}\n'
            "3. Complete task:\n"
            '   {"type": "finish", "message": "<completion_summary>"}\n'
            "Do not include any explanation or commentary outside the JSON."
        )

        files_context = "\n".join(snapshot.files[:50])
        user_prompt = (
            f"Task Issue: {issue.title}\n"
            f"Description: {issue.description}\n\n"
            f"Acceptance Criteria: {', '.join(issue.acceptance_criteria) if issue.acceptance_criteria else 'None'}\n\n"
            f"Current Step: {step.step_id} - {step.objective}\n"
            f"Expected Files: {', '.join(step.expected_files) if step.expected_files else 'None'}\n\n"
            f"Repository Instructions:\n{chr(10).join(snapshot.instructions.values())[:16000] or 'None'}\n\n"
            f"Available Files in Repository:\n{files_context}\n\n"
            f"Current contents of expected existing files:\n{json.dumps(snapshot.target_file_contents, ensure_ascii=False) if snapshot.target_file_contents else 'None'}\n\n"
            f"Planned Action Type: {step.action_type}\n"
            "For a file mutation, the action type must match the planned action_type exactly. "
            "Choose the action that fits the current step. For implementation steps, "
            "Use edit_file for localized changes; one action may contain multiple exact replacements in the same file. "
            "Copy every old_text exactly from the provided contents; each must occur exactly once, and edits must not overlap. "
            "Keep edits small and preserve unrelated content. Use write_file only for new files or explicitly requested full replacement. "
            "When adding tests, include every required import in the same file action as the test body. "
            "Tests must import every symbol they use; add a separate edit for any missing import. "
            "Use finish only if no change is needed."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.chat(messages, format="json")

    def propose_plan(
        self,
        issue: IssueSpec,
        snapshot: RepositorySnapshot,
        state: TaskState,
        budget: BudgetLedger,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        """Generate a JSON-Schema-constrained plan proposal with bounded context."""
        files_context = "\n".join(snapshot.files[:100])
        instructions = "\n\n".join(
            f"--- {path} ---\n{text[:8000]}"
            for path, text in list(snapshot.instructions.items())[:8]
        )[:16000] or "None"
        user_prompt = (
            f"Issue title: {issue.title}\n"
            f"Issue description: {issue.description}\n"
            f"Acceptance criteria: {json.dumps(issue.acceptance_criteria)}\n\n"
            f"Current task status: {state.status.value}\n"
            f"Completed steps: {json.dumps(state.completed_steps)}\n"
            f"Durable verified context: {json.dumps(state.transient.get('persistent_context'))}\n"
            f"Available files (these are the only files known to exist):\n{files_context}\n\n"
            f"Project instructions:\n{instructions}\n\n"
            f"Test command: {json.dumps(snapshot.test_command)}\n"
            f"Git status: {snapshot.git_status[:2000]}\n\n"
            "Ground the plan in the issue, acceptance criteria, available files, and instructions. "
            "When durable verified context is present, plan only its pending requirements and do not repeat completed work. "
            "Treat any path absent from the available file list as a proposed new file, not as an existing file. "
            "Choose a small sequence of file mutation steps. Set action_type to edit_file for localized changes "
            "to files listed as available, and write_file for new files or intentional full replacements. "
            "Dependencies must refer to step IDs in this plan. "
            "Plan implementation and test changes separately when both are required. "
            "Return only the object matching the provided JSON Schema."
        )
        budget_context = {
            "remaining_llm_calls": max(0, budget.max_llm_calls - budget.llm_calls),
            "remaining_tool_actions": max(0, budget.max_tool_actions - budget.tool_actions),
            "remaining_token_budget": max(
                0, budget.max_token_estimate - budget.estimated_tokens - budget.actual_tokens,
            ),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You create concise software change plans. The JSON Schema is authoritative. "
                    "Do not include execution results or claim that any step has already completed."
                ),
            },
            {"role": "user", "content": user_prompt + "\n\nBudget available for this run: "
             + json.dumps(budget_context)},
        ]
        return self.chat(
            messages,
            format=response_schema,
            options={"temperature": 0.0, "num_predict": 1536},
        )
