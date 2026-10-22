"""Ollama provider for local LLM inference via HTTP API."""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from typing import Any

from ..errors import LLMConnectionError, LLMResponseError
from ..models import IssueSpec, PlanStep, RepositorySnapshot, TaskState
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
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

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
                raw_response = json.loads(raw_bytes.decode("utf-8"))
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
        message = raw.get("message", {})
        content = message.get("content", "").strip()

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

        prompt_tokens = int(raw.get("prompt_eval_count", 0))
        output_tokens = int(raw.get("eval_count", 0))
        total_duration_ns = int(raw.get("total_duration", 0))
        duration_ms = round(total_duration_ns / 1_000_000, 2)

        return LLMResponse(
            content=parsed,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            duration_ms=duration_ms,
            model=raw.get("model", self.model),
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
            "You are an autonomous software engineering assistant. "
            "You must respond ONLY with a single valid JSON object representing the next action.\n"
            "Action schemas allowed:\n"
            "1. Write a file:\n"
            '   {"type": "write_file", "path": "<relative_path>", "content": "<file_content>"}\n'
            "2. Complete task:\n"
            '   {"type": "finish", "message": "<completion_summary>"}\n'
            "Do not include any explanation or commentary outside the JSON."
        )

        files_context = "\n".join(snapshot.files[:50])
        user_prompt = (
            f"Task Issue: {issue.title}\n"
            f"Description: {issue.description}\n\n"
            f"Current Step: {step.step_id} - {step.objective}\n"
            f"Expected Files: {', '.join(step.expected_files) if step.expected_files else 'None'}\n\n"
            f"Available Files in Repository:\n{files_context}\n\n"
            "Propose the next action as JSON."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.chat(messages, format="json")
