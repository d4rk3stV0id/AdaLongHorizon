"""Append-only JSONL event logging for task telemetry."""

from __future__ import annotations

import json
from pathlib import Path

from ..models import RunEvent


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: RunEvent) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    def read_all(self) -> list[dict]:
        """Read all events from the log file."""
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
