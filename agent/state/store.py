from __future__ import annotations

import json
from pathlib import Path

from ..models import TaskState
from .progress import PersistentTaskState


class StateStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state: TaskState) -> Path:
        path = self.root / f"{state.task_id}.json"
        path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        return path

    def load(self, task_id: str) -> TaskState:
        return TaskState.from_dict(json.loads((self.root / f"{task_id}.json").read_text(encoding="utf-8")))

    def save_progress(self, state: PersistentTaskState) -> Path:
        path = self.root / f"{state.task_id}.progress.json"
        path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load_progress(self, task_id: str) -> PersistentTaskState:
        value = json.loads((self.root / f"{task_id}.progress.json").read_text(encoding="utf-8"))
        return PersistentTaskState.from_dict(value)
