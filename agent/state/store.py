from __future__ import annotations

import json
import os
from pathlib import Path

from ..models import TaskState


class StateStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state: TaskState) -> Path:
        path = self.root / f"{state.task_id}.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        os.replace(temp, path)
        return path

    def load(self, task_id: str) -> TaskState:
        return TaskState.from_dict(json.loads((self.root / f"{task_id}.json").read_text(encoding="utf-8")))
