from __future__ import annotations

import json
import os
import hashlib
import threading
from pathlib import Path
from uuid import uuid4

from ..models import JournalIntent, JournalOutcome, RecoveryCheckpoint, TaskState
from .progress import PersistentTaskState


class TaskOwnershipError(RuntimeError):
    """The task is already owned by another execution or recovery process."""


class TaskOwnershipLock:
    """Nonblocking OS lock, released by the OS if its process exits."""

    _guard = threading.RLock()
    _owned: dict[str, tuple[int, object, int]] = {}

    def __init__(self, root: Path, task_id: str):
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        self.path = root / f"{digest}.task-lock"
        self._key = str(self.path.resolve())
        self._entered = False

    def __enter__(self):
        thread_id = threading.get_ident()
        with self._guard:
            current = self._owned.get(self._key)
            if current:
                owner_thread, stream, depth = current
                if owner_thread != thread_id:
                    raise TaskOwnershipError("task is already owned in this process")
                self._owned[self._key] = (owner_thread, stream, depth + 1)
                self._entered = True
                return self

            stream = self.path.open("a+b")
            try:
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as exc:
                stream.close()
                raise TaskOwnershipError("task is already owned by another process") from exc
            self._owned[self._key] = (thread_id, stream, 1)
            self._entered = True
            return self

    def __exit__(self, exc_type, exc, tb):
        if not self._entered:
            return False
        with self._guard:
            owner_thread, stream, depth = self._owned[self._key]
            if depth > 1:
                self._owned[self._key] = (owner_thread, stream, depth - 1)
            else:
                del self._owned[self._key]
                try:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                finally:
                    stream.close()
        self._entered = False
        return False


class StateStore:
    CHECKPOINT_VERSION = 1
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def task_lock(self, task_id: str) -> TaskOwnershipLock:
        if not task_id:
            raise ValueError("task_id must not be empty")
        return TaskOwnershipLock(self.root, task_id)

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

    def _atomic_json(self, path: Path, value: dict) -> Path:
        """Write a same-directory snapshot; replacement never exposes a partial file."""
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        payload = json.dumps(value, indent=2, sort_keys=True)
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)
        return path

    def _checkpoint_path(self, task_id: str, previous: bool = False) -> Path:
        suffix = ".recovery.previous.json" if previous else ".recovery.json"
        return self.root / f"{task_id}{suffix}"

    def journal_path(self, task_id: str) -> Path:
        return self.root / f"{task_id}.journal.jsonl"

    def save_recovery_checkpoint(self, checkpoint: RecoveryCheckpoint) -> Path:
        self._validate_checkpoint(checkpoint)
        path = self._checkpoint_path(checkpoint.task_id)
        previous = self._checkpoint_path(checkpoint.task_id, previous=True)
        if path.exists():
            # Preserve the last complete snapshot before replacement.
            self._atomic_json(previous, json.loads(path.read_text(encoding="utf-8")))
        return self._atomic_json(path, checkpoint.__dict__)

    def load_recovery_checkpoint(self, task_id: str, repository_id: str) -> tuple[RecoveryCheckpoint, bool]:
        failures: list[str] = []
        for fallback, path in ((False, self._checkpoint_path(task_id)), (True, self._checkpoint_path(task_id, True))):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = RecoveryCheckpoint(**value)
                self._validate_checkpoint(checkpoint)
                if checkpoint.task_id != task_id or checkpoint.repository_id != repository_id:
                    raise ValueError("checkpoint identity does not match recovery target")
                return checkpoint, fallback
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                failures.append(f"{path.name}: {exc}")
        raise ValueError("no valid recovery checkpoint: " + "; ".join(failures))

    def append_journal(self, task_id: str, record: JournalIntent | JournalOutcome) -> None:
        if record.task_id != task_id or record.sequence < 1:
            raise ValueError("invalid journal record identity or sequence")
        path = self.journal_path(task_id)
        payload = json.dumps(record.__dict__, sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def read_journal(self, task_id: str, repository_id: str) -> list[dict]:
        path = self.journal_path(task_id)
        if not path.exists():
            return []
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        records: list[dict] = []
        expected = 1
        for index, line in enumerate(lines):
            final = index == len(lines) - 1
            if not line.endswith((b"\n", b"\r")):
                # A non-newline final tail is safe to ignore only if all prior records validate.
                if final:
                    break
                raise ValueError("journal has a malformed non-final record")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if final:
                    break
                raise ValueError("journal has malformed record") from exc
            if (not isinstance(value, dict) or value.get("sequence") != expected
                    or value.get("task_id") != task_id or value.get("repository_id") != repository_id
                    or value.get("kind") not in {"intent", "outcome"}):
                raise ValueError("journal sequence or identity integrity failure")
            records.append(value)
            expected += 1
        return records

    def validate_journal_prefix(self, task_id: str, repository_id: str, position: int) -> list[dict]:
        """Read and validate the entire journal, then return only its post-checkpoint suffix."""
        records = self.read_journal(task_id, repository_id)
        if position < 0 or position > len(records):
            raise ValueError("checkpoint journal_position is outside the valid journal")
        return records[position:]

    @classmethod
    def _validate_checkpoint(cls, checkpoint: RecoveryCheckpoint) -> None:
        if checkpoint.schema_version != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported recovery checkpoint schema")
        if not checkpoint.task_id or not checkpoint.repository_id or checkpoint.journal_position < 0:
            raise ValueError("malformed recovery checkpoint identity")
        if not isinstance(checkpoint.runtime_state, dict) or not isinstance(checkpoint.repository_observation, dict):
            raise ValueError("malformed recovery checkpoint payload")
        if checkpoint.replanning_state is not None and not isinstance(checkpoint.replanning_state, dict):
            raise ValueError("malformed recovery checkpoint replanning_state")
        try:
            state = TaskState.from_dict(checkpoint.runtime_state)
        except Exception as exc:
            raise ValueError(f"malformed runtime state payload: {exc}") from exc
        if state.task_id != checkpoint.task_id:
            raise ValueError("runtime state task_id does not match checkpoint task_id")
        if state.plan is not None:
            plan_errors = state.plan.validate()
            if plan_errors:
                raise ValueError(f"malformed runtime state plan: {'; '.join(plan_errors)}")
