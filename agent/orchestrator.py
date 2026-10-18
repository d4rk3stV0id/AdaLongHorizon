"""Task lifecycle orchestrator — coordinates inspection, planning,
execution, and verification into a single run.

Phase 3 scope: verification is limited to running the discovered test
command and checking that all plan steps completed.  Independent auditing,
adaptive verification, and sophisticated recovery belong to Phase 4+.
"""

from __future__ import annotations

from pathlib import Path

from .errors import BudgetExceeded
from .executor import Executor
from .llm.provider import DeterministicProvider
from .models import (
    AgentConfig, BudgetLedger, IssueSpec, Lifecycle, ObservationBundle,
    RunEvent, StepStatus, TaskState,
)
from .planner import Planner
from .state.store import StateStore
from .telemetry.logger import EventLogger
from .tools.repository import RepositoryAdapter


class Orchestrator:
    def __init__(self, config: AgentConfig, provider=None):
        self.config = config
        self.repository = RepositoryAdapter(
            config.workspace, timeout=config.command_timeout_seconds,
        )
        state_dir = Path(config.workspace) / config.state_dir
        self.store = StateStore(state_dir)
        self.events = EventLogger(state_dir / "events.jsonl")
        self.provider = provider or DeterministicProvider()
        self.planner = Planner()

    def run(
        self, issue: IssueSpec, budget: BudgetLedger | None = None,
    ) -> tuple[TaskState, ObservationBundle]:
        state = TaskState(
            issue,
            budget=budget or BudgetLedger(command_timeout_seconds=self.config.command_timeout_seconds),
        )
        obs = ObservationBundle(state.task_id)
        self._save_event(state, "task_created")

        try:
            # --- INSPECTING ---
            state.transition(Lifecycle.INSPECTING)
            snapshot = self.repository.inspect()
            state.transient["checkpoint"] = self.repository.checkpoint()
            self._save_event(state, "repository_inspected", {"file_count": len(snapshot.files)})

            # --- PLANNING ---
            state.transition(Lifecycle.PLANNING)
            state.plan = self.planner.create_plan(issue, snapshot, state, state.budget)
            self._save_event(state, "plan_created", {"steps": [s.step_id for s in state.plan.steps]})

            # --- EXECUTING ---
            state.transition(Lifecycle.EXECUTING)
            executor = Executor(self.repository, self.provider)
            while executor.eligible_step(state):
                if not executor.execute_next(state, snapshot, obs):
                    break
                self._save_event(state, "step_executed", {"step": state.transient.get("last_step")})

            # --- VERIFYING ---
            state.transition(Lifecycle.VERIFYING)
            test_command = snapshot.test_command or ["pytest", "-q"]
            state.budget.record("tool")
            test_result = self.repository.run_command(test_command, timeout=state.budget.command_timeout_seconds)
            obs.results.append(test_result)
            obs.notes.append(
                "Command output is an observation; task completion is not independently audited in this phase."
            )

            all_steps_done = state.plan is not None and all(
                step.status == StepStatus.COMPLETED for step in state.plan.steps
            )

            if test_result.success and all_steps_done:
                state.budget.clear_failures()
                state.transition(Lifecycle.COMPLETED)
            elif not all_steps_done:
                state.transition(Lifecycle.BLOCKED)
                obs.notes.append("Plan execution stopped before all steps were completed.")
            else:
                state.budget.record_failure(f"{test_result.exit_code}:{test_result.stderr[-200:]}")
                state.transition(Lifecycle.BLOCKED)

        except BudgetExceeded as exc:
            if state.status != Lifecycle.COMPLETED:
                state.transition(Lifecycle.BUDGET_EXHAUSTED)
            obs.notes.append(str(exc))
        except Exception as exc:
            state.transient["error"] = str(exc)
            terminal = {Lifecycle.COMPLETED, Lifecycle.BLOCKED, Lifecycle.FAILED, Lifecycle.BUDGET_EXHAUSTED}
            if state.status not in terminal:
                state.transition(Lifecycle.FAILED)
            obs.notes.append(f"{type(exc).__name__}: {exc}")

        self._save_event(state, "task_finished", {"status": state.status.value})
        return state, obs

    def _save_event(self, state: TaskState, event_type: str, payload=None):
        self.store.save(state)
        self.events.append(RunEvent(state.task_id, event_type, payload or {}))
