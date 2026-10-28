"""Task lifecycle orchestrator — coordinates inspection, planning,
execution, and verification into a single run.

Phase 3 scope: verification is limited to running the discovered test
command and checking that all plan steps completed.  Independent auditing,
adaptive verification, and sophisticated recovery belong to Phase 4+.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .errors import BudgetExceeded
from .executor import Executor
from .llm.provider import DeterministicProvider
from .models import (
    AgentConfig, BudgetLedger, IssueSpec, Lifecycle, ObservationBundle,
    RunEvent, StepStatus, TaskState,
)
from .planner import Planner
from .state.store import StateStore
from .state.progress import (
    PersistentTaskState, ProgressStatus, RoundTrace, RunTrace,
    create_progress_state, summarize_provider_usage, update_from_verification,
)
from .verifier import Verifier
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
        if provider is not None:
            self.provider = provider
        elif config.provider == "ollama":
            from .llm.ollama import OllamaProvider
            self.provider = OllamaProvider(host=config.ollama_host, model=config.llm_model)
        else:
            self.provider = DeterministicProvider()
        self.planner = Planner(self.provider if config.provider == "ollama" else None)
        self.verifier = Verifier()
        self.last_run_trace: RunTrace | None = None

    def run(
        self, issue: IssueSpec, budget: BudgetLedger | None = None,
        persistent_state: PersistentTaskState | None = None,
        requirement_checks: dict[str, list[str]] | None = None,
        round_id: int = 1,
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
            state.plan = self.planner.create_plan(
                issue, snapshot, state, state.budget, persistent_state=persistent_state,
            )
            self._save_event(state, "plan_created", {
                "steps": [s.step_id for s in state.plan.steps],
                "llm_usage": state.transient.get("last_llm_usage"),
            })

            # --- EXECUTING ---
            state.transition(Lifecycle.EXECUTING)
            executor = Executor(self.repository, self.provider)
            while executor.eligible_step(state):
                if not executor.execute_next(state, snapshot, obs):
                    break
                self._save_event(state, "step_executed", {
                    "step": state.transient.get("last_step"),
                    "llm_usage": state.transient.get("last_llm_usage"),
                })

            # --- VERIFYING ---
            state.transition(Lifecycle.VERIFYING)
            test_command = snapshot.test_command or ["pytest", "-q"]
            state.budget.record("tool")
            test_result = self.repository.run_command(test_command, timeout=state.budget.command_timeout_seconds)
            obs.results.append(test_result)
            verification = self.verifier.verify(self.repository, obs.execution_records, test_result)
            state.transient["verification_result"] = verification.to_dict()
            obs.notes.append(f"Independent verification: {verification.status.value}.")
            if verification.failures:
                obs.notes.extend(verification.failures)

            if persistent_state is not None and requirement_checks is not None:
                update_from_verification(
                    persistent_state, verification, requirement_checks, test_result, round_id=round_id,
                )
                self.store.save_progress(persistent_state)

            all_steps_done = state.plan is not None and all(
                step.status == StepStatus.COMPLETED for step in state.plan.steps
            )

            if verification.verified and all_steps_done:
                state.budget.clear_failures()
                state.transition(Lifecycle.COMPLETED)
            elif not all_steps_done or not verification.verified:
                state.transition(Lifecycle.BLOCKED)
                if not all_steps_done:
                    obs.notes.append("Plan execution stopped before all steps were completed.")
                state.budget.record_failure("verification:" + "|".join(verification.failures)[:200])
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

        self._save_event(state, "task_finished", {
            "status": state.status.value,
            "error": state.transient.get("error"),
            "llm_usage": state.transient.get("last_llm_usage"),
            "verification_result": state.transient.get("verification_result"),
        })
        return state, obs

    def run_persistent_round(
        self,
        persistent_state: PersistentTaskState,
        requirement_checks: dict[str, list[str]],
        budget: BudgetLedger | None = None,
    ) -> tuple[TaskState, ObservationBundle]:
        """Execute exactly one fresh runtime round against durable verified state."""
        issue = IssueSpec(
            title=persistent_state.issue["title"],
            description=persistent_state.issue["description"],
            acceptance_criteria=list(persistent_state.issue["acceptance_criteria"]),
            issue_id=persistent_state.issue["issue_id"],
        )
        return self.run(
            issue, budget, persistent_state=persistent_state,
            requirement_checks=requirement_checks,
        )

    def run_persistent(
        self, issue: IssueSpec, budget: BudgetLedger | None = None,
    ) -> tuple[PersistentTaskState, list[tuple[TaskState, ObservationBundle]]]:
        """Run fresh bounded execution contexts against one durable task state."""
        run_started = perf_counter()
        self.last_run_trace = None
        shared_budget = budget or BudgetLedger(command_timeout_seconds=self.config.command_timeout_seconds)
        task_id = uuid4().hex
        run_id = uuid4().hex
        progress = create_progress_state(task_id, issue)
        self.store.save_progress(progress)
        rounds: list[tuple[TaskState, ObservationBundle]] = []
        max_rounds = len(progress.requirements)
        for round_id in range(1, max_rounds + 1):
            if progress.status != ProgressStatus.IN_PROGRESS:
                break
            requirement = next(
                (item for item in progress.requirements if item.status.value == "pending"), None,
            )
            if requirement is None:
                break
            pending_issue = IssueSpec(
                title=issue.title,
                description=issue.description + f"\nCurrent round requirement ({requirement.requirement_id}): {requirement.description}",
                acceptance_criteria=[requirement.description],
                issue_id=issue.issue_id,
            )
            started = perf_counter()
            previously_completed = {
                item.requirement_id for item in progress.requirements
                if item.status.value == "completed"
            }
            pending_before = sum(item.status.value == "pending" for item in progress.requirements)
            pending_ids_before = [
                item.requirement_id for item in progress.requirements
                if item.status.value == "pending"
            ]
            tools_before = shared_budget.tool_actions
            state, observations = self.run(
                pending_issue, shared_budget, persistent_state=progress, round_id=round_id,
            )
            rounds.append((state, observations))
            raw = state.transient.get("verification_result", {})
            verification = self.verifier.verify(
                self.repository, observations.execution_records,
                next((r for r in reversed(observations.results) if r.operation == "run_command"), None),
            )
            mapped_checks = sorted({
                check.name
                for index, record in enumerate(observations.execution_records)
                if record.tool_result.success
                for check in verification.checks
                if check.name.startswith(f"execution_record_{index}_")
            })
            test_result = next(
                (r for r in reversed(observations.results) if r.operation == "run_command"), None,
            )
            if mapped_checks:
                update_from_verification(
                    progress, verification, {requirement.requirement_id: mapped_checks},
                    test_result, round_id,
                )
            newly_verified = sum(
                item.status.value == "completed" and item.requirement_id not in previously_completed
                for item in progress.requirements
            )
            trace = progress.rounds[-1] if progress.rounds and progress.rounds[-1].round_id == round_id else None
            if trace is None:
                trace = RoundTrace(task_id, round_id)
                progress.rounds.append(trace)
            usage = state.transient.get("llm_usage", [])
            provider_usage = summarize_provider_usage(
                usage, llm_provider_enabled=self.config.provider == "ollama",
            )
            trace.llm_calls = provider_usage.llm_calls
            trace.input_tokens = provider_usage.input_tokens
            trace.output_tokens = provider_usage.output_tokens
            trace.total_tokens = provider_usage.total_tokens
            trace.provider_latency_ms = provider_usage.provider_latency_ms
            trace.tool_calls = shared_budget.tool_actions - tools_before
            trace.planning_calls = sum(
                item.get("purpose") == "planning" for item in usage
            ) if self.config.provider == "ollama" else 0
            trace.verification_calls = 1 + int("verification_result" in state.transient)
            trace.execution_time_seconds = perf_counter() - started
            trace.planning_result = "accepted" if state.plan is not None else "failed"
            trace.verification_result = verification.status.value
            trace.pending_requirements = pending_before
            trace.pending_requirement_ids = pending_ids_before
            trace.newly_verified_requirements = newly_verified
            trace.final_outcome = state.status.value.lower()
            if requirement.status.value != "completed" or newly_verified == 0:
                progress.status = ProgressStatus.BLOCKED
            for record in observations.execution_records:
                if record.tool_result.success and record.action.get("path"):
                    if record.action["path"] not in progress.artifacts:
                        progress.artifacts.append(record.action["path"])
            self.store.save_progress(progress)
            # The next round deliberately reconstructs its durable input from disk.
            progress = self.store.load_progress(task_id)
            if progress.status == ProgressStatus.BLOCKED:
                break
        self.last_run_trace = RunTrace.from_rounds(
            task_id=task_id,
            rounds=progress.rounds,
            final_outcome=progress.status.value,
            elapsed_seconds=perf_counter() - run_started,
            verification_result=(progress.rounds[-1].verification_result if progress.rounds else None),
            run_id=run_id,
        )
        return progress, rounds

    def _save_event(self, state: TaskState, event_type: str, payload=None):
        self.store.save(state)
        self.events.append(RunEvent(state.task_id, event_type, payload or {}))
