"""Task lifecycle orchestrator — coordinates inspection, planning,
execution, and verification into a single run.

Phase 3 scope: verification is limited to running the discovered test
command and checking that all plan steps completed.  Independent auditing,
adaptive verification, and sophisticated recovery belong to Phase 4+.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .errors import BudgetExceeded
from .executor import Executor
from .manager import BoundedExecutionContract, ControlDecisionKind, MEAManager
from .replanning import ReplanDecisionKind, ReplanningController
from .recovery import RecoveryJournal, checkpoint_runtime, reconcile_recovery, repository_identity
from .llm.provider import DeterministicProvider
from .models import (
    AgentConfig, BudgetLedger, ExecutionRecord, IssueSpec, Lifecycle,
    ObservationBundle, RunEvent, StepStatus, TaskState, ToolResult,
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
        self.manager = MEAManager()
        self.last_run_trace: RunTrace | None = None

    def run(
        self, issue: IssueSpec, budget: BudgetLedger | None = None,
        persistent_state: PersistentTaskState | None = None,
        requirement_checks: dict[str, list[str]] | None = None,
        round_id: int = 1,
        contract: BoundedExecutionContract | None = None,
    ) -> tuple[TaskState, ObservationBundle]:
        task_id = uuid4().hex
        with self.store.task_lock(task_id):
            if persistent_state is None:
                return self._run_with_ownership(
                    issue, budget, persistent_state, requirement_checks, round_id, contract, task_id,
                )
            with self.store.task_lock(persistent_state.task_id):
                return self._run_with_ownership(
                    issue, budget, persistent_state, requirement_checks, round_id, contract, task_id,
                )

    def _run_with_ownership(
        self, issue: IssueSpec, budget: BudgetLedger | None,
        persistent_state: PersistentTaskState | None,
        requirement_checks: dict[str, list[str]] | None, round_id: int,
        contract: BoundedExecutionContract | None, task_id: str,
    ) -> tuple[TaskState, ObservationBundle]:
        state = TaskState(
            issue,
            budget=budget or BudgetLedger(command_timeout_seconds=self.config.command_timeout_seconds),
            task_id=task_id,
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
                contract=contract,
            )
            if contract is not None:
                state.plan.requirement_ids = list(contract.requirement_ids)
            recovery_id = repository_identity(snapshot.root, snapshot.baseline_commit)
            journal = RecoveryJournal(self.store, state.task_id, recovery_id)
            checkpoint_runtime(
                self.store, state, recovery_id, state.transient["checkpoint"], journal.sequence,
                persistent_state.task_id if persistent_state is not None else None,
            )
            self._save_event(state, "plan_created", {
                "steps": [s.step_id for s in state.plan.steps],
                "llm_usage": state.transient.get("last_llm_usage"),
            })

            return self._execute_and_verify(
                state, snapshot, obs, recovery_id, journal,
                persistent_state=persistent_state,
                requirement_checks=requirement_checks,
                round_id=round_id, contract=contract,
            )
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

    def _execute_and_verify(
        self,
        state: TaskState,
        snapshot: RepositorySnapshot,
        obs: ObservationBundle,
        recovery_id: str,
        journal: RecoveryJournal,
        persistent_state: PersistentTaskState | None = None,
        requirement_checks: dict[str, list[str]] | None = None,
        round_id: int = 1,
        contract: BoundedExecutionContract | None = None,
    ) -> tuple[TaskState, ObservationBundle]:
        try:
            # Seed obs.execution_records from recovered records if any
            if state.transient.get("recovered_execution_records") and not obs.execution_records:
                for rec_dict in state.transient["recovered_execution_records"]:
                    tool_res = ToolResult(**rec_dict["tool_result"]) if isinstance(rec_dict.get("tool_result"), dict) else rec_dict.get("tool_result")
                    obs.execution_records.append(ExecutionRecord(
                        step_id=rec_dict["step_id"],
                        expected_files=rec_dict["expected_files"],
                        action=rec_dict["action"],
                        before_content=rec_dict.get("before_content"),
                        tool_result=tool_res,
                        active=rec_dict.get("active", True),
                    ))

            # --- EXECUTING ---
            if state.status != Lifecycle.EXECUTING:
                state.transition(Lifecycle.EXECUTING)
            executor = Executor(self.repository, self.provider, journal=journal)
            replanning = ReplanningController(self.config, self.verifier, state=state)
            while executor.eligible_step(state):
                before_records = len(obs.execution_records)
                try:
                    executed = executor.execute_next(state, snapshot, obs)
                    error = None
                except Exception as exc:
                    executed = False
                    error = exc
                record = obs.execution_records[-1] if len(obs.execution_records) > before_records else None
                assessment = self.verifier.assess_execution_record(self.repository, record) if record else None
                outcome = replanning.classify(record, assessment, error)
                if isinstance(error, BudgetExceeded):
                    raise error
                decision = replanning.decide(outcome, assessment)
                state.transient.setdefault("execution_outcomes", []).append({
                    "kind": outcome.kind.value, "reason": outcome.reason.value if outcome.reason else None,
                    "signature": outcome.signature, "detail": outcome.detail,
                    "assessment": assessment.status.value if assessment else None,
                })
                checkpoint_runtime(
                    self.store, state, recovery_id, self.repository.checkpoint(), journal.sequence,
                    persistent_state.task_id if persistent_state is not None else None,
                )
                if decision == ReplanDecisionKind.CONTINUE and executed:
                    self._save_event(state, "step_executed", {
                        "step": state.transient.get("last_step"),
                        "llm_usage": state.transient.get("last_llm_usage"),
                    })
                    continue
                if decision == ReplanDecisionKind.RETRY:
                    if record is not None:
                        record.active = False
                    step = next((item for item in state.plan.steps if item.step_id == outcome.step_id), None)
                    if step is not None:
                        step.status = StepStatus.PENDING
                    continue
                if decision == ReplanDecisionKind.REPLAN and contract is not None and state.plan is not None:
                    old_plan = state.plan
                    replan_snapshot = self.repository.inspect()
                    try:
                        candidate = self.planner.create_revised_plan(
                            state.issue, replan_snapshot, state, state.budget,
                            replanning.context(state, contract, replan_snapshot),
                        )
                        errors = replanning.validate_candidate(candidate, state, contract, replan_snapshot)
                    except Exception as exc:
                        candidate, errors = None, [f"revision proposal failed: {type(exc).__name__}: {exc}"]
                    if candidate is not None and not errors:
                        replanning.activate(old_plan, candidate, contract, replan_snapshot, outcome)
                        if record is not None:
                            record.active = False
                        state.plan = candidate
                        state.transient["replan_history"] = [asdict(item) for item in replanning.history]
                        checkpoint_runtime(
                            self.store, state, recovery_id, self.repository.checkpoint(), journal.sequence,
                            persistent_state.task_id if persistent_state is not None else None,
                        )
                        self._save_event(state, "plan_revised", {"version": candidate.version, "reason": candidate.revision_reason})
                        continue
                    replanning.reject(old_plan, outcome, "; ".join(errors))
                    state.transient["replan_history"] = [asdict(item) for item in replanning.history]
                if error is not None:
                    state.transient["error"] = f"{type(error).__name__}: {error}"
                obs.notes.append(f"Execution stopped: {outcome.kind.value}: {outcome.detail}")
                break
            state.transient["replan_history"] = [asdict(item) for item in replanning.history]

            # --- VERIFYING ---
            state.transition(Lifecycle.VERIFYING)
            test_command = snapshot.test_command or ["pytest", "-q"]
            state.budget.record("tool")
            test_result = self.repository.run_command(test_command, timeout=state.budget.command_timeout_seconds)
            obs.results.append(test_result)
            verification = self.verifier.verify(self.repository, obs.execution_records, test_result)
            obs.audit_result = verification
            state.transient["verification_result"] = verification.to_dict()
            checkpoint_runtime(
                self.store, state, recovery_id, self.repository.checkpoint(), journal.sequence,
                persistent_state.task_id if persistent_state is not None else None,
            )
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

    def recover(self, task_id: str) -> TaskState:
        """Load/reconcile a runtime checkpoint; callers must resume normal verification afterward."""
        with self.store.task_lock(task_id):
            return self._recover_with_ownership(task_id)

    def _recover_with_ownership(self, task_id: str) -> TaskState:
        snapshot = self.repository.inspect()
        state = reconcile_recovery(
            self.store, self.verifier, self.repository, task_id,
            repository_identity(snapshot.root, snapshot.baseline_commit),
        )
        self._save_event(state, "recovery_reconciled", state.transient.get("recovery", {}))
        return state

    def resume(
        self,
        task_id: str,
        persistent_state: PersistentTaskState | None = None,
        requirement_checks: dict[str, list[str]] | None = None,
        round_id: int = 1,
        contract: BoundedExecutionContract | None = None,
    ) -> tuple[TaskState, ObservationBundle]:
        """Resume execution of an interrupted task after recovering its checkpoint and journal."""
        with self.store.task_lock(task_id):
            state = self.recover(task_id)
            snapshot = self.repository.inspect()
            recovery_id = repository_identity(snapshot.root, snapshot.baseline_commit)
            journal = RecoveryJournal(self.store, state.task_id, recovery_id)
            obs = ObservationBundle(state.task_id)
            return self._execute_and_verify(
                state, snapshot, obs, recovery_id, journal,
                persistent_state=persistent_state,
                requirement_checks=requirement_checks,
                round_id=round_id, contract=contract,
            )

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
        """Create durable progress, then run the explicit bounded M3 MEA loop."""
        task_id = uuid4().hex
        with self.store.task_lock(task_id):
            return self._run_persistent_with_ownership(issue, budget, task_id)

    def _run_persistent_with_ownership(
        self, issue: IssueSpec, budget: BudgetLedger | None, task_id: str,
    ) -> tuple[PersistentTaskState, list[tuple[TaskState, ObservationBundle]]]:
        run_started = perf_counter()
        self.last_run_trace = None
        shared_budget = budget or BudgetLedger(command_timeout_seconds=self.config.command_timeout_seconds)
        run_id = uuid4().hex
        progress = create_progress_state(task_id, issue)
        self.store.save_progress(progress)
        # Hold the durable task ownership across every M3 runtime round.
        with self.store.task_lock(task_id):
            return self._run_persistent_rounds(issue, shared_budget, progress, task_id, run_started, run_id)

    def _run_persistent_rounds(self, issue, shared_budget, progress, task_id, run_started, run_id):
        rounds: list[tuple[TaskState, ObservationBundle]] = []
        while True:
            decision = self.manager.decide(
                progress, shared_budget, max_rounds=self.config.max_rounds,
            )
            if decision.kind == ControlDecisionKind.DONE:
                break
            if decision.kind == ControlDecisionKind.BLOCKED:
                progress.status = ProgressStatus.BLOCKED
                self.store.save_progress(progress)
                break

            contract = decision.contract
            if contract is None:  # Defensive: EXECUTE decisions must always carry a contract.
                progress.status = ProgressStatus.BLOCKED
                self.store.save_progress(progress)
                break
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
                contract.to_issue(progress.issue), shared_budget, persistent_state=progress,
                round_id=contract.round_id, contract=contract,
            )
            rounds.append((state, observations))
            verification = observations.audit_result
            if verification is None:
                # A failed runtime cannot manufacture audit success.
                progress.status = ProgressStatus.BLOCKED
                self.store.save_progress(progress)
                progress = self.store.load_progress(task_id)
                break
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
                    progress, verification, {contract.requirement_ids[0]: mapped_checks},
                    test_result, contract.round_id,
                )
            newly_verified = sum(
                item.status.value == "completed" and item.requirement_id not in previously_completed
                for item in progress.requirements
            )
            trace = progress.rounds[-1] if progress.rounds and progress.rounds[-1].round_id == contract.round_id else None
            if trace is None:
                trace = RoundTrace(task_id, contract.round_id)
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
            trace.revision_history = list(state.transient.get("replan_history", []))
            requirement = next(
                item for item in progress.requirements
                if item.requirement_id == contract.requirement_ids[0]
            )
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
