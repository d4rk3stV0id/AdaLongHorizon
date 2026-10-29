# Multi-Round Manage–Execute–Audit (M3) Design

## Intent

M3 makes the existing persistent multi-round baseline an explicit,
deterministic Manage–Execute–Audit (MEA) loop.  It retains the project's
Planner, Executor, Verifier, StateStore, and Orchestrator terminology rather
than duplicating those components under new names.

## Responsibility mapping

| MEA responsibility | Existing/new component | Responsibility |
| --- | --- | --- |
| Manage | `MEAManager` used by `Orchestrator` | Read durable verified state and the current budget/round boundary; emit a structural `EXECUTE`, `DONE`, or `BLOCKED` decision. |
| Plan | `Planner` | Turn the manager's bounded contract into a validated bounded plan. |
| Execute | `Executor` | Run validated actions through `RepositoryAdapter` in one fresh `TaskState`. |
| Audit | `Verifier` | Read-only inspection of execution records, repository contents, and the round test result. |
| Durable update | `update_from_verification` and `StateStore` | Advance only requirement-specific, verifier-backed progress and persist/reload it. |

`MEAManager` does not mutate the repository, invoke a provider, or decide that
a requirement is complete.  The Executor does not establish durable truth.

## Control decisions and contracts

`ControlDecision` has a typed kind:

- `EXECUTE`: contains one `BoundedExecutionContract` for the first eligible
  pending requirement, using deterministic requirement order.
- `DONE`: all durable requirements are completed.
- `BLOCKED`: the durable task is already blocked, no eligible pending work
  exists, or the configured round/budget boundary does not permit another
  round.  A structured reason is retained.

A contract contains the task/round identifiers, immediate objective, selected
requirement ID and acceptance criterion, bounded verified facts/artifacts,
constraints, and the expected verification condition.  It intentionally does
not contain raw executor history or previous runtime `TaskState` data.

## Round lifecycle

```text
PersistentTaskState
  -> manager decision
  -> EXECUTE contract (or DONE/BLOCKED terminal decision)
  -> fresh TaskState + ObservationBundle
  -> Planner -> Executor -> read-only Verifier
  -> update_from_verification
  -> StateStore save/load
  -> next manager decision
```

The Orchestrator invokes the manager before every round and after the durable
save/load boundary.  It does not iterate merely because requirements exist.
The current policy selects only one pending requirement per contract and does
not rewrite plans from failed observations.

## State and transition rules

- `PersistentTaskState` is durable: original issue, requirement statuses,
  verifier-derived facts/artifacts, and round traces.
- `TaskState`, `ObservationBundle`, plans, and executor interactions are fresh
  per round and are not converted into durable truth.
- A requirement completes only when its explicitly mapped verifier checks pass
  **and** the round test command succeeds.
- After an accepted audit/update, the progress record is saved and reloaded
  before the next manager decision.
- A rejected audit or no verified progress leaves the requirement pending and
  changes task progress to `blocked`; there is no retry or recovery policy.

## Boundaries and terminals

The manager returns `DONE` only for fully verified durable completion.  It
returns `BLOCKED` for no eligible work, a configured round limit, or a budget
that cannot start another round.  A runtime budget exception is also persisted
as blocked without manufacturing completion.  M3 introduces neither dynamic
replanning (M4), retry/rollback/checkpoint recovery (M5), nor a new telemetry
or cost framework (M6).

## Resume semantics

`StateStore.load_progress` and `run_persistent_round` remain available for a
caller that explicitly loads durable progress.  M3 does not add a task service
or automatic resume API: `run_persistent` deliberately creates a new task, and
blocked tasks are terminal under this milestone.  A broader resume policy is
deferred until task ownership and budget persistence requirements are defined.
