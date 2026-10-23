# Persistent Verified State — Design

## Intent

Build Milestone 2 for AdaLongHorizon: let verified task progress survive distinct execution rounds (and process restarts) so later planning can focus on unfinished requirements. The runtime remains a deterministic software-engineering agent, and the existing independent, read-only verifier remains the sole authority for completion evidence.

## Goals

- Represent task requirements and their `pending` / `completed` / `blocked` lifecycle, affected artifacts, independently verified facts, and overall `in_progress` / `completed` / `blocked` status.
- Persist that progress as inspectable JSON using the existing local state-store approach.
- Keep durable progress separate from a fresh per-round `TaskState`, budget, plan, and execution observations.
- Update durable progress only through an explicit operation consuming a successful `VerificationResult` and its evidence; executor reports alone cannot complete requirements.
- Run bounded rounds: inspect, plan from pending requirements plus verified facts, execute a bounded plan, run the existing test command, verify, apply state update, persist, and stop on completion or a rejected/failed round. No retry or recovery policy is added.
- Record optional run/round identifiers, call/token/tool counts, planning and verification counts, elapsed time, verification outcome, and final outcome without inventing unavailable values.
- Preserve existing one-round behavior and Ollama usage reporting.

## Non-goals and constraints

No confidence or evidence-quality scoring, adaptive planning/auditing, semantic memory, state graphs, databases, rollback/checkpointing, retry policy, loop detection, cost/risk controller, RL, fine-tuning, SWE-bench, retrieval optimization, or multi-agent runtime. State management has no repository mutation capability. The verifier remains read-only. No commit is part of this work.

## Approaches

1. **Separate durable progress from runtime `TaskState` (recommended).** Add a task-progress model/store and pass a read-only progress snapshot into planning; each round receives fresh runtime state. This honors the requested isolation and limits serialized data to durable task facts, at the cost of a distinct model and explicit update boundary.
2. **Extend and reuse `TaskState`.** Add requirements/facts/artifacts to the currently persisted lifecycle object and reset its transient fields per round. This changes fewer persistence pieces but couples durable truth, per-round budget, and execution lifecycle; it conflicts with the fresh-round separation requirement.

The selected direction is approach 1, as confirmed by the user. Use standard-library dataclasses/enums and JSON, consistent with the current project; do not add dependencies.

### Persistence trade-off

The current prototype writes state directly to its destination rather than using temporary-file replacement. In this development workspace, replacing an existing file with `os.replace()` fails intermittently under Windows despite no application-level handle, concurrency, JSON, or shared-state dependency; direct overwrite remains reliable. This is a deliberate reliability trade-off for the prototype and does not provide the crash-atomicity guarantees of atomic replacement.

## Proposed architecture

- Add a durable `PersistentTaskState` (name may follow repository conventions) containing task ID, original issue/requirements, requirement status, artifact paths, verified facts, overall status, and round history/trace summaries. Validation rejects malformed or unknown enum values.
- Extend the existing state store or add a narrowly scoped state-store method for deterministic JSON save/load and safe rejection of corrupt state. Keep existing `TaskState` event persistence compatible.
- Provide an explicit state updater that accepts the durable state and a verified result (plus round requirement mapping), records evidence-backed facts/artifacts, and transitions only requirements whose mapped evidence checks passed. Rejected verification, missing evidence, failed tests, execution failure, or malformed input must not mark completion. State update code receives no repository adapter.
- Requirement completion uses a deterministic requirement-to-verifier-check mapping: each requirement's mapped evidence checks must pass AND the round-level test command must succeed. A passing round test is necessary, never sufficient by itself; requirements not explicitly mapped to passing evidence remain incomplete. If a requirement names a particular test, its evidence mapping must include that test.
- Each round creates a new runtime `TaskState` and `ObservationBundle`; durable state is passed separately to `Planner`. A provider-backed plan prompt receives pending requirements and verified facts. Deterministic planning must also use only pending requirements.
- To allow incremental progress without weakening the current verifier, each round must leave the discovered repository test command passing. Do not mark a requirement complete unless the verifier accepts the round's mutation records and test result. A rejected round stops the run as `blocked`; there is no automatic retry.
- Track task-level budget across rounds while keeping per-round plan/context/observations fresh. Bound rounds by pending requirements and existing budget limits; stop if a round makes no verified progress.
- Keep trace measurements optional when unavailable. Distinguish measured values from estimates already identified by the current budget telemetry.
- Add a deterministic multi-round CLI demo in a disposable repository, with at least two independently verified requirements and visible state before/after each round; inspect and report its temporary diff.

## Verification and review focus

Tests must prove initial pending state; verified-only transitions; rejected/malformed/failed execution and failing tests cannot complete; serialization round-trip and corrupt-file rejection; multi-round planner visibility and remaining-requirement focus; completion only when all requirements are verified; fresh round contexts; trace values remain unavailable rather than fabricated; and existing single-round/verifier/Ollama contracts remain intact. Run focused tests first, then the full suite, deterministic demo, multi-round demo, optional Ollama demo if available, and `git diff --check`.

## Open implementation calls for the builder

- Map acceptance criteria to stable requirement IDs at task creation; preserve the original task statement in the durable state.
- Keep state update and persistence boundaries explicit and deterministic; update atomically via a requirement-to-verifier-check mapping and successful round test, rather than trusting action text or executor completion status.
- Use a new dedicated multi-round demo command if that keeps the existing `demo` contract stable; otherwise extend the existing demo without changing its default output semantics.
- Do not modify unrelated user work already present in the dirty worktree.
