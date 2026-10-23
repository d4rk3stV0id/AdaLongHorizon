# Persistent Verified State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, serializable task progress across fresh execution rounds, where a requirement completes only from its explicitly mapped passing verifier checks and a passing round test command.

**Architecture:** A durable progress model/store is separate from per-round `TaskState` and observations. A deterministic updater stages all requirement transitions from one accepted `VerificationResult` plus per-round test success, commits updates atomically, and persists. Planner receives pending requirements and verified facts; orchestration creates bounded fresh rounds while carrying task budget and aggregates optional trace metrics.

**Tech Stack:** Python 3.11+, dataclasses/enums, JSON, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-23-persistent-verified-state-design.md`

## Global Constraints

- Only independently verified environment observations may establish persistent completion.
- Requirement completion requires mapped requirement-specific verifier checks AND successful round-level tests.
- Keep executor result, verifier result, and persistent state update distinct.
- Rejected/failed rounds make no partial requirement completion updates.
- Keep each execution round's runtime context bounded and fresh.
- Record unavailable metrics as unavailable; do not fabricate measurements.
- Preserve existing single-round behavior and tests; no later research features; do not commit.

## Review Focus

- Mapping gaps or unknown verifier check IDs must not complete requirements — test explicit mapping validation and unmapped requirements.
- Mixed round outcomes must be atomic and requirement-specific — test one mapped requirement passing while another is unmapped/failing.
- Repeated criteria/state reload must preserve stable IDs and reject corrupt input — test round-trip and malformed JSON/schema.
- Planner must see prior verified facts but not unbounded execution history — test subsequent-round pending scope and fresh observations.
- Instrumentation must distinguish measured/unavailable values — test absent provider usage and deterministic timing fields remain `None` when unavailable.

---

### Task 1: Durable progress model and atomic update boundary

**Files:**
- Modify: `agent/models.py`
- Create or modify: `agent/state/progress.py`
- Modify: `agent/state/store.py`
- Test: `tests/unit/test_state.py`

**Interfaces:**
- Produces typed requirement/progress/task-trace records, deterministic initial-state construction, `StateStore.save_progress/load_progress`, and an updater requiring `VerificationResult`, requirement-to-check IDs, and round test result.
- State update stages changes and only commits when verification is accepted, round test succeeded, and all mapped checks for each requirement passed; no repository access.

- [ ] Add tests for initial pending requirements, status, serialization round-trip, malformed-state rejection, executor-only/rejected/failing-test no completion, explicit check mapping, unmapped requirements, and atomic multi-requirement behavior.
- [ ] Run focused state tests and confirm expected failures.
- [ ] Implement the minimal progress types, JSON store, and atomic updater.
- [ ] Run focused state tests.

### Task 2: Planning against pending verified progress

**Files:**
- Modify: `agent/planner.py`
- Modify: `agent/llm/ollama.py`
- Modify: `agent/llm/base.py` only if typed provider input is needed
- Test: `tests/unit/test_planner_executor.py` and `tests/unit/test_planner_llm.py`

**Interfaces:**
- Planner consumes durable progress and creates plans only for pending requirements while including bounded verified facts/artifacts.
- Existing planner API remains backward-compatible when progress is absent.

- [ ] Add deterministic/provider planner tests proving completed requirements are excluded and pending requirements are explicitly represented.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement minimal progress-aware planning and bounded Ollama prompt context.
- [ ] Run focused planner tests.

### Task 3: Multi-round orchestration and persistence

**Files:**
- Modify: `agent/orchestrator.py`
- Modify: `agent/models.py` for trace fields if needed
- Modify: `agent/state/progress.py` for verified round summary
- Test: `tests/integration/test_orchestrator.py`

**Interfaces:**
- `Orchestrator.run` preserves its return contract and supports bounded rounds over one durable progress record; each round gets fresh `TaskState` and `ObservationBundle`, shared task budget, monotonically numbered round ID.
- Save durable progress after verified atomic updates; rejected round blocks without progress mutation.
- Record task/round IDs, optional call/token/tool/planning/verification/time metrics and final status without fabricated values.

- [ ] Add integration tests for two rounds where round 2 sees round 1 facts, focuses on remaining requirements, and completes only after each requirement is individually mapped and verified; test rejected/failed round rollback of all progress changes and trace aggregation.
- [ ] Run focused integration tests and confirm expected failures.
- [ ] Implement bounded round coordination, progress persistence, and optional trace collection while preserving the current single-round API behavior.
- [ ] Run focused integration tests and existing orchestration tests.

### Task 4: Deterministic multi-round demonstration and full verification

**Files:**
- Modify: `agent/__main__.py`
- Modify: `README.md`
- Test: `tests/integration/test_orchestrator.py` or new `tests/integration/test_multi_round_demo.py`

**Interfaces:**
- Add a deterministic multi-round demo entry point with visible per-round requirement statuses; retain existing `python -m agent demo` behavior.

- [ ] Add an integration test for deterministic demo behavior and output states.
- [ ] Run it and confirm expected failure.
- [ ] Implement the smallest demo path and explain usage in README.
- [ ] Run full `python -m pytest`, existing deterministic demo, new multi-round demo, optional Ollama demo if accessible, and `git diff --check`.
- [ ] Inspect temporary demo repository diffs and verify existing verifier/single-round behavior.
