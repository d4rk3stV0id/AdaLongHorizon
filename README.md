# AdaLongHorizon

AdaLongHorizon is a research prototype for a budget-aware software engineering
agent. It inspects a local Git repository, creates a structured plan, executes
validated file operations, and independently verifies the result with the
repository's test command.

The project makes progress and completion distinct. An executor can report an
operation succeeded, but a requirement becomes durable only after the
independent verifier confirms the relevant evidence and tests pass.

## What it demonstrates

- Structured planning with versioned `Plan` and `PlanStep` models.
- Repository inspection, constrained file operations, and command execution.
- Independent verification of changed artifacts and test results.
- Persistent requirement state across bounded runtime rounds.
- Dynamic replanning for a selected requirement, with bounded retries
  and replans.
- Atomic checkpoints, action journaling, process ownership, and recovery of
  safely reconcilable interrupted work.
- A deterministic provider for repeatable demos and an optional local Ollama
  provider for structured model-backed planning and execution.

## Requirements

- Python 3.11 or later
- Git, for repository inspection and the disposable demos
- `pytest`, installed by the development extra

Optional Ollama demos also require a running [Ollama](https://ollama.com/)
service and an installed model.

## Install

```powershell
py -m pip install -e ".[dev]"
```

The package exposes the `ada-agent` console entry point. The examples below
use `py -m agent`, which works directly from a checkout.

## Quick start

Run the deterministic end-to-end example:

```powershell
py -m agent demo
```

It creates a temporary Git repository, plans a small implementation task,
writes the proposed files through the executor, runs pytest, and prints a JSON
summary. The current checkout is not changed.

Choose the alternate deterministic task:

```powershell
py -m agent demo --task normalize
```

## Architecture

```text
Issue specification
        |
Repository inspection --> Planner --> versioned active plan
                                      |
                                      v
                               Executor + action journal
                                      |
                                      v
                         Intermediate postcondition assessment
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
              bounded retry/replan              final verification
                                                        |
                                                        v
                                      durable verified requirement progress
```

| Component | Responsibility |
| --- | --- |
| `RepositoryAdapter` | Inspects a workspace, reads and writes approved files, and discovers a pytest command. |
| `Planner` | Produces structured plans and revised remaining plans. |
| `Executor` | Validates and applies one planned operation at a time. |
| `Verifier` | Assesses intermediate postconditions and performs final independent verification. |
| `MEAManager` | Selects the current requirement and bounds persistent rounds. |
| `ReplanningController` | Classifies execution outcomes and permits bounded retry or replanning within one round. |
| `StateStore` and recovery module | Persist verified progress, checkpoints, action journals, and task ownership locks. |

### Completion and recovery guarantees

- Executor success and intermediate assessment are not durable requirement
  completion evidence.
- Only normal final verification can update durable verified requirement state.
- Plan revisions apply only to the requirement selected for the current runtime
  round.
  Plans are not restored across later rounds.
- Before retrying an ambiguous action such as a timeout, the agent examines the
  intended postcondition. An achieved postcondition is never blindly repeated.
- The recovery system writes atomic, versioned checkpoints and records intent before a side
  effect and outcome afterward.
- Recovery validates the full journal, reconciles only records after the
  checkpoint position, and blocks when an in-flight action cannot be safely
  reconciled.
- Ownership is task-scoped and process-level. On Windows it uses `msvcrt`
  byte-range locking; nested acquisition is permitted only on the owning thread.

## Demos

All demos print JSON and use disposable workspaces unless noted otherwise.

| Command | Provider | What it runs |
| --- | --- | --- |
| `py -m agent demo` | Deterministic | A complete single-task planning, execution, and verification cycle. |
| `py -m agent demo --task normalize` | Deterministic | The whitespace-normalization variant of the single-task demo. |
| `py -m agent multi-round-demo` | Deterministic | Two independently verified requirements through persistent rounds. |
| `py -m agent llm-smoke-test` | Ollama | One schema-constrained model request; it does not inspect or modify a repository. |
| `py -m agent executor-ollama-smoke` | Ollama | One model-generated executor action and its fixture test. |
| `py -m agent ollama-demo` | Ollama | Full planning, execution, and verification in a temporary repository. |
| `py -m agent ollama-multi-round-demo` | Ollama | Two persistent rounds with provider usage and run-trace output. |

The deterministic demos make no model calls. `multi-round-demo` and the
Ollama multi-round demo report round traces, including verified progress and
available provider usage.

## Ollama setup

Set the local service and model before running any Ollama command:

```powershell
$env:ADA_OLLAMA_HOST = "http://localhost:11434"
$env:ADA_LLM_MODEL = "qwen3:8b"
py -m agent ollama-demo
```

The provider requests JSON-schema-constrained output derived from the plan
models. It records provider-reported token usage when available. The current
implementation sends `think=false`; it does not infer separate reasoning-token
counts from Ollama responses.

## Configuration

Configuration comes from environment variables. Defaults are chosen for the
deterministic local demos.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ADA_AGENT_PROVIDER` | `deterministic` | Provider selected by the orchestrator; set to `ollama` for model-backed execution. |
| `ADA_OLLAMA_HOST` | `http://localhost:11434` | Ollama HTTP endpoint. |
| `ADA_LLM_MODEL` | `qwen3:8b` | Ollama model name. |
| `ADA_AGENT_STATE_DIR` | `.agent-state` | Directory for task state, checkpoints, journals, and events. |
| `ADA_AGENT_COMMAND_TIMEOUT` | `30` | Per-command timeout in seconds. |
| `ADA_AGENT_MAX_ROUNDS` | `12` | Maximum persistent rounds. |
| `ADA_AGENT_MAX_REPLANS` | `2` | Maximum plan revisions per round. |
| `ADA_AGENT_MAX_SAFE_RETRIES` | `1` | Maximum safe retries per round. |
| `ADA_AGENT_NO_PROGRESS_THRESHOLD` | `2` | Repeated no-progress threshold before execution stops. |

## Testing

The repository configures pytest to use importlib import mode, so the normal
commands work even though unit and integration tests intentionally share some
module basenames.

```powershell
py -m pytest tests/unit -q
py -m pytest tests/integration -q
py -m pytest -q
```

The suite covers planner and executor validation, final verifier authority,
persistent state, replanning boundaries, and checkpoint, journal, recovery,
and task-ownership behavior.

## Project layout

```text
agent/
  __main__.py          CLI demos
  orchestrator.py      lifecycle coordination
  planner.py           structured plan generation
  executor.py          validated operations and journal integration
  verifier.py          intermediate assessment and final verification
  replanning.py        bounded replanning control
  recovery.py          checkpoint and journal reconciliation
  state/               persistent progress, checkpoints, and locks
  tools/               repository and command adapters
tests/
  unit/                isolated component tests
  integration/         lifecycle and repository-boundary tests
```

## Current scope

AdaLongHorizon supports one local process and one local workspace per task.
It does not implement distributed coordination, multi-agent recovery, or
additional capabilities beyond the local single-agent model. A checkpoint
records the agent's prior knowledge; recovery still rechecks relevant
repository state before it resumes work.
