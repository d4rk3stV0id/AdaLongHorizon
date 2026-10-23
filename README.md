# AdaLongHorizon

Budget-aware long-horizon software engineering agent research prototype.

## Requirements

Python 3.11+ and Git. The demonstration uses pytest in a disposable temporary repository.

## Run

```powershell
py -m pip install -e ".[dev]"
py -m agent demo
py -m agent llm-smoke-test
py -m agent executor-ollama-smoke
py -m agent ollama-demo
py -m pytest
```

The deterministic demo remains the default. The separate `llm-smoke-test`
command sends one JSON-schema-constrained request to Ollama and prints the
structured response and Ollama-reported usage. Configure it with
`ADA_OLLAMA_HOST` (default `http://localhost:11434`) and `ADA_LLM_MODEL`
(default `qwen3:8b`). It does not inspect or modify a repository.

`demo` explicitly uses the deterministic Planner and provider. Set
`ADA_AGENT_PROVIDER=ollama` to select the Ollama-backed Planner and Executor
through the Orchestrator. `ollama-demo` is the explicit disposable end-to-end
path and uses `ADA_OLLAMA_HOST` and `ADA_LLM_MODEL` for its local model.

`executor-ollama-smoke` runs one model-generated action through the existing
Executor in a disposable workspace, then runs its small pytest fixture there.
It never falls back to the deterministic provider.

The LLM Planner uses Ollama JSON-Schema structured output derived from the
existing `Plan` and `PlanStep` dataclasses. Actual input/output usage is stored
in the `BudgetLedger` when returned by the provider; only calls without usage
metadata remain estimated.

Ollama requests explicitly set `think=false` for this milestone. The API can
return a separate `message.thinking` field; the provider exposes whether that
field was present. Ollama reports one `eval_count` value without a separate
thinking-token count, so the agent does not infer or subtract reasoning tokens.
See the [Ollama thinking API documentation](https://docs.ollama.com/capabilities/thinking).

The demo creates a tiny Python repository, inspects it, plans two file changes,
executes validated write actions, runs its discovered test command, and prints
structured observations. It does not modify this checkout.

For Ollama action generation, the Executor includes only existing files named
by the current step's `expected_files`, capped at 8,000 characters total.
Localized changes use `edit_file`, which atomically replaces one or more exact,
unique, non-overlapping text spans in the same file. Missing or ambiguous text
rejects the whole action without modifying the file. `write_file`
remains available for new files and intentional full-file replacement.

The Orchestrator records each validated mutation separately from its execution
result, then runs a deterministic read-only verifier over the actual target file
contents and test-command result. `COMPLETED` now requires both completed plan
steps and a verified result. A rejected result leaves the round `BLOCKED`; the
verifier does not retry or modify repository state. Persistent cross-round state,
recovery, adaptive budgets, and benchmark infrastructure remain outside this
phase.
