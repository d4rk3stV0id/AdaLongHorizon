# AdaLongHorizon

Budget-aware long-horizon software engineering agent research prototype.

## Requirements

Python 3.11+ and Git. The demonstration uses pytest in a disposable temporary repository.

## Run

```powershell
py -m pip install -e ".[dev]"
py -m agent demo
py -m pytest
```

The demo creates a tiny Python repository, inspects it, plans two file changes,
executes validated write actions, runs its discovered test command, and prints
structured observations. It does not modify this checkout.

The initial prototype implements typed task state, validated lifecycle changes,
JSON state persistence, append-only JSONL events, deterministic budgets, a
bounded Git-backed repository adapter, a provider interface, and a minimal
planner/executor. Verification is currently limited to returning command
observations; durable verified completion, advanced recovery, adaptive budgets,
and benchmark infrastructure are outside this phase.
