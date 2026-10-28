"""CLI entry point for running agent demonstrations and tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from .config import load_config
from .errors import BudgetExceeded, LLMProviderError, RepositoryError
from .executor import Executor
from .llm.ollama import OllamaProvider
from .models import IssueSpec, ObservationBundle, Plan, PlanStep, TaskState
from .orchestrator import Orchestrator
from .tools.repository import RepositoryAdapter


def demo(task: str = "multiply") -> int:
    with tempfile.TemporaryDirectory(prefix="ada-agent-demo-") as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\n"
            "testpaths = ['tests']\n"
            "pythonpath = ['.']\n"
            "addopts = '-p no:cacheprovider --basetemp=.pytest_tmp'\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "demo@test.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Demo"], cwd=root, check=True)

        if task == "multiply":
            issue = IssueSpec(
                title="Add multiply function",
                description="Add a utility function multiply(a, b) and tests",
                acceptance_criteria=["function returns product of a and b", "pytest passes"],
            )
        else:
            issue = IssueSpec(
                title="Normalize whitespace",
                description="Add a function that collapses repeated whitespace.",
                acceptance_criteria=["function returns trimmed single-space text", "pytest passes"],
            )

        config = load_config(root)
        config.provider = "deterministic"
        orchestrator = Orchestrator(config)
        state, observations = orchestrator.run(issue)

        events_file = root / ".agent-state" / "events.jsonl"
        events = (
            [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if events_file.exists()
            else []
        )

        output = {
            "task": task,
            "status": state.status.value,
            "verification_result": state.transient.get("verification_result"),
            "plan": [s.step_id for s in state.plan.steps] if state.plan else [],
            "completed_steps": state.completed_steps,
            "files_in_workspace": [
                p.relative_to(root).as_posix()
                for p in root.rglob("*.py")
                if not any(part.startswith(".") for part in p.parts)
            ],
            "event_transitions": [e["event_type"] for e in events],
            "observations": [
                {
                    "operation": r.operation,
                    "success": r.success,
                    "exit_code": r.exit_code,
                    "stdout": r.stdout.strip(),
                    "stderr": r.stderr.strip(),
                }
                for r in observations.results
            ],
            "notes": observations.notes,
        }
        print(json.dumps(output, indent=2))
        return 0 if state.status.value == "COMPLETED" else 1


def multi_round_demo() -> int:
    """Demonstrate deterministic verified progress across persisted rounds."""
    with tempfile.TemporaryDirectory(
        prefix="ada-multi-round-demo-", dir=Path.cwd(),
    ) as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (root / "tests" / "test_smoke.py").write_text(
            "def test_smoke():\n    assert True\n", encoding="utf-8",
        )
        (root / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\n"
            "testpaths = ['tests']\n"
            "pythonpath = ['.']\n"
            "addopts = '-p no:cacheprovider --basetemp=.pytest_tmp'\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "multi-round-demo@test.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Multi-Round Demo"], cwd=root, check=True)
        subprocess.run(["git", "add", "--", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "temporary demo baseline"], cwd=root, check=True)

        issue = IssueSpec(
            title="Add two independent utilities",
            description="Add a function and corresponding tests for each requirement.",
            acceptance_criteria=[
                "multiply function returns the product of its arguments",
                "normalize_whitespace function collapses repeated whitespace",
            ],
        )
        config = load_config(root)
        config.provider = "deterministic"
        orchestrator = Orchestrator(config)
        progress, _ = orchestrator.run_persistent(issue)
        trace = orchestrator.last_run_trace
        if trace is None:
            print(json.dumps({"success": False, "error": "multi-round trace was not produced"}, indent=2))
            return 1

        expected_files = [
            "src/utils.py", "tests/test_utils.py",
            "src/text_tools.py", "tests/test_text_tools.py",
        ]
        subprocess.run(["git", "add", "-N", "--", *expected_files], cwd=root, check=True)
        repo = orchestrator.repository
        round_summaries = [
            {
                "round_id": item.round_id,
                "pending_requirement_ids": item.pending_requirement_ids,
                "pending_requirements": item.pending_requirements,
                "newly_verified_requirements": item.newly_verified_requirements,
                "planning_result": item.planning_result,
                "verification_result": item.verification_result,
                "status": (
                    "accepted"
                    if item.verification_result == "verified" and (item.newly_verified_requirements or 0) > 0
                    else "blocked"
                ),
            }
            for item in trace.rounds
        ]
        completed_count = sum(item.status.value == "completed" for item in progress.requirements)
        output = {
            "success": progress.status.value == "completed" and completed_count == len(progress.requirements),
            "workspace_is_temporary": True,
            "provider": "deterministic",
            "model": None,
            "test_command": ["python", "-m", "pytest", "-q"],
            "persistence_boundary": "StateStore saves and reloads progress between rounds",
            "state_store_reload_count": trace.rounds_attempted,
            "rounds": round_summaries,
            "final_state": {
                "task_id": progress.task_id,
                "requirements": [
                    {"requirement_id": item.requirement_id, "status": item.status.value}
                    for item in progress.requirements
                ],
                "completed_requirements": completed_count,
                "requirement_count": len(progress.requirements),
                "outcome": progress.status.value.upper(),
            },
            "trace": {
                "run_id": trace.run_id,
                "task_id": trace.task_id,
                "rounds": trace.rounds_attempted,
                "llm_calls": trace.llm_calls,
                "tool_actions": trace.tool_actions,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "total_tokens": trace.total_tokens,
                "token_metrics": "unavailable: deterministic provider made no LLM calls",
                "provider_latency_ms": trace.provider_latency_ms,
                "elapsed_seconds": trace.elapsed_seconds,
                "verification_result": trace.verification_result,
                "final_outcome": trace.final_outcome.upper(),
            },
            "final_diff": repo.diff(),
            "changed_files": [
                path for path in repo.changed_files()
                if Path(path).parts[0] != ".agent-state"
                and ".pytest_tmp" not in Path(path).parts
                and "__pycache__" not in Path(path).parts
            ],
        }
        print(json.dumps(output, indent=2))
        return 0 if output["success"] else 1


def llm_smoke_test() -> int:
    """Make one explicit structured request to the configured local Ollama model."""
    config = load_config()
    provider = OllamaProvider(host=config.ollama_host, model=config.llm_model)
    schema = {
        "type": "object",
        "properties": {"product": {"type": "integer"}},
        "required": ["product"],
        "additionalProperties": False,
    }
    try:
        response = provider.chat(
            [{"role": "user", "content": "Return the product of 17 and 23."}],
            format=schema,
        )
    except Exception as exc:
        print(json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps({
        "success": True,
        "provider": response.provider,
        "model": response.model,
        "response": response.content,
        "prompt_tokens": response.prompt_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": response.total_tokens,
        "usage_available": response.usage_available,
        "duration_ms": response.duration_ms,
        "think": provider.think,
        "thinking_present": bool(response.thinking),
        "done": response.done,
        "done_reason": response.done_reason,
    }, indent=2))
    return 0


def executor_ollama_smoke_test() -> int:
    """Run one Ollama-generated action in a disposable repository."""
    with tempfile.TemporaryDirectory(prefix="ada-executor-ollama-") as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (root / "tests" / "test_utils.py").write_text(
            "from src.utils import multiply\n\n\n"
            "def test_multiply():\n"
            "    assert multiply(3, 4) == 12\n"
            "    assert multiply(-2, 3) == -6\n",
            encoding="utf-8",
        )
        (root / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\n"
            "testpaths = ['tests']\n"
            "pythonpath = ['.']\n"
            "addopts = '-p no:cacheprovider --basetemp=.pytest_tmp'\n",
            encoding="utf-8",
        )

        config = load_config(root)
        config.provider = "ollama"
        orchestrator = Orchestrator(config)
        repository: RepositoryAdapter = orchestrator.repository
        issue = IssueSpec(
            title="Add multiply(a, b) to src/utils.py",
            description="Implement a Python function that returns the product of its two numeric arguments.",
            acceptance_criteria=["multiply(3, 4) returns 12", "multiply(-2, 3) returns -6"],
        )
        step = PlanStep(
            step_id="implementation",
            objective="Implement multiply(a, b) in src/utils.py and return a * b.",
            expected_files=["src/utils.py"],
            action_type="write_file",
            acceptance_checks=["the provided multiply tests pass"],
        )
        state = TaskState(issue, plan=Plan(issue.issue_id, [step]))
        snapshot = repository.inspect()
        observations = ObservationBundle(state.task_id)
        executed = False
        test_result = None
        error = None
        try:
            executor = Executor(repository, orchestrator.provider)
            executed = executor.execute_next(state, snapshot, observations)
            if executed:
                test_result = repository.run_command(["python", "-m", "pytest", "-q"])
        except (LLMProviderError, ValueError, BudgetExceeded, RepositoryError) as exc:
            error = f"{type(exc).__name__}: {exc}"

        output = {
            "success": executed and test_result is not None and test_result.success and not error,
            "workspace_is_temporary": True,
            "provider": "ollama",
            "model": config.llm_model,
            "action": state.transient.get("last_action"),
            "action_validation_passed": "last_action" in state.transient,
            "action_executed": executed,
            "llm_usage": state.transient.get("last_llm_response"),
            "llm_calls": state.budget.llm_calls,
            "actual_tokens_in_budget": state.budget.actual_tokens,
            "estimated_tokens_recorded_in_budget": state.budget.estimated_tokens,
            "final_files": sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.relative_to(root).parts
                and not any(part.startswith(".") for part in path.relative_to(root).parts)
            ),
            "test_result": None if test_result is None else {
                "success": test_result.success,
                "exit_code": test_result.exit_code,
                "stdout": test_result.stdout.strip(),
                "stderr": test_result.stderr.strip(),
            },
            "error": error,
        }
        print(json.dumps(output, indent=2))
        return 0 if output["success"] else 1


def ollama_demo() -> int:
    """Run planning, execution, and pytest in a disposable repository using Ollama."""
    with tempfile.TemporaryDirectory(prefix="ada-ollama-demo-", dir=Path.cwd()) as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (root / "src" / "utils.py").write_text('"""Small numeric utility functions."""\n', encoding="utf-8")
        (root / "tests" / "test_utils.py").write_text(
            '"""Tests for numeric utility functions."""\n\n'
            "def test_existing_behavior():\n"
            "    assert 2 + 2 == 4\n",
            encoding="utf-8",
        )
        (root / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\n"
            "testpaths = ['tests']\n"
            "pythonpath = ['.']\n"
            "addopts = '-p no:cacheprovider --basetemp=.pytest_tmp'\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "ollama-demo@test.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Ollama Demo"], cwd=root, check=True)
        subprocess.run(["git", "add", "--", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "temporary baseline"], cwd=root, check=True)

        config = load_config(root)
        config.provider = "ollama"
        orchestrator = Orchestrator(config)
        issue = IssueSpec(
            title="Add multiply(a, b) and a unit test",
            description=(
                "Add a multiply(a, b) function to src/utils.py that returns the product "
                "of its arguments, and add a pytest unit test in tests/test_utils.py that "
                "imports multiply from src.utils and checks positive and negative inputs. "
                "Use edit_file for these localized additions, preserve both existing module "
                "docstrings, and keep any existing test behavior intact."
            ),
            acceptance_criteria=[
                "multiply(3, 4) returns 12",
                "multiply(-2, 3) returns -6",
                "tests/test_utils.py imports multiply from src.utils and covers multiply",
                "both original module docstrings and the existing test remain",
            ],
        )
        state, observations = orchestrator.run(issue)
        repo = orchestrator.repository
        test_result = next(
            (result for result in reversed(observations.results) if result.operation == "run_command"),
            None,
        )
        output = {
            "success": state.status.value == "COMPLETED",
            "workspace_is_temporary": True,
            "planner": "ollama",
            "provider": type(orchestrator.provider).__name__,
            "model": config.llm_model,
            "status": state.status.value,
            "verification_result": state.transient.get("verification_result"),
            "issue": {"title": issue.title, "acceptance_criteria": issue.acceptance_criteria},
            "plan": [
                {
                    "step_id": step.step_id,
                    "objective": step.objective,
                    "dependencies": step.dependencies,
                    "expected_files": step.expected_files,
                    "status": step.status.value,
                }
                for step in state.plan.steps
            ] if state.plan else None,
            "generated_actions": state.transient.get("generated_actions", []),
            "usage": state.transient.get("llm_usage", []),
            "budget": {
                "llm_calls": state.budget.llm_calls,
                "tool_actions": state.budget.tool_actions,
                "actual_input_tokens": state.budget.actual_input_tokens,
                "actual_output_tokens": state.budget.actual_output_tokens,
                "actual_tokens": state.budget.actual_tokens,
                "estimated_tokens": state.budget.estimated_tokens,
            },
            "test_result": None if test_result is None else {
                "success": test_result.success,
                "exit_code": test_result.exit_code,
                "stdout": test_result.stdout.strip(),
                "stderr": test_result.stderr.strip(),
            },
            "final_diff": repo.diff(),
            "changed_files": [
                path for path in repo.changed_files()
                if "__pycache__" not in Path(path).parts
                and ".pytest_tmp" not in Path(path).parts
                and Path(path).parts[0] != ".agent-state"
            ],
            "notes": observations.notes,
            "error": state.transient.get("error"),
        }
        print(json.dumps(output, indent=2))
        return 0 if output["success"] else 1


def ollama_multi_round_demo() -> int:
    """Run two independently verified persistent rounds using Ollama."""
    with tempfile.TemporaryDirectory(
        prefix="ada-ollama-multi-round-demo-", dir=Path.cwd(),
    ) as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (root / "src" / "text_tools.py").write_text(
            '"""Small text utility functions."""\n\n'
            "def normalize_whitespace(text):\n"
            '    return " ".join(text.split())\n',
            encoding="utf-8",
        )
        (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
        (root / "tests" / "test_utils.py").write_text(
            '"""Tests for numeric utility functions."""\n\n'
            "from src.multiply import multiply\n\n\n"
            "def test_existing_behavior():\n"
            "    assert 2 + 2 == 4\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_text_tools.py").write_text(
            '"""Tests for text utility functions."""\n\n'
            "from src.text_tools import normalize_whitespace\n\n\n"
            "def test_existing_text_behavior():\n"
            '    assert normalize_whitespace("ready") == "ready"\n',
            encoding="utf-8",
        )
        (root / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\n"
            "testpaths = ['tests']\n"
            "pythonpath = ['.']\n"
            "addopts = '-p no:cacheprovider --basetemp=.pytest_tmp'\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "ollama-multi-round@test.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Ollama Multi-Round Demo"], cwd=root, check=True)
        subprocess.run(["git", "add", "--", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "temporary multi-round baseline"], cwd=root, check=True)

        config = load_config(root)
        config.provider = "ollama"
        orchestrator = Orchestrator(config)
        issue = IssueSpec(
            title="Complete the current acceptance criterion",
            description=(
                "This is a bounded multi-round task. In each invocation, implement only the single current-round "
                "acceptance criterion appended to this description. Do not work on other still-pending requirements; "
                "they will be handled in later persisted rounds. Preserve existing behavior."
            ),
            acceptance_criteria=[
                "Create src/multiply.py with multiply(a, b), and add positive and negative assertions to the existing tests/test_utils.py test, which already imports it from src.multiply.",
                "Extend tests/test_text_tools.py with coverage that mixed spaces, tabs, and newlines collapse to one space and that surrounding whitespace is trimmed. normalize_whitespace already exists and is imported from src.text_tools.",
            ],
        )
        progress, round_results = orchestrator.run_persistent(issue)
        trace = orchestrator.last_run_trace
        if trace is None:
            print(json.dumps({"success": False, "error": "multi-round trace was not produced"}, indent=2))
            return 1

        repo = orchestrator.repository
        if (root / "src" / "multiply.py").is_file():
            subprocess.run(
                ["git", "add", "-N", "--", "src/multiply.py"], cwd=root, check=True,
            )
        completed_count = sum(item.status.value == "completed" for item in progress.requirements)
        changed_files = [
            path for path in repo.changed_files()
            if "__pycache__" not in Path(path).parts
            and ".pytest_tmp" not in Path(path).parts
            and Path(path).parts[0] != ".agent-state"
        ]
        round_summaries = []
        for index, item in enumerate(trace.rounds):
            state, observations = round_results[index]
            test_result = next(
                (result for result in reversed(observations.results) if result.operation == "run_command"),
                None,
            )
            verification = state.transient.get("verification_result", {})
            round_summaries.append({
                "round_id": item.round_id,
                "pending_requirement_ids": item.pending_requirement_ids,
                "pending_requirements": item.pending_requirements,
                "newly_verified_requirements": item.newly_verified_requirements,
                "planning_result": item.planning_result,
                "verification_result": item.verification_result,
                "outcome": state.status.value,
                "llm_calls": item.llm_calls,
                "tool_actions": item.tool_calls,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "total_tokens": item.total_tokens,
                "provider_latency_ms": item.provider_latency_ms,
                "test_passed": test_result.success if test_result is not None else None,
                "test_stdout": test_result.stdout if test_result is not None else None,
                "test_stderr": test_result.stderr if test_result is not None else None,
                "verification_failures": verification.get("failures", []),
                "error": state.transient.get("error"),
                "notes": observations.notes,
            })

        output = {
            "success": (
                progress.status.value == "completed"
                and completed_count == len(progress.requirements)
                and trace.rounds_attempted >= 2
            ),
            "workspace_is_temporary": True,
            "provider": "ollama",
            "model": config.llm_model,
            "persistence_boundary": "StateStore.save → StateStore.load after each round",
            "state_store_reload_count": trace.rounds_attempted,
            "round_details": round_summaries,
            "usage": {
                "rounds": trace.rounds_attempted,
                "llm_calls": trace.llm_calls,
                "tool_actions": trace.tool_actions,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "total_tokens": trace.total_tokens,
                "provider_latency_ms": trace.provider_latency_ms,
            },
            "final_verification": trace.verification_result,
            "final_outcome": trace.final_outcome.upper(),
            "final_state": {
                "requirements": [
                    {"requirement_id": item.requirement_id, "status": item.status.value}
                    for item in progress.requirements
                ],
                "completed_requirements": completed_count,
                "requirement_count": len(progress.requirements),
                "task_status": progress.status.value.upper(),
            },
            "test_command": ["python", "-m", "pytest", "-q"],
            "final_diff": repo.diff(),
            "changed_files": changed_files,
        }
        print(json.dumps(output, indent=2))
        return 0 if output["success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Budget-aware software engineering prototype")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run deterministic end-to-end demonstration")
    demo_parser.add_argument(
        "--task",
        choices=["multiply", "normalize"],
        default="multiply",
        help="Demo task to execute (default: multiply)",
    )

    subparsers.add_parser(
        "multi-round-demo",
        help="Run a deterministic persistent-state multi-round demonstration",
    )

    subparsers.add_parser("llm-smoke-test", help="Send one structured request to configured Ollama model")
    subparsers.add_parser(
        "executor-ollama-smoke",
        help="Run one Qwen-generated action through Executor in a temporary workspace",
    )
    subparsers.add_parser(
        "ollama-demo",
        help="Run a full Ollama planning and execution task in a temporary workspace",
    )
    subparsers.add_parser(
        "ollama-multi-round-demo",
        help="Run a two-requirement persistent multi-round demonstration using Ollama",
    )

    args = parser.parse_args()
    if args.command == "demo":
        return demo(task=args.task)
    if args.command == "multi-round-demo":
        return multi_round_demo()
    if args.command == "llm-smoke-test":
        return llm_smoke_test()
    if args.command == "executor-ollama-smoke":
        return executor_ollama_smoke_test()
    if args.command == "ollama-demo":
        return ollama_demo()
    if args.command == "ollama-multi-round-demo":
        return ollama_multi_round_demo()

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
