import json
import sys
from types import SimpleNamespace
from pathlib import Path

from agent import __main__
from agent.models import Lifecycle
from agent.tools.repository import RepositoryAdapter
from agent.state.progress import (
    ProgressStatus, RequirementStatus, RoundTrace, RunTrace, create_progress_state,
)


def test_ollama_multi_round_demo_uses_persistent_orchestration_and_reports_trace(
    monkeypatch, capsys,
):
    observed = {}

    class FakeOrchestrator:
        def __init__(self, config):
            observed["provider"] = config.provider
            self.repository = RepositoryAdapter(config.workspace)
            self.last_run_trace = None

        def run_persistent(self, issue):
            observed["criteria"] = issue.acceptance_criteria
            (Path(self.repository.root) / "src" / "multiply.py").write_text(
                "def multiply(a, b):\n    return a * b\n", encoding="utf-8",
            )
            progress = create_progress_state("demo-task", issue)
            for item in progress.requirements:
                item.status = RequirementStatus.COMPLETED
            progress.status = ProgressStatus.COMPLETED
            traces = [
                RoundTrace(
                    "demo-task", 1, llm_calls=2, tool_calls=3,
                    input_tokens=100, output_tokens=30, total_tokens=130,
                    provider_latency_ms=1250,
                    pending_requirements=2, pending_requirement_ids=["R1", "R2"],
                    newly_verified_requirements=1, planning_result="accepted",
                    verification_result="verified", final_outcome="completed",
                ),
                RoundTrace(
                    "demo-task", 2, llm_calls=2, tool_calls=3,
                    input_tokens=120, output_tokens=40, total_tokens=160,
                    provider_latency_ms=1400,
                    pending_requirements=1, pending_requirement_ids=["R2"],
                    newly_verified_requirements=1, planning_result="accepted",
                    verification_result="verified", final_outcome="completed",
                ),
            ]
            self.last_run_trace = RunTrace.from_rounds(
                task_id="demo-task", rounds=traces, final_outcome="completed",
                elapsed_seconds=3.0, verification_result="verified",
            )
            round_results = [
                (
                    SimpleNamespace(
                        status=Lifecycle.COMPLETED,
                        transient={"verification_result": {"failures": []}},
                    ),
                        SimpleNamespace(results=[SimpleNamespace(
                            operation="run_command", success=True,
                            stdout="2 passed", stderr="",
                        )], notes=[]),
                )
                for _ in traces
            ]
            return progress, round_results

    monkeypatch.setattr(__main__, "Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(sys, "argv", ["agent", "ollama-multi-round-demo"])

    assert __main__.main() == 0
    output = json.loads(capsys.readouterr().out)

    assert observed["provider"] == "ollama"
    assert len(observed["criteria"]) == 2
    assert output["usage"] == {
        "rounds": 2,
        "llm_calls": 4,
        "tool_actions": 6,
        "input_tokens": 220,
        "output_tokens": 70,
        "total_tokens": 290,
        "provider_latency_ms": 2650,
    }
    assert output["round_details"][0]["pending_requirement_ids"] == ["R1", "R2"]
    assert output["round_details"][1]["pending_requirement_ids"] == ["R2"]
    assert output["final_state"]["completed_requirements"] == 2
    assert output["final_state"]["task_status"] == "COMPLETED"
    assert "src/multiply.py" in output["final_diff"]
    assert "def multiply" in output["final_diff"]
    assert output["changed_files"] == ["src/multiply.py"]
