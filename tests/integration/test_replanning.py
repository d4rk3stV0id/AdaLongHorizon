"""Deterministic M4 integration coverage for bounded runtime replanning."""

from pathlib import Path

from agent.models import AgentConfig, IssueSpec, Lifecycle, ToolResult
from agent.orchestrator import Orchestrator
from agent.state.progress import RequirementStatus


class ReplanProvider:
    def __init__(self, plans, actions):
        self.plans = plans
        self.actions = actions
        self.plan_calls = 0

    def propose_plan(self, issue, snapshot, state, budget, response_schema):
        plan = self.plans[min(self.plan_calls, len(self.plans) - 1)]
        self.plan_calls += 1
        return {"steps": [plan]}

    def propose_action(self, issue, step, snapshot, state):
        return self.actions[step.step_id]


def _step(step_id, path, action_type="write_file"):
    return {
        "step_id": step_id, "objective": f"satisfy the selected requirement via {path}",
        "dependencies": [], "expected_files": [path], "action_type": action_type,
        "acceptance_checks": ["artifact exists"], "risk_level": "low", "estimated_cost": 1,
    }


def _orchestrator(git_repo, provider, **config):
    (git_repo / "tests").mkdir()
    (git_repo / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
    (git_repo / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths=['tests']\n", encoding="utf-8")
    return Orchestrator(AgentConfig(workspace=str(git_repo), provider="ollama", **config), provider)


def test_operation_blocker_replans_to_alternative_strategy(git_repo: Path):
    provider = ReplanProvider(
        [_step("missing-edit", "missing.py", "edit_file"), _step("write-alternative", "created.py")],
        {
            "missing-edit": {"type": "edit_file", "path": "missing.py", "edits": [{"old_text": "x", "new_text": "y"}]},
            "write-alternative": {"type": "write_file", "path": "created.py", "content": "value = 1\n"},
        },
    )
    orchestrator = _orchestrator(git_repo, provider)
    progress, rounds = orchestrator.run_persistent(IssueSpec("Create value", "Create value", ["value exists"]))

    state, _ = rounds[0]
    assert state.status == Lifecycle.COMPLETED
    assert (git_repo / "created.py").is_file()
    assert state.plan.version == 2
    assert progress.requirements[0].status == RequirementStatus.COMPLETED
    assert progress.rounds[0].revision_history[0]["result"] == "activated"


def test_terminal_permission_blocker_does_not_replan(git_repo: Path):
    provider = ReplanProvider([_step("write", "blocked.py")], {
        "write": {"type": "write_file", "path": "blocked.py", "content": "value = 1\n"},
    })
    orchestrator = _orchestrator(git_repo, provider)
    orchestrator.repository.write_file = lambda path, content: ToolResult("write_file", False, stderr="permission denied")

    progress, rounds = orchestrator.run_persistent(IssueSpec("Write", "Write", ["value exists"]))

    assert rounds[0][0].status == Lifecycle.BLOCKED
    assert provider.plan_calls == 1
    assert progress.requirements[0].status == RequirementStatus.PENDING


def test_invalid_revised_plan_is_rejected_without_replacing_active_plan(git_repo: Path):
    invalid = _step("bad", "bad.py")
    invalid["dependencies"] = ["unknown"]
    provider = ReplanProvider([_step("missing-edit", "missing.py", "edit_file"), invalid], {
        "missing-edit": {"type": "edit_file", "path": "missing.py", "edits": [{"old_text": "x", "new_text": "y"}]},
    })
    orchestrator = _orchestrator(git_repo, provider)
    progress, rounds = orchestrator.run_persistent(IssueSpec("Write", "Write", ["value exists"]))

    state, _ = rounds[0]
    assert state.status == Lifecycle.BLOCKED
    assert state.plan.version == 1
    assert state.plan.steps[0].step_id == "missing-edit"
    assert state.transient["replan_history"][0]["result"] == "rejected"
    assert progress.requirements[0].status == RequirementStatus.PENDING


def test_final_verification_is_mandatory_after_achieved_assessment(git_repo: Path):
    provider = ReplanProvider([_step("write", "value.py")], {
        "write": {"type": "write_file", "path": "value.py", "content": "value = 1\n"},
    })
    orchestrator = _orchestrator(git_repo, provider)
    (git_repo / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert False\n", encoding="utf-8")

    progress, rounds = orchestrator.run_persistent(IssueSpec("Write", "Write", ["value exists"]))

    assert rounds[0][0].transient["execution_outcomes"][0]["assessment"] == "achieved"
    assert rounds[0][0].transient["verification_result"]["status"] == "rejected"
    assert progress.requirements[0].status == RequirementStatus.PENDING


def test_replan_limit_exhaustion_blocks_safely(git_repo: Path):
    provider = ReplanProvider(
        [_step("first", "one.py", "edit_file"), _step("second", "two.py", "edit_file"), _step("third", "three.py", "edit_file")],
        {
            name: {"type": "edit_file", "path": path, "edits": [{"old_text": "x", "new_text": "y"}]}
            for name, path in [("first", "one.py"), ("second", "two.py"), ("third", "three.py")]
        },
    )
    orchestrator = _orchestrator(git_repo, provider, max_replans=1)
    progress, rounds = orchestrator.run_persistent(IssueSpec("Write", "Write", ["value exists"]))

    state, _ = rounds[0]
    assert state.status == Lifecycle.BLOCKED
    assert provider.plan_calls == 2
    assert state.plan.version == 2
    assert len(state.transient["replan_history"]) == 1
    assert progress.requirements[0].status == RequirementStatus.PENDING


def test_previously_verified_requirement_is_preserved_when_next_round_blocks(git_repo: Path):
    class TwoRoundProvider(ReplanProvider):
        def propose_plan(self, issue, snapshot, state, budget, response_schema):
            self.plan_calls += 1
            return {"steps": [_step("r1" if "(R1)" in issue.description else "r2", "first.py" if "(R1)" in issue.description else "second.py")]}

    provider = TwoRoundProvider([], {
        "r1": {"type": "write_file", "path": "first.py", "content": "first = True\n"},
        "r2": {"type": "write_file", "path": "second.py", "content": "second = True\n"},
    })
    orchestrator = _orchestrator(git_repo, provider)
    original_write = orchestrator.repository.write_file

    def write_or_deny(path, content):
        if path == "second.py":
            return ToolResult("write_file", False, stderr="permission denied")
        return original_write(path, content)

    orchestrator.repository.write_file = write_or_deny
    progress, rounds = orchestrator.run_persistent(IssueSpec("Two", "Two", ["first exists", "second exists"]))

    assert len(rounds) == 2
    assert [item.status for item in progress.requirements] == [RequirementStatus.COMPLETED, RequirementStatus.PENDING]
