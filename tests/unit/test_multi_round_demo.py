import json

from agent import __main__
from agent.state.store import StateStore


def test_multi_round_demo_completes_and_crosses_state_store_boundary(monkeypatch, capsys):
    original_load = StateStore.load_progress
    loaded_states = []

    def recording_load(store, task_id):
        state = original_load(store, task_id)
        loaded_states.append(state.to_dict())
        return state

    monkeypatch.setattr(StateStore, "load_progress", recording_load)

    exit_code = __main__.multi_round_demo()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["workspace_is_temporary"] is True
    assert output["state_store_reload_count"] == 2
    assert len(loaded_states) == 2
    assert [item["status"] for item in loaded_states[0]["requirements"]] == ["completed", "pending"]
    assert output["rounds"] == [
        {
            "round_id": 1, "pending_requirement_ids": ["R1", "R2"],
            "pending_requirements": 2, "newly_verified_requirements": 1,
            "planning_result": "accepted", "verification_result": "verified",
            "status": "accepted",
        },
        {
            "round_id": 2, "pending_requirement_ids": ["R2"],
            "pending_requirements": 1, "newly_verified_requirements": 1,
            "planning_result": "accepted", "verification_result": "verified",
            "status": "accepted",
        },
    ]
    assert output["final_state"]["completed_requirements"] == 2
    assert output["final_state"]["requirement_count"] == 2
    assert output["final_state"]["outcome"] == "COMPLETED"
    assert output["trace"]["llm_calls"] == 0
    assert output["trace"]["token_metrics"] == "unavailable: deterministic provider made no LLM calls"
    assert output["trace"]["input_tokens"] is None
    assert output["trace"]["output_tokens"] is None
    assert "src/utils.py" in output["final_diff"]
    assert "src/text_tools.py" in output["final_diff"]
