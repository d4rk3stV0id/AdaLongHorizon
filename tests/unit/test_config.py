"""Unit tests for configuration loading."""

import os
from unittest import mock

from agent.config import load_config


def test_load_config_defaults(tmp_path):
    config = load_config(tmp_path)
    assert config.workspace == str(tmp_path.resolve())
    assert config.state_dir == ".agent-state"
    assert config.command_timeout_seconds == 30.0
    assert config.provider == "deterministic"


def test_load_config_env_overrides(tmp_path):
    env = {
        "ADA_AGENT_STATE_DIR": "custom-state",
        "ADA_AGENT_COMMAND_TIMEOUT": "45.5",
        "ADA_AGENT_PROVIDER": "custom_provider",
    }
    with mock.patch.dict(os.environ, env):
        config = load_config(tmp_path)
        assert config.state_dir == "custom-state"
        assert config.command_timeout_seconds == 45.5
        assert config.provider == "custom_provider"


def test_orchestrator_selects_ollama_provider_from_config(tmp_path):
    from agent.models import AgentConfig
    from agent.orchestrator import Orchestrator

    provider = object()
    config = AgentConfig(
        workspace=str(tmp_path), provider="ollama", ollama_host="http://ollama.local:11434",
        llm_model="qwen3:8b",
    )
    with mock.patch("agent.llm.ollama.OllamaProvider", return_value=provider) as factory:
        orchestrator = Orchestrator(config)

    factory.assert_called_once_with(host="http://ollama.local:11434", model="qwen3:8b")
    assert orchestrator.provider is provider
    assert orchestrator.planner.provider is provider
