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
