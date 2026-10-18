"""Configuration loading from explicit values and environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from .models import AgentConfig


def load_config(workspace: str | Path = ".") -> AgentConfig:
    return AgentConfig(workspace=str(Path(workspace).resolve()),
                       state_dir=os.getenv("ADA_AGENT_STATE_DIR", ".agent-state"),
                       command_timeout_seconds=float(os.getenv("ADA_AGENT_COMMAND_TIMEOUT", "30")),
                       provider=os.getenv("ADA_AGENT_PROVIDER", "deterministic"))
