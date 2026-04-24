"""
Configuration: YAML loading with ${ENV_VAR} and ${ENV_VAR:-default} interpolation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Env-var interpolation
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env(value: Any) -> Any:
    """Recursively replace ${VAR} and ${VAR:-default} in strings."""
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            expr = m.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                return os.environ.get(var.strip(), default.strip())
            return os.environ.get(expr.strip(), m.group(0))
        return _ENV_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0


class MemoryConfig(BaseModel):
    backend: Literal["sqlite"] = "sqlite"
    db_path: str = "./data/memory.db"
    max_history: int = 100


class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "sse", "streamable_http"]
    # stdio fields
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # sse / streamable_http fields
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_fields(self) -> "MCPServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"MCP server '{self.name}': 'command' required for stdio transport")
        if self.transport in ("sse", "streamable_http") and not self.url:
            raise ValueError(f"MCP server '{self.name}': 'url' required for {self.transport} transport")
        return self


class ToolsConfig(BaseModel):
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)


class CLIConfig(BaseModel):
    enabled: bool = True
    stream: bool = True
    session_id: str = "cli-default"


class APIConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None


class InterfacesConfig(BaseModel):
    cli: CLIConfig = Field(default_factory=CLIConfig)
    api: APIConfig = Field(default_factory=APIConfig)


class STTServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8081
    enabled: bool = True
    args: dict[str, str | None] = Field(default_factory=dict)


class TTSConfig(BaseModel):
    api_key: str = ""
    url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    model: str = "qwen3-tts-flash"
    voice: str = "Cherry"
    language: str = "Chinese"
    phone_ip: str = "192.168.1.125"
    phone_port: int = 9999
    enabled: bool = True


class AgentConfig(BaseModel):
    name: str = "Agent"
    system_prompt: str = "You are a helpful assistant."
    max_iterations: int = 20
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    interfaces: InterfacesConfig = Field(default_factory=InterfacesConfig)
    stt: STTServerConfig | None = Field(default=None)
    tts: TTSConfig | None = Field(default=None)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config_from_dict(raw: dict) -> AgentConfig:
    raw = _resolve_env(raw)
    return AgentConfig.model_validate(raw)

def load_config(path: str = "config/default.yaml") -> AgentConfig:
    config_path = Path(path)
    raw: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    raw = _resolve_env(raw)
    return AgentConfig.model_validate(raw)
