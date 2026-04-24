"""
Tests for: multimodal helpers, config loading, tool base models.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile

import pytest

from src.config import load_config, AgentConfig
from src.multimodal import build_user_message, text_item, image_file_item
from src.tools.base import ToolDefinition, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_load_default_config(tmp_path):
    yaml_content = """
name: "TestBot"
system_prompt: "You are a test bot."
llm:
  model: "gpt-4o-mini"
  api_key: "sk-test"
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.name == "TestBot"
    assert cfg.llm.model == "gpt-4o-mini"


def test_env_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-from-env")
    yaml_content = "llm:\n  api_key: '${TEST_KEY}'\n"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.llm.api_key == "sk-from-env"


def test_env_default_value(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    yaml_content = "llm:\n  base_url: '${MISSING_VAR:-http://localhost}'\n"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.llm.base_url == "http://localhost"


def test_missing_config_returns_defaults():
    cfg = load_config("/nonexistent/path.yaml")
    assert isinstance(cfg, AgentConfig)
    assert cfg.llm.model == "gpt-4o"


def test_mcp_server_config_validation(tmp_path):
    from src.config import MCPServerConfig
    with pytest.raises(Exception):
        # stdio without command should fail
        MCPServerConfig(name="bad", transport="stdio")
    with pytest.raises(Exception):
        # sse without url should fail
        MCPServerConfig(name="bad", transport="sse")


# ---------------------------------------------------------------------------
# Multimodal
# ---------------------------------------------------------------------------

def test_text_only_message():
    msg = build_user_message(text="Hello")
    assert msg["role"] == "user"
    assert msg["content"] == "Hello"  # optimised to plain string


def test_text_item():
    item = text_item("hi")
    assert item == {"type": "text", "text": "hi"}


def test_image_url_message():
    msg = build_user_message(
        text="Look at this",
        images=["https://example.com/img.jpg"]
    )
    assert isinstance(msg["content"], list)
    types = [p["type"] for p in msg["content"]]
    assert "text" in types
    assert "image_url" in types


def test_image_file_message(tmp_path):
    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)  # minimal JPEG-like bytes
    item = image_file_item(img)
    assert item["type"] == "image_url"
    assert "data:image/jpeg;base64," in item["image_url"]["url"]


def test_empty_message_raises():
    with pytest.raises(ValueError):
        build_user_message()


def test_mixed_content_message(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 10)
    msg = build_user_message(
        text="Describe",
        images=[str(img)],
    )
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 2


# ---------------------------------------------------------------------------
# Tool base models
# ---------------------------------------------------------------------------

def test_tool_definition_from_parts():
    td = ToolDefinition.from_parts(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    assert td.name == "get_weather"
    assert td.type == "function"


def test_tool_definition_serialise():
    td = ToolDefinition.from_parts("fn", "desc", {"type": "object"})
    d = td.model_dump()
    assert d["type"] == "function"
    assert d["function"]["name"] == "fn"


def test_tool_result_to_message():
    r = ToolResult(tool_call_id="c1", name="fn", content="result")
    msg = r.to_message()
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "c1"
    assert msg["content"] == "result"


def test_tool_call_model():
    tc = ToolCall(id="c1", name="fn", arguments='{"x":1}')
    assert tc.name == "fn"
    assert json.loads(tc.arguments) == {"x": 1}
