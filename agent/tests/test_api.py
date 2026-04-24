"""
Tests for the FastAPI OpenAI-compatible web API.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.agent import Agent
from src.config import AgentConfig, APIConfig, InterfacesConfig, LLMConfig
from src.interfaces.api import create_app
from src.memory import Memory
from src.tools.base import ToolCall, ToolResult
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cfg(api_key: str | None = None) -> AgentConfig:
    cfg = AgentConfig()
    cfg.interfaces.api.api_key = api_key
    return cfg


async def _make_memory(tmp_path) -> Memory:
    from src.config import MemoryConfig
    m = Memory(MemoryConfig(db_path=str(tmp_path / "api_test.db")))
    await m.initialize()
    return m


def _mock_agent(response_content: str = "Hello from agent") -> Agent:
    agent = MagicMock(spec=Agent)
    agent._memory = MagicMock()
    agent._memory.list_sessions = AsyncMock(return_value=[])
    agent._memory.clear = AsyncMock()

    async def _run(messages, session_id):
        return {"role": "assistant", "content": response_content}

    async def _run_stream(messages, session_id):
        for token in response_content.split():
            yield token + " "

    agent.run = _run
    agent.run_stream = _run_stream
    return agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client_no_auth():
    cfg = _build_cfg(api_key=None)
    agent = _mock_agent("Hello!")
    app = create_app(agent, cfg)
    return TestClient(app)


@pytest.fixture
def client_with_auth():
    cfg = _build_cfg(api_key="secret-key")
    agent = _mock_agent("Secured response")
    app = create_app(agent, cfg)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client_no_auth):
    r = client_no_auth.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_list_models(client_no_auth):
    r = client_no_auth.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


# ---------------------------------------------------------------------------
# Chat completions – non-streaming
# ---------------------------------------------------------------------------

def test_chat_completions_basic(client_no_auth):
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    r = client_no_auth.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_session_id(client_no_auth):
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "user": "my-session-123",
        "stream": False,
    }
    r = client_no_auth.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200


def test_chat_completions_multipart_content(client_no_auth):
    """Content can be a list of content parts."""
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        ],
        "stream": False,
    }
    r = client_no_auth.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Chat completions – streaming
# ---------------------------------------------------------------------------

def test_chat_completions_streaming(client_no_auth):
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    with client_no_auth.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        # Read the response body to avoid httpx.ResponseNotRead
        r.read()
        raw = r.text

    lines = [l for l in raw.split("\n") if l.startswith("data:")]
    assert any("[DONE]" in l for l in lines)

    # Parse content chunks
    chunks = []
    for line in lines:
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            continue
        obj = json.loads(data)
        delta = obj["choices"][0]["delta"]
        if delta.get("content"):
            chunks.append(delta["content"])
    assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def test_auth_required_missing_key(client_with_auth):
    r = client_with_auth.get("/v1/models")
    assert r.status_code == 401


def test_auth_required_wrong_key(client_with_auth):
    r = client_with_auth.get(
        "/v1/models",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401


def test_auth_correct_key(client_with_auth):
    r = client_with_auth.get(
        "/v1/models",
        headers={"Authorization": "Bearer secret-key"},
    )
    assert r.status_code == 200


def test_auth_correct_key_completions(client_with_auth):
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    r = client_with_auth.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer secret-key"},
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def test_list_sessions(client_no_auth):
    r = client_no_auth.get("/v1/sessions")
    assert r.status_code == 200
    assert "sessions" in r.json()


def test_clear_session(client_no_auth):
    r = client_no_auth.delete("/v1/sessions/my-session")
    assert r.status_code == 200
    assert r.json()["cleared"] == "my-session"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_missing_messages_field(client_no_auth):
    r = client_no_auth.post("/v1/chat/completions", json={"model": "gpt-4o"})
    assert r.status_code == 422  # Unprocessable Entity


def test_tool_messages_forwarded(client_no_auth):
    """Tool messages in the request body should pass validation."""
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Use a tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ],
        "stream": False,
    }
    r = client_no_auth.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
