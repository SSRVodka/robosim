"""
Tests for the Agent loop using mocked LLM and registry.
Verifies: single-turn, multi-turn tool calls, streaming, max_iterations guard.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent import Agent, AgentError
from src.config import AgentConfig, MemoryConfig
from src.llm import LLMClient
from src.memory import Memory
from src.tools.base import ToolCall, ToolResult
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_cfg():
    return AgentConfig(
        name="TestAgent",
        system_prompt="You are a test assistant.",
        max_iterations=5,
    )


@pytest.fixture
async def memory(tmp_path):
    cfg = MemoryConfig(db_path=str(tmp_path / "test.db"))
    m = Memory(cfg)
    await m.initialize()
    yield m
    await m.close()


def _make_agent(agent_cfg, memory, llm, registry):
    return Agent(agent_cfg, llm, memory, registry)


# ---------------------------------------------------------------------------
# Helper mocks
# ---------------------------------------------------------------------------

def _mock_registry(tools=None, return_content="tool result"):
    reg = MagicMock(spec=ToolRegistry)
    reg.get_tool_definitions.return_value = tools or []
    async def _call_all(tool_calls):
        return [
            ToolResult(tool_call_id=tc.id, name=tc.name, content=return_content)
            for tc in tool_calls
        ]
    reg.call_all = _call_all
    return reg


def _mock_llm_single(content="Hello from assistant"):
    """LLM that returns a single text response."""
    llm = MagicMock(spec=LLMClient)
    assistant_msg = {"role": "assistant", "content": content}
    llm.complete = AsyncMock(return_value=(assistant_msg, [], "stop"))

    async def _stream_acc(messages, tools=None):
        # Yield a fake text chunk then a final chunk
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = content
        chunk1.choices[0].finish_reason = None
        yield chunk1, None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = None
        chunk2.choices[0].finish_reason = "stop"
        acc = MagicMock()
        acc.finish_reason = "stop"
        acc.tool_calls = []
        acc.to_message.return_value = assistant_msg
        yield chunk2, acc

    llm.stream_accumulated = _stream_acc
    return llm


def _mock_llm_with_tool(tool_name="get_weather", final_content="It is sunny."):
    """LLM that first returns a tool call, then a final answer."""
    llm = MagicMock(spec=LLMClient)
    call_count = 0

    tc = ToolCall(id="call-1", name=tool_name, arguments='{"city":"NYC"}')
    tool_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call-1", "type": "function",
             "function": {"name": tool_name, "arguments": '{"city":"NYC"}'}}
        ],
    }
    final_msg = {"role": "assistant", "content": final_content}

    async def _complete(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_msg, [tc], "tool_calls"
        return final_msg, [], "stop"

    llm.complete = _complete

    # streaming variant
    stream_call_count = 0

    async def _stream_acc(messages, tools=None):
        nonlocal stream_call_count
        stream_call_count += 1
        if stream_call_count == 1:
            # First call → tool_calls finish
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = None
            chunk.choices[0].finish_reason = "tool_calls"
            acc = MagicMock()
            acc.finish_reason = "tool_calls"
            acc.tool_calls = [tc]
            acc.to_message.return_value = tool_msg
            yield chunk, acc
        else:
            # Second call → text + stop
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = final_content
            chunk1.choices[0].finish_reason = None
            yield chunk1, None

            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = None
            chunk2.choices[0].finish_reason = "stop"
            acc2 = MagicMock()
            acc2.finish_reason = "stop"
            acc2.tool_calls = []
            acc2.to_message.return_value = final_msg
            yield chunk2, acc2

    llm.stream_accumulated = _stream_acc
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_turn(agent_cfg, memory):
    llm = _mock_llm_single("Hello!")
    reg = _mock_registry()
    agent = _make_agent(agent_cfg, memory, llm, reg)
    result = await agent.run([{"role": "user", "content": "Hi"}], "s1")
    assert result["content"] == "Hello!"


@pytest.mark.asyncio
async def test_history_persisted(agent_cfg, memory):
    llm = _mock_llm_single("Answer")
    reg = _mock_registry()
    agent = _make_agent(agent_cfg, memory, llm, reg)
    await agent.run([{"role": "user", "content": "First"}], "s2")
    history = await memory.load("s2")
    roles = [m["role"] for m in history]
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_tool_call_round_trip(agent_cfg, memory):
    llm = _mock_llm_with_tool("get_weather", "It is sunny.")
    reg = _mock_registry(
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        return_content="Sunny and 75°F",
    )
    agent = _make_agent(agent_cfg, memory, llm, reg)
    result = await agent.run([{"role": "user", "content": "Weather?"}], "s3")
    assert result["content"] == "It is sunny."

    # History should contain user, assistant (tool call), tool result, assistant (final)
    history = await memory.load("s3")
    roles = [m["role"] for m in history]
    assert roles.count("assistant") == 2
    assert "tool" in roles


@pytest.mark.asyncio
async def test_streaming_no_tool(agent_cfg, memory):
    llm = _mock_llm_single("Stream token")
    reg = _mock_registry()
    agent = _make_agent(agent_cfg, memory, llm, reg)
    tokens = []
    async for tok in agent.run_stream([{"role": "user", "content": "Hi"}], "s4"):
        tokens.append(tok)
    assert "Stream token" in "".join(tokens)


@pytest.mark.asyncio
async def test_streaming_with_tool(agent_cfg, memory):
    llm = _mock_llm_with_tool("calc", "42")
    reg = _mock_registry(
        tools=[{"type": "function", "function": {"name": "calc"}}],
        return_content="42",
    )
    agent = _make_agent(agent_cfg, memory, llm, reg)
    tokens = []
    async for tok in agent.run_stream([{"role": "user", "content": "1+1?"}], "s5"):
        tokens.append(tok)
    assert "42" in "".join(tokens)


@pytest.mark.asyncio
async def test_max_iterations_exceeded(agent_cfg, memory):
    """Agent should raise AgentError if tool_calls never resolves."""
    llm = MagicMock(spec=LLMClient)
    tc = ToolCall(id="c1", name="loop", arguments="{}")
    tool_msg = {"role": "assistant", "content": None, "tool_calls": []}
    llm.complete = AsyncMock(return_value=(tool_msg, [tc], "tool_calls"))
    reg = _mock_registry()
    agent = _make_agent(agent_cfg, memory, llm, reg)

    with pytest.raises(AgentError, match="max_iterations"):
        await agent.run([{"role": "user", "content": "loop"}], "s6")


@pytest.mark.asyncio
async def test_history_loaded_in_context(agent_cfg, memory):
    """Prior history should be included in subsequent calls."""
    await memory.save("s7", [
        {"role": "user", "content": "Previous"},
        {"role": "assistant", "content": "Remembered"},
    ])

    captured_messages = []
    llm = MagicMock(spec=LLMClient)
    async def _complete(messages, tools=None):
        captured_messages.extend(messages)
        return {"role": "assistant", "content": "OK"}, [], "stop"
    llm.complete = _complete

    reg = _mock_registry()
    agent = _make_agent(agent_cfg, memory, llm, reg)
    await agent.run([{"role": "user", "content": "New"}], "s7")

    contents = [m.get("content") for m in captured_messages]
    assert "Previous" in contents
    assert "Remembered" in contents
