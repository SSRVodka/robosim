"""
Tests for MCP client and ToolRegistry using mock MCP sessions.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import MCPServerConfig
from src.tools.base import ToolCall, ToolDefinition, ToolResult
from src.tools.mcp_client import MCPConnection, MCPManager
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str, description: str = "A tool"):
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = {"type": "object", "properties": {"x": {"type": "string"}}}
    return t


def _make_content_block(text: str):
    block = MagicMock()
    block.text = text
    return block


def _mock_session(tool_names: list[str], call_result_text: str = "ok"):
    session = MagicMock()
    tools = [_make_tool(n) for n in tool_names]
    list_resp = MagicMock()
    list_resp.tools = tools
    session.list_tools = AsyncMock(return_value=list_resp)
    session.initialize = AsyncMock()

    call_resp = MagicMock()
    call_resp.content = [_make_content_block(call_result_text)]
    call_resp.isError = False
    session.call_tool = AsyncMock(return_value=call_resp)
    return session


# ---------------------------------------------------------------------------
# MCPConnection unit tests (session mocked; no real process)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_connection_list_tools():
    cfg = MCPServerConfig(
        name="test-server", transport="stdio",
        command="echo", args=[]
    )
    conn = MCPConnection(cfg)
    conn.session = _mock_session(["tool_a", "tool_b"])
    await conn._refresh_tools()
    tools = conn.list_tools()
    assert len(tools) == 2
    assert tools[0].name == "tool_a"
    assert tools[1].name == "tool_b"


@pytest.mark.asyncio
async def test_mcp_connection_call_tool():
    cfg = MCPServerConfig(
        name="test-server", transport="stdio",
        command="echo", args=[]
    )
    conn = MCPConnection(cfg)
    conn.session = _mock_session(["my_tool"], "result text")

    # Manually populate tool list
    await conn._refresh_tools()

    tc = ToolCall(id="c1", name="my_tool", arguments='{"x": "hello"}')
    result = await conn.call_tool(tc)
    assert result.tool_call_id == "c1"
    assert result.name == "my_tool"
    assert result.content == "result text"


@pytest.mark.asyncio
async def test_mcp_connection_invalid_json_arguments():
    cfg = MCPServerConfig(
        name="test-server", transport="stdio",
        command="echo", args=[]
    )
    conn = MCPConnection(cfg)
    conn.session = _mock_session(["fn"])
    await conn._refresh_tools()

    tc = ToolCall(id="c2", name="fn", arguments="NOT JSON{{")
    result = await conn.call_tool(tc)
    assert "[error]" in result.content.lower()


@pytest.mark.asyncio
async def test_mcp_connection_tool_error_propagated():
    cfg = MCPServerConfig(
        name="test-server", transport="stdio",
        command="echo", args=[]
    )
    conn = MCPConnection(cfg)
    session = _mock_session(["boom"])
    session.call_tool = AsyncMock(side_effect=RuntimeError("Server crashed"))
    conn.session = session
    await conn._refresh_tools()

    tc = ToolCall(id="c3", name="boom", arguments="{}")
    result = await conn.call_tool(tc)
    assert "[error]" in result.content.lower()


@pytest.mark.asyncio
async def test_mcp_connection_isError_flag():
    cfg = MCPServerConfig(
        name="test-server", transport="stdio",
        command="echo", args=[]
    )
    conn = MCPConnection(cfg)
    session = _mock_session(["risky"])
    call_resp = MagicMock()
    call_resp.content = [_make_content_block("something bad")]
    call_resp.isError = True
    session.call_tool = AsyncMock(return_value=call_resp)
    conn.session = session
    await conn._refresh_tools()

    tc = ToolCall(id="c4", name="risky", arguments="{}")
    result = await conn.call_tool(tc)
    assert "[error]" in result.content.lower()


# ---------------------------------------------------------------------------
# MCPManager unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_manager_list_tools():
    manager = MCPManager([])

    # Inject two pre-connected mock connections
    conn_a = MagicMock()
    conn_a.list_tools.return_value = [
        ToolDefinition.from_parts("tool_a", "desc a", {"type": "object"}),
    ]
    conn_b = MagicMock()
    conn_b.list_tools.return_value = [
        ToolDefinition.from_parts("tool_b", "desc b", {"type": "object"}),
    ]
    manager._connections = {"srv-a": conn_a, "srv-b": conn_b}

    tools = manager.list_tools()
    names = [t.name for t in tools]
    assert "tool_a" in names
    assert "tool_b" in names


@pytest.mark.asyncio
async def test_mcp_manager_routes_call():
    manager = MCPManager([])

    td = ToolDefinition.from_parts("weather", "Get weather", {"type": "object"})
    conn = MagicMock()
    conn.list_tools.return_value = [td]
    expected_result = ToolResult(tool_call_id="c1", name="weather", content="Sunny")
    conn.call_tool = AsyncMock(return_value=expected_result)
    manager._connections = {"weather-srv": conn}

    tc = ToolCall(id="c1", name="weather", arguments='{"city": "LA"}')
    result = await manager.call_tool(tc)
    assert result.content == "Sunny"
    conn.call_tool.assert_called_once_with(tc)


@pytest.mark.asyncio
async def test_mcp_manager_unknown_tool():
    manager = MCPManager([])
    manager._connections = {}

    tc = ToolCall(id="c1", name="nonexistent", arguments="{}")
    result = await manager.call_tool(tc)
    assert "[error]" in result.content.lower()
    assert "nonexistent" in result.content


@pytest.mark.asyncio
async def test_mcp_manager_has_tool():
    manager = MCPManager([])
    conn = MagicMock()
    conn.list_tools.return_value = [
        ToolDefinition.from_parts("exists", "desc", {"type": "object"}),
    ]
    manager._connections = {"srv": conn}

    assert manager.has_tool("exists") is True
    assert manager.has_tool("missing") is False


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_registry_get_definitions():
    manager = MagicMock()
    manager.list_tools.return_value = [
        ToolDefinition.from_parts("fn1", "desc1", {"type": "object"}),
        ToolDefinition.from_parts("fn2", "desc2", {"type": "object"}),
    ]
    reg = ToolRegistry(manager)
    defs = reg.get_tool_definitions()
    assert len(defs) == 2
    names = [d["function"]["name"] for d in defs]
    assert "fn1" in names and "fn2" in names


@pytest.mark.asyncio
async def test_registry_call_known_tool():
    manager = MagicMock()
    manager.has_tool.return_value = True
    expected = ToolResult(tool_call_id="c1", name="fn1", content="done")
    manager.call_tool = AsyncMock(return_value=expected)
    reg = ToolRegistry(manager)

    tc = ToolCall(id="c1", name="fn1", arguments="{}")
    result = await reg.call(tc)
    assert result.content == "done"


@pytest.mark.asyncio
async def test_registry_call_unknown_tool():
    manager = MagicMock()
    manager.has_tool.return_value = False
    reg = ToolRegistry(manager)

    tc = ToolCall(id="c1", name="ghost", arguments="{}")
    result = await reg.call(tc)
    assert "[error]" in result.content.lower()


@pytest.mark.asyncio
async def test_registry_call_all_concurrent():
    """call_all should dispatch multiple tool calls concurrently."""
    import asyncio

    call_order = []

    async def _slow_call(tc: ToolCall) -> ToolResult:
        call_order.append(tc.name)
        await asyncio.sleep(0.05)
        return ToolResult(tool_call_id=tc.id, name=tc.name, content=tc.name)

    manager = MagicMock()
    manager.has_tool.return_value = True
    manager.call_tool = _slow_call
    reg = ToolRegistry(manager)

    tcs = [ToolCall(id=str(i), name=f"tool{i}", arguments="{}") for i in range(4)]
    results = await reg.call_all(tcs)
    assert len(results) == 4
    result_names = {r.name for r in results}
    assert result_names == {"tool0", "tool1", "tool2", "tool3"}
