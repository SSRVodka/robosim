"""
MCP Client: connects to MCP servers over stdio, SSE, or streamable HTTP.

Each server connection is held open for the application lifetime and
protected by an asyncio.Lock to serialise concurrent tool calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

try:
    from mcp.client.sse import sse_client          # older SDK path
    _HAS_SSE = True
except ImportError:
    _HAS_SSE = False

try:
    from mcp.client.streamable_http import streamablehttp_client
    _HAS_STREAMABLE = True
except ImportError:
    _HAS_STREAMABLE = False

from ..config import MCPServerConfig
from .base import ToolDefinition, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class MCPConnection:
    """A single live connection to one MCP server."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
    ) -> None:
        self.config = config
        self.session: ClientSession | None = None
        self._lock = asyncio.Lock()
        self._stack = AsyncExitStack()
        self._tools: list[ToolDefinition] = []
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        transport = self.config.transport
        try:
            if transport == "stdio":
                await self._connect_stdio()
            elif transport == "sse":
                await self._connect_sse()
            elif transport == "streamable_http":
                await self._connect_streamable_http()
            else:
                raise ValueError(f"Unknown MCP transport: {transport}")

            await self.session.initialize()
            await self._refresh_tools()
            logger.info("MCP server '%s' connected (%s)", self.config.name, transport)
        except Exception as exc:
            logger.error("Failed to connect MCP server '%s': %s", self.config.name, exc)
            raise

    async def _connect_stdio(self) -> None:
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env={**{}, **self.config.env} or None,
        )
        read, write = await self._stack.enter_async_context(
            stdio_client(params)
        )
        self.session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )

    async def _connect_sse(self) -> None:
        if not _HAS_SSE:
            raise RuntimeError(
                "SSE transport not available – install mcp[sse] or upgrade the mcp package"
            )
        read, write = await self._stack.enter_async_context(
            sse_client(self.config.url, headers=self.config.headers)
        )
        self.session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )

    async def _connect_streamable_http(self) -> None:
        if not _HAS_STREAMABLE:
            raise RuntimeError(
                "Streamable HTTP transport not available – upgrade the mcp package to >=1.5"
            )
        read, write, _ = await self._stack.enter_async_context(
            streamablehttp_client(self.config.url, headers=self.config.headers)
        )
        self.session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )

    async def disconnect(self) -> None:
        await self._stack.aclose()
        self.session = None
        self._tools.clear()
        logger.info("MCP server '%s' disconnected", self.config.name)

    # ------------------------------------------------------------------
    # Reconnect
    # ------------------------------------------------------------------

    async def _reconnect(self) -> None:
        """Tear down and re-establish the connection and re-fetch tool list."""
        logger.warning("MCP server '%s' – reconnecting…", self.config.name)
        await self._safe_close()
        await self.connect()

    async def _safe_close(self) -> None:
        """Close the exit stack without raising, resetting session and tools."""
        try:
            await self._stack.aclose()
        except Exception as exc:
            logger.debug("MCP server '%s' _safe_close: %s", self.config.name, exc)
        self.session = None
        self._tools.clear()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def _refresh_tools(self) -> None:
        assert self.session
        response = await self.session.list_tools()
        self._tools = [
            ToolDefinition.from_parts(
                name=t.name,
                description=t.description or "",
                parameters=t.inputSchema or {"type": "object", "properties": {}},
            )
            for t in response.tools
        ]
        logger.debug(
            "MCP server '%s' – refreshed %d tool(s): %s",
            self.config.name,
            len(self._tools),
            [t.name for t in self._tools],
        )

    def list_tools(self) -> list[ToolDefinition]:
        """Return cached tool definitions (prefixed with server name)."""
        return self._tools

    # ------------------------------------------------------------------
    # Tool invocation with auto-reconnect
    # ------------------------------------------------------------------

    async def call_tool(self, tool_call: ToolCall, *, timeout: float | None = 30.0) -> ToolResult:
        """
        Call a tool with automatic reconnection on failure.

        Retries up to max_retries times. On each retry:
          1. Acquires the per-connection lock (serialises concurrent calls)
          2. Calls the tool (or reconnects once if session is dead then retries)
          3. Exponential back-off between retries
        """
        args = self._parse_args(tool_call)
        if isinstance(args, ToolResult):
            return args  # parse error, already packaged

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            async with self._lock:
                session = self.session
                if session is None:
                    if attempt == 0:
                        logger.warning(
                            "MCP '%s' – no active session, attempting reconnect before tool '%s'",
                            self.config.name,
                            tool_call.name,
                        )
                        try:
                            await self._reconnect()
                            session = self.session
                        except Exception as exc:
                            logger.error(
                                "MCP '%s' – reconnect failed: %s",
                                self.config.name,
                                exc,
                            )
                            return self._error_result(
                                tool_call,
                                f"[error] MCP server unreachable: {exc}",
                            )
                    else:
                        return self._error_result(
                            tool_call,
                            "[error] MCP session not available after reconnect",
                        )

                try:
                    return await self._execute(session, tool_call, args, timeout)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "MCP '%s' tool '%s' attempt %d/%d failed: %s – will retry",
                        self.config.name,
                        tool_call.name,
                        attempt + 1,
                        self._max_retries,
                        exc,
                    )
                    # Mark session as dead; reconnect on next attempt
                    await self._safe_close()

                    if attempt < self._max_retries - 1:
                        delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                        logger.debug(
                            "MCP '%s' – backing off %.1fs before retry",
                            self.config.name,
                            delay,
                        )
                        await asyncio.sleep(delay)

        # All retries exhausted
        return self._error_result(
            tool_call,
            f"[error] Tool '{tool_call.name}' failed after {self._max_retries} attempts: {last_exc}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_args(self, tool_call: ToolCall) -> dict | ToolResult:
        try:
            return json.loads(tool_call.arguments or "{}")
        except json.JSONDecodeError as exc:
            logger.warning(
                "MCP '%s' tool '%s' – invalid JSON arguments: %s",
                self.config.name,
                tool_call.name,
                exc,
            )
            return self._error_result(tool_call, f"[error] Invalid JSON arguments: {exc}")

    async def _execute(
        self,
        session: ClientSession,
        tool_call: ToolCall,
        args: dict,
        timeout: float | None,
    ) -> ToolResult:
        coro = session.call_tool(tool_call.name, args)
        response = await asyncio.wait_for(coro, timeout=timeout) if timeout else await coro

        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                parts.append(f"[binary data len={len(block.data)}]")
            else:
                parts.append(str(block))
        result_str = "\n".join(parts) if parts else ""

        if response.isError:
            result_str = f"[error] {result_str}"
            logger.warning(
                "MCP '%s' tool '%s' returned error response",
                self.config.name,
                tool_call.name,
            )
        else:
            logger.debug(
                "MCP '%s' tool '%s' ok – %d char(s)",
                self.config.name,
                tool_call.name,
                len(result_str),
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=result_str,
        )

    def _error_result(self, tool_call: ToolCall, content: str) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MCPManager:
    """
    Manages a collection of MCP server connections.

    Usage::

        manager = MCPManager(configs)
        await manager.start()
        ...
        await manager.stop()
    """

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = configs
        # name -> connection
        self._connections: dict[str, MCPConnection] = {}

    async def start(self) -> None:
        logger.info("MCPManager starting – %d server(s) to connect", len(self._configs))
        tasks = [self._start_one(cfg) for cfg in self._configs]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            "MCPManager started – %d connection(s) active",
            len(self._connections),
        )

    async def _start_one(self, cfg: MCPServerConfig) -> None:
        conn = MCPConnection(cfg)
        try:
            await conn.connect()
            self._connections[cfg.name] = conn
            logger.debug("MCPManager registered server '%s'", cfg.name)
        except Exception:
            logger.warning("Skipping MCP server '%s' (connection failed)", cfg.name)

    async def stop(self) -> None:
        logger.info("MCPManager stopping – disconnecting %d server(s)", len(self._connections))
        tasks = [c.disconnect() for c in self._connections.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._connections.clear()

    def list_tools(self) -> list[ToolDefinition]:
        """Return all tool definitions from all connected servers."""
        tools: list[ToolDefinition] = []
        for conn in self._connections.values():
            tools.extend(conn.list_tools())
        return tools

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        """Route a tool call to the owning MCP server."""
        # Find which server owns this tool by name
        for conn in self._connections.values():
            if any(t.name == tool_call.name for t in conn.list_tools()):
                return await conn.call_tool(tool_call)
        logger.warning("MCPManager – no server provides tool '%s'", tool_call.name)
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=f"[error] No MCP server provides tool '{tool_call.name}'",
        )

    def has_tool(self, name: str) -> bool:
        return any(
            t.name == name
            for conn in self._connections.values()
            for t in conn.list_tools()
        )
