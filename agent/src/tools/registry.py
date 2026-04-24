"""
Tool registry: unified source-of-truth for all tool definitions and
the routing layer for tool invocations.

Supports two backends:
  - MCP servers  – registered via MCPManager (stdio / SSE / streamable HTTP)
  - Local Python functions – registered via register_local()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from .base import ToolDefinition, ToolCall, ToolResult
from .mcp_client import MCPManager

logger = logging.getLogger(__name__)

LocalExecutor = Callable[..., Coroutine[Any, Any, str]]


class _LocalTool:
    """A registered local Python function and its metadata."""

    def __init__(
        self,
        definition: ToolDefinition,
        executor: LocalExecutor,
    ) -> None:
        self.definition = definition
        self.executor = executor


class ToolRegistry:
    def __init__(self, mcp_manager: MCPManager) -> None:
        self._mcp = mcp_manager
        self._local: dict[str, _LocalTool] = {}

    def get_registry_info(self) -> str:
        return f"MCP tools: {len(self._mcp.list_tools())}, Local tools: {len(self._local)}"

    # ------------------------------------------------------------------
    # Local function registration
    # ------------------------------------------------------------------

    def register_local(
        self,
        definition: ToolDefinition,
        executor: LocalExecutor,
    ) -> None:
        """
        Register a Python function as a function tool.

        *definition* is the OpenAI-format tool description (name, description,
        parameters).  *executor* is an async callable that receives the raw
        ``{argument_name: value}`` dict parsed from the LLM's JSON arguments
        and returns a plain ``str`` result.

        Example::

            async def speak_executor(args: dict) -> str:
                return await speak(args["text"])

            registry.register_local(
                get_tool_definition(),
                speak_executor,
            )
        """
        self._local[definition.name] = _LocalTool(definition, executor)
        logger.info("Local tool registered: '%s'", definition.name)

    def unregister_local(self, name: str) -> bool:
        """Remove a local tool. Returns True if it existed."""
        return self._local.pop(name, None) is not None

    # ------------------------------------------------------------------
    # Definitions (OpenAI format)
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions from MCP servers and local functions."""
        defs = [t.model_dump() for t in self._mcp.list_tools()]
        defs.extend(t.definition.model_dump() for t in self._local.values())
        logger.debug("Returning %d tool definition(s)", len(defs))
        return defs

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def call(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch a tool call to the appropriate backend."""
        args_preview = tool_call.arguments[:120] + ("…" if len(tool_call.arguments) > 120 else "")
        logger.info("Executing tool: %s(%s)", tool_call.name, args_preview)

        # MCP takes priority; fall back to local
        if self._mcp.has_tool(tool_call.name):
            result = await self._mcp.call_tool(tool_call)
        elif tool_call.name in self._local:
            result = await self._call_local(tool_call)
        else:
            logger.warning("Unknown tool requested: '%s'", tool_call.name)
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"[error] Unknown tool '{tool_call.name}'",
            )

        content_preview = result.content[:200] + ("…" if len(result.content) > 200 else "")
        logger.info("Tool '%s' result: %s", tool_call.name, content_preview)
        return result

    async def _call_local(self, tool_call: ToolCall) -> ToolResult:
        """Parse JSON args and invoke a local function tool."""
        import json
        try:
            args: dict[str, Any] = json.loads(tool_call.arguments or "{}")
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON arguments for tool '%s': %s", tool_call.name, exc)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"[error] Invalid JSON arguments: {exc}",
            )

        try:
            content: str = await self._local[tool_call.name].executor(args)
        except Exception as exc:
            logger.exception("Local tool '%s' raised", tool_call.name)
            content = f"[error] {exc}"

        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
        )

    async def call_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls concurrently."""
        results = list(await asyncio.gather(*[self.call(tc) for tc in tool_calls]))
        logger.debug("Executed %d tool call(s) concurrently", len(results))
        return results
