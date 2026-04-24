"""Expose gRPC interfaces as OpenAI function tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from ._common import TOOL_DEFINITIONS, create_tool_implementations
from .client import RobosimClient


def get_tools() -> list[dict[str, Any]]:
    """Get all function tools with implementations bound to the client."""
    return [{"type": "function", "function": d} for d in TOOL_DEFINITIONS]


def get_tool_implementations(client: RobosimClient) -> dict[str, Callable[..., Any]]:
    """Get tool name -> async implementation mapping."""
    return create_tool_implementations(client)


def call_tool(client: RobosimClient, name: str, arguments: str | dict[str, Any]) -> str:
    """Call a tool by name with arguments (JSON string or dict)."""
    impls = get_tool_implementations(client)
    if name not in impls:
        err_obj = {"error": f"Unknown tool: {name}"}
        return json.dumps(err_obj)
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    try:
        return asyncio.run(impls[name](args))
    except Exception as e:
        err_obj = {"error": str(e)}
        return json.dumps(err_obj)
