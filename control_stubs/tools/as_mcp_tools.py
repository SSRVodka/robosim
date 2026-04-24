"""Expose gRPC interfaces as MCP (Model Context Protocol) tools.

Supports both stdio and streamable_http transports.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from mcp import Tool
from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp.types import CallToolResult, TextContent

from ._common import MCP_TOOLS, create_tool_implementations
from .client import RobosimClient


def create_mcp_server(client: RobosimClient) -> tuple[Server, list[Tool]]:
    """Create an MCP server with all tools registered."""
    server = Server("robosim")
    impls = create_tool_implementations(client)

    tools = [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
        )
        for t in MCP_TOOLS
    ]

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        if name not in impls:
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))],
                isError=True,
            )
        try:
            result = await impls[name](arguments or {})
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result))])
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"error": str(e)}))],
                isError=True,
            )

    return server, tools


async def run_stdio(client: RobosimClient) -> None:
    """Run the MCP server using stdio transport."""
    from mcp.server.stdio import stdio_server

    server, _ = create_mcp_server(client)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def run_streamable_http(client: RobosimClient, host: str = "127.0.0.1", port: int = 3000) -> None:
    """Run the MCP server using streamable HTTP transport."""
    import uvicorn

    server, _ = create_mcp_server(client)

    class MCPHTTPApp:
        def __init__(self, mcp_server: Server) -> None:
            self.mcp_server = mcp_server
            self.transport = StreamableHTTPServerTransport(mcp_session_id=None, is_json_response_enabled=True)

        async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
            await self.transport.handle_request(scope, receive, send)

    app = MCPHTTPApp(server)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_uvicorn = uvicorn.Server(config)
    await server_uvicorn.serve()


def main() -> None:
    """Entry point for MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Robosim MCP Server")
    parser.add_argument("--host", default="localhost", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable_http"],
        default="stdio",
        help="MCP transport type (default: stdio)",
    )
    parser.add_argument("--http-host", default="127.0.0.1", help="HTTP server host (for streamable_http)")
    parser.add_argument("--http-port", type=int, default=3000, help="HTTP server port (for streamable_http)")
    args = parser.parse_args()

    client = RobosimClient(host=args.host, port=args.port)
    try:
        if args.transport == "stdio":
            asyncio.run(run_stdio(client))
        else:
            asyncio.run(run_streamable_http(client, host=args.http_host, port=args.http_port))
    finally:
        client.close()
