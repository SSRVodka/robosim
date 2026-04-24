import os
from typing import Optional

import click

from agent.main import cli as agent_cli
from agent.main import register_local_tool
from agent.src.tools.base import ToolDefinition
from control_stubs.tools import TOOL_DEFINITIONS, RobosimClient, get_tool_implementations


@click.group(
    name="agent_orchestrator",
    help="Robosim Agent Orchestrator - runs agent with gRPC tools"
)
@click.option(
    "--config", "-c",
    default="config/default.yaml",
    show_default=True,
    help="Path to YAML configuration file."
)
@click.option(
    "--grpc-host", "-gh",
    default=None,
    help="gRPC server host (overrides GRPC_HOST environment variable)"
)
@click.option(
    "--grpc-port", "-gp",
    default=None,
    type=int,
    help="gRPC server port (overrides GRPC_PORT environment variable)"
)
@click.pass_context
def main_cli(
    ctx: click.Context,
    config: str,
    grpc_host: Optional[str],
    grpc_port: Optional[int],
):
    """Robosim Agent Orchestrator - runs agent.main.cli with tools registered.

    Example:
      python agent_orchestrator.py chat --config my.yaml
    """
    ctx.obj = ctx.obj or {}
    ctx.obj["config_path"] = config

    actual_host = grpc_host or os.environ.get("GRPC_HOST", "localhost")
    actual_port = grpc_port or int(os.environ.get("GRPC_PORT", "50051"))

    client = RobosimClient(host=actual_host, port=actual_port)
    tool_impls = get_tool_implementations(client)
    for definition in TOOL_DEFINITIONS:
        tool_name = definition["name"]
        tool_async_func = tool_impls[tool_name]
        tool_def = ToolDefinition.from_parts(
            name=tool_name,
            description=definition["description"],
            parameters=definition["parameters"],
        )
        register_local_tool(tool_def, tool_async_func)

for _, cmd_obj in agent_cli.commands.items():
    main_cli.add_command(cmd_obj)


if __name__ == "__main__":
    main_cli()
