"""
CLI interface: interactive REPL and single-shot modes.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from ..agent import Agent
from ..config import AgentConfig
from ..multimodal import build_user_message

logger = logging.getLogger("agent.cli")
console = Console()


async def _run_once(
    agent: Agent,
    session_id: str,
    text: str,
    images: list[str],
    audio: list[str],
    stream: bool,
) -> None:
    logger.info(
        "CLI _run_once – session=%s text_len=%d images=%d audio=%d stream=%s",
        session_id,
        len(text) if text else 0,
        len(images),
        len(audio),
        stream,
    )
    msg = build_user_message(
        text=text or None,
        images=images or None,
        audio=audio or None,
    )
    if stream:
        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
        async for token in agent.run_stream([msg], session_id):
            console.print(token, end="", highlight=False)
        console.print()
    else:
        with console.status("Thinking…"):
            response = await agent.run([msg], session_id)
        content = response.get("content") or ""
        console.print(
            Panel(Markdown(content), title="[bold cyan]Assistant[/bold cyan]")
        )


async def _repl(agent: Agent, cfg: AgentConfig) -> None:
    session_id = cfg.interfaces.cli.session_id
    stream = cfg.interfaces.cli.stream
    loop = asyncio.get_running_loop()
    logger.info(
        "CLI REPL started – agent='%s' session=%s stream=%s",
        cfg.name,
        session_id,
        stream,
    )
    console.print(
        Panel(
            f"[bold green]{cfg.name}[/bold green] – type [bold]/quit[/bold] to exit, "
            "[bold]/clear[/bold] to reset session, [bold]/session <id>[/bold] to switch.",
            title="Agent REPL",
        )
    )

    while True:
        try:
            # Prompt.ask is a synchronous blocking call; run it in a thread
            # so it never blocks the event loop.
            user_input = await loop.run_in_executor(
                None, Prompt.ask, "[bold yellow]You[/bold yellow]"
            )
        except (KeyboardInterrupt, EOFError):
            logger.info("CLI REPL exited (keyboard interrupt)")
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input.strip():
            continue
        if user_input.strip() == "/quit":
            logger.info("CLI REPL exited (user command)")
            console.print("[dim]Goodbye.[/dim]")
            break
        if user_input.strip() == "/clear":
            await agent._memory.clear(session_id)
            logger.info("CLI REPL – session '%s' cleared", session_id)
            console.print("[dim]Session cleared.[/dim]")
            continue
        if user_input.strip().startswith("/session "):
            new_sid = user_input.strip().split(" ", 1)[1].strip()
            session_id = new_sid
            logger.info("CLI REPL – switched to session '%s'", session_id)
            console.print(f"[dim]Switched to session '{session_id}'[/dim]")
            continue

        logger.debug("CLI REPL – user input accepted, session='%s'", session_id)
        try:
            await _run_once(agent, session_id, user_input, [], [], stream)
        except Exception as exc:
            logger.exception("CLI REPL – error during agent run")
            console.print(f"[bold red]Error:[/bold red] {exc}")


async def run_cli(agent: Agent, cfg: AgentConfig) -> None:
    await _repl(agent, cfg)
