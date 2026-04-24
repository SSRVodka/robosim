"""
Agent: orchestrates the LLM <-> tool call loop.

Two entry points:
  run()        – returns the final assistant message (non-streaming)
  run_stream() – async generator yielding text tokens; tool calls are
                 handled transparently between rounds.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from .config import AgentConfig
from .llm import LLMClient
from .memory import Memory
from .tools.base import ToolCall, ToolResult
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

Message = dict[str, Any]


class AgentError(Exception):
    pass


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMClient,
        memory: Memory,
        registry: ToolRegistry,
    ) -> None:
        self._config = config
        self._llm = llm
        self._memory = memory
        self._registry = registry
        logger.info(
            "Agent created – name='%s' max_iterations=%d",
            config.name,
            config.max_iterations,
        )
        logger.info("Agent registry info: %s", self._registry.get_registry_info())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _system_message(self) -> Message:
        return {"role": "system", "content": self._config.system_prompt}

    def _tools(self) -> list[dict] | None:
        defs = self._registry.get_tool_definitions()
        return defs if defs else None

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[Message]:
        results: list[ToolResult] = await self._registry.call_all(tool_calls)
        return [r.to_message() for r in results]

    def _build_context(
        self, history: list[Message], new_messages: list[Message]
    ) -> list[Message]:
        return [self._system_message()] + history + new_messages

    # ------------------------------------------------------------------
    # Non-streaming run
    # ------------------------------------------------------------------

    async def run(
        self,
        new_messages: list[Message],
        session_id: str,
    ) -> Message:
        """
        Run the agent to completion and return the final assistant message.
        History is loaded before and saved after.
        """
        logger.info("Agent.run started – session=%s", session_id)
        history = await self._memory.load(session_id)
        logger.debug("Loaded %d history message(s) for session '%s'", len(history), session_id)
        context = self._build_context(history, new_messages)
        tools = self._tools()
        iterations = 0

        while iterations < self._config.max_iterations:
            iterations += 1
            logger.debug(
                "Iteration %d/%d – LLM call (context_len=%d, tools=%s)",
                iterations,
                self._config.max_iterations,
                len(context),
                "yes" if tools else "no",
            )
            assistant_msg, tool_calls, finish = await self._llm.complete(
                context, tools=tools
            )
            context.append(assistant_msg)

            if finish == "tool_calls" and tool_calls:
                tool_names = [tc.name for tc in tool_calls]
                logger.info(
                    "Iteration %d: %d tool call(s) -> %s",
                    iterations,
                    len(tool_calls),
                    tool_names,
                )
                tool_messages = await self._execute_tool_calls(tool_calls)
                context.extend(tool_messages)
                continue

            # Final answer – persist and return
            # Save: everything after the system message
            content_preview = (assistant_msg.get("content") or "")[:80].replace("\n", " ")
            logger.info(
                "Agent.run done – session=%s iterations=%d content_preview='%s'",
                session_id,
                iterations,
                content_preview,
            )
            logger.info("Full response: %s", assistant_msg.get("content"))
            await self._memory.save(session_id, context[1:])
            return assistant_msg

        raise AgentError(
            f"Agent exceeded max_iterations={self._config.max_iterations}"
        )

    # ------------------------------------------------------------------
    # Streaming run
    # ------------------------------------------------------------------

    async def run_stream(
        self,
        new_messages: list[Message],
        session_id: str,
    ) -> AsyncIterator[str]:
        """
        Async generator that yields text tokens.

        Tool call rounds are handled internally: when the LLM returns
        tool calls the generator pauses, executes them, then starts a new
        streaming call transparently.
        """
        logger.info("Agent.run_stream started – session=%s", session_id)
        history = await self._memory.load(session_id)
        logger.debug("Loaded %d history message(s) for session '%s'", len(history), session_id)
        context = self._build_context(history, new_messages)
        tools = self._tools()
        iterations = 0

        while iterations < self._config.max_iterations:
            iterations += 1
            logger.debug(
                "Stream iteration %d/%d – context_len=%d, tools=%s",
                iterations,
                self._config.max_iterations,
                len(context),
                "yes" if tools else "no",
            )

            async for chunk, accumulated in self._llm.stream_accumulated(
                context, tools=tools
            ):
                if accumulated is None:
                    # Mid-stream text token
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                else:
                    # Final chunk of this LLM turn
                    assistant_msg = accumulated.to_message()
                    context.append(assistant_msg)

                    if accumulated.finish_reason == "tool_calls" and accumulated.tool_calls:
                        tool_names = [tc.name for tc in accumulated.tool_calls]
                        logger.info(
                            "Stream iteration %d: %d tool call(s) -> %s",
                            iterations,
                            len(accumulated.tool_calls),
                            tool_names,
                        )
                        tool_messages = await self._execute_tool_calls(
                            accumulated.tool_calls
                        )
                        context.extend(tool_messages)
                        # Break inner loop to start a new LLM call
                        break
                    else:
                        # Conversation is done – persist and return
                        content_preview = (accumulated.content or "")[:80].replace("\n", " ")
                        logger.info(
                            "Agent.run_stream done – session=%s iterations=%d content_preview='%s'",
                            session_id,
                            iterations,
                            content_preview,
                        )
                        await self._memory.save(session_id, context[1:])
                        return
            else:
                # Inner for-loop completed without break → done
                await self._memory.save(session_id, context[1:])
                return

        raise AgentError(
            f"Agent exceeded max_iterations={self._config.max_iterations}"
        )
