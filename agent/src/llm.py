"""
LLM client: thin async wrapper around openai.AsyncOpenAI.

Supports:
  - complete()  – single blocking call (returns full message)
  - stream()    – async generator yielding raw stream chunks

Both support tool definitions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import openai
from openai.types.chat import ChatCompletionChunk

from .config import LLMConfig
from .tools.base import ToolCall

logger = logging.getLogger(__name__)

Message = dict[str, Any]


class _AccumulatedResponse:
    """Accumulates a streaming response into a final message."""

    def __init__(self) -> None:
        self.content = ""
        self.tool_calls: list[ToolCall] = []
        self.finish_reason: str | None = None
        # Internal: keyed by index for accumulating partial tool calls
        self._tc_buffers: dict[int, dict] = {}

    def feed(self, chunk: ChatCompletionChunk) -> None:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            return
        if choice.finish_reason:
            self.finish_reason = choice.finish_reason

        delta = choice.delta
        if delta.content:
            self.content += delta.content

        for tc_delta in delta.tool_calls or []:
            idx = tc_delta.index
            if idx not in self._tc_buffers:
                self._tc_buffers[idx] = {"id": "", "name": "", "arguments": ""}
            buf = self._tc_buffers[idx]
            if tc_delta.id:
                buf["id"] += tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    buf["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    buf["arguments"] += tc_delta.function.arguments

    def finalise(self) -> None:
        self.tool_calls = [
            ToolCall(id=v["id"], name=v["name"], arguments=v["arguments"])
            for v in self._tc_buffers.values()
        ]

    def to_message(self) -> Message:
        """Build an assistant message suitable for appending to history."""
        msg: Message = {"role": "assistant"}
        if self.content:
            msg["content"] = self.content
        else:
            msg["content"] = None
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return msg


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = openai.AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "sk-placeholder",
            timeout=config.timeout,
        )
        logger.info(
            "LLMClient initialised – model=%s base_url=%s timeout=%.0fs",
            config.model,
            config.base_url,
            config.timeout,
        )

    def _base_params(self, tools: list[dict] | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        return params

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> tuple[Message, list[ToolCall], str | None]:
        """
        Returns (assistant_message, tool_calls, finish_reason).
        """
        params = self._base_params(tools)
        logger.info(
            "LLM complete – %d message(s) in context, tools=%s",
            len(messages),
            "yes" if tools else "no",
        )
        response = await self._client.chat.completions.create(
            messages=messages, stream=False, **params
        )
        choice = response.choices[0]
        msg_obj = choice.message
        finish = choice.finish_reason

        tool_calls: list[ToolCall] = []
        if msg_obj.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in msg_obj.tool_calls
            ]

        assistant_msg: Message = {"role": "assistant", "content": msg_obj.content}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in tool_calls
            ]

        logger.info(
            "LLM complete done – finish=%s content_len=%d tool_calls=%d",
            finish,
            len(msg_obj.content or ""),
            len(tool_calls),
        )
        return assistant_msg, tool_calls, finish

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Yield raw ChatCompletionChunk objects from the SSE stream."""
        params = self._base_params(tools)
        logger.debug("LLM stream start – %d message(s), tools=%s", len(messages), "yes" if tools else "no")
        token_count = 0
        async with await self._client.chat.completions.create(
            messages=messages, stream=True, **params
        ) as stream_ctx:
            async for chunk in stream_ctx:
                if chunk.choices and chunk.choices[0].delta.content:
                    token_count += len(chunk.choices[0].delta.content)
                yield chunk
        logger.info(
            "LLM stream complete – %d token(s) accumulated",
            token_count,
        )

    async def stream_accumulated(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[tuple[ChatCompletionChunk, "_AccumulatedResponse | None"]]:
        """
        Yields (chunk, accumulated_on_finish).
        accumulated_on_finish is non-None only for the final chunk.
        """
        acc = _AccumulatedResponse()
        async for chunk in self.stream(messages, tools):
            acc.feed(chunk)
            if chunk.choices and chunk.choices[0].finish_reason:
                acc.finalise()
                logger.info(
                    "LLM stream round done – finish=%s content_len=%d tool_calls=%d",
                    acc.finish_reason,
                    len(acc.content),
                    len(acc.tool_calls),
                )
                yield chunk, acc
            else:
                yield chunk, None
