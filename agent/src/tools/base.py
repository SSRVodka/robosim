"""
Tool base: shared data models used across the tools subsystem.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class ToolDefinition(BaseModel):
    """OpenAI-format function tool definition."""
    type: str = "function"
    function: dict[str, Any]

    @classmethod
    def from_parts(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> "ToolDefinition":
        return cls(
            function={
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )

    @property
    def name(self) -> str:
        return self.function["name"]


class ToolCall(BaseModel):
    """A single tool call requested by the LLM."""
    id: str
    name: str
    arguments: str   # raw JSON string from the LLM


class ToolResult(BaseModel):
    """Result returned after executing a tool call."""
    tool_call_id: str
    name: str
    content: str     # stringified result

    def to_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
        }
