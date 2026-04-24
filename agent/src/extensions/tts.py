"""
TTS tool: converts text to speech via Alibaba qwen3-tts-flash and streams
the resulting PCM audio directly to a phone over TCP.

Provides two integration points for the agent:
  1. A standalone async function  – speak(text)
  2. A tool-definition dict + executor for use with ToolRegistry

Configuration is read from a shared mutable dict so that values set at
runtime (e.g. from a YAML config file) are visible to speak() without
requiring re-imports.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import logging
import os
import socket
from typing import Any, Callable

import requests

from ..tools.base import ToolDefinition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults – all overridden by applying TTSConfig values to this dict
# ---------------------------------------------------------------------------

_tts_cfg: dict[str, Any] = {
    "api_key": "",
    "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    "model": "qwen3-tts-flash",
    "voice": "Cherry",
    "language": "Chinese",
    "phone_ip": "192.168.1.125",
    "phone_port": 9999,
}


def apply_config(cfg: dict[str, Any]) -> None:
    """Merge values from a TTSConfig dict into the runtime config."""
    _tts_cfg.update({k: v for k, v in cfg.items() if v is not None})


# ---------------------------------------------------------------------------
# Speak callbacks
# ---------------------------------------------------------------------------

_ON_SPEAK_START_CB: list[Callable] = []
_ON_SPEAK_END_CB: list[Callable] = []


def register_on_speak_start(on_speak_start_callback: Callable):
    _ON_SPEAK_START_CB.append(on_speak_start_callback)


def regsiter_on_speak_end(on_speak_end_callback: Callable):
    _ON_SPEAK_END_CB.append(on_speak_end_callback)


# ---------------------------------------------------------------------------
# Core TTS logic
# ---------------------------------------------------------------------------

async def speak(text: str) -> str:
    """
    Synthesise *text* using Alibaba qwen3-tts-flash and stream PCM audio
    to the phone at *phone_ip:phone_port*.

    Returns a human-readable status string on success, or an error message
    on failure.
    """
    key = _tts_cfg.get("api_key") or ""
    if not key:
        return "[error] TTS API key not configured – set tts.api_key in config"

    ip = _tts_cfg.get("phone_ip", "192.168.1.125")
    port: int = _tts_cfg.get("phone_port", 9999)

    logger.info("tts.speak – text=%r -> %s:%d", text[:60], ip, port)

    for start_cb in _ON_SPEAK_START_CB:
        start_cb()

    # --- TCP socket to phone ------------------------------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((ip, port))
    except ConnectionRefusedError:
        logger.error("Connection refused by %s:%d", ip, port)
        return f"[error] Connection refused – phone not reachable at {ip}:{port}"
    except Exception as e:
        logger.error("Socket error connecting to %s:%d: %s", ip, port, e)
        return f"[error] Network error: {e}"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "enable",
    }
    payload = {
        "model": _tts_cfg.get("model", "qwen3-tts-flash"),
        "input": {
            "text": text,
            "voice": _tts_cfg.get("voice", "Cherry"),
            "language_type": _tts_cfg.get("language", "Chinese"),
        },
        "parameters": {"format": "pcm"},
    }

    total_bytes = 0
    # first frame audio playback start time
    playback_start_time: float | None = None

    try:
        response = requests.post(
            _tts_cfg.get("url") or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers=headers, json=payload, stream=True, verify=False
        )

        if response.status_code != 200:
            msg = f"[error] TTS request failed: {response.status_code} – {response.text}"
            logger.error(msg)
            return msg

        frame_count = 0
        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode("utf-8").strip()
            if not decoded_line.startswith("data:"):
                continue

            json_str = decoded_line[5:].strip()
            if not json_str:
                continue

            try:
                frame_data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            output_node = frame_data.get("output", {})
            audio_b64: str | None = None

            audio_node = output_node.get("audio")
            if isinstance(audio_node, dict):
                audio_b64 = audio_node.get("data")
            elif isinstance(audio_node, str):
                audio_b64 = audio_node

            if not audio_b64:
                continue

            # Decode base64 → raw PCM bytes → send to phone
            pcm_bytes = base64.b64decode(audio_b64)

            if playback_start_time is None:
                playback_start_time = time.monotonic()

            sock.sendall(pcm_bytes)
            total_bytes += len(pcm_bytes)
            frame_count += 1

        logger.info("tts.speak done – %d audio frame(s) sent", frame_count)

    except requests.exceptions.RequestException as e:
        msg = f"[error] TTS request exception: {e}"
        logger.error(msg)
        return msg
    except Exception as e:
        msg = f"[error] TTS runtime exception: {e}"
        logger.error(msg)
        return msg
    finally:
        sock.close()

    # Block until complete playing audio
    if playback_start_time is not None and total_bytes > 0:
        _BYTES_PER_SECOND = 24000 * 2 * 1  # 48000
        total_duration = total_bytes / _BYTES_PER_SECOND
        elapsed = time.monotonic() - playback_start_time
        remaining = total_duration - elapsed
        if remaining > 0:
            logger.info("tts.speak – waiting %.2fs for playback to finish", remaining)
            await asyncio.sleep(remaining)
    
    for end_cb in _ON_SPEAK_END_CB:
        end_cb()

    return f"[ok] Sent {frame_count} audio frame(s) to {ip}:{port}"

    # logger.info("tts.speak – text=%r", text)
    # return "[ok] TTS request sent"


# ---------------------------------------------------------------------------
# Tool-definition interface (for use with ToolRegistry / MCP fallback)
# ---------------------------------------------------------------------------

TOOL_NAME = "speak"
TOOL_DESCRIPTION = (
    "Convert the provided text (any language) to speech and play it on the phone. "
    "Use this tool whenever you want to talk to a user or say something out loud. "
    "Input: the text you want the robot to speak."
)

TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "The Chinese text to be spoken by the robot.",
        },
    },
    "required": ["text"],
}


def get_tool_definition() -> ToolDefinition:
    """Return the OpenAI-format tool definition for the speak tool."""
    return ToolDefinition.from_parts(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        parameters=TOOL_PARAMETERS,
    )


async def call_tool(args: dict[str, Any]) -> str:
    """
    Execute the speak tool: synthesise *text* and stream it to the phone.
    Receives the raw JSON arguments dict from the registry and returns a plain str.
    """
    text = args.get("text")
    if not text:
        return "[error] Missing required argument: 'text'"
    return await speak(text)