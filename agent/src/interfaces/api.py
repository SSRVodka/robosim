"""
Web API: OpenAI-compatible /v1/chat/completions endpoint.

Session IDs are taken from the `user` field in the request body.
Multimodal content is forwarded as-is (the client is responsible for
encoding images/audio into OpenAI content-block format).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..agent import Agent
from ..config import AgentConfig
from ..memory import Memory
from ..tools.mcp_client import MCPManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: dict | None = None
    input_audio: dict | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None          # used as session_id
    tools: list[dict] | None = None  # ignored; agent manages its own tools
    tool_choice: Any = None


def _msg_to_dict(m: ChatMessage) -> dict:
    """Convert request message to plain dict for the agent."""
    d: dict[str, Any] = {"role": m.role}
    if isinstance(m.content, list):
        d["content"] = [p.model_dump(exclude_none=True) for p in m.content]
    else:
        d["content"] = m.content
    if m.tool_calls:
        d["tool_calls"] = m.tool_calls
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    if m.name:
        d["name"] = m.name
    return d


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(data: str) -> str:
    return f"data: {data}\n\n"


def _chunk_json(
    chunk_id: str,
    model: str,
    delta: dict,
    finish_reason: str | None = None,
) -> str:
    obj = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return json.dumps(obj)


async def _stream_response(
    agent: Agent,
    messages: list[dict],
    session_id: str,
    model: str,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    logger.info("API stream started – session=%s model=%s", session_id, model)
    # role header
    yield _sse(_chunk_json(chunk_id, model, {"role": "assistant", "content": ""}))

    try:
        async for token in agent.run_stream(messages, session_id):
            yield _sse(_chunk_json(chunk_id, model, {"content": token}))
    except Exception as exc:
        logger.exception("Agent error during streaming")
        yield _sse(
            _chunk_json(chunk_id, model, {"content": f"\n\n[error: {exc}]"}, "stop")
        )
        yield _sse("[DONE]")
        return

    yield _sse(_chunk_json(chunk_id, model, {}, "stop"))
    yield _sse("[DONE]")
    logger.info("API stream complete – session=%s", session_id)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    agent: Agent,
    cfg: AgentConfig,
    mcp: MCPManager | None = None,
    mem: Memory | None = None,
    stt_proc: subprocess.Popen | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if stt_proc is not None and stt_proc.poll() is None:
            logger.info("Stopping stt_server (pid=%d)", stt_proc.pid)
            stt_proc.terminate()
            try:
                stt_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stt_proc.kill()
                stt_proc.wait()
        if mcp is not None:
            await mcp.stop()
        if mem is not None:
            await mem.close()
        logger.info("Shutdown complete")

    app = FastAPI(title=cfg.name, version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_key = cfg.interfaces.api.api_key

    def _auth(authorization: str | None = Header(default=None)) -> None:
        if not api_key:
            return
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if token != api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    # ------------------------------------------------------------------
    # Models endpoint
    # ------------------------------------------------------------------

    @app.get("/v1/models", dependencies=[Depends(_auth)])
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": cfg.llm.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "agent-framework",
                }
            ],
        }

    # ------------------------------------------------------------------
    # Chat completions endpoint
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions", dependencies=[Depends(_auth)])
    async def chat_completions(req: ChatCompletionRequest):
        session_id = req.user or f"api-{uuid.uuid4().hex[:8]}"
        messages = [_msg_to_dict(m) for m in req.messages]
        logger.info(
            "API chat completions – session=%s stream=%s msg_count=%d",
            session_id,
            req.stream,
            len(messages),
        )

        if req.stream:
            return StreamingResponse(
                _stream_response(agent, messages, session_id, req.model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming
        try:
            response = await agent.run(messages, session_id)
        except Exception as exc:
            logger.exception("Agent error (non-streaming)")
            raise HTTPException(status_code=500, detail=str(exc))

        content = response.get("content") or ""
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        logger.info(
            "API chat completions done – session=%s completion_id=%s content_len=%d",
            session_id,
            completion_id,
            len(content),
        )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ------------------------------------------------------------------
    # Session management endpoints
    # ------------------------------------------------------------------

    @app.get("/v1/sessions", dependencies=[Depends(_auth)])
    async def list_sessions():
        sessions = await agent._memory.list_sessions()
        logger.info("API list_sessions – %d session(s)", len(sessions))
        return {"sessions": sessions}

    @app.delete("/v1/sessions/{session_id}", dependencies=[Depends(_auth)])
    async def clear_session(session_id: str):
        await agent._memory.clear(session_id)
        logger.info("API cleared session '%s'", session_id)
        return {"cleared": session_id}

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
