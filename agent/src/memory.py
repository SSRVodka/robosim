"""
Memory: persistent conversation history backed by SQLite.

Each session is identified by a string session_id.
Messages are stored as a JSON array and trimmed to max_history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from .config import MemoryConfig

logger = logging.getLogger(__name__)

# OpenAI-format message type alias
Message = dict[str, Any]


class Memory:
    """Async, concurrency-safe SQLite message store."""

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._db_path = Path(config.db_path)
        self._db: aiosqlite.Connection | None = None
        # Per-session locks prevent race conditions on concurrent writes
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                messages   TEXT NOT NULL DEFAULT '[]',
                updated_at REAL NOT NULL
            )
            """
        )
        await self._db.commit()
        logger.info("Memory initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        async with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = asyncio.Lock()
            return self._locks[session_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(self, session_id: str) -> list[Message]:
        """Return all stored messages for the session (oldest first)."""
        assert self._db, "Memory not initialized"
        async with await self._get_lock(session_id):
            async with self._db.execute(
                "SELECT messages FROM sessions WHERE session_id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                logger.debug("Session '%s' not found, returning empty history", session_id)
                return []
            messages = json.loads(row["messages"])
            logger.debug(
                "Loaded %d message(s) for session '%s'",
                len(messages),
                session_id,
            )
            return messages

    async def save(self, session_id: str, messages: list[Message]) -> None:
        """Persist messages (trimmed to max_history) for the session."""
        assert self._db, "Memory not initialized"
        trimmed = messages[-self._config.max_history :]
        payload = json.dumps(trimmed, ensure_ascii=False)
        async with await self._get_lock(session_id):
            await self._db.execute(
                """
                INSERT INTO sessions (session_id, messages, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    messages   = excluded.messages,
                    updated_at = excluded.updated_at
                """,
                (session_id, payload, time.time()),
            )
            await self._db.commit()
        logger.debug(
            "Saved %d message(s) (trimmed to %d) for session '%s'",
            len(trimmed),
            self._config.max_history,
            session_id,
        )

    async def append(self, session_id: str, message: Message) -> None:
        """Atomically append a single message."""
        async with await self._get_lock(session_id):
            assert self._db, "Memory not initialized"
            async with self._db.execute(
                "SELECT messages FROM sessions WHERE session_id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
            msgs: list[Message] = json.loads(row["messages"]) if row else []
            msgs.append(message)
            trimmed = msgs[-self._config.max_history :]
            payload = json.dumps(trimmed, ensure_ascii=False)
            await self._db.execute(
                """
                INSERT INTO sessions (session_id, messages, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    messages   = excluded.messages,
                    updated_at = excluded.updated_at
                """,
                (session_id, payload, time.time()),
            )
            await self._db.commit()

    async def clear(self, session_id: str) -> None:
        """Delete all history for a session."""
        assert self._db, "Memory not initialized"
        async with await self._get_lock(session_id):
            await self._db.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            await self._db.commit()
        logger.info("Cleared session '%s'", session_id)

    async def list_sessions(self) -> list[dict]:
        """Return metadata for all sessions."""
        assert self._db, "Memory not initialized"
        async with self._db.execute(
            "SELECT session_id, updated_at FROM sessions ORDER BY updated_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]
