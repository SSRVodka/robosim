"""Tests for Memory (SQLite persistence)."""

from __future__ import annotations

import asyncio
import tempfile
import pytest

from src.config import MemoryConfig
from src.memory import Memory


@pytest.fixture
def mem_cfg(tmp_path):
    return MemoryConfig(db_path=str(tmp_path / "test.db"), max_history=10)


@pytest.fixture
async def mem(mem_cfg):
    m = Memory(mem_cfg)
    await m.initialize()
    yield m
    await m.close()


@pytest.mark.asyncio
async def test_load_empty(mem):
    result = await mem.load("sess-1")
    assert result == []


@pytest.mark.asyncio
async def test_save_and_load(mem):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    await mem.save("sess-1", messages)
    loaded = await mem.load("sess-1")
    assert loaded == messages


@pytest.mark.asyncio
async def test_max_history_trimming(mem):
    messages = [{"role": "user", "content": str(i)} for i in range(20)]
    await mem.save("sess-trim", messages)
    loaded = await mem.load("sess-trim")
    assert len(loaded) == 10
    assert loaded[-1]["content"] == "19"


@pytest.mark.asyncio
async def test_append(mem):
    await mem.save("sess-2", [{"role": "user", "content": "a"}])
    await mem.append("sess-2", {"role": "assistant", "content": "b"})
    loaded = await mem.load("sess-2")
    assert len(loaded) == 2
    assert loaded[1]["content"] == "b"


@pytest.mark.asyncio
async def test_clear(mem):
    await mem.save("sess-3", [{"role": "user", "content": "x"}])
    await mem.clear("sess-3")
    assert await mem.load("sess-3") == []


@pytest.mark.asyncio
async def test_list_sessions(mem):
    await mem.save("sess-a", [{"role": "user", "content": "1"}])
    await mem.save("sess-b", [{"role": "user", "content": "2"}])
    sessions = await mem.list_sessions()
    ids = [s["session_id"] for s in sessions]
    assert "sess-a" in ids
    assert "sess-b" in ids


@pytest.mark.asyncio
async def test_concurrent_writes(mem):
    """Multiple coroutines writing to the same session should not corrupt data."""
    async def _write(i: int):
        await mem.append("shared", {"role": "user", "content": str(i)})

    await asyncio.gather(*[_write(i) for i in range(20)])
    loaded = await mem.load("shared")
    # All 20 messages should be present (subject to max_history=10)
    assert len(loaded) == 10


@pytest.mark.asyncio
async def test_independent_sessions(mem):
    await mem.save("a", [{"role": "user", "content": "A"}])
    await mem.save("b", [{"role": "user", "content": "B"}])
    assert (await mem.load("a"))[0]["content"] == "A"
    assert (await mem.load("b"))[0]["content"] == "B"
