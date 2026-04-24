#!/usr/bin/env python3
"""
Standalone test runner – no pytest required, no network required.
Stubs all external dependencies before importing framework modules.

Run:
    python3 tests/run_tests.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# ── Bootstrap: install stubs BEFORE any src import ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
import tests._stubs  # noqa: F401  (side-effect: registers stubs)

# ── Now safe to import framework modules ──────────────────────────────────────
from src.config import (
    AgentConfig, LLMConfig, MemoryConfig,
    MCPServerConfig, load_config, _resolve_env,
)
from src.memory import Memory
from src.multimodal import (
    build_user_message, text_item, image_url_item, image_file_item,
)
from src.tools.base import ToolDefinition, ToolCall, ToolResult
from src.tools.mcp_client import MCPConnection, MCPManager
from src.tools.registry import ToolRegistry
from src.agent import Agent, AgentError
from src.llm import LLMClient


# ── helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


def _tmp_db():
    d = tempfile.mkdtemp()
    return os.path.join(d, "test.db")


async def _make_memory(max_history: int = 100) -> Memory:
    m = Memory(MemoryConfig(db_path=_tmp_db(), max_history=max_history))
    await m.initialize()
    return m


def _tool(name: str):
    return ToolDefinition.from_parts(name, "desc", {"type": "object"})


# ── Config tests ──────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = load_config("/nonexistent/path.yaml")
        self.assertIsInstance(cfg, AgentConfig)
        self.assertEqual(cfg.llm.model, "gpt-4o")

    def test_env_interpolation(self):
        os.environ["_TEST_KEY"] = "hello"
        result = _resolve_env("${_TEST_KEY}")
        self.assertEqual(result, "hello")
        del os.environ["_TEST_KEY"]

    def test_env_default_value(self):
        os.environ.pop("_MISSING_VAR_XYZ", None)
        result = _resolve_env("${_MISSING_VAR_XYZ:-default_val}")
        self.assertEqual(result, "default_val")

    def test_env_interpolation_in_dict(self):
        os.environ["_URL"] = "http://example.com"
        result = _resolve_env({"url": "${_URL}", "other": "static"})
        self.assertEqual(result["url"], "http://example.com")
        self.assertEqual(result["other"], "static")
        del os.environ["_URL"]

    def test_mcp_stdio_requires_command(self):
        with self.assertRaises(Exception):
            MCPServerConfig(name="bad", transport="stdio")

    def test_mcp_sse_requires_url(self):
        with self.assertRaises(Exception):
            MCPServerConfig(name="bad", transport="sse")

    def test_mcp_valid_stdio(self):
        cfg = MCPServerConfig(name="ok", transport="stdio", command="echo")
        self.assertEqual(cfg.name, "ok")

    def test_mcp_valid_sse(self):
        cfg = MCPServerConfig(name="ok", transport="sse", url="http://localhost/sse")
        self.assertEqual(cfg.url, "http://localhost/sse")

    def test_agent_config_defaults(self):
        cfg = AgentConfig()
        self.assertEqual(cfg.max_iterations, 20)

    def test_load_config_from_yaml(self):
        import tempfile, yaml  # yaml is installed
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump({"name": "YamlBot", "system_prompt": "You are YamlBot.", "llm": {"model": "gpt-4o-mini"}}, f)
            fname = f.name
        cfg = load_config(fname)
        self.assertEqual(cfg.name, "YamlBot")
        self.assertEqual(cfg.llm.model, "gpt-4o-mini")
        os.unlink(fname)


# ── Memory tests ──────────────────────────────────────────────────────────────

class TestMemory(unittest.TestCase):

    def test_load_empty(self):
        mem = run(_make_memory())
        result = run(mem.load("new-session"))
        self.assertEqual(result, [])
        run(mem.close())

    def test_save_and_load(self):
        mem = run(_make_memory())
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        run(mem.save("s1", msgs))
        loaded = run(mem.load("s1"))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["content"], "hi")
        run(mem.close())

    def test_max_history_trimming(self):
        mem = run(_make_memory(max_history=5))
        msgs = [{"role": "user", "content": str(i)} for i in range(10)]
        run(mem.save("s-trim", msgs))
        loaded = run(mem.load("s-trim"))
        self.assertEqual(len(loaded), 5)
        self.assertEqual(loaded[-1]["content"], "9")
        run(mem.close())

    def test_append_single_message(self):
        mem = run(_make_memory())
        run(mem.save("s2", [{"role": "user", "content": "a"}]))
        run(mem.append("s2", {"role": "assistant", "content": "b"}))
        loaded = run(mem.load("s2"))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[1]["content"], "b")
        run(mem.close())

    def test_clear_session(self):
        mem = run(_make_memory())
        run(mem.save("s3", [{"role": "user", "content": "x"}]))
        run(mem.clear("s3"))
        self.assertEqual(run(mem.load("s3")), [])
        run(mem.close())

    def test_list_sessions(self):
        mem = run(_make_memory())
        run(mem.save("sess-a", [{"role": "user", "content": "1"}]))
        run(mem.save("sess-b", [{"role": "user", "content": "2"}]))
        sessions = run(mem.list_sessions())
        ids = [s["session_id"] for s in sessions]
        self.assertIn("sess-a", ids)
        self.assertIn("sess-b", ids)
        run(mem.close())

    def test_independent_sessions_not_mixed(self):
        mem = run(_make_memory())
        run(mem.save("a", [{"role": "user", "content": "A-content"}]))
        run(mem.save("b", [{"role": "user", "content": "B-content"}]))
        self.assertEqual(run(mem.load("a"))[0]["content"], "A-content")
        self.assertEqual(run(mem.load("b"))[0]["content"], "B-content")
        run(mem.close())

    def test_overwrite_session(self):
        mem = run(_make_memory())
        run(mem.save("s4", [{"role": "user", "content": "first"}]))
        run(mem.save("s4", [{"role": "user", "content": "replaced"}]))
        loaded = run(mem.load("s4"))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["content"], "replaced")
        run(mem.close())

    def test_concurrent_appends_no_corruption(self):
        """Concurrent appends to the same session must all land."""
        mem = run(_make_memory(max_history=50))

        async def _write_all():
            await asyncio.gather(*[
                mem.append("concurrent", {"role": "user", "content": str(i)})
                for i in range(20)
            ])

        run(_write_all())
        loaded = run(mem.load("concurrent"))
        self.assertEqual(len(loaded), 20)
        run(mem.close())


# ── Multimodal tests ──────────────────────────────────────────────────────────

class TestMultimodal(unittest.TestCase):

    def test_text_item(self):
        item = text_item("hello")
        self.assertEqual(item, {"type": "text", "text": "hello"})

    def test_text_only_optimised_to_string(self):
        msg = build_user_message(text="Hello")
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"], "Hello")

    def test_image_url_message(self):
        msg = build_user_message(
            text="Look",
            images=["https://example.com/img.jpg"]
        )
        self.assertIsInstance(msg["content"], list)
        types = [p["type"] for p in msg["content"]]
        self.assertIn("text", types)
        self.assertIn("image_url", types)

    def test_image_url_item(self):
        item = image_url_item("https://example.com/x.png")
        self.assertEqual(item["type"], "image_url")
        self.assertEqual(item["image_url"]["url"], "https://example.com/x.png")

    def test_local_image_file_encoded(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 16)
            fname = f.name
        item = image_file_item(fname)
        self.assertEqual(item["type"], "image_url")
        self.assertIn("data:image/jpeg;base64,", item["image_url"]["url"])
        os.unlink(fname)

    def test_data_url_passed_through(self):
        data_url = "data:image/png;base64,abc123"
        msg = build_user_message(images=[data_url])
        self.assertIsInstance(msg["content"], list)
        url_item = next(p for p in msg["content"] if p["type"] == "image_url")
        self.assertEqual(url_item["image_url"]["url"], data_url)

    def test_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_user_message()

    def test_mixed_text_and_image(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
            fname = f.name
        msg = build_user_message(text="Describe this", images=[fname])
        self.assertEqual(len(msg["content"]), 2)
        os.unlink(fname)

    def test_multiple_images(self):
        urls = ["https://a.com/1.jpg", "https://b.com/2.jpg"]
        msg = build_user_message(images=urls)
        image_items = [p for p in msg["content"] if p["type"] == "image_url"]
        self.assertEqual(len(image_items), 2)


# ── Tool base tests ───────────────────────────────────────────────────────────

class TestToolBase(unittest.TestCase):

    def test_tool_definition_name_property(self):
        td = ToolDefinition.from_parts("fn", "d", {"type": "object"})
        self.assertEqual(td.name, "fn")

    def test_tool_definition_serialise(self):
        td = ToolDefinition.from_parts("get_weather", "Get weather",
                                       {"type": "object", "properties": {}})
        d = td.model_dump()
        self.assertEqual(d["type"], "function")
        self.assertEqual(d["function"]["name"], "get_weather")

    def test_tool_result_to_message(self):
        r = ToolResult(tool_call_id="c1", name="fn", content="result")
        msg = r.to_message()
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "c1")
        self.assertEqual(msg["content"], "result")

    def test_tool_call_json_arguments(self):
        tc = ToolCall(id="c1", name="fn", arguments='{"x": 1}')
        self.assertEqual(json.loads(tc.arguments), {"x": 1})


# ── MCP Connection / Manager tests ───────────────────────────────────────────

class TestMCPConnection(unittest.TestCase):

    def _make_conn(self):
        cfg = MCPServerConfig(name="test", transport="stdio", command="echo")
        return MCPConnection(cfg)

    def _mock_session(self, tool_names, call_text="ok", is_error=False):
        session = MagicMock()
        tools = []
        for n in tool_names:
            t = MagicMock()
            t.name = n
            t.description = "desc"
            t.inputSchema = {"type": "object"}
            tools.append(t)
        list_resp = MagicMock()
        list_resp.tools = tools
        session.list_tools = AsyncMock(return_value=list_resp)
        session.initialize = AsyncMock()
        block = MagicMock()
        block.text = call_text
        call_resp = MagicMock()
        call_resp.content = [block]
        call_resp.isError = is_error
        session.call_tool = AsyncMock(return_value=call_resp)
        return session

    def test_refresh_tools(self):
        conn = self._make_conn()
        conn.session = self._mock_session(["tool_a", "tool_b"])
        run(conn._refresh_tools())
        tools = conn.list_tools()
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0].name, "tool_a")

    def test_call_tool_success(self):
        conn = self._make_conn()
        conn.session = self._mock_session(["my_tool"], "result text")
        run(conn._refresh_tools())
        tc = ToolCall(id="c1", name="my_tool", arguments='{"x": "hello"}')
        result = run(conn.call_tool(tc))
        self.assertEqual(result.content, "result text")
        self.assertEqual(result.tool_call_id, "c1")

    def test_call_tool_invalid_json(self):
        conn = self._make_conn()
        conn.session = self._mock_session(["fn"])
        run(conn._refresh_tools())
        tc = ToolCall(id="c2", name="fn", arguments="NOT JSON{{")
        result = run(conn.call_tool(tc))
        self.assertIn("[error]", result.content.lower())

    def test_call_tool_exception_handled(self):
        conn = self._make_conn()
        session = self._mock_session(["boom"])
        session.call_tool = AsyncMock(side_effect=RuntimeError("crash"))
        conn.session = session
        run(conn._refresh_tools())
        tc = ToolCall(id="c3", name="boom", arguments="{}")
        result = run(conn.call_tool(tc))
        self.assertIn("[error]", result.content.lower())

    def test_call_tool_is_error_flag(self):
        conn = self._make_conn()
        conn.session = self._mock_session(["risky"], "bad result", is_error=True)
        run(conn._refresh_tools())
        tc = ToolCall(id="c4", name="risky", arguments="{}")
        result = run(conn.call_tool(tc))
        self.assertIn("[error]", result.content.lower())


class TestMCPManager(unittest.TestCase):

    def _manager_with_conns(self, conns: dict):
        mgr = MCPManager([])
        mgr._connections = conns
        return mgr

    def _mock_conn(self, tool_names):
        conn = MagicMock()
        conn.list_tools.return_value = [_tool(n) for n in tool_names]
        return conn

    def test_list_tools_aggregated(self):
        mgr = self._manager_with_conns({
            "a": self._mock_conn(["tool_a"]),
            "b": self._mock_conn(["tool_b"]),
        })
        names = [t.name for t in mgr.list_tools()]
        self.assertIn("tool_a", names)
        self.assertIn("tool_b", names)

    def test_has_tool_found(self):
        mgr = self._manager_with_conns({"srv": self._mock_conn(["exists"])})
        self.assertTrue(mgr.has_tool("exists"))

    def test_has_tool_not_found(self):
        mgr = self._manager_with_conns({"srv": self._mock_conn(["exists"])})
        self.assertFalse(mgr.has_tool("missing"))

    def test_routes_call_to_correct_server(self):
        conn_a = self._mock_conn(["weather"])
        conn_a.call_tool = AsyncMock(return_value=ToolResult(
            tool_call_id="c1", name="weather", content="Sunny"
        ))
        mgr = self._manager_with_conns({"weather-srv": conn_a})
        tc = ToolCall(id="c1", name="weather", arguments='{}')
        result = run(mgr.call_tool(tc))
        self.assertEqual(result.content, "Sunny")
        conn_a.call_tool.assert_called_once_with(tc)

    def test_unknown_tool_returns_error(self):
        mgr = self._manager_with_conns({})
        tc = ToolCall(id="c1", name="ghost", arguments="{}")
        result = run(mgr.call_tool(tc))
        self.assertIn("[error]", result.content.lower())
        self.assertIn("ghost", result.content)


# ── ToolRegistry tests ────────────────────────────────────────────────────────

class TestToolRegistry(unittest.TestCase):

    def _reg(self, tool_names=None, call_content="done"):
        mgr = MagicMock()
        mgr.list_tools.return_value = [_tool(n) for n in (tool_names or [])]
        mgr.has_tool = lambda name: name in (tool_names or [])
        expected = ToolResult(tool_call_id="x", name="fn", content=call_content)
        mgr.call_tool = AsyncMock(return_value=expected)
        return ToolRegistry(mgr)

    def test_get_tool_definitions(self):
        reg = self._reg(["fn1", "fn2"])
        defs = reg.get_tool_definitions()
        self.assertEqual(len(defs), 2)
        names = [d["function"]["name"] for d in defs]
        self.assertIn("fn1", names)

    def test_call_known_tool(self):
        reg = self._reg(["my_fn"], "great result")
        tc = ToolCall(id="c1", name="my_fn", arguments="{}")
        result = run(reg.call(tc))
        self.assertEqual(result.content, "great result")

    def test_call_unknown_tool_error(self):
        reg = self._reg([])
        tc = ToolCall(id="c1", name="unknown_fn", arguments="{}")
        result = run(reg.call(tc))
        self.assertIn("[error]", result.content.lower())

    def test_call_all_dispatches_concurrently(self):
        """call_all should return one result per input tool call."""
        results_store = {}

        async def _call(tc: ToolCall) -> ToolResult:
            await asyncio.sleep(0.01)
            r = ToolResult(tool_call_id=tc.id, name=tc.name, content=f"res-{tc.name}")
            results_store[tc.name] = r
            return r

        mgr = MagicMock()
        mgr.has_tool = lambda name: True
        mgr.call_tool = _call
        reg = ToolRegistry(mgr)

        tcs = [ToolCall(id=str(i), name=f"t{i}", arguments="{}") for i in range(4)]
        results = run(reg.call_all(tcs))
        self.assertEqual(len(results), 4)
        names = {r.name for r in results}
        self.assertEqual(names, {"t0", "t1", "t2", "t3"})


# ── Agent tests ───────────────────────────────────────────────────────────────

def _mock_registry(tool_names=None, call_content="tool result"):
    reg = MagicMock(spec=ToolRegistry)
    reg.get_tool_definitions.return_value = [
        {"type": "function", "function": {"name": n}} for n in (tool_names or [])
    ]
    async def _call_all(tool_calls):
        return [
            ToolResult(tool_call_id=tc.id, name=tc.name, content=call_content)
            for tc in tool_calls
        ]
    reg.call_all = _call_all
    return reg


def _mock_llm_simple(content="Hello!"):
    llm = MagicMock(spec=LLMClient)
    msg = {"role": "assistant", "content": content}
    llm.complete = AsyncMock(return_value=(msg, [], "stop"))

    async def _stream_acc(messages, tools=None):
        c1 = MagicMock()
        c1.choices = [MagicMock()]
        c1.choices[0].delta.content = content
        c1.choices[0].finish_reason = None
        yield c1, None

        c2 = MagicMock()
        c2.choices = [MagicMock()]
        c2.choices[0].delta.content = None
        c2.choices[0].finish_reason = "stop"
        acc = MagicMock()
        acc.finish_reason = "stop"
        acc.tool_calls = []
        acc.to_message.return_value = msg
        yield c2, acc

    llm.stream_accumulated = _stream_acc
    return llm


def _mock_llm_with_one_tool_call(tool_name="calc", final="42"):
    llm = MagicMock(spec=LLMClient)
    tc = ToolCall(id="c1", name=tool_name, arguments='{"x":1}')
    tool_msg = {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": tool_name, "arguments": '{"x":1}'}}]
    }
    final_msg = {"role": "assistant", "content": final}
    call_count = [0]

    async def _complete(messages, tools=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return tool_msg, [tc], "tool_calls"
        return final_msg, [], "stop"
    llm.complete = _complete

    stream_count = [0]

    async def _stream_acc(messages, tools=None):
        stream_count[0] += 1
        if stream_count[0] == 1:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = None
            chunk.choices[0].finish_reason = "tool_calls"
            acc = MagicMock()
            acc.finish_reason = "tool_calls"
            acc.tool_calls = [tc]
            acc.to_message.return_value = tool_msg
            yield chunk, acc
        else:
            c1 = MagicMock()
            c1.choices = [MagicMock()]
            c1.choices[0].delta.content = final
            c1.choices[0].finish_reason = None
            yield c1, None

            c2 = MagicMock()
            c2.choices = [MagicMock()]
            c2.choices[0].delta.content = None
            c2.choices[0].finish_reason = "stop"
            acc2 = MagicMock()
            acc2.finish_reason = "stop"
            acc2.tool_calls = []
            acc2.to_message.return_value = final_msg
            yield c2, acc2

    llm.stream_accumulated = _stream_acc
    return llm


class TestAgent(unittest.TestCase):

    def _agent(self, llm, reg, max_iterations=5):
        cfg = AgentConfig(
            name="Test", system_prompt="You are a test bot.", max_iterations=max_iterations
        )
        mem = run(_make_memory())
        return Agent(cfg, llm, mem, reg), mem

    def test_single_turn_returns_response(self):
        llm = _mock_llm_simple("Hello!")
        reg = _mock_registry()
        agent, mem = self._agent(llm, reg)
        result = run(agent.run([{"role": "user", "content": "Hi"}], "s1"))
        self.assertEqual(result["content"], "Hello!")
        run(mem.close())

    def test_history_persisted_after_run(self):
        llm = _mock_llm_simple("Answer")
        reg = _mock_registry()
        agent, mem = self._agent(llm, reg)
        run(agent.run([{"role": "user", "content": "Q"}], "s2"))
        history = run(mem.load("s2"))
        roles = [m["role"] for m in history]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        run(mem.close())

    def test_tool_call_round_trip(self):
        llm = _mock_llm_with_one_tool_call("weather", "Sunny")
        reg = _mock_registry(["weather"], "Sunny and 75°F")
        agent, mem = self._agent(llm, reg)
        result = run(agent.run([{"role": "user", "content": "Weather?"}], "s3"))
        self.assertEqual(result["content"], "Sunny")
        history = run(mem.load("s3"))
        roles = [m["role"] for m in history]
        self.assertIn("tool", roles)
        self.assertEqual(roles.count("assistant"), 2)
        run(mem.close())

    def test_history_included_in_context(self):
        """Prior history messages must appear in the LLM call context."""
        captured = []

        async def _complete(messages, tools=None):
            captured.extend(messages)
            return {"role": "assistant", "content": "OK"}, [], "stop"

        llm = MagicMock(spec=LLMClient)
        llm.complete = _complete
        reg = _mock_registry()
        cfg = AgentConfig(name="T", system_prompt="Sys", max_iterations=5)
        mem = run(_make_memory())
        run(mem.save("s4", [
            {"role": "user", "content": "Prior"},
            {"role": "assistant", "content": "Memory"},
        ]))
        agent = Agent(cfg, llm, mem, reg)
        run(agent.run([{"role": "user", "content": "New"}], "s4"))
        contents = [m.get("content") for m in captured]
        self.assertIn("Prior", contents)
        self.assertIn("Memory", contents)
        run(mem.close())

    def test_system_prompt_always_first(self):
        captured = []

        async def _complete(messages, tools=None):
            captured.extend(messages)
            return {"role": "assistant", "content": "OK"}, [], "stop"

        llm = MagicMock(spec=LLMClient)
        llm.complete = _complete
        reg = _mock_registry()
        cfg = AgentConfig(name="T", system_prompt="System instruction", max_iterations=5)
        mem = run(_make_memory())
        agent = Agent(cfg, llm, mem, reg)
        run(agent.run([{"role": "user", "content": "Hello"}], "sys-test"))
        self.assertEqual(captured[0]["role"], "system")
        self.assertEqual(captured[0]["content"], "System instruction")
        run(mem.close())

    def test_max_iterations_exceeded(self):
        llm = MagicMock(spec=LLMClient)
        tc = ToolCall(id="loop", name="looper", arguments="{}")
        tool_msg = {"role": "assistant", "content": None, "tool_calls": []}
        llm.complete = AsyncMock(return_value=(tool_msg, [tc], "tool_calls"))
        reg = _mock_registry(["looper"])
        agent, mem = self._agent(llm, reg, max_iterations=3)
        with self.assertRaises(AgentError):
            run(agent.run([{"role": "user", "content": "loop"}], "loop-sess"))
        run(mem.close())

    def test_streaming_yields_tokens(self):
        llm = _mock_llm_simple("streaming content")
        reg = _mock_registry()
        agent, mem = self._agent(llm, reg)

        async def _collect():
            tokens = []
            async for tok in agent.run_stream([{"role": "user", "content": "Hi"}], "str1"):
                tokens.append(tok)
            return tokens

        tokens = run(_collect())
        self.assertIn("streaming content", "".join(tokens))
        run(mem.close())

    def test_streaming_with_tool_call(self):
        llm = _mock_llm_with_one_tool_call("calc", "42")
        reg = _mock_registry(["calc"], "42")
        agent, mem = self._agent(llm, reg)

        async def _collect():
            tokens = []
            async for tok in agent.run_stream([{"role": "user", "content": "1+1?"}], "str2"):
                tokens.append(tok)
            return tokens

        tokens = run(_collect())
        self.assertIn("42", "".join(tokens))
        run(mem.close())

    def test_empty_tool_list_when_no_tools(self):
        captured_tools = []

        async def _complete(messages, tools=None):
            captured_tools.append(tools)
            return {"role": "assistant", "content": "OK"}, [], "stop"

        llm = MagicMock(spec=LLMClient)
        llm.complete = _complete
        reg = _mock_registry([])  # no tools
        cfg = AgentConfig(name="T", system_prompt="S", max_iterations=5)
        mem = run(_make_memory())
        agent = Agent(cfg, llm, mem, reg)
        run(agent.run([{"role": "user", "content": "Hi"}], "notool"))
        # When no tools are available, None is passed (not empty list)
        self.assertIsNone(captured_tools[0])
        run(mem.close())


# ── Test runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestConfig,
        TestMemory,
        TestMultimodal,
        TestToolBase,
        TestMCPConnection,
        TestMCPManager,
        TestToolRegistry,
        TestAgent,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
