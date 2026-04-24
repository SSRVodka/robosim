"""
Stub external packages for offline testing.
Run:  python3 tests/run_tests.py
"""

import sys
import types
import asyncio
import json
import re
from typing import Any


# ── helpers ──────────────────────────────────────────────────────────────────

def _mod(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# ── pydantic (minimal but correct) ───────────────────────────────────────────

import inspect as _inspect
import typing as _typing

_SENTINEL = object()

class _Field:
    def __init__(self, *a, **kw):
        self.default_factory = kw.get("default_factory", None)
        self.default = kw.get("default", _SENTINEL)

def _field(*a, **kw):
    return _Field(*a, **kw)

# Registry of model_validators keyed by class + method name
_VALIDATORS: dict = {}

def _model_validator(*, mode):
    """Register a post-init validator on the decorated method."""
    def decorator(fn):
        fn.__is_model_validator__ = True
        fn.__validator_mode__ = mode
        return fn
    return decorator

def _collect_fields(cls):
    """Walk MRO to collect (name, annotation, default/Field) triples."""
    fields = {}
    for base in reversed(cls.__mro__):
        hints = {}
        try:
            hints = _typing.get_type_hints(base)
        except Exception:
            hints = getattr(base, "__annotations__", {})
        for name, hint in hints.items():
            if name.startswith("_"):
                continue
            raw = base.__dict__.get(name, _SENTINEL)
            if raw is _SENTINEL:
                fields[name] = (hint, _SENTINEL, None)
            elif isinstance(raw, _Field):
                fields[name] = (hint, raw.default, raw.default_factory)
            else:
                fields[name] = (hint, raw, None)
    return fields

def _get_origin(tp):
    return getattr(tp, "__origin__", None)

def _get_args(tp):
    return getattr(tp, "__args__", ())

def _coerce(value, hint):
    """Attempt to coerce value into the expected type (best-effort)."""
    if value is None:
        return None
    origin = _get_origin(hint)
    args = _get_args(hint)

    # Union / Optional
    if origin is _typing.Union:
        non_none = [a for a in args if a is not type(None)]
        for t in non_none:
            try:
                return _coerce(value, t)
            except Exception:
                pass
        return value

    # List
    if origin is list:
        item_hint = args[0] if args else None
        if isinstance(value, list) and item_hint is not None:
            return [_coerce(i, item_hint) for i in value]
        return value

    # Dict
    if origin is dict:
        return value

    # Subclass of BaseModel
    if isinstance(hint, type) and issubclass(hint, BaseModel):
        if isinstance(value, hint):
            return value
        if isinstance(value, dict):
            return hint(**value)
        return value

    return value


class _ModelMeta(type):
    pass

class BaseModel(metaclass=_ModelMeta):
    model_config: dict = {}

    def __init__(self, **data):
        fields = _collect_fields(self.__class__)
        for name, (hint, default, factory) in fields.items():
            if name in ("model_config",):
                continue
            if name in data:
                setattr(self, name, _coerce(data[name], hint))
            else:
                if factory is not None:
                    setattr(self, name, factory())
                elif default is not _SENTINEL:
                    setattr(self, name, default)
                # else: required field – leave unset (will raise AttributeError if accessed)

        # Run model_validators (mode="after")
        for attr_name in dir(self.__class__):
            fn = self.__class__.__dict__.get(attr_name)
            if fn and getattr(fn, "__is_model_validator__", False):
                result = fn(self)
                if result is not None and result is not self:
                    for k, v in result.__dict__.items():
                        setattr(self, k, v)

    @classmethod
    def model_validate(cls, data: dict):
        return cls(**data)

    def model_dump(self, *, exclude_none: bool = False):
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if exclude_none and v is None:
                continue
            if isinstance(v, BaseModel):
                result[k] = v.model_dump(exclude_none=exclude_none)
            elif isinstance(v, list):
                result[k] = [
                    i.model_dump(exclude_none=exclude_none) if isinstance(i, BaseModel) else i
                    for i in v
                ]
            else:
                result[k] = v
        return result

    def __repr__(self):
        fields = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({fields})"

pydantic_mod = _mod("pydantic")
pydantic_mod.BaseModel = BaseModel
pydantic_mod.Field = _field
pydantic_mod.model_validator = _model_validator

pydantic_v1 = _mod("pydantic.v1")

# ── aiosqlite ─────────────────────────────────────────────────────────────────
import sqlite3, contextlib, pathlib, time as _time

class _AsyncCursor:
    def __init__(self, cursor): self._c = cursor
    async def fetchone(self): return self._c.fetchone()
    async def fetchall(self): return self._c.fetchall()
    async def __aenter__(self): return self
    async def __aexit__(self, *_): self._c.close()

async def _noop_cursor(cur):
    return cur

class _AwaitableCtx:
    """Mimics aiosqlite's _ContextManager: both awaitable and async context manager."""
    def __init__(self, cursor):
        self._cursor = cursor

    def __await__(self):
        # allows:  cur = await conn.execute(...)
        return _noop_cursor(self._cursor).__await__()

    async def __aenter__(self):
        return self._cursor

    async def __aexit__(self, *_):
        pass

    async def fetchone(self): return await self._cursor.fetchone()
    async def fetchall(self): return await self._cursor.fetchall()

class _AioConn:
    def __init__(self, path):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    @property
    def row_factory(self): return sqlite3.Row
    @row_factory.setter
    def row_factory(self, v): self._conn.row_factory = v

    def execute(self, sql, params=()):
        # Synchronously execute; return dual-mode wrapper
        raw_cur = self._conn.execute(sql, params)
        async_cur = _AsyncCursor(raw_cur)
        return _AwaitableCtx(async_cur)

    async def commit(self): self._conn.commit()
    async def close(self): self._conn.close()

    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.close()

async def _aiosqlite_connect(path):
    return _AioConn(path)

aiosqlite_mod = _mod("aiosqlite")
aiosqlite_mod.connect = _aiosqlite_connect
aiosqlite_mod.Connection = _AioConn
aiosqlite_mod.Row = sqlite3.Row

# ── mcp stubs ─────────────────────────────────────────────────────────────────
for name in ["mcp", "mcp.client", "mcp.client.stdio", "mcp.client.sse",
             "mcp.client.streamable_http"]:
    _mod(name)

class _ClientSession:
    async def initialize(self): pass
    async def list_tools(self): ...
    async def call_tool(self, name, args): ...
    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass

sys.modules["mcp"].ClientSession = _ClientSession

class _StdioParams:
    def __init__(self, command, args, env=None):
        self.command = command; self.args = args; self.env = env

sys.modules["mcp.client.stdio"].stdio_client = None
sys.modules["mcp.client.stdio"].StdioServerParameters = _StdioParams
sys.modules["mcp.client.sse"].sse_client = None
sys.modules["mcp.client.streamable_http"].streamablehttp_client = None

# ── openai stubs ──────────────────────────────────────────────────────────────
for name in ["openai", "openai.types", "openai.types.chat"]:
    _mod(name)

class _AsyncOpenAI:
    def __init__(self, **kw): pass

sys.modules["openai"].AsyncOpenAI = _AsyncOpenAI
sys.modules["openai.types.chat"].ChatCompletionChunk = object

# ── fastapi stubs ─────────────────────────────────────────────────────────────
for name in ["fastapi", "fastapi.middleware", "fastapi.middleware.cors",
             "fastapi.responses", "starlette", "starlette.testclient",
             "uvicorn"]:
    _mod(name)

class _FastAPI:
    def __init__(self, **kw):
        self.routes = []
    def add_middleware(self, cls, **kw): pass
    def get(self, path, **kw):
        def dec(fn): return fn
        return dec
    def post(self, path, **kw):
        def dec(fn): return fn
        return dec
    def delete(self, path, **kw):
        def dec(fn): return fn
        return dec

class _CORSMiddleware: pass
class _JSONResponse: pass
class _StreamingResponse: pass
class _Header:
    @staticmethod
    def __call__(**kw): return None
class _HTTPException(Exception):
    def __init__(self, status_code, detail=""): self.status_code = status_code; self.detail = detail
class _Depends:
    def __call__(self, fn): return fn

fastapi_mod = sys.modules["fastapi"]
fastapi_mod.FastAPI = _FastAPI
fastapi_mod.Depends = lambda fn: None
fastapi_mod.Header = _Header()
fastapi_mod.HTTPException = _HTTPException
fastapi_mod.Request = object
fastapi_mod.status = types.SimpleNamespace(HTTP_401_UNAUTHORIZED=401)
sys.modules["fastapi.middleware.cors"].CORSMiddleware = _CORSMiddleware
sys.modules["fastapi.responses"].JSONResponse = _JSONResponse
sys.modules["fastapi.responses"].StreamingResponse = _StreamingResponse

class _UvicornConfig:
    def __init__(self, app, **kw): pass
class _UvicornServer:
    def __init__(self, cfg): pass
    async def serve(self): pass
uvicorn_mod = sys.modules["uvicorn"]
uvicorn_mod.Config = _UvicornConfig
uvicorn_mod.Server = _UvicornServer

# ── rich / click already installed ───────────────────────────────────────────
# click is installed; rich is not but tests don't invoke CLI directly
for name in ["rich", "rich.console", "rich.markdown", "rich.panel", "rich.prompt"]:
    m = _mod(name)
rich_mod = sys.modules["rich.console"]
rich_mod.Console = type("Console", (), {
    "print": lambda *a, **kw: None,
    "status": lambda self, *a, **kw: _DummyCtx(),
    "__init__": lambda self: None,
})
class _DummyCtx:
    def __enter__(self): return self
    def __exit__(self, *_): pass
sys.modules["rich.markdown"].Markdown = lambda s: s
sys.modules["rich.panel"].Panel = lambda *a, **kw: None
sys.modules["rich.prompt"].Prompt = type("Prompt", (), {"ask": staticmethod(lambda *a, **kw: "")})

# ── sse-starlette / python-multipart ─────────────────────────────────────────
for name in ["sse_starlette", "sse_starlette.sse", "multipart"]:
    _mod(name)

# ── anyio / aiofiles ─────────────────────────────────────────────────────────
for name in ["anyio", "aiofiles"]:
    _mod(name)

print("[stubs] All external dependency stubs installed.", flush=True)
