# A Minimal Agent Framework

A modular, async-first AI agent framework that supports multimodal input, persistent memory,
multi-round tool invocation, MCP tool servers, and both CLI and OpenAI-compatible Web API interfaces.

Copyright © 2025-2026 SSRVodka. All Rights Reserved.

---

## Architecture

```
agent/
├── config/
│   └── default.yaml          # Default configuration (all values overridable)
├── src/
│   ├── config.py             # YAML loader + Pydantic models + ${ENV_VAR} interpolation
│   ├── memory.py             # SQLite-backed persistent conversation history
│   ├── llm.py                # Async OpenAI client (streaming + non-streaming)
│   ├── multimodal.py         # Text / image / audio content builders
│   ├── agent.py              # Core agent loop (multi-round tool orchestration)
│   ├── tools/
│   │   ├── base.py           # ToolDefinition / ToolCall / ToolResult data models
│   │   ├── mcp_client.py     # MCP connections (stdio · SSE · streamable HTTP)
│   │   └── registry.py       # Unified tool registry + concurrent dispatch
│   └── interfaces/
│       ├── cli.py            # Rich REPL + single-shot CLI
│       └── api.py            # FastAPI OpenAI-compatible REST API
├── tests/
│   ├── _stubs.py             # Offline dependency stubs (no network required)
│   ├── run_tests.py          # Standalone test runner (no pytest required)
│   ├── test_memory.py        # pytest-compatible memory tests
│   ├── test_agent.py         # pytest-compatible agent tests
│   ├── test_tools.py         # pytest-compatible tool + config tests
│   └── test_api.py           # pytest-compatible API tests
├── main.py                   # CLI entry point (click)
└── requirements.txt
```

### Key design decisions

| Concern | Choice | Rationale |
|---|---|---|
| LLM backend | Any OpenAI-API-compatible endpoint | Works with OpenAI, Anthropic (via proxy), Ollama, vLLM, etc. |
| Memory | SQLite via aiosqlite + per-session `asyncio.Lock` | No extra infrastructure; concurrent-safe; portable |
| MCP transports | stdio · SSE · streamable HTTP (all in `mcp_client.py`) | Covers the full MCP spec; per-server locks prevent interleaving |
| Tool dispatch | Concurrent `asyncio.gather` in `ToolRegistry.call_all` | Parallel execution when the LLM issues multiple tool calls |
| Streaming | Raw SSE chunk forwarding via async generator | Zero buffering latency; compatible with any SSE consumer |
| Agent loop | Iterative while-loop with `max_iterations` guard | Simple, debuggable; prevents infinite tool-call loops |

---

## Installation

```bash
pip install -r requirements.txt
```

Set your API key (and optionally base URL):

```bash
export OPENAI_API_KEY=sk-...
# export OPENAI_BASE_URL=https://api.openai.com/v1   # default
```

---

## Configuration

Edit `config/default.yaml` or point to your own file with `--config`.

### Environment variable interpolation

All string values support `${VAR}` and `${VAR:-default}` syntax:

```yaml
llm:
  api_key: "${OPENAI_API_KEY}"
  base_url: "${OPENAI_BASE_URL:-https://api.openai.com/v1}"
```

### MCP server examples

```yaml
tools:
  mcp_servers:
    # stdio: spawn a local process
    - name: "filesystem"
      transport: "stdio"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

    # SSE: connect to a running SSE server
    - name: "weather"
      transport: "sse"
      url: "http://localhost:8001/sse"

    # Streamable HTTP: modern MCP transport
    - name: "search"
      transport: "streamable_http"
      url: "http://localhost:8002/mcp"
      headers:
        Authorization: "Bearer ${SEARCH_API_KEY}"
```

---

## CLI Usage

```bash
# Interactive REPL
python main.py chat

# Single-shot query
python main.py ask "What is the capital of France?"

# Multimodal: text + image
python main.py ask --image ./photo.jpg "Describe this image"

# Multimodal: text + audio
python main.py ask --audio ./clip.mp3 "Transcribe this"

# Use a specific session
python main.py ask --session my-project "Continue our conversation"

# Disable streaming
python main.py ask --no-stream "Quick answer please"

# Use a custom config
python main.py --config config/my-agent.yaml chat

# Session management
python main.py sessions          # list all sessions
python main.py clear my-session  # delete a session's history
```

### REPL commands

| Command | Effect |
|---|---|
| `/quit` | Exit |
| `/clear` | Clear current session history |
| `/session <id>` | Switch to a different session |

---

## Web API

Start the server:

```bash
python main.py serve
# or with overrides:
python main.py serve --host 127.0.0.1 --port 9000
```

The server exposes an **OpenAI-compatible** API, so any OpenAI SDK client works out of the box.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat (streaming + non-streaming) |
| `GET` | `/v1/sessions` | List persisted sessions |
| `DELETE` | `/v1/sessions/{id}` | Clear a session |

### Example: streaming with the Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="ignored")

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
    user="my-session-id",    # <-- used as session_id for history persistence
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Example: multimodal image request

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ]
    }],
    user="vision-session",
)
print(response.choices[0].message.content)
```

### Authentication

Set `interfaces.api.api_key` in your config to require a Bearer token:

```yaml
interfaces:
  api:
    api_key: "${API_SECRET_KEY}"
```

---

## Running Tests

### Offline (no packages required beyond stdlib + PyYAML)

```bash
python3 tests/run_tests.py
```

### With pytest (when packages are installed)

```bash
pytest -v
```

---

## Memory

Conversations are persisted per `session_id` in a SQLite database (`data/memory.db` by default).

- **Concurrency**: each session has its own `asyncio.Lock`; a global lock prevents races when creating new lock objects.
- **Trimming**: only the last `max_history` messages are retained (configurable).
- **Isolation**: sessions are fully independent.

---

## Extending

### Add a native Python tool

1. Create a `ToolDefinition` in `src/tools/base.py`.
2. Register it in `ToolRegistry` and add a dispatch branch in `call()`.

### Add a new memory backend

Implement the same async interface as `Memory` (`initialize`, `close`, `load`, `save`, `append`, `clear`, `list_sessions`) and swap it in `main.py`.

### Use a different LLM provider

Point `llm.base_url` at any OpenAI-compatible endpoint:

```yaml
llm:
  base_url: "http://localhost:11434/v1"   # Ollama
  model: "llama3.2"
  api_key: "ollama"
```
