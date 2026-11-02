# Agentic Loop Web Console

A small, standalone FastAPI + HTML/CSS/JavaScript chat UI for exercising the
agentic-loop project. All web files are contained in this `web/` directory.
The adapter imports and calls the existing `core.agentic_loop.agentic_loop`;
it does not alter the parent system's Python modules, prompts, configuration,
schemas, or memory implementation.

## Install

Run these commands from the **repository root** (the directory containing
`main.py`):

```bash
python -m pip install -r requirements.txt
python -m pip install -r web/requirements.txt
```

The first command installs the existing agent system's dependencies. The
second installs only the web-serving dependencies.

## Start

From the repository root:

```bash
python -m web
```

Or, for development with auto-reload:

```bash
uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. FastAPI's local API documentation is available at
<http://127.0.0.1:8000/api/docs>.

## How it works

- Each browser tab gets its own server-side `agentic_loop` instance and chat
  history. Turns in one session are serialized; separate sessions may run in
  parallel.
- Requests call the existing synchronous `agentic_loop.run()` in FastAPI's
  worker pool. The browser receives the agent's final response, status, runtime
  state, and elapsed time.
- The **Run details** panel exposes the structured result for inspection.
- Agent responses render common Markdown as clean UI elements: headings, paragraphs, bold/italic text, lists, links, tables, quotes, inline code, and copyable fenced code blocks. Rendering uses safe DOM nodes rather than injecting model output as HTML.
- Starting a new chat closes the previous session and calls its existing
  `end_session()` method, allowing configured episodic memory to consolidate.
- Open chats are held in server memory. A server restart ends those sessions;
  the browser will ask you to start a fresh chat if its old session is gone.
- The health indicator reports local app/configuration state. It does not claim
  that Ollama or MCP servers are reachable; actual connectivity is exercised
  during a prompt run.

## Existing configuration note

The current parent repository's `configs/mcp_settings.py` sets the hybrid
retrieval server's working directory with a hard-coded `PROJECT_ROOT`. This
web adapter intentionally leaves that file untouched. When hybrid retrieval
is configured, that existing path must resolve to this checkout on the device;
otherwise the MCP process may fail to start. This is a parent-project
configuration constraint, not a web-server setting.

## Scope

This directory is an optional testing surface, not a permanent dependency of
the agent runtime. Remove `web/` at any time to remove the web console; the
existing CLI and core agent system remain independent.
