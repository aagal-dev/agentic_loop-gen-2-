# Repository Guidelines

## Project Structure & Module Organization

`main.py` starts the interactive CLI and delegates requests to
`core/agentic_loop.py`. Keep orchestration and runtime-state transitions in
`core/`; `core/runtime_state.py` contains the Pydantic-backed state model.
Place agent implementations in `agents/`, provider adapters in
`integrations/`, and environment-backed settings/MCP server declarations in
`configs/`. Structured input and output contracts belong in `schemas/`, while
agent prompts live in `prompts/`. The vendored MCP client package is under
`packages/MCP/`.

## Build, Test, and Development Commands

Create and activate a virtual environment, then install the declared runtime
dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py       # start the interactive agent; enter q to quit
python test.py       # exercise the RuntimeStateManager demonstration
python -m pip install -e '.[test]'
python -m pytest -q tests/  # run the automated regression suite
```

Use `python -m pip install -e .` when validating package metadata or imports.
There is no configured formatter or linter.

## Coding Style & Naming Conventions

Write Python targeting 3.10+ with four-space indentation, type annotations on
public functions, and `snake_case` for modules, functions, and variables.
Use `PascalCase` for classes and Pydantic models (for example,
`RuntimeStateManager` and `AgentLoopResult`). Keep schemas strict with
`ConfigDict(extra="forbid")` where appropriate, validate external/LLM data at
the boundary, and return structured result models instead of untyped payloads.
Avoid committing generated `__pycache__/` files.

## Testing Guidelines

Add focused tests alongside the code they cover or in a future `tests/`
directory using names such as `test_complete_step_rejects_unfinished_runtime`.
Cover valid transitions and error paths for runtime state, model validation,
and MCP/LLM failures. Keep networked Ollama and MCP calls mocked in automated
tests; use `python test.py` only as a local smoke check.

## Configuration & Contributions

Copy `.env.example` to `.env` for local secrets and endpoints. Never commit
tokens, service URLs that are private, or `.env`. When adding a setting, update
both files and keep the corresponding type/default in `configs/settings.py`.

This checkout has no Git history, so no existing commit convention can be
derived. Use short imperative commit subjects (for example,
`Add runtime-state validation`) and submit PRs with a concise summary, test
evidence, configuration notes, and screenshots only for user-facing changes.
