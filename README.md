# new_agentic_loop

Personal AI project scaffold.

## Run

```sh
source .venv/bin/activate
python main.py
```

Run the automated tests with `python -m pip install -e '.[test]'` followed by
`python -m pytest -q tests/`. Tests mock external Ollama and MCP services.

## Configuration

- `.env` contains local settings and is excluded from Git.
- `.env.example` documents the environment variables; update both when adding new settings.
- Zapier MCP is optional. Without `ZAPIER_URL`, direct responses work with no remote MCP connection.
- Put each agent's system prompt in `prompts/`.
- Add Pydantic input/output models in `schemas/` and agent implementations in `agents/`.
- Copy your customized MCP SDK source into `packages/MCP/` when ready.

The project is intentionally framework-agnostic and does not install or depend on a separate agent framework.
