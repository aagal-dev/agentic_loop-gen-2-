"""Run the web console with ``python -m web`` from the repository root."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)
