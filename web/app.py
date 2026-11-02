"""FastAPI adapter for the existing agentic-loop runtime.

This package intentionally lives in ``web/`` and does not modify the parent
system. Run it from the repository root so the existing project modules and
its root-level ``.env`` can be imported normally.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Make the repository root importable without changing its existing modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.mcp_settings import mcp_settings  # noqa: E402
from configs.settings import settings  # noqa: E402
from core.agentic_loop import agentic_loop  # noqa: E402


logger = logging.getLogger("agentic_loop.web")
STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=32, max_length=64)
    message: str = Field(min_length=1, max_length=12_000)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be empty.")
        return value


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: str


class ChatResponse(BaseModel):
    session_id: str
    request_id: str
    status: str
    response: str | None = None
    error: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: int


class AgentSession:
    """One isolated agent instance and conversation history per browser session."""

    def __init__(self) -> None:
        self.agent = agentic_loop()
        self.lock = asyncio.Lock()
        self.created_at = datetime.now(timezone.utc).isoformat()


_sessions: dict[str, AgentSession] = {}
_sessions_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # Consolidate any open episodic-memory sessions during a clean shutdown.
    async with _sessions_lock:
        remaining = list(_sessions.values())
        _sessions.clear()
    for session in remaining:
        try:
            await run_in_threadpool(session.agent.end_session)
        except Exception:
            logger.exception("Could not consolidate an agent session during shutdown")


app = FastAPI(
    title="Agentic Loop Web Console",
    description="A lightweight chat-style test interface for the existing agentic loop.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Report local app/configuration status; this does not ping Ollama or MCP."""
    try:
        configured_servers = [server.name for server in mcp_settings.servers]
        mcp_error = None
    except Exception as exc:  # Keep the UI available even if optional config fails.
        configured_servers = []
        mcp_error = str(exc)[:500]

    return {
        "status": "ok",
        "service": "Agentic Loop Web Console",
        "model": settings.ollama_model,
        "ollama_host_configured": bool(settings.ollama_host),
        "mcp_servers": configured_servers,
        "mcp_configuration_error": mcp_error,
        "episodic_memory_configured": bool(
            settings.qdrant_url and settings.cohere_api_key
        ),
        "project_root": PROJECT_ROOT.name,
        "note": "Configuration status only; provider connectivity is checked during a run.",
    }


@app.post(
    "/api/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session() -> CreateSessionResponse:
    session_id = uuid4().hex
    session = AgentSession()
    async with _sessions_lock:
        _sessions[session_id] = session
    return CreateSessionResponse(session_id=session_id, created_at=session.created_at)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, str]:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This test session is no longer active.",
        )
    return {"session_id": session_id, "created_at": session.created_at, "status": "active"}


@app.delete("/api/sessions/{session_id}")
async def close_session(session_id: str) -> dict[str, str]:
    async with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        return {"status": "already_closed"}

    async with session.lock:
        try:
            await run_in_threadpool(session.agent.end_session)
        except Exception:
            logger.exception("Session consolidation failed for %s", session_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The session ended, but episodic-memory consolidation failed.",
            )
    return {"status": "closed"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    session = _sessions.get(payload.session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This test session is no longer active. Start a new chat session.",
        )

    request_id = uuid4().hex[:12]
    started = time.perf_counter()

    # Serialize turns within one conversation while allowing separate sessions
    # to run concurrently. The agent loop itself is synchronous, so it runs in
    # FastAPI's worker pool rather than blocking the event loop.
    async with session.lock:
        try:
            result = await run_in_threadpool(session.agent.run, payload.message)
            history_snapshot = session.agent.chat_history.snapshot().get("messages", [])
        except Exception as exc:
            logger.exception("Unhandled agent failure (request_id=%s)", request_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent execution failed. Reference: {request_id}",
            ) from exc

    duration_ms = round((time.perf_counter() - started) * 1000)
    try:
        result_data = result.model_dump(mode="json")
    except Exception:
        logger.exception("Could not serialize agent result (request_id=%s)", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent returned a result that could not be serialized. Reference: {request_id}",
        )

    return ChatResponse(
        session_id=payload.session_id,
        request_id=request_id,
        status=str(result_data.get("status", "failed")),
        response=(
            result_data.get("decision")
            if isinstance(result_data.get("decision"), str)
            else None
        ),
        error=result_data.get("error"),
        state=result_data.get("state") or {},
        history=history_snapshot,
        duration_ms=duration_ms,
    )
