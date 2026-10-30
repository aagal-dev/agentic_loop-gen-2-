"""Small, typed connection configuration models."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncContextManager, Callable, Mapping
from urllib.parse import urlsplit

from .exceptions import MCPValidationError


@dataclass(frozen=True)
class StdioTransport:
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None

    def __post_init__(self) -> None:
        if not self.command or not self.command.strip():
            raise MCPValidationError("StdioTransport.command is required")


@asynccontextmanager
async def _streamable_http_connection(
    url: str, headers: Mapping[str, str], terminate_on_close: bool, timeout: float
):
    """Own the SDK HTTP client for the full life of one MCP session."""
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    # Keep the HTTP client timeout consistent with the MCP session timeout.
    async with create_mcp_http_client(headers=dict(headers), timeout=timeout) as http_client:
        async with streamable_http_client(
            url, http_client=http_client, terminate_on_close=terminate_on_close
        ) as streams:
            yield streams


@dataclass(frozen=True)
class StreamableHTTPTransport:
    """Remote MCP over Streamable HTTP, including header-based authentication."""

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    bearer_token: str | None = field(default=None, repr=False)
    terminate_on_close: bool = True
    http_timeout: float = 30.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise MCPValidationError("StreamableHTTPTransport.url must be an absolute http(s) URL")
        if self.bearer_token is not None and not self.bearer_token.strip():
            raise MCPValidationError("StreamableHTTPTransport.bearer_token cannot be blank")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.headers.items()):
            raise MCPValidationError("StreamableHTTPTransport.headers must contain string keys and values")
        if self.bearer_token and any(key.lower() == "authorization" for key in self.headers):
            raise MCPValidationError("provide either bearer_token or an Authorization header, not both")
        if self.http_timeout <= 0:
            raise MCPValidationError("StreamableHTTPTransport.http_timeout must be positive")

    def factory(self, *, timeout: float | None = None) -> AsyncContextManager[tuple[Any, Any]]:
        headers = dict(self.headers)
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return _streamable_http_connection(
            self.url, headers, self.terminate_on_close, timeout or self.http_timeout
        )


@dataclass(frozen=True)
class SessionFactoryTransport:
    """Adapter for any official SDK transport context manager (HTTP, SSE, etc.)."""
    factory: Callable[[], AsyncContextManager[tuple[Any, Any]]]

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise MCPValidationError("SessionFactoryTransport.factory must be callable")


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: StdioTransport | StreamableHTTPTransport | SessionFactoryTransport
    timeout: float = 600.0

    def __post_init__(self) -> None:
        if not self.name or "__" in self.name or not self.name.replace("_", "").isalnum():
            raise MCPValidationError("server name must be non-empty alphanumeric/underscore and cannot contain '__'")
        if self.timeout <= 0:
            raise MCPValidationError("timeout must be positive")
