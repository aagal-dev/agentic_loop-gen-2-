"""MCP server registrations for the agentic loop."""

import sys
from pathlib import Path

from configs.settings import settings
from packages.MCP import MCPServerConfig, StdioTransport, StreamableHTTPTransport


PROJECT_ROOT = Path("/sdcard/Acode/AI Prototypes/new_agentic_loop")


class MCPSettings:
    """Build the configured local and optional remote MCP server list."""

    @property
    def servers(self) -> list[MCPServerConfig]:
        servers: list[MCPServerConfig] = []

        # Existing optional Zapier server.
        if settings.zapier_url:
            servers.append(
                MCPServerConfig(
                    name="zapier",
                    transport=StreamableHTTPTransport(
                        url=settings.zapier_url,
                        bearer_token=settings.zapier_token or None,
                        terminate_on_close=False,
                    ),
                )
            )

        # Enable hybrid retrieval only when its providers are configured.
        memory_is_configured = bool(
            settings.cohere_api_key
            and settings.cohere_api_key.strip()
            and settings.qdrant_url
            and settings.qdrant_url.strip()
        )

        if memory_is_configured:
            servers.append(
                MCPServerConfig(
                    name="hybrid_retrieval",
                    transport=StdioTransport(
                        command=sys.executable,
                        args=(
                            "-u",
                            "-m",
                            "mcp_servers.hybrid_retrieval.server",
                        ),
                        cwd=str(PROJECT_ROOT),
                    ),
                )
            )

        return servers


mcp_settings = MCPSettings()