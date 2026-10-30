
from pydantic_settings import BaseSettings
from .settings import settings
from packages.MCP import MCPServerConfig, StdioTransport, StreamableHTTPTransport

# ============================================
#     MCP SERVER CONFIGURATIONS
# ============================================

class MCPSettings(BaseSettings):
    @property
    def servers(self) -> list[MCPServerConfig]:
        if not settings.zapier_url:
            return []
        return [
            MCPServerConfig(
                name="zapier",
                transport=StreamableHTTPTransport(
                    url=settings.zapier_url,
                    bearer_token=settings.zapier_token or None,
                    terminate_on_close=False,
                ),
            )
        ]


mcp_settings = MCPSettings()
