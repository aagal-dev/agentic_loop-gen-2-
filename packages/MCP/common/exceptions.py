"""Public errors raised by mcp_wrapper."""


class MCPError(Exception):
    """Base class for wrapper errors."""


class MCPConnectionError(MCPError): pass
class MCPTransportError(MCPError): pass
class MCPTimeoutError(MCPError): pass
class MCPProtocolError(MCPError): pass
class MCPValidationError(MCPError): pass
class MCPToolError(MCPError): pass
class MCPResourceError(MCPError): pass
class MCPPromptError(MCPError): pass
class MCPLifecycleError(MCPError): pass
