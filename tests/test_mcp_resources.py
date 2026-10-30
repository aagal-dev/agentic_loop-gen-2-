from types import SimpleNamespace

from packages.MCP.client.client import MCPClient
from packages.MCP.client.result import resource_agent_view


def test_resource_catalog_schema_is_compact_and_json_ready():
    resource = SimpleNamespace(
        name="guide",
        uri="file://guide",
        description="  Project guide  ",
        mimeType="text/markdown",
        size=128,
    )

    assert MCPClient._resource_schema(resource, "docs__guide", "docs") == {
        "name": "docs__guide",
        "server": "docs",
        "uri": "file://guide",
        "description": "Project guide",
        "mime_type": "text/markdown",
        "size": 128,
    }


def test_resource_agent_view_truncates_text_and_omits_blob_data():
    raw = SimpleNamespace(contents=[
        SimpleNamespace(uri="file://text", mimeType="text/plain", text="abcdef"),
        SimpleNamespace(uri="file://binary", mimeType="image/png", blob="large-base64-data"),
    ])

    assert resource_agent_view(raw, max_text_chars=4) == {
        "contents": [
            {"uri": "file://text", "mime_type": "text/plain", "text": "abcd"},
            {"uri": "file://binary", "mime_type": "image/png", "blob_available": True},
        ],
        "truncated": True,
    }
