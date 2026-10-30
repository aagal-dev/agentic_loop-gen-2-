"""Normalized MCP tool results, with a JSON-native view for the agent runtime."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _json_value(value: Any) -> Any:
    """Convert MCP/Pydantic values to JSON-native data without losing fields."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_text(text: str) -> Any:
    """Use structured JSON returned as text when a server did not set structuredContent."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


@dataclass(frozen=True)
class MCPResult:
    ok: bool
    data: Any
    text: str | None
    content: list[dict[str, Any]]
    is_error: bool
    raw: Any

    def agent_view(self) -> dict[str, Any]:
        """Small, serializable result payload to append to the next agent state."""
        return {
            "ok": self.ok,
            "result": self.data,
            "text": self.text,
            "is_error": self.is_error,
        }


def normalize(raw: Any) -> MCPResult:
    raw_content = list(getattr(raw, "content", None) or [])
    content = [_json_value(item) for item in raw_content]
    structured = getattr(raw, "structuredContent", None)
    if structured is None:
        structured = getattr(raw, "structured_content", None)

    texts = [item["text"] for item in content if item.get("type") == "text" and isinstance(item.get("text"), str)]
    text = "\n".join(texts) or None
    if structured is not None:
        data = _json_value(structured)
    elif len(texts) == 1:
        data = _parse_text(texts[0])
    elif texts:
        data = [_parse_text(item) for item in texts]
    else:
        data = content

    is_error = bool(getattr(raw, "isError", getattr(raw, "is_error", False)))
    return MCPResult(not is_error, data, text, content, is_error, raw)


def resource_agent_view(raw: Any, *, max_text_chars: int = 24_000) -> dict[str, Any]:
    """Return useful text resource content without passing binary blobs to the LLM."""
    if max_text_chars < 0:
        raise ValueError("max_text_chars cannot be negative")
    contents = []
    remaining = max_text_chars
    truncated = False
    for item in getattr(raw, "contents", None) or []:
        content = {
            "uri": str(getattr(item, "uri", "")),
            "mime_type": getattr(item, "mimeType", None) or getattr(item, "mime_type", None),
        }
        text = getattr(item, "text", None)
        if isinstance(text, str):
            content["text"] = text[:remaining]
            remaining -= len(content["text"])
            if len(text) > len(content["text"]):
                truncated = True
        elif getattr(item, "blob", None) is not None:
            content["blob_available"] = True
        contents.append(content)
    return {"contents": contents, "truncated": truncated}
