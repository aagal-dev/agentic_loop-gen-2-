import json
import time

from typing import Any, Generic, Optional, Type, TypeVar

import ollama

from pydantic import BaseModel, ValidationError

from configs.settings import settings


T = TypeVar("T", bound=BaseModel)


MAX_STRUCTURED_OUTPUT_ATTEMPTS = 3

MAX_REPAIR_CONTEXT_CHARS = 8000

RETRY_DELAY_SECONDS = 0.15


# ============================================================
# RESPONSE HELPERS
# ============================================================

def _read_field(value: Any, field: str) -> Any:
    """
    Read a field from either a dictionary or an object.

    Supports Ollama SDK response objects and dictionary-based
    responses used in tests.
    """

    if isinstance(value, dict):
        return value.get(field)

    return getattr(value, field, None)


def _extract_content(response: Any) -> str:
    """
    Safely extract assistant content from an Ollama response.

    Missing message/content fields are treated as empty output.
    """

    message = _read_field(response, "message")

    if message is None:
        return ""

    content = _read_field(message, "content")

    return content if isinstance(content, str) else ""


# ============================================================
# STATE SERIALIZATION
# ============================================================

def _json_default(value: Any) -> Any:
    """
    Convert non-native objects into JSON-compatible values.
    """

    if isinstance(value, BaseModel):
        return value.model_dump(
            mode="json",
            exclude_none=True,
        )

    if isinstance(value, (set, frozenset)):
        return list(value)

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def _serialize_state(state: Any) -> str:
    """
    Serialize agent state into compact JSON.

    Also removes duplicate MCP result text when the text is
    exactly equivalent to its already-normalized result.

    The original state object is never modified.
    """

    if not isinstance(state, dict):
        return str(state)

    payload = dict(state)

    tool_results = payload.get("tool_results")

    if isinstance(tool_results, list):

        compact_results = []

        for result in tool_results:

            if not isinstance(result, dict):
                compact_results.append(result)
                continue

            item = dict(result)

            result_text = item.get("text")

            if (
                isinstance(result_text, str)
                and "result" in item
            ):

                try:
                    parsed_text = json.loads(result_text)

                    if parsed_text == item["result"]:
                        item.pop("text", None)

                except (json.JSONDecodeError, TypeError):
                    pass

            compact_results.append(item)

        payload["tool_results"] = compact_results

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


# ============================================================
# JSON VALIDATION HELPERS
# ============================================================

def _escape_json_string_control_characters(content: str) -> str:
    """
    Escape literal control characters inside JSON strings.
    """

    escaped: list[str] = []

    in_string = False
    after_backslash = False

    for char in content:

        if in_string:

            if after_backslash:
                escaped.append(char)
                after_backslash = False

            elif char == "\\":
                escaped.append(char)
                after_backslash = True

            elif char == '"':
                escaped.append(char)
                in_string = False

            elif ord(char) < 0x20:
                escaped.append(f"\\u{ord(char):04x}")

            else:
                escaped.append(char)

        else:

            escaped.append(char)

            if char == '"':
                in_string = True

    return "".join(escaped)


def _validation_issues(exc: ValidationError) -> list[dict[str, str]]:
    """
    Convert Pydantic validation errors into compact diagnostics.
    """

    issues = []

    for error in exc.errors(include_url=False):

        location = ".".join(
            str(part) for part in error.get("loc", ())
        ) or "<root>"

        issues.append({
            "location": location,
            "issue": error.get("msg", "Validation failed."),
            "type": error.get("type", "validation_error"),
        })

    return issues


# ============================================================
# RESPONSE MODEL
# ============================================================

class LLMResponse(BaseModel, Generic[T]):

    success: bool

    output: Optional[str] = None

    data: Optional[T] = None

    error: Optional[str] = None


# ============================================================
# OLLAMA CLIENT
# ============================================================

class OllamaClient:

    def __init__(
        self,
        model: Optional[str] = None,
        max_token_output: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:

        self.model = (
            model
            if model is not None
            else settings.ollama_model
        )

        self.max_token_output = (
            max_token_output
            if max_token_output is not None
            else settings.ollama_max_token_output
        )

        self.temperature = (
            temperature
            if temperature is not None
            else settings.ollama_temperature
        )

        self.client = (
            ollama.Client(host=settings.ollama_host)
            if settings.ollama_host
            else ollama.Client()
        )

    # ========================================================
    # GENERATION
    # ========================================================

    def generate(
        self,
        state,
        response_model: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse[T]:

        try:

            output_format = (
                response_model.model_json_schema()
                if response_model
                else None
            )

            # Serialize state once.
            state_content = _serialize_state(state)

            base_messages = [
                {
                    "role": "system",
                    "content": system_prompt or "",
                },
                {
                    "role": "user",
                    "content": state_content,
                },
            ]

            attempts = (
                MAX_STRUCTURED_OUTPUT_ATTEMPTS
                if response_model
                else 1
            )

            last_content: Optional[str] = None

            last_issues: list[dict[str, str]] = []

            # Some Ollama models return an empty message for JSON-schema
            # constrained generation. Retry those responses in JSON mode,
            # then validate locally against the requested Pydantic model.
            use_json_mode = False

            # =================================================
            # RETRY LOOP
            # =================================================

            for attempt in range(attempts):

                messages = list(base_messages)

                # ---------------------------------------------
                # RETRY CONTEXT
                # ---------------------------------------------

                if attempt > 0:

                    time.sleep(
                        min(
                            RETRY_DELAY_SECONDS * attempt,
                            0.3,
                        )
                    )

                    if last_content:

                        previous_content = last_content[
                            :MAX_REPAIR_CONTEXT_CHARS
                        ]

                        messages.extend([
                            {
                                "role": "assistant",
                                "content": previous_content,
                            },
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response did not match "
                                    "the required JSON response schema.\n\n"
                                    f"Validation issues: "
                                    f"{json.dumps(last_issues, ensure_ascii=False)}\n\n"
                                    "Return exactly one valid JSON object "
                                    "matching the required schema. "
                                    "Do not include explanations, "
                                    "Markdown, or additional text."
                                ),
                            },
                        ])

                    else:

                        # IMPORTANT:
                        # Never append an empty assistant message.
                        # Request a fresh generation instead.

                        messages.append({
                            "role": "user",
                            "content": (
                                "Your previous generation returned "
                                "an empty response.\n\n"
                                "Generate a fresh response now. "
                                "Return exactly one valid JSON object "
                                "matching the required schema. "
                                "Do not return an empty response, "
                                "prose, or Markdown."
                            ),
                        })

                # ---------------------------------------------
                # OLLAMA REQUEST
                # ---------------------------------------------

                response = self.client.chat(

                    model=self.model,

                    messages=messages,

                    format=("json" if use_json_mode else output_format),

                    stream=False,

                    options={

                        # Deterministic retries.
                        "temperature": (
                            0.0
                            if response_model and attempt > 0
                            else self.temperature
                        ),

                        "num_predict": self.max_token_output,
                    },
                )

                content = _extract_content(response)

                # ---------------------------------------------
                # UNSTRUCTURED RESPONSE
                # ---------------------------------------------

                if response_model is None:

                    if not content.strip():

                        return LLMResponse[T](
                            success=False,
                            output=content,
                            error=(
                                "Ollama returned an empty response."
                            ),
                        )

                    return LLMResponse[T](
                        success=True,
                        output=content,
                    )

                # ---------------------------------------------
                # EMPTY RESPONSE DETECTION
                # ---------------------------------------------

                if not content.strip():

                    last_content = None
                    use_json_mode = True

                    last_issues = [{
                        "location": "<root>",
                        "issue": (
                            "Ollama returned an empty message.content. "
                            "No JSON content was generated."
                        ),
                        "type": "empty_response",
                    }]

                    done_reason = _read_field(
                        response,
                        "done_reason",
                    )

                    if done_reason:
                        last_issues[0]["done_reason"] = str(
                            done_reason
                        )

                    continue

                # ---------------------------------------------
                # JSON VALIDATION
                # ---------------------------------------------

                validated_content = content

                try:

                    data = response_model.model_validate_json(
                        content
                    )

                    return LLMResponse[T](
                        success=True,
                        output=validated_content,
                        data=data,
                    )

                except ValidationError as exc:

                    last_issues = _validation_issues(exc)

                # ---------------------------------------------
                # CONTROL CHARACTER REPAIR
                # ---------------------------------------------

                repaired_content = (
                    _escape_json_string_control_characters(
                        content
                    )
                )

                if repaired_content != content:

                    try:

                        data = response_model.model_validate_json(
                            repaired_content
                        )

                        return LLMResponse[T](
                            success=True,
                            output=repaired_content,
                            data=data,
                        )

                    except ValidationError as repair_exc:

                        last_issues = _validation_issues(
                            repair_exc
                        )

                # ---------------------------------------------
                # PREPARE NEXT RETRY
                # ---------------------------------------------

                last_content = content

            # =================================================
            # FINAL FAILURE
            # =================================================

            return LLMResponse[T](
                success=False,

                output=last_content,

                error=(
                    "Ollama did not return content matching "
                    "the required JSON response schema after "
                    f"{attempts} attempts. "

                    "Output mode: json. "

                    "Validation issues: "
                    + json.dumps(
                        last_issues,
                        ensure_ascii=False,
                    )
                ),
            )

        except Exception as exc:

            return LLMResponse[T](
                success=False,
                error=str(exc),
            )
