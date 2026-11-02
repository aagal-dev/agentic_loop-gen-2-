from typing import Generic, Optional, Type, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from configs.settings import settings

T = TypeVar("T", bound=BaseModel)
MAX_STRUCTURED_OUTPUT_ATTEMPTS = 3


def _escape_json_string_control_characters(content: str) -> str:
    """Escape literal control characters inside JSON strings from model output."""
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


class LLMResponse(BaseModel, Generic[T]):
    success: bool
    output: Optional[str] = None
    data: Optional[T] = None
    error: Optional[str] = None


class OllamaClient:
    def __init__(
        self,
        model: Optional[str] = None,
        max_token_output: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.model = model if model is not None else settings.ollama_model
        self.max_token_output = (
            max_token_output
            if max_token_output is not None
            else settings.ollama_max_token_output
        )
        self.temperature = (
            temperature if temperature is not None else settings.ollama_temperature
        )
        self.client = (
            ollama.Client(host=settings.ollama_host)
            if settings.ollama_host
            else ollama.Client()
        )

    def generate(
        self,
        state,
        response_model: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse[T]:
        try:
            output_format = response_model.model_json_schema() if response_model else None
            messages = [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": str(state)},
            ]
            attempts = MAX_STRUCTURED_OUTPUT_ATTEMPTS if response_model else 1

            for attempt in range(attempts):
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    format=output_format,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_token_output,
                    },
                )
                content = response["message"]["content"]

                if response_model is None:
                    return LLMResponse[T](success=True, output=content)

                try:
                    data = response_model.model_validate_json(content)
                except ValidationError:
                    repaired_content = _escape_json_string_control_characters(content)
                    if repaired_content != content:
                        try:
                            data = response_model.model_validate_json(repaired_content)
                        except ValidationError:
                            data = None
                    else:
                        data = None

                if data is not None:
                    return LLMResponse[T](success=True, output=content, data=data)

                if attempt + 1 < attempts:
                    messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous reply did not match the required JSON schema. "
                                    "Return only one valid JSON object matching the schema. "
                                    "Do not include prose, markdown fences, or analysis."
                                ),
                            },
                        ]
                    )

            return LLMResponse[T](
                success=False,
                output=content,
                error=(
                    "Ollama did not return content matching the required JSON response "
                    f"schema after {attempts} attempts."
                ),
            )
        except Exception as exc:
            return LLMResponse[T](success=False, error=str(exc))
