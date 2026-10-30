from typing import Any, Optional, Type

from pydantic import BaseModel

from integrations.ollama_client import OllamaClient


class BaseAgent:
    MAX_STATE_CHARS = 64_000

    def __init__(
        self,
        system_prompt: Optional[str],
        response_model: Type[BaseModel],
        client: Optional[OllamaClient] = None,
    ) -> None:
        if not isinstance(system_prompt, (str, type(None))):
            raise TypeError("system_prompt must be a string or None.")
        if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
            raise TypeError("response_model must be a Pydantic BaseModel class.")

        self.client = client or OllamaClient()
        self.system_prompt = system_prompt or ""
        self.response_model = response_model

    def invoke(self, state: dict[str, Any]) -> BaseModel:
        if not isinstance(state, dict):
            return self._error("Agent state must be a dictionary.")

        try:
            if len(str(state)) > self.MAX_STATE_CHARS:
                return self._error("Agent state is too large.")
        except Exception:
            return self._error("Agent state could not be serialized.")

        state = dict(state)

        try:
            response = self.client.generate(
                state=state,
                response_model=self.response_model,
                system_prompt=self.system_prompt,
            )
        except Exception as exc:
            return self._error(
                f"LLM client failure: {str(exc)[:2000] or type(exc).__name__}"
            )

        if response is None:
            return self._error("LLM client returned no response.")
        if not response.success:
            return self._error(response.error or "LLM generation failed.")
        if response.data is None:
            return self._error("LLM returned success but no structured data.")
        if not isinstance(response.data, self.response_model):
            return self._error("LLM client returned an unexpected response type.")

        return response.data

    def _error(self, message: str) -> BaseModel:
        return self.response_model.model_validate({"type": "error", "error": message})
