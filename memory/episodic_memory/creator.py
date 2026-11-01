from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import (
    ChatHistoryMessage,
    EpisodicCreatorResult,
    EpisodicMemoryCreatorState,
)


DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "episodic-memory-creator.md"


class EpisodicMemoryCreator:
    """Runs one structured-output agent for eligibility and episode extraction."""

    def __init__(self, agent: Any) -> None:
        if agent is None or not callable(getattr(agent, "invoke", None)):
            raise TypeError("agent must expose an invoke(state) method.")
        self.agent = agent

    def create(
        self,
        chat_history: list[ChatHistoryMessage],
        relevant_context: dict[str, list[dict[str, Any]]] | None = None,
    ) -> EpisodicCreatorResult:
        state: EpisodicMemoryCreatorState = {
            "chat_history": chat_history,
            "relevant_context": relevant_context
            or {
                "episodic_memory": [],
                "semantic_memory": [],
                "procedural_memory": [],
            },
        }
        try:
            response = self.agent.invoke(state)
            if isinstance(response, EpisodicCreatorResult):
                return response
            if isinstance(response, dict):
                return EpisodicCreatorResult.model_validate(response)
            if callable(getattr(response, "model_dump", None)):
                return EpisodicCreatorResult.model_validate(
                    response.model_dump(mode="python")
                )
            raise TypeError(
                "Agent invoke() must return EpisodicCreatorResult, a dict, "
                "or a Pydantic model containing that schema."
            )
        except ValidationError as exc:
            return EpisodicCreatorResult(error=f"Invalid creator response: {exc}")
        except Exception as exc:
            return EpisodicCreatorResult(
                error=f"Episodic memory creator failure: {str(exc)[:2000]}"
            )
