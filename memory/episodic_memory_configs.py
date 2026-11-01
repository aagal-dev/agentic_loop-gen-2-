"""Application configuration for the reusable episodic memory component."""

from memory.episodic_memory import EpisodicMemory, EpisodicCreatorResult
from integrations.base_agent import BaseAgent
from configs.settings import settings


def configure_episodic_memory() -> EpisodicMemory | None:
    """Configure episodic memory when its required providers are available."""
    if not settings.qdrant_url or not settings.cohere_api_key:
        return None

    memory = EpisodicMemory()
    creator_agent = BaseAgent(
        system_prompt=memory.creator_prompt,
        response_model=EpisodicCreatorResult,
    )
    memory.configure(
        agent=creator_agent,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        cohere_api_key=settings.cohere_api_key,
        qdrant_collection=settings.qdrant_collection,
        cohere_model=settings.cohere_embed_model,
        embedding_dimensions=settings.cohere_output_dimension,
    )
    return memory
