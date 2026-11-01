from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .episodic_memory import EpisodeRecord, EpisodicEpisode


class MemoryRetrievalResultItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
    source: Literal["hybrid", "dense", "sparse"] = "hybrid"
    episode: EpisodicEpisode

    def as_context(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "source": self.source,
            **self.episode.model_dump(),
        }


class MemoryRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[MemoryRetrievalResultItem] = Field(default_factory=list)
    error: str | None = None

    def as_context(self) -> list[dict[str, Any]]:
        return [result.as_context() for result in self.results]


class EpisodicConsolidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[EpisodeRecord] = Field(default_factory=list)
    error: str | None = None
    indexing_errors: list[str] = Field(default_factory=list)

    @property
    def created_any(self) -> bool:
        return bool(self.records)


class MemoryContextResult(BaseModel):
    """Context-ready episodic memories returned by retrieve_context()."""

    model_config = ConfigDict(extra="forbid")

    query: str
    context: str = ""
    memories: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
