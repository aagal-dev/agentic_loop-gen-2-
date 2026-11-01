from datetime import datetime
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatHistoryMessage(TypedDict):
    role: str
    content: str


class EpisodicMemoryCreatorState(TypedDict):
    chat_history: list[ChatHistoryMessage]
    relevant_context: dict[str, list[dict[str, Any]]]


class EpisodicEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    summary: str
    state: str | None = None
    change: str | None = None
    outcome: str | None = None
    learned_lesson: str | None = None
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
    event_time: str | None = None
    evidence: list[str] = Field(default_factory=list)
    action: Literal["create", "update", "delete"] = "create"
    target_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_evidence(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        evidence = value.get("evidence")
        if not isinstance(evidence, list):
            return value

        normalized: list[Any] = []
        changed = False
        for item in evidence:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                role = item.get("role")
                if isinstance(role, str) and role.strip():
                    normalized.append(f"{role}: {item['content']}")
                else:
                    normalized.append(item["content"])
                changed = True
            else:
                normalized.append(item)

        if not changed:
            return value

        normalized_value = dict(value)
        normalized_value["evidence"] = normalized
        return normalized_value

    @model_validator(mode="after")
    def validate_summary(self) -> "EpisodicEpisode":
        if self.action != "delete" and not self.summary.strip():
            raise ValueError("Episode summary must not be empty.")
        if self.action == "create" and self.target_id:
            raise ValueError("Create actions must not contain target_id.")
        if self.action in {"update", "delete"} and not self.target_id:
            raise ValueError(f"{self.action.title()} actions require target_id.")
        return self


class EpisodicCreatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_create: bool = False
    episodes: list[EpisodicEpisode] = Field(default_factory=list)
    error: Optional[str] = None


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    timestamp: datetime
    action: Literal["create", "update", "delete"]
    target_id: str | None = None
    episode: EpisodicEpisode | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "EpisodeRecord":
        if self.action in {"create", "update"} and self.episode is None:
            raise ValueError(
                f"Episode action '{self.action}' requires an episode payload."
            )
        if self.action in {"update", "delete"} and not self.target_id:
            raise ValueError(
                f"Episode action '{self.action}' requires target_id."
            )
        if self.action == "delete" and self.episode is not None:
            raise ValueError("Delete records must not contain an episode payload.")
        return self
