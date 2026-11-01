from typing import Any

from memory.episodic_memory.integrations import CohereClient, QdrantClient
from memory.episodic_memory.schemas import (
    EpisodeRecord,
    MemoryRetrievalResult,
    MemoryRetrievalResultItem,
)
from memory.episodic_memory.storage import EpisodicMemoryStore


class MemoryRetrieval:
    """Owns embedding, Qdrant hybrid search, indexing, and canonical lookup."""

    def __init__(
        self,
        cohere_client: CohereClient,
        qdrant_client: QdrantClient,
        store: EpisodicMemoryStore,
        scope: str = "default",
    ) -> None:
        self.cohere = cohere_client
        self.qdrant = qdrant_client
        self.store = store
        self.scope = scope

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        query_filter: dict[str, Any] | None = None,
    ) -> MemoryRetrievalResult:
        if not isinstance(query, str) or not query.strip():
            return MemoryRetrievalResult(query="", error="Query must be a non-empty string.")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            return MemoryRetrievalResult(query=query, error="Limit must be a positive integer.")

        try:
            dense_vector = self.cohere.embed_query(query)
            combined_filter = self._combine_filters(query_filter)
            points = self.qdrant.hybrid_search(
                query=query,
                dense_vector=dense_vector,
                limit=limit,
                query_filter=combined_filter,
            )
            ids = [self._point_id(point) for point in points]
            canonical = self.store.get_active_by_ids(
                [point_id for point_id in ids if point_id]
            )
            results: list[MemoryRetrievalResultItem] = []
            for point, point_id in zip(points, ids):
                if point_id is None:
                    continue
                record = canonical.get(point_id)
                if record is None or record.episode is None:
                    continue
                results.append(
                    MemoryRetrievalResultItem(
                        id=record.id,
                        score=self._score(point),
                        source="hybrid",
                        episode=record.episode,
                    )
                )
            return MemoryRetrievalResult(query=query, results=results)
        except Exception as exc:
            return MemoryRetrievalResult(
                query=query,
                error=f"Memory retrieval failed: {str(exc)[:2000]}",
            )

    def index_record(self, record: EpisodeRecord) -> str | None:
        """Index after canonical persistence; indexing failure is recoverable."""
        try:
            target_id = record.target_id or record.id
            if record.action == "delete":
                self.qdrant.delete_episode(target_id)
                return None
            if record.episode is None:
                return "Episode indexing skipped because the payload is missing."

            embedding_text = self._embedding_text(record.episode)
            dense_vector = self.cohere.embed_document(embedding_text)
            self.qdrant.upsert_episode(
                episode_id=target_id,
                summary=embedding_text,
                dense_vector=dense_vector,
                payload={
                    "type": record.episode.type,
                    "tags": record.episode.tags,
                    "scope": self.scope,
                },
            )
            return None
        except Exception as exc:
            return f"Episode {record.id} was stored but not indexed: {str(exc)[:2000]}"

    def rebuild_index(self) -> list[str]:
        """Re-create the derived Qdrant index from canonical lifecycle records."""
        errors: list[str] = []
        # Retry previously failed deletions as well as re-indexing active episodes.
        deleted_ids = {
            record.target_id
            for record in self.store.list_records()
            if record.action == "delete" and record.target_id
        }
        for episode_id in deleted_ids:
            try:
                self.qdrant.delete_episode(episode_id)
            except Exception as exc:
                errors.append(f"Could not remove deleted episode {episode_id}: {str(exc)[:1000]}")
        for record in self.store.list_active():
            error = self.index_record(record)
            if error:
                errors.append(error)
        return errors

    def _combine_filters(
        self,
        query_filter: dict[str, Any] | None,
    ) -> dict[str, Any]:
        scope_condition = {"key": "scope", "match": {"value": self.scope}}
        if not query_filter:
            return {"must": [scope_condition]}
        if not isinstance(query_filter, dict):
            raise TypeError("query_filter must be a Qdrant filter dictionary.")
        combined = dict(query_filter)
        existing_must = combined.get("must", [])
        if existing_must is None:
            existing_must = []
        if not isinstance(existing_must, list):
            raise TypeError("query_filter['must'] must be a list of Qdrant conditions.")
        combined["must"] = [*existing_must, scope_condition]
        return combined

    @staticmethod
    def _embedding_text(episode: Any) -> str:
        fields = [
            f"Type: {episode.type}",
            f"Summary: {episode.summary}",
        ]
        for label, value in (
            ("State", episode.state),
            ("Change", episode.change),
            ("Outcome", episode.outcome),
            ("Lesson", episode.learned_lesson),
            ("Event time", episode.event_time),
        ):
            if value:
                fields.append(f"{label}: {value}")
        if episode.tags:
            fields.append("Tags: " + ", ".join(episode.tags))
        return "\n".join(fields)

    @staticmethod
    def _point_id(point: dict[str, Any]) -> str | None:
        point_id = point.get("id")
        if point_id is None:
            payload = point.get("payload")
            if isinstance(payload, dict):
                point_id = payload.get("episode_id")
        return str(point_id) if point_id is not None else None

    @staticmethod
    def _score(point: dict[str, Any]) -> float:
        try:
            return float(point.get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0
