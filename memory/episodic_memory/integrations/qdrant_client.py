import json
import os
import time
from typing import Any
from urllib.parse import quote

import httpx


class QdrantClientError(RuntimeError):
    """Raised when the Qdrant REST API request fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class QdrantClient:
    """Small HTTP-only client for Qdrant collection, hybrid search, and indexing."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str = "episodic_memory",
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "text",
        sparse_model: str = "qdrant/bm25",
        timeout: float = 20.0,
        max_retries: int = 2,
        retry_delay: float = 0.25,
    ) -> None:
        self.url = (url or os.getenv("QDRANT_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection = collection
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.sparse_model = sparse_model
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)

    def ensure_collection(self, dense_size: int) -> None:
        if not self.url:
            raise QdrantClientError("QDRANT_URL is not configured.")
        if dense_size <= 0:
            raise QdrantClientError("Dense vector size must be positive.")

        path = f"/collections/{quote(self.collection, safe='')}"
        try:
            self._request("GET", path)
            return
        except QdrantClientError as exc:
            if exc.status_code != 404:
                raise

        try:
            self._request(
                "PUT",
                path,
                {
                    "vectors": {
                        self.dense_vector_name: {
                            "size": dense_size,
                            "distance": "Cosine",
                        }
                    },
                    "sparse_vectors": {
                        self.sparse_vector_name: {"modifier": "idf"}
                    },
                },
            )
        except QdrantClientError as exc:
            # Another worker may have created the collection after our GET.
            if exc.status_code != 409:
                raise

    def upsert_episode(
        self,
        episode_id: str,
        summary: str,
        dense_vector: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_collection(len(dense_vector))
        point_payload = {
            "episode_id": episode_id,
            "summary": summary,
            **(payload or {}),
        }
        self._request(
            "PUT",
            f"/collections/{quote(self.collection, safe='')}/points?wait=true",
            {
                "points": [
                    {
                        "id": episode_id,
                        "vector": {
                            self.dense_vector_name: dense_vector,
                            self.sparse_vector_name: {
                                "text": summary,
                                "model": self.sparse_model,
                            },
                        },
                        "payload": point_payload,
                    }
                ]
            },
        )

    def delete_episode(self, episode_id: str) -> None:
        self._ensure_configured()
        self._request(
            "POST",
            f"/collections/{quote(self.collection, safe='')}/points/delete?wait=true",
            {"points": [episode_id]},
        )

    def hybrid_search(
        self,
        query: str,
        dense_vector: list[float],
        limit: int,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_collection(len(dense_vector))
        prefetch_limit = max(limit * 3, limit)
        payload: dict[str, Any] = {
            "prefetch": [
                {
                    "query": dense_vector,
                    "using": self.dense_vector_name,
                    "limit": prefetch_limit,
                },
                {
                    "query": {"text": query, "model": self.sparse_model},
                    "using": self.sparse_vector_name,
                    "limit": prefetch_limit,
                },
            ],
            # Current Qdrant Query API syntax for Reciprocal Rank Fusion.
            "query": {"rrf": {}},
            "limit": limit,
            "with_payload": True,
        }
        if query_filter:
            payload["filter"] = query_filter

        response = self._request(
            "POST",
            f"/collections/{quote(self.collection, safe='')}/points/query",
            payload,
        )
        result = response.get("result")
        if isinstance(result, dict):
            result = result.get("points")
        if not isinstance(result, list):
            raise QdrantClientError("Qdrant returned an invalid points response.")
        return [point for point in result if isinstance(point, dict)]

    def _ensure_configured(self) -> None:
        if not self.url:
            raise QdrantClientError("QDRANT_URL is not configured.")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_configured()
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(
                        method,
                        f"{self.url}{path}",
                        headers=headers,
                        json=payload,
                    )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                response.raise_for_status()
                parsed = response.json() if response.content else {}
                if not isinstance(parsed, dict):
                    raise QdrantClientError("Qdrant returned non-object JSON.")
                return parsed
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                body = exc.response.text[:1000]
                if status in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    last_error = exc
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise QdrantClientError(
                    f"Qdrant HTTP {status}: {body}", status_code=status
                ) from exc
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise QdrantClientError("Qdrant request timed out.") from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise QdrantClientError(f"Qdrant network failure: {exc}") from exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise QdrantClientError("Qdrant returned invalid JSON.") from exc

        raise QdrantClientError(f"Qdrant request failed: {last_error}")
