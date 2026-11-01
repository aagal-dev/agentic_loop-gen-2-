import os
from typing import Any

import httpx


class CohereClientError(RuntimeError):
    """Raised when a Cohere embedding request fails."""


class CohereClient:
    """Small Cohere Embed v2 client using the documented REST API."""

    BASE_URL = "https://api.cohere.com/v2/embed"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-v4.0",
        output_dimensions: int = 1024,
        timeout: float = 20.0,
    ) -> None:
        self.model = model
        self.output_dimensions = output_dimensions
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.timeout = timeout

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not self.api_key:
            raise CohereClientError(
                "COHERE_API_KEY is not configured; embeddings are unavailable."
            )
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise CohereClientError("Embedding input must contain non-empty text.")

        payload: dict[str, Any] = {
            "model": self.model,
            "input_type": input_type,
            "texts": texts,
            "embedding_types": ["float"],
        }
        if self.output_dimensions is not None:
            payload["output_dimension"] = self.output_dimensions

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise CohereClientError("Cohere embedding request timed out.") from exc
        except httpx.RequestError as exc:
            raise CohereClientError(f"Cohere network error: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:1000]
            raise CohereClientError(
                f"Cohere HTTP {exc.response.status_code}: {body}"
            ) from exc
        except ValueError as exc:
            raise CohereClientError("Cohere returned invalid JSON.") from exc

        embedding_data = data.get("embeddings") if isinstance(data, dict) else None
        embeddings = (
            embedding_data.get("float")
            if isinstance(embedding_data, dict)
            else None
        )
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise CohereClientError("Unexpected embedding response from Cohere.")

        vectors: list[list[float]] = []
        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise CohereClientError("Cohere returned an empty or invalid vector.")
            try:
                parsed = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise CohereClientError("Cohere returned non-numeric vector values.") from exc
            if self.output_dimensions and len(parsed) != self.output_dimensions:
                raise CohereClientError(
                    "Embedding dimension mismatch: expected "
                    f"{self.output_dimensions}, received {len(parsed)}."
                )
            vectors.append(parsed)
        return vectors

    def embed_document(self, text: str) -> list[float]:
        return self._embed([text], input_type="search_document")[0]

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="search_query")[0]
