import io
import json
from urllib.error import HTTPError
from urllib.request import Request
from uuid import UUID

import pytest

from memory.episodic import CohereEmbeddingModel, QdrantVectorDatabase
from memory.episodic import providers


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_cohere_posts_embedding_request_over_http(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return Response({"embeddings": {"float": [[0.1, 0.2]]}})

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    embedder = CohereEmbeddingModel(
        "test-key", model="embed-v4.0", output_dimension=2
    )

    assert embedder.embed_document("episode summary") == [0.1, 0.2]
    request, timeout = calls[0]
    assert isinstance(request, Request)
    assert request.full_url == "https://api.cohere.com/v2/embed"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert json.loads(request.data) == {
        "texts": ["episode summary"],
        "model": "embed-v4.0",
        "input_type": "search_document",
        "embedding_types": ["float"],
        "output_dimension": 2,
    }
    assert timeout == 30


def test_qdrant_creates_collection_and_upserts_over_http(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        if request.get_method() == "GET":
            raise HTTPError(request.full_url, 404, "missing", {}, io.BytesIO(b"{}"))
        return Response({"result": {"status": "ok"}})

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    vector_db = QdrantVectorDatabase(
        "https://qdrant.example/", collection_name="episode test", vector_size=2,
        api_key="qdrant-key",
    )
    vector_db.upsert("episode-1", [0.1, 0.2], {"summary": "a preference"})

    assert [request.get_method() for request in calls] == ["GET", "PUT", "PUT"]
    assert calls[0].full_url == "https://qdrant.example/collections/episode%20test"
    assert calls[0].get_header("Api-key") == "qdrant-key"
    assert json.loads(calls[1].data) == {
        "vectors": {"size": 2, "distance": "Cosine"}
    }
    assert calls[2].full_url.endswith("/points?wait=true")
    point = json.loads(calls[2].data)["points"][0]
    assert UUID(point["id"])
    assert point["vector"] == [0.1, 0.2]
    assert point["payload"] == {"summary": "a preference"}
    with pytest.raises(ValueError, match="dimension"):
        vector_db.upsert("episode-2", [0.1], {})


def test_qdrant_checks_existing_collection_dimension(monkeypatch):
    def fake_urlopen(request, timeout):
        return Response(
            {"result": {"config": {"params": {"vectors": {"size": 3}}}}}
        )

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="expected 2"):
        QdrantVectorDatabase(
            "https://qdrant.example", collection_name="episodes", vector_size=2
        )


def test_qdrant_uses_single_named_vector_from_existing_collection(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        if request.get_method() == "GET":
            return Response(
                {
                    "result": {
                        "config": {
                            "params": {
                                "vectors": {"abstract-dense-vector": {"size": 2}}
                            }
                        }
                    }
                }
            )
        return Response({"result": {"status": "ok"}})

    monkeypatch.setattr(providers, "urlopen", fake_urlopen)
    vector_db = QdrantVectorDatabase(
        "https://qdrant.example", collection_name="episodes", vector_size=2
    )
    vector_db.upsert("episode-1", [0.1, 0.2], {"summary": "test"})

    assert vector_db.vector_name == "abstract-dense-vector"
    point = json.loads(calls[-1].data)["points"][0]
    assert point["vector"] == {"abstract-dense-vector": [0.1, 0.2]}


def test_http_provider_configuration_comes_from_environment(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "cohere-key")
    monkeypatch.setenv("COHERE_EMBED_MODEL", "embed-v4.0")
    monkeypatch.setenv("COHERE_OUTPUT_DIMENSION", "2")
    monkeypatch.setenv("COHERE_EMBED_URL", "https://cohere.example/embed")
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-key")
    monkeypatch.setenv("QDRANT_COLLECTION", "episode-test")
    monkeypatch.setenv("QDRANT_VECTOR_NAME", "vector-test")

    embedder = CohereEmbeddingModel.from_env()
    assert embedder.api_key == "cohere-key"
    assert embedder.url == "https://cohere.example/embed"
    assert embedder.model == "embed-v4.0"
    assert embedder.output_dimension == 2

    monkeypatch.setattr(
        providers,
        "urlopen",
        lambda request, timeout: Response(
            {
                "result": {
                    "config": {
                        "params": {"vectors": {"vector-test": {"size": 2}}}
                    }
                }
            }
        ),
    )
    vector_db = QdrantVectorDatabase.from_env(vector_size=2)
    assert vector_db.url == "https://qdrant.example"
    assert vector_db.api_key == "qdrant-key"
    assert vector_db.collection_name == "episode-test"
    assert vector_db.vector_name == "vector-test"


def test_cohere_rejects_wrong_dimension_and_invalid_response(monkeypatch):
    monkeypatch.setattr(
        providers, "urlopen", lambda request, timeout: Response({"embeddings": {"float": [[1.0]]}})
    )
    with pytest.raises(RuntimeError, match="dimension"):
        CohereEmbeddingModel("key", output_dimension=2).embed_document("text")

    monkeypatch.setattr(providers, "urlopen", lambda request, timeout: Response({}))
    with pytest.raises(RuntimeError, match="did not contain"):
        CohereEmbeddingModel("key", output_dimension=2).embed_document("text")
