import json
from datetime import datetime, timezone

import pytest

from memory.episodic import (
    EligibilityDecision,
    Episode,
    EpisodeCreationResult,
    EpisodeExtraction,
    EpisodeFileStore,
    EpisodicMemory,
)


class FakeEligibility:
    def __init__(self, decision=True):
        self.decision = decision
        self.history = None
        self.context = None
        self.calls = 0

    def evaluate(self, chat_history, relevant_context):
        self.calls += 1
        self.history = chat_history
        self.context = relevant_context
        return EligibilityDecision(should_create=self.decision, reason="durable detail")


class FakeExtractor:
    def __init__(self, episodes=None):
        self.episodes = episodes or [
            Episode(
                type="user_preference",
                summary="The user prefers concise technical explanations.",
                state="No preference recorded before this conversation.",
                change="The user stated a preference for concise technical explanations.",
                importance=0.8,
                confidence=0.97,
                tags=["communication", "preference"],
                evidence=[
                    {
                        "role": "user",
                        "content": "Please keep technical explanations concise.",
                    }
                ],
            )
        ]
        self.history = None
        self.context = None
        self.calls = 0

    def extract(self, chat_history, relevant_context):
        self.calls += 1
        self.history = chat_history
        self.context = relevant_context
        return EpisodeExtraction(episodes=self.episodes)


class FakeEmbedder:
    def __init__(self):
        self.texts = []

    def embed_document(self, text):
        self.texts.append(text)
        return [0.25, 0.5, 0.75]


class FakeVectorDatabase:
    def __init__(self):
        self.upserts = []

    def upsert(self, item_id, vector, payload):
        self.upserts.append((item_id, list(vector), dict(payload)))


def make_memory(tmp_path, *, eligible=True, episodes=None, vector_db=None):
    eligibility = FakeEligibility(eligible)
    extractor = FakeExtractor(episodes)
    embedder = FakeEmbedder()
    vector_db = vector_db or FakeVectorDatabase()
    memory = EpisodicMemory(
        eligibility_agent=eligibility,
        extractor_agent=extractor,
        embedding_model=embedder,
        vector_database=vector_db,
        episodes_dir=tmp_path / "episodes",
    )
    return memory, eligibility, extractor, embedder, vector_db


def test_create_episode_runs_both_agents_and_indexes_summary(tmp_path):
    memory, eligibility, extractor, embedder, vectors = make_memory(tmp_path)
    history = [
        {"role": "user", "content": f"message {index}", "metadata": {"n": index}}
        for index in range(35)
    ]
    history[12]["resource_read"] = {"server": "docs", "uri": "file://guide"}
    context = {"episodic_memory": [{"id": "prior-episode"}]}

    result = memory.create_episode(
        history,
        relevant_context=context,
        timestamp=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )

    assert result.status == "created"
    assert result.created_any is True
    assert eligibility.history == history
    assert extractor.history == history
    assert eligibility.context == context == extractor.context
    assert len(history) == 35
    assert embedder.texts == [result.records[0].episode.summary]
    assert len(vectors.upserts) == 1
    record_id, vector, payload = vectors.upserts[0]
    assert record_id == result.records[0].id
    assert vector == [0.25, 0.5, 0.75]
    assert payload["episode_id"] == record_id
    assert payload["episode"]["summary"] == embedder.texts[0]
    assert not hasattr(memory, "search")
    assert not hasattr(memory, "retrieve_context")

    path = tmp_path / "episodes" / "ep-2026-10-30.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert len(document) == 1
    assert set(document[0]) == {"id", "timestamp", "action", "target_id", "episode"}
    assert document[0]["action"] == "create"
    assert document[0]["target_id"] is None
    assert document[0]["episode"]["type"] == "user_preference"
    assert document[0]["episode"]["evidence"][0]["role"] == "user"


def test_ineligible_history_skips_extractor_embedding_and_persistence(tmp_path):
    memory, eligibility, extractor, embedder, vectors = make_memory(
        tmp_path, eligible=False
    )

    result = memory.create_episode([{"role": "user", "content": "Hi."}])

    assert result.status == "skipped"
    assert result.should_create is False
    assert result.reason == "durable detail"
    assert eligibility.calls == 1
    assert extractor.calls == 0
    assert embedder.texts == []
    assert vectors.upserts == []
    assert not (tmp_path / "episodes").exists()


def test_chat_history_snapshot_is_unwrapped_without_dropping_message_fields(tmp_path):
    memory, eligibility, extractor, _, _ = make_memory(tmp_path)
    snapshot = {
        "messages": [
            {"role": "assistant", "content": "", "tool_call": {"name": "lookup"}},
            {"role": "user", "content": "Remember my preference."},
        ]
    }

    memory.create_episode(snapshot)

    assert eligibility.history == snapshot["messages"]
    assert extractor.history == snapshot["messages"]
    assert eligibility.history[0]["tool_call"] == {"name": "lookup"}


def test_agent_input_mutation_cannot_change_history_seen_by_next_stage(tmp_path):
    memory, _, extractor, _, _ = make_memory(tmp_path)

    class MutatingEligibility:
        def evaluate(self, chat_history, relevant_context):
            chat_history[0]["content"] = "mutated by eligibility agent"
            relevant_context["episodic_memory"].append({"id": "mutated"})
            return EligibilityDecision(should_create=True)

    memory.eligibility_agent = MutatingEligibility()
    original = [{"role": "user", "content": "Keep the history unchanged."}]
    memory.create_episode(original, relevant_context={"episodic_memory": []})

    assert original[0]["content"] == "Keep the history unchanged."
    assert extractor.history == original
    assert extractor.context == {"episodic_memory": []}


def test_empty_history_and_extraction_without_episodes_are_clean_skips(tmp_path):
    memory, eligibility, extractor, _, _ = make_memory(tmp_path)
    empty_result = memory.create_episode([])
    assert empty_result.status == "skipped"
    assert eligibility.calls == 0

    extractor.episodes = []
    result = memory.create_episode([{"role": "user", "content": "History"}])
    assert result.status == "skipped"
    assert result.should_create is True
    assert result.records == []


def test_vector_failure_is_reported_after_file_is_saved_and_rebuild_retries(tmp_path):
    class FlakyVectorDatabase(FakeVectorDatabase):
        fail = True

        def upsert(self, item_id, vector, payload):
            if self.fail:
                raise RuntimeError("vector service offline")
            super().upsert(item_id, vector, payload)

    vectors = FlakyVectorDatabase()
    memory, _, _, _, _ = make_memory(tmp_path, vector_db=vectors)
    history = [{"role": "user", "content": "Please remember this preference."}]

    first = memory.create_episode(history)
    assert first.status == "created"
    assert first.records
    assert "vector service offline" in first.indexing_errors[0]
    assert len(memory.store.list_active()) == 1

    vectors.fail = False
    assert memory.rebuild_index() == []
    assert len(vectors.upserts) == 1


def test_agent_or_persistence_errors_are_structured(tmp_path):
    class BrokenEligibility:
        def evaluate(self, chat_history, relevant_context):
            raise RuntimeError("provider unavailable")

    memory, _, _, embedder, vectors = make_memory(tmp_path)
    memory.eligibility_agent = BrokenEligibility()
    result = memory.create_episode([{"role": "user", "content": "A fact"}])
    assert result.status == "failed"
    assert "provider unavailable" in result.error
    assert embedder.texts == []
    assert vectors.upserts == []

    broken_store = EpisodeFileStore(tmp_path / "episodes")
    created_at = datetime.now(timezone.utc)
    broken_path = broken_store.path_for_day(created_at.date())
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_text("not json", encoding="utf-8")
    memory.store = broken_store
    memory.eligibility_agent = FakeEligibility()
    persistence_result = memory.create_episode(
        [{"role": "user", "content": "A supported durable fact."}],
        timestamp=created_at,
    )
    assert persistence_result.status == "failed"
    assert "Episode file persistence failed" in persistence_result.error


def test_pydantic_contracts_reject_invalid_episode_scores():
    with pytest.raises(ValueError):
        Episode(type="event", summary="Bad", confidence=1.5)
    with pytest.raises(ValueError):
        EpisodeExtraction(type="error")
