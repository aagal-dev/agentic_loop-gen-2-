import pytest

from core.runtime_state import RuntimeStateManager, RuntimeStateNotInitializedError


def test_runtime_transitions_and_records_step_result():
    runtime = RuntimeStateManager()
    created = runtime.create(
        "Research the question", steps=["Find sources", "Summarize findings"], automated_id=True
    )

    assert created.current_step_id == 1
    assert [step.id for step in created.steps] == [1, 2]

    runtime.complete_step({"sources": 3})
    runtime.start_next_step()
    runtime.complete_step("Summary")
    completed = runtime.complete()

    assert completed.status == "completed"
    assert runtime.is_complete
    assert completed.steps[0].result == {"sources": 3}


def test_failed_step_can_be_retried():
    runtime = RuntimeStateManager()
    runtime.create("Do task", steps=["Run task"], automated_id=True)

    runtime.fail_step("temporary failure")
    assert runtime.agent_context()["step_results"] == [
        {"id": "001", "status": "failed", "error": "temporary failure"}
    ]

    runtime.start_step(1)
    assert runtime.get().steps[0].status == "in_progress"
    assert runtime.get().steps[0].error is None


def test_external_state_is_copied_and_restore_is_validated():
    runtime = RuntimeStateManager()
    runtime.create("Do task", steps=["One"], automated_id=True)
    snapshot = runtime.snapshot()

    exposed = runtime.get()
    exposed.steps[0].description = "changed outside manager"
    assert runtime.get().steps[0].description == "One"

    restored = RuntimeStateManager()
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot


def test_complete_rejects_unfinished_steps():
    runtime = RuntimeStateManager()
    runtime.create("Do task", steps=["One"], automated_id=True)

    with pytest.raises(RuntimeError, match="current step"):
        runtime.complete()


def test_access_before_create_is_rejected():
    with pytest.raises(RuntimeStateNotInitializedError):
        RuntimeStateManager().snapshot()
