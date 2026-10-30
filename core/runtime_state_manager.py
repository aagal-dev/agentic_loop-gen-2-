from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    step: str
    result: str | dict[str, Any] | None = None
    status: Literal["pending", "in_progress", "completed"] = "pending"


class RuntimeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_request: str
    objective: str
    steps: list[RuntimeStep] = Field(default_factory=list)
    current_step_id: str | None = None

    @model_validator(mode="after")
    def validate_step_references(self) -> "RuntimeState":
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Runtime step IDs must be unique.")
        if self.current_step_id is not None and self.current_step_id not in step_ids:
            raise ValueError("current_step_id must reference an existing runtime step.")
        return self


RuntimeStepStatus = Literal["pending", "in_progress", "completed"]


class RuntimeStateNotInitializedError(RuntimeError):
    """Raised when a runtime state is requested before a runtime starts."""


class RuntimeStateManager:
    """Owns the lifecycle of the semantic state for one active runtime."""

    def __init__(self) -> None:
        self._state: RuntimeState | None = None
        self._completed = False

    @property
    def is_complete(self) -> bool:
        return self._completed

    def create(
        self,
        user_request: str,
        objective: str | None = None,
        steps: Sequence[RuntimeStep | Mapping[str, Any]] | None = None,
        current_step_id: str | None = None,
    ) -> RuntimeState:
        runtime_steps = [
            step if isinstance(step, RuntimeStep) else RuntimeStep.model_validate(step)
            for step in (steps or [])
        ]
        self._state = RuntimeState(
            user_request=user_request,
            objective=objective if objective is not None else user_request,
            steps=runtime_steps,
            current_step_id=current_step_id,
        )
        self._validate_current_step()
        self._completed = False
        return self.get()

    def get(self) -> RuntimeState:
        if self._state is None:
            raise RuntimeStateNotInitializedError("Runtime state has not been initialized.")
        return self._state.model_copy(deep=True)

    def update(self, state: RuntimeState | Mapping[str, Any]) -> RuntimeState:
        updated_state = state if isinstance(state, RuntimeState) else RuntimeState.model_validate(state)
        self._state = updated_state.model_copy(deep=True)
        self._validate_current_step()
        self._completed = False
        return self.get()

    def save(self, state: RuntimeState | Mapping[str, Any] | None = None) -> RuntimeState:
        """Persist the current semantic state in this active runtime manager."""
        return self.update(state) if state is not None else self.get()

    def persist(self, state: RuntimeState | Mapping[str, Any] | None = None) -> RuntimeState:
        """Alias for save, keeping persistence separate from the semantic model."""
        return self.save(state)

    def set_current_step(self, step_id: str | None) -> RuntimeState:
        state = self._require_state()
        if step_id is None:
            state.current_step_id = None
            self._state = state
            return self.get()

        step = self._find_step(step_id)
        if state.current_step_id and state.current_step_id != step_id:
            previous_step = self._find_step(state.current_step_id)
            if previous_step.status == "in_progress":
                previous_step.status = "completed"
        step.status = "in_progress"
        state.current_step_id = step_id
        self._state = state
        self._completed = False
        return self.get()

    def update_step_status(self, step_id: str, status: RuntimeStepStatus) -> RuntimeState:
        state = self._require_state()
        step = self._find_step(step_id)
        step.status = status

        if status == "in_progress":
            if state.current_step_id and state.current_step_id != step_id:
                previous_step = self._find_step(state.current_step_id)
                if previous_step.status == "in_progress":
                    previous_step.status = "completed"
            state.current_step_id = step_id
        elif state.current_step_id == step_id:
            state.current_step_id = None

        self._state = state
        self._completed = False
        return self.get()

    def update_step_result(
        self, step_id: str, result: str | dict[str, Any] | None
    ) -> RuntimeState:
        """Attach a step result without changing its status."""
        state = self._require_state()
        self._find_step(step_id).result = result
        self._state = state
        self._completed = False
        return self.get()

    def add_step(self, step: RuntimeStep | Mapping[str, Any]) -> RuntimeState:
        """Add a step, for example when a direct request invokes an MCP tool."""
        state = self._require_state()
        runtime_step = step if isinstance(step, RuntimeStep) else RuntimeStep.model_validate(step)
        if any(existing.id == runtime_step.id for existing in state.steps):
            raise ValueError(f"Runtime step ID already exists: {runtime_step.id}")
        state.steps.append(runtime_step)
        self._state = state
        self._validate_current_step()
        self._completed = False
        return self.get()

    def complete(self) -> RuntimeState:
        state = self._require_state()
        if state.current_step_id is not None:
            current_step = self._find_step(state.current_step_id)
            if current_step.status == "in_progress":
                current_step.status = "completed"
            state.current_step_id = None
        self._state = state
        self._completed = True
        return self.get()

    def _require_state(self) -> RuntimeState:
        if self._state is None:
            raise RuntimeStateNotInitializedError("Runtime state has not been initialized.")
        return self._state

    def _find_step(self, step_id: str) -> RuntimeStep:
        state = self._require_state()
        for step in state.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"Unknown runtime step: {step_id}")

    def _validate_current_step(self) -> None:
        state = self._require_state()
        if state.current_step_id is not None:
            self._find_step(state.current_step_id)
