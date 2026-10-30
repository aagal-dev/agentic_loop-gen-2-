"""Small, reusable runtime-state manager for agentic systems.

The manager owns one active runtime. Returned state is always a deep copy,
so callers cannot accidentally mutate the manager's internal state.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

StepStatus = Literal["pending", "in_progress", "completed", "failed"]
RuntimeStatus = Literal["active", "completed"]


class RuntimeStep(BaseModel):
    """One actionable item in a runtime plan."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    description: str = Field(min_length=1)
    status: StepStatus = "pending"
    result: JsonValue = None
    error: str | None = None


class RuntimeState(BaseModel):
    """Serializable semantic state for one agent run."""

    model_config = ConfigDict(extra="forbid")

    user_request: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    steps: list[RuntimeStep] = Field(default_factory=list)
    current_step_id: int | None = None
    status: RuntimeStatus = "active"
    automated_id: bool = False

    @model_validator(mode="after")
    def validate_consistency(self) -> "RuntimeState":
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Runtime step IDs must be unique.")

        in_progress = [step for step in self.steps if step.status == "in_progress"]
        if len(in_progress) > 1:
            raise ValueError("Only one runtime step can be in progress.")

        if self.current_step_id is None:
            if in_progress:
                raise ValueError("current_step_id must reference the in-progress step.")
        else:
            current = next((step for step in self.steps if step.id == self.current_step_id), None)
            if current is None:
                raise ValueError("current_step_id must reference an existing runtime step.")
            if current.status != "in_progress":
                raise ValueError("The current runtime step must be in progress.")

        if self.status == "completed" and any(step.status != "completed" for step in self.steps):
            raise ValueError("A runtime can be completed only when every step is completed.")

        return self


class RuntimeStateNotInitializedError(RuntimeError):
    """Raised when runtime state is accessed before a runtime starts."""


class RuntimeStateManager:
    """Own state transitions and provide a compact view for agent context."""

    def __init__(self) -> None:
        self._state: RuntimeState | None = None

    @property
    def is_complete(self) -> bool:
        return self._state is not None and self._state.status == "completed"

    def create(
        self,
        user_request: str,
        objective: str | None = None,
        steps: Sequence[str | RuntimeStep | Mapping[str, Any]] | None = None,
        current_step_id: int | None = None,
        automated_id: bool = False,
    ) -> RuntimeState:
        """Start a runtime; optionally generate integer step IDs 1, 2, 3, ... ."""
        request = user_request.strip()
        target_objective = (objective if objective is not None else user_request).strip()
        if not request:
            raise ValueError("user_request cannot be empty.")
        if not target_objective:
            raise ValueError("objective cannot be empty.")

        runtime_steps: list[RuntimeStep] = []
        for step_input in steps or []:
            if automated_id:
                description = self._step_description(step_input)
                runtime_steps.append(
                    RuntimeStep(
                        id=self._next_step_id(runtime_steps),
                        description=description,
                    )
                )
            else:
                runtime_steps.append(self._validate_manual_step(step_input))

        if current_step_id is None:
            active_steps = [step for step in runtime_steps if step.status == "in_progress"]
            if active_steps:
                current_step_id = active_steps[0].id
            else:
                next_step = next((step for step in runtime_steps if step.status == "pending"), None)
                if next_step is not None:
                    next_step.status = "in_progress"
                    current_step_id = next_step.id
        else:
            current_step = next(
                (step for step in runtime_steps if step.id == current_step_id), None
            )
            if current_step is not None:
                if current_step.status == "pending":
                    current_step.status = "in_progress"
                elif current_step.status != "in_progress":
                    raise ValueError("The requested current step must be pending or in progress.")

        state = RuntimeState(
            user_request=request,
            objective=target_objective,
            steps=runtime_steps,
            current_step_id=current_step_id,
            status="active",
            automated_id=automated_id,
        )
        self._state = state
        return self.get()

    def get(self) -> RuntimeState:
        """Return a deep copy; edits to it never mutate the managed state."""
        return self._require_state()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-ready snapshot, including the ID-generation setting."""
        return self.get().model_dump(mode="json")

    def restore(self, snapshot: RuntimeState | Mapping[str, Any]) -> RuntimeState:
        """Restore and validate a previously captured snapshot."""
        data = snapshot.model_dump(mode="python") if isinstance(snapshot, RuntimeState) else snapshot
        restored = RuntimeState.model_validate(data)
        self._state = restored
        return self.get()

    def agent_context(self, max_results: int = 3) -> dict[str, Any]:
        """Build a compact, JSON-ready state view intended for an agent prompt."""
        if max_results < 0:
            raise ValueError("max_results cannot be negative.")

        state = self._require_state()
        current = self._find_step(state, state.current_step_id) if state.current_step_id else None
        finished_with_output = [
            step
            for step in state.steps
            if step.status in ("completed", "failed")
            and (step.result is not None or step.error is not None)
        ]
        selected_results = finished_with_output[-max_results:] if max_results else []

        return {
            "user_request": state.user_request,
            "objective": state.objective,
            "runtime_status": state.status,
            "current_step": (
                {"id": self._format_step_id(current.id), "description": current.description}
                if current
                else None
            ),
            "plan": [
                {
                    "id": self._format_step_id(step.id),
                    "description": step.description,
                    "status": step.status,
                }
                for step in state.steps
            ],
            "step_results": [
                {
                    "id": self._format_step_id(step.id),
                    "status": step.status,
                    **({"result": step.result} if step.result is not None else {}),
                    **({"error": step.error} if step.error is not None else {}),
                }
                for step in selected_results
            ],
        }

    def start_step(self, step_id: int) -> RuntimeState:
        """Start a pending step, or retry a failed step."""
        state = self._require_state()
        self._ensure_active(state)

        if state.current_step_id == step_id:
            return self.get()  # Idempotent when the requested step is already active.
        if state.current_step_id is not None:
            raise RuntimeError("Complete or fail the current step before starting another.")

        step = self._find_step(state, step_id)
        if step.status == "completed":
            raise ValueError(f"Runtime step is already completed: {step_id}")

        step.status = "in_progress"
        step.result = None
        step.error = None
        state.current_step_id = step_id
        return self._commit(state)

    def start_next_step(self) -> RuntimeState:
        """Start the first pending step; leave state unchanged if none are pending."""
        state = self._require_state()
        self._ensure_active(state)
        if state.current_step_id is not None:
            return self.get()

        next_step = next((step for step in state.steps if step.status == "pending"), None)
        if next_step is None:
            return self.get()

        next_step.status = "in_progress"
        state.current_step_id = next_step.id
        return self._commit(state)

    def complete_step(self, result: JsonValue = None) -> RuntimeState:
        """Complete the current step and optionally store its JSON-compatible result."""
        state = self._require_state()
        self._ensure_active(state)
        if state.current_step_id is None:
            raise RuntimeError("There is no current step to complete.")

        step = self._find_step(state, state.current_step_id)
        step.status = "completed"
        step.result = result
        step.error = None
        state.current_step_id = None
        return self._commit(state)

    def fail_step(self, error: str) -> RuntimeState:
        """Mark the current step as failed, preserving the error for agent context."""
        state = self._require_state()
        self._ensure_active(state)
        if state.current_step_id is None:
            raise RuntimeError("There is no current step to fail.")
        if not error.strip():
            raise ValueError("error cannot be empty.")

        step = self._find_step(state, state.current_step_id)
        step.status = "failed"
        step.result = None
        step.error = error.strip()
        state.current_step_id = None
        return self._commit(state)

    def add_step(self, step: str | RuntimeStep | Mapping[str, Any]) -> RuntimeState:
        """Append a pending step, generating its ID when this runtime uses automated IDs."""
        state = self._require_state()
        self._ensure_active(state)

        if state.automated_id:
            description = self._step_description(step)
            new_step = RuntimeStep(
                id=self._next_step_id(state.steps),
                description=description,
            )
        else:
            new_step = self._validate_manual_step(step)

        if new_step.status != "pending":
            raise ValueError("New runtime steps must start with status='pending'.")
        if any(existing.id == new_step.id for existing in state.steps):
            raise ValueError(f"Runtime step ID already exists: {new_step.id}")

        state.steps.append(new_step)
        return self._commit(state)

    def complete(self) -> RuntimeState:
        """Complete the runtime only after every step has completed successfully."""
        state = self._require_state()
        self._ensure_active(state)
        if state.current_step_id is not None:
            raise RuntimeError("Complete or fail the current step before completing the runtime.")

        unfinished = [step.id for step in state.steps if step.status != "completed"]
        if unfinished:
            raise RuntimeError(f"Runtime has unfinished steps: {', '.join(map(str, unfinished))}")

        state.status = "completed"
        return self._commit(state)

    @staticmethod
    def _next_step_id(existing_steps: Sequence[RuntimeStep]) -> int:
        """Return the next unused sequential integer ID (1, 2, 3, ...)."""
        return max((step.id for step in existing_steps), default=0) + 1

    @staticmethod
    def _format_step_id(step_id: int) -> str:
        """Format integer IDs as three digits for the agent-facing context view."""
        return f"{step_id:03d}"

    @staticmethod
    def _step_description(step_input: str | RuntimeStep | Mapping[str, Any]) -> str:
        """Extract a planner's step text when the runtime is generating IDs."""
        if isinstance(step_input, RuntimeStep):
            raise ValueError(
                "With automated_id=True, provide each step as text or a mapping with "
                "'description' (or 'step'); do not provide a RuntimeStep with an ID."
            )

        if isinstance(step_input, str):
            description = step_input.strip()
        elif isinstance(step_input, Mapping):
            if "id" in step_input:
                raise ValueError("With automated_id=True, omit step IDs; the runtime generates them.")
            unexpected = set(step_input) - {"description", "step"}
            if unexpected:
                raise ValueError(
                    "With automated_id=True, each step should contain only 'description' or 'step'."
                )
            description = step_input.get("description", step_input.get("step", ""))
            if isinstance(description, str):
                description = description.strip()
        else:
            raise TypeError("Each step must be text or a mapping containing its description.")

        if not isinstance(description, str) or not description:
            raise ValueError("Each runtime step must have a non-empty description.")
        return description

    @staticmethod
    def _validate_manual_step(step_input: RuntimeStep | Mapping[str, Any] | str) -> RuntimeStep:
        """Validate a step that supplies its own integer ID and description."""
        if isinstance(step_input, RuntimeStep):
            data: dict[str, Any] = step_input.model_dump(mode="python")
        elif isinstance(step_input, Mapping):
            data = dict(step_input)
        else:
            raise TypeError(
                "With automated_id=False, each step must provide an integer 'id' "
                "and a 'description' (or 'step')."
            )

        if "step" in data:
            data.setdefault("description", data.pop("step"))
        if "id" not in data or "description" not in data:
            raise ValueError(
                "With automated_id=False, each step must provide an integer 'id' "
                "and a 'description' (or 'step')."
            )
        return RuntimeStep.model_validate(data)

    def _require_state(self) -> RuntimeState:
        if self._state is None:
            raise RuntimeStateNotInitializedError("Runtime state has not been initialized.")
        return self._state.model_copy(deep=True)

    @staticmethod
    def _find_step(state: RuntimeState, step_id: int) -> RuntimeStep:
        for step in state.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"Unknown runtime step: {step_id}")

    @staticmethod
    def _ensure_active(state: RuntimeState) -> None:
        if state.status != "active":
            raise RuntimeError("This runtime is already completed.")

    def _commit(self, state: RuntimeState) -> RuntimeState:
        # Re-validate every transition before committing it to internal state.
        committed = RuntimeState.model_validate(state.model_dump(mode="python"))
        self._state = committed
        return self.get()
