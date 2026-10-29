"""Backend-qualified model identifiers and model catalogue entries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelRef:
    """A model identifier qualified by the backend that provides it."""

    backend_id: str
    model_id: str

    def __post_init__(self) -> None:
        if not self.backend_id or not self.model_id:
            raise ValueError("Model references require a backend and model ID.")
        if ":" in self.backend_id or ":" in self.model_id:
            raise ValueError("Backend and model IDs must not contain ':'.")

    @classmethod
    def parse(cls, value: str) -> "ModelRef":
        try:
            backend_id, model_id = value.split(":", 1)
        except ValueError as error:
            raise ValueError(
                f"Invalid model reference {value!r}; expected 'backend:model'."
            ) from error
        return cls(backend_id=backend_id, model_id=model_id)

    def __str__(self) -> str:
        return f"{self.backend_id}:{self.model_id}"


@dataclass(frozen=True)
class ModelDescriptor:
    """A model advertised by one registered backend plugin."""

    ref: ModelRef
    display_name: str
    engine: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
