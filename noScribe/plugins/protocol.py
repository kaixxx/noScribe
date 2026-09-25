"""Wire-level protocol primitives for local and future external workers.

External workers will exchange one JSON object per line. Requests use
``WorkerRequest``; events use the same message shapes already emitted through
the multiprocessing queues by the built-in workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .manifest import PLUGIN_PROTOCOL_VERSION


EVENT_TYPES = frozenset({"log", "status", "progress", "segment", "result"})
METHODS = frozenset({"transcribe", "diarize", "cancel", "health"})


@dataclass(frozen=True)
class WorkerRequest:
    request_id: str
    method: str
    params: Mapping[str, Any] = field(default_factory=dict)
    protocol_version: int = PLUGIN_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("Worker requests require an ID.")
        if self.method not in METHODS:
            raise ValueError(f"Unsupported worker method: {self.method!r}")
        if self.protocol_version != PLUGIN_PROTOCOL_VERSION:
            raise ValueError("Unsupported worker protocol version.")

    def to_mapping(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "id": self.request_id,
            "method": self.method,
            "params": dict(self.params),
        }


def validate_worker_event(value: Any) -> dict:
    """Validate the common event envelope without constraining event payloads."""
    if not isinstance(value, dict):
        raise ValueError("Worker events must be JSON objects.")
    event_type = value.get("type")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unsupported worker event type: {event_type!r}")
    if event_type == "result" and "ok" not in value:
        raise ValueError("Worker result events require an 'ok' flag.")
    return value
