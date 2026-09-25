"""Configuration for the standalone noScribe inference server."""

from __future__ import annotations

import ipaddress
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8766
    max_queued_jobs: int = 50
    max_upload_bytes: int = 1024 * 1024 * 1024
    max_audio_hours: float = 24.0
    reservation_ttl_seconds: float = 300.0
    upload_ready_ttl_seconds: float = 30.0
    terminal_job_ttl_seconds: float = 60.0
    upload_timeout_seconds: float = 3600.0
    job_timeout_seconds: float = 24 * 3600.0
    force_cpu: bool = False
    cpu_threads: int = 4
    vad_threshold: float = 0.5
    whisper_models_dir: Path = Path("models")
    runtime_dir: Path = Path(tempfile.gettempdir()) / "noscribe-server"
    require_tmpfs: bool = False

    def __post_init__(self) -> None:
        try:
            is_loopback = ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            is_loopback = self.host.casefold() == "localhost"
        if not is_loopback:
            raise ValueError("The noScribe server must listen on a loopback address.")
        if not 1 <= self.port <= 65535:
            raise ValueError("Server port must be between 1 and 65535.")
        if self.max_queued_jobs < 0:
            raise ValueError("max_queued_jobs must not be negative.")
        if self.max_upload_bytes <= 0 or self.max_audio_hours <= 0:
            raise ValueError("Audio limits must be positive.")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive.")
        if not 0 <= self.vad_threshold <= 1:
            raise ValueError("vad_threshold must be between 0 and 1.")
        for name in (
            "reservation_ttl_seconds",
            "upload_ready_ttl_seconds",
            "terminal_job_ttl_seconds",
            "upload_timeout_seconds",
            "job_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        object.__setattr__(self, "runtime_dir", Path(self.runtime_dir))
        object.__setattr__(self, "whisper_models_dir", Path(self.whisper_models_dir))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ServerConfig":
        if not isinstance(value, Mapping):
            raise ValueError("Server configuration must be a YAML mapping.")
        known = {item.name for item in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(
                f"Unknown server configuration fields: {', '.join(sorted(unknown))}"
            )
        return cls(**dict(value))


def load_server_config(path: Path) -> ServerConfig:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return ServerConfig.from_mapping(value or {})
