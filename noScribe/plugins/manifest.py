"""Versioned manifest format shared by built-in and installable plugins."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


PLUGIN_PROTOCOL_VERSION = 1


class ExecutionType(str, Enum):
    BUILTIN = "builtin"
    EXTERNAL = "external"
    REMOTE = "remote"


@dataclass(frozen=True)
class PluginExecution:
    type: ExecutionType
    adapter: str | None = None
    command: str | None = None
    driver: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PluginExecution":
        try:
            execution_type = ExecutionType(str(value["type"]))
        except (KeyError, ValueError) as error:
            raise ValueError("Plugin execution requires a supported type.") from error

        execution = cls(
            type=execution_type,
            adapter=value.get("adapter"),
            command=value.get("command"),
            driver=value.get("driver"),
        )
        required_field = {
            ExecutionType.BUILTIN: execution.adapter,
            ExecutionType.EXTERNAL: execution.command,
            ExecutionType.REMOTE: execution.driver,
        }[execution_type]
        if not required_field:
            raise ValueError(f"Execution type {execution_type.value!r} is incomplete.")
        return execution


@dataclass(frozen=True)
class PluginManifest:
    id: str
    name: str
    version: str
    protocol_version: int
    engine: str | None
    execution: PluginExecution
    capabilities: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PluginManifest":
        try:
            manifest = cls(
                id=str(value["id"]),
                name=str(value["name"]),
                version=str(value["version"]),
                protocol_version=int(value["protocol_version"]),
                engine=(str(value["engine"]) if value.get("engine") else None),
                execution=PluginExecution.from_mapping(value["execution"]),
                capabilities=frozenset(str(item) for item in value.get("capabilities", [])),
            )
        except KeyError as error:
            raise ValueError(f"Plugin manifest is missing {error.args[0]!r}.") from error
        if not manifest.id or ":" in manifest.id:
            raise ValueError("Plugin IDs must be non-empty and must not contain ':'.")
        if manifest.protocol_version != PLUGIN_PROTOCOL_VERSION:
            raise ValueError(
                f"Plugin {manifest.id!r} uses protocol {manifest.protocol_version}; "
                f"this app supports {PLUGIN_PROTOCOL_VERSION}."
            )
        return manifest

    @classmethod
    def from_file(cls, path: Path) -> "PluginManifest":
        with path.open("r", encoding="utf-8") as stream:
            return cls.from_mapping(json.load(stream))
