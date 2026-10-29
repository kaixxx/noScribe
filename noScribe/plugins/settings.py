"""Installation-owned settings files, independent of inference jobs.

The store knows only paths and YAML. Each plugin owns its settings schema and
defaults; adding a plugin never requires extending the application's config.
"""

from pathlib import Path
import re
import os
import tempfile

import yaml


class PluginSettingsStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def path_for(self, backend_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", backend_id):
            raise ValueError(f"Invalid plugin settings ID: {backend_id!r}")
        return self.directory / f"{backend_id}.yml"

    def load(self, backend_id: str) -> dict:
        path = self.path_for(backend_id)
        if not path.exists():
            return {}
        try:
            with path.open(encoding="utf-8") as stream:
                value = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid plugin settings file: {path}") from error
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"Plugin settings must be a YAML mapping: {path}")
        return value

    def save(self, backend_id: str, value: dict) -> None:
        """Atomically replace settings without touching the plugin package."""
        path = self.path_for(backend_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def settings_values(value: dict, allowed: set[str]) -> dict:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown plugin settings: {', '.join(map(str, sorted(unknown, key=str)))}")
    return dict(value)


def validate_device(device: str, allowed: set[str]) -> None:
    if not isinstance(device, str) or device not in allowed:
        raise ValueError(f"Device must be one of: {', '.join(sorted(allowed))}.")
