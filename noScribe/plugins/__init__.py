"""Plugin manifests, registry, and built-in backend adapters."""

from .manifest import ExecutionType, PluginManifest
from .registry import BackendRegistry

__all__ = ["BackendRegistry", "ExecutionType", "PluginManifest"]
