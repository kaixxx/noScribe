"""Plugin manifests, registry, and built-in backend adapters."""

from .manifest import ExecutionType, PluginManifest
from .remote_profiles import RemoteBackendProfile, load_remote_profiles
from .registry import BackendRegistry

__all__ = [
    "BackendRegistry",
    "ExecutionType",
    "PluginManifest",
    "RemoteBackendProfile",
    "load_remote_profiles",
]
