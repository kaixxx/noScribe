"""Configuration discovery for independently managed remote backends."""

from __future__ import annotations

import ipaddress
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import yaml


REMOTE_PROFILE_SCHEMA_VERSION = 1
_RFC1918_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


@dataclass(frozen=True)
class RemoteBackendProfile:
    """One configured remote server connection."""

    id: str
    name: str
    driver: str
    url: str
    api_key: str
    enabled: bool = True
    source: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        source: Path | None = None,
    ) -> "RemoteBackendProfile":
        if not isinstance(value, Mapping):
            raise ValueError("Remote backend profile must be a YAML mapping.")

        schema_version = value.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != REMOTE_PROFILE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported remote backend schema version {schema_version!r}; "
                f"expected {REMOTE_PROFILE_SCHEMA_VERSION}."
            )

        profile_id = _required_string(value, "id")
        if ":" in profile_id:
            raise ValueError("Remote backend profile IDs must not contain ':'.")

        name = _required_string(value, "name")
        driver = _required_string(value, "driver")
        url = _validate_url(_required_string(value, "url"))
        api_key = _required_string(value, "api_key")

        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("Remote backend field 'enabled' must be a boolean.")

        return cls(
            id=profile_id,
            name=name,
            driver=driver,
            url=url,
            api_key=api_key,
            enabled=enabled,
            source=source,
        )


@dataclass(frozen=True)
class RemoteProfileError:
    path: Path
    message: str


@dataclass(frozen=True)
class RemoteProfileLoadResult:
    profiles: tuple[RemoteBackendProfile, ...]
    errors: tuple[RemoteProfileError, ...]


def load_remote_profiles(directory: Path) -> RemoteProfileLoadResult:
    """Load all YAML profiles while isolating errors to individual files."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (*directory.glob("*.yml"), *directory.glob("*.yaml")),
        key=lambda path: path.name.casefold(),
    )
    profiles: list[RemoteBackendProfile] = []
    errors: list[RemoteProfileError] = []
    profile_sources: dict[str, Path] = {}

    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = yaml.safe_load(stream)
            profile = RemoteBackendProfile.from_mapping(value, source=path)
            if profile.id in profile_sources:
                first_path = profile_sources[profile.id]
                raise ValueError(
                    f"Duplicate remote backend profile ID {profile.id!r}; "
                    f"already defined in {first_path.name!r}."
                )
            profiles.append(profile)
            profile_sources[profile.id] = path
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
            errors.append(RemoteProfileError(path=path, message=str(error)))

    return RemoteProfileLoadResult(tuple(profiles), tuple(errors))


def new_remote_profile_id(name: str, existing_ids: set[str]) -> str:
    """Return a readable, filesystem-safe ID that remains stable after save."""
    base = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    base = base or "remote-server"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def save_remote_profile(
    directory: Path, profile: RemoteBackendProfile
) -> RemoteBackendProfile:
    """Atomically persist one profile and return it with its source path."""
    directory.mkdir(parents=True, exist_ok=True)
    root = directory.resolve()
    source = profile.source
    if source is not None and source.resolve().parent == root:
        target = source
    else:
        target = directory / f"{profile.id}.yml"
        suffix = 2
        while target.exists():
            target = directory / f"{profile.id}-{suffix}.yml"
            suffix += 1

    value = {
        "schema_version": REMOTE_PROFILE_SCHEMA_VERSION,
        "id": profile.id,
        "name": profile.name,
        "driver": profile.driver,
        "enabled": profile.enabled,
        "url": profile.url,
        "api_key": profile.api_key,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}-", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return RemoteBackendProfile.from_mapping(value, source=target)


def delete_remote_profile(
    directory: Path, profile: RemoteBackendProfile
) -> None:
    """Delete only a profile file contained in the configured profile folder."""
    if profile.source is None:
        return
    root = directory.resolve()
    source = profile.source.resolve()
    if source.parent != root:
        raise ValueError("Remote profile file is outside the profile directory.")
    source.unlink(missing_ok=True)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Remote backend field {key!r} must be a non-empty string.")
    return item.strip()


def _validate_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Remote backend field 'url' must be an HTTP(S) URL.")
    if parsed.username or parsed.password:
        raise ValueError("Remote backend URLs must not contain credentials.")
    if parsed.scheme == "http" and not _is_private_http_host(parsed.hostname):
        raise ValueError(
            "Remote backend URLs require HTTPS except for loopback and "
            "RFC 1918 private IPv4 addresses."
        )
    return value.rstrip("/")


def _is_private_http_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return isinstance(address, ipaddress.IPv4Address) and any(
        address in network for network in _RFC1918_NETWORKS
    )
