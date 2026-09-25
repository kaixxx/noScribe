"""Configuration discovery for independently managed remote backends."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import yaml


REMOTE_PROFILE_SCHEMA_VERSION = 1


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
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError(
            "Remote backend URLs require HTTPS except for loopback connections."
        )
    return value.rstrip("/")


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
