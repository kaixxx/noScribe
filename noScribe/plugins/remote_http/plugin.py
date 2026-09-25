"""Remote backend adapter for the streaming noScribe HTTP API."""

from __future__ import annotations

import json
import mimetypes
import threading
from pathlib import Path
from typing import Callable, Iterable

import requests

from ...inference import (
    DiarizationSegment,
    InferenceCancelled,
    InferenceWorkerError,
    TranscriptionInfo,
    TranscriptionSegment,
)
from ...models import ModelDescriptor, ModelRef
from ..manifest import PLUGIN_PROTOCOL_VERSION, PluginManifest
from ..protocol import validate_worker_event
from ..remote_profiles import RemoteBackendProfile


NOSCRIBE_HTTP_DRIVER = "noscribe-http-v1"
_MODEL_TIMEOUT = (3.05, 10)
_REQUEST_TIMEOUT = (10, None)
_CANCEL_TIMEOUT = (3.05, 5)


class RemoteHttpPlugin:
    """Expose one configured remote server through the backend interface."""

    def __init__(
        self,
        profile: RemoteBackendProfile,
        *,
        session: requests.Session | None = None,
    ):
        if profile.driver != NOSCRIBE_HTTP_DRIVER:
            raise ValueError(f"Unsupported remote backend driver: {profile.driver!r}")
        self.profile = profile
        self._session = session or requests.Session()
        self._cancel_event = threading.Event()
        self._active_lock = threading.Lock()
        self._active_response: requests.Response | None = None
        self._active_job_id: str | None = None
        self._models, server_version = self._fetch_models()
        capabilities = frozenset(
            capability
            for model in self._models
            for capability in model.capabilities
        )
        self.manifest = PluginManifest.from_mapping({
            "id": profile.id,
            "name": profile.name,
            "version": server_version,
            "protocol_version": PLUGIN_PROTOCOL_VERSION,
            "engine": None,
            "execution": {
                "type": "remote",
                "driver": NOSCRIBE_HTTP_DRIVER,
            },
            "capabilities": sorted(capabilities),
        })

    def list_models(self) -> list[ModelDescriptor]:
        return list(self._models)

    def transcribe(
        self,
        request,
        on_segment: Callable[[TranscriptionSegment], None],
        on_log=lambda _level, _message: None,
        on_status=lambda _message_id, _params, _level: None,
        on_progress=lambda _percent, _detail: None,
        is_cancelled=lambda: False,
    ) -> TranscriptionInfo:
        fields = {
            "model": request.model.model_id,
            "multilingual": _form_bool(request.multilingual),
            "include_disfluencies": _form_bool(request.include_disfluencies),
            "response_format": "noscribe_jsonl",
        }
        if request.language:
            fields["language"] = request.language

        streamed_segments = False
        result = None
        for event in self._stream_audio_request(
            "/v1/audio/transcriptions",
            request.audio_path,
            fields,
            is_cancelled,
        ):
            event_type = event["type"]
            if event_type == "log":
                on_log(str(event.get("level", "info")), str(event.get("msg", "")))
            elif event_type == "status":
                on_status(
                    str(event.get("message_id", "")),
                    event.get("params") or {},
                    str(event.get("level", "info")),
                )
            elif event_type == "progress":
                if event.get("pct") is not None:
                    on_progress(float(event["pct"]), event.get("detail"))
            elif event_type == "segment":
                streamed_segments = True
                on_segment(TranscriptionSegment.from_mapping(event.get("segment") or {}))
            elif event_type == "result":
                result = self._require_success(event, "Transcription failed")

        if result is None:
            raise InferenceWorkerError("Remote transcription ended without a result.")
        if not streamed_segments:
            for segment in result.get("segments") or []:
                on_segment(TranscriptionSegment.from_mapping(segment))
        info = result.get("info") or {
            key: result.get(key)
            for key in ("duration", "language", "language_probability", "sample_rate")
            if result.get(key) is not None
        }
        return TranscriptionInfo.from_mapping(info)

    def diarize(
        self,
        request,
        on_log=lambda _level, _message: None,
        on_progress=lambda _step, _percent: None,
        is_cancelled=lambda: False,
    ) -> list[DiarizationSegment]:
        fields = {
            "model": request.model.model_id,
            "response_format": "noscribe_jsonl",
        }
        if request.num_speakers is not None:
            fields["num_speakers"] = str(request.num_speakers)

        segments: list[DiarizationSegment] = []
        result = None
        for event in self._stream_audio_request(
            "/v1/audio/diarizations",
            request.audio_path,
            fields,
            is_cancelled,
        ):
            event_type = event["type"]
            if event_type == "log":
                on_log(str(event.get("level", "info")), str(event.get("msg", "")))
            elif event_type == "progress":
                on_progress(str(event.get("step", "")), int(event.get("pct", 0)))
            elif event_type == "segment":
                segments.append(_diarization_segment(event.get("segment") or {}))
            elif event_type == "result":
                result = self._require_success(event, "Diarization failed")

        if result is None:
            raise InferenceWorkerError("Remote diarization ended without a result.")
        if not segments:
            segments = [
                _diarization_segment(segment)
                for segment in (result.get("segments") or [])
            ]
        return segments

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._active_lock:
            response = self._active_response
            job_id = self._active_job_id
        if response is not None:
            response.close()
        if job_id:
            try:
                requests.delete(
                    f"{self.profile.url}/v1/jobs/{job_id}",
                    headers=self._headers(),
                    timeout=_CANCEL_TIMEOUT,
                )
            except requests.RequestException:
                pass

    def close(self) -> None:
        self.cancel()
        self._session.close()

    def _fetch_models(self) -> tuple[tuple[ModelDescriptor, ...], str]:
        try:
            response = self._session.get(
                f"{self.profile.url}/v1/models",
                headers=self._headers(),
                timeout=_MODEL_TIMEOUT,
            )
            self._raise_for_status(response, "Could not retrieve remote models")
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            if isinstance(error, InferenceWorkerError):
                raise
            raise InferenceWorkerError(
                f"Could not retrieve models from {self.profile.name}: {error}"
            ) from error

        if not isinstance(payload, dict):
            raise InferenceWorkerError("Remote model response must be a JSON object.")
        protocol_version = payload.get("protocol_version")
        if (
            not isinstance(protocol_version, int)
            or isinstance(protocol_version, bool)
            or protocol_version != PLUGIN_PROTOCOL_VERSION
        ):
            raise InferenceWorkerError(
                f"Remote server uses protocol {protocol_version!r}; "
                f"this app supports {PLUGIN_PROTOCOL_VERSION}."
            )
        raw_models = payload.get("data", payload.get("models"))
        if not isinstance(raw_models, list):
            raise InferenceWorkerError("Remote model response has no model list.")

        models = []
        seen_ids = set()
        supported_capabilities = {"transcription", "diarization"}
        for value in raw_models:
            if not isinstance(value, dict):
                raise InferenceWorkerError("Remote model entries must be JSON objects.")
            raw_capabilities = value.get("capabilities")
            if raw_capabilities is None:
                continue
            if not isinstance(raw_capabilities, list) or not all(
                isinstance(capability, str) for capability in raw_capabilities
            ):
                raise InferenceWorkerError(
                    "Remote model capabilities must be a list of strings."
                )
            capabilities = frozenset(raw_capabilities)
            capabilities &= supported_capabilities
            if not capabilities:
                continue
            model_id = value.get("id")
            if not isinstance(model_id, str) or not model_id or ":" in model_id:
                raise InferenceWorkerError("Remote model IDs must be non-empty strings without ':'.")
            if model_id in seen_ids:
                raise InferenceWorkerError(f"Duplicate remote model ID: {model_id!r}")
            seen_ids.add(model_id)
            display_name = value.get("name") or model_id
            engine = value.get("engine") or "remote"
            models.append(ModelDescriptor(
                ref=ModelRef(self.profile.id, model_id),
                display_name=str(display_name),
                engine=str(engine),
                capabilities=capabilities,
            ))
        if not models:
            raise InferenceWorkerError("Remote server offers no supported inference models.")
        return tuple(models), str(payload.get("server_version") or "unknown")

    def _stream_audio_request(
        self,
        endpoint: str,
        audio_path: str,
        fields: dict[str, str],
        is_cancelled: Callable[[], bool],
    ) -> Iterable[dict]:
        self._cancel_event.clear()
        path = Path(audio_path)
        if path.suffix.lower() != ".opus":
            raise InferenceWorkerError(
                "Remote inference requires a locally prepared Opus audio file."
            )
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            with path.open("rb") as audio_stream:
                response = self._session.post(
                    f"{self.profile.url}{endpoint}",
                    headers=self._headers(),
                    data=fields,
                    files={"file": (path.name, audio_stream, mime_type)},
                    stream=True,
                    timeout=_REQUEST_TIMEOUT,
                )
            self._raise_for_status(response, "Remote inference request failed")
            with self._active_lock:
                self._active_response = response
                self._active_job_id = response.headers.get("X-noScribe-Job-ID")
            try:
                for line in response.iter_lines():
                    if self._cancel_event.is_set() or is_cancelled():
                        self.cancel()
                        raise InferenceCancelled("Inference canceled")
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        yield validate_worker_event(event)
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                        raise InferenceWorkerError(
                            f"Invalid event from remote backend: {error}"
                        ) from error
            finally:
                response.close()
                with self._active_lock:
                    if self._active_response is response:
                        self._active_response = None
                        self._active_job_id = None
        except InferenceCancelled:
            raise
        except InferenceWorkerError:
            raise
        except (OSError, requests.RequestException) as error:
            if self._cancel_event.is_set() or is_cancelled():
                raise InferenceCancelled("Inference canceled") from error
            raise InferenceWorkerError(f"Remote inference request failed: {error}") from error

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.profile.api_key}",
            "Accept": "application/x-ndjson",
        }

    @staticmethod
    def _require_success(event: dict, fallback: str) -> dict:
        if not event.get("ok"):
            raise InferenceWorkerError(
                str(event.get("error") or fallback),
                event.get("trace"),
            )
        return event

    @staticmethod
    def _raise_for_status(response: requests.Response, prefix: str) -> None:
        if response.ok:
            return
        detail = response.text.strip()[:500]
        suffix = f": {detail}" if detail else ""
        raise InferenceWorkerError(f"{prefix} (HTTP {response.status_code}){suffix}")


def _form_bool(value: bool) -> str:
    return "true" if value else "false"


def _diarization_segment(value: dict) -> DiarizationSegment:
    return DiarizationSegment(
        start_ms=round(float(value.get("start") or 0) * 1000),
        end_ms=round(float(value.get("end") or 0) * 1000),
        label=str(value.get("speaker") or value.get("label") or ""),
    )
