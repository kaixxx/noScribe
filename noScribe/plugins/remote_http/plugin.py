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
    InferenceWorkflowRequest,
    InferenceWorkflowResult,
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
_QUEUE_POLL_INTERVAL = 0.5


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
        self._active_job_token: str | None = None
        self._models, server_version, features = self._fetch_models()
        self.supports_workflows = "queued_workflows" in features
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
        if self.supports_workflows:
            def handle(event):
                if event.get("operation") not in {None, "transcription"}:
                    return
                event_type = event.get("type")
                if event_type == "segment":
                    on_segment(TranscriptionSegment.from_mapping(
                        event.get("segment") or {}
                    ))
                elif event_type == "log":
                    on_log(str(event.get("level", "info")), str(event.get("msg", "")))
                elif event_type == "status":
                    on_status(
                        str(event.get("message_id", "")),
                        event.get("params") or {},
                        str(event.get("level", "info")),
                    )
                elif event_type == "progress" and event.get("pct") is not None:
                    on_progress(float(event["pct"]), event.get("detail"))

            result = self.run_workflow(
                InferenceWorkflowRequest(
                    audio_path=request.audio_path, transcription=request
                ),
                on_event=handle,
                is_cancelled=is_cancelled,
            )
            if result.transcription_info is None:
                raise InferenceWorkerError("Remote transcription returned no metadata.")
            return result.transcription_info

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
        if self.supports_workflows:
            def handle(event):
                if event.get("operation") not in {None, "diarization"}:
                    return
                event_type = event.get("type")
                if event_type == "log":
                    on_log(str(event.get("level", "info")), str(event.get("msg", "")))
                elif event_type == "progress":
                    on_progress(str(event.get("step", "")), int(event.get("pct", 0)))

            result = self.run_workflow(
                InferenceWorkflowRequest(
                    audio_path=request.audio_path, diarization=request
                ),
                on_event=handle,
                is_cancelled=is_cancelled,
            )
            return list(result.diarization_segments)

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

    def run_workflow(
        self,
        request: InferenceWorkflowRequest,
        on_event=lambda _event: None,
        is_cancelled=lambda: False,
    ) -> InferenceWorkflowResult:
        if not self.supports_workflows:
            raise InferenceWorkerError(
                "Remote server does not support queued workflows."
            )
        path = Path(request.audio_path)
        if path.suffix.casefold() != ".flac":
            raise InferenceWorkerError(
                "Remote inference requires a locally prepared FLAC audio file."
            )
        tasks = []
        if request.diarization is not None:
            options = {}
            if request.diarization.num_speakers is not None:
                options["num_speakers"] = request.diarization.num_speakers
            tasks.append({
                "type": "diarization",
                "model": request.diarization.model.model_id,
                "options": options,
            })
        if request.transcription is not None:
            options = {
                "multilingual": bool(request.transcription.multilingual),
                "include_disfluencies": bool(
                    request.transcription.include_disfluencies
                ),
            }
            if request.transcription.language:
                options["language"] = request.transcription.language
            tasks.append({
                "type": "transcription",
                "model": request.transcription.model.model_id,
                "options": options,
            })

        self._cancel_event.clear()
        try:
            response = self._session.post(
                f"{self.profile.url}/v1/audio/jobs",
                headers={**self._headers(), "Content-Type": "application/json"},
                json={
                    "tasks": tasks,
                    "audio": {"filename": path.name, "size": path.stat().st_size},
                },
                timeout=_MODEL_TIMEOUT,
            )
            self._raise_for_status(response, "Could not reserve remote job")
            admission = response.json()
            job_id = _required_response_string(admission, "job_id")
            job_token = _required_response_string(admission, "job_token")
            with self._active_lock:
                self._active_job_id = job_id
                self._active_job_token = job_token

            state = str(admission.get("state") or "")
            while state == "queued":
                if self._cancel_event.wait(_QUEUE_POLL_INTERVAL) or is_cancelled():
                    self.cancel()
                    raise InferenceCancelled("Inference canceled")
                status_response = self._session.get(
                    f"{self.profile.url}/v1/audio/jobs/{job_id}",
                    headers=self._job_headers(job_token),
                    timeout=_MODEL_TIMEOUT,
                )
                self._raise_for_status(status_response, "Remote queue status failed")
                status_value = status_response.json()
                state = str(status_value.get("state") or "")
                on_event({
                    "type": "status",
                    "message_id": "server_queue_wait",
                    "params": {"position": status_value.get("position", 0)},
                    "level": "info",
                })
            if state != "ready_for_upload":
                raise InferenceWorkerError(f"Remote job entered state {state!r}.")
            if self._cancel_event.is_set() or is_cancelled():
                self.cancel()
                raise InferenceCancelled("Inference canceled")

            with path.open("rb") as audio_stream:
                response = self._session.post(
                    f"{self.profile.url}/v1/audio/jobs/{job_id}/audio",
                    headers={
                        **self._job_headers(job_token),
                        "Content-Type": "audio/flac",
                        "Content-Length": str(path.stat().st_size),
                    },
                    data=audio_stream,
                    stream=True,
                    timeout=_REQUEST_TIMEOUT,
                )
            self._raise_for_status(response, "Remote audio upload failed")
            with self._active_lock:
                self._active_response = response

            transcription_segments = []
            transcription_info = None
            diarization_segments = []
            final_result = None
            try:
                for line in response.iter_lines(chunk_size=1):
                    if self._cancel_event.is_set() or is_cancelled():
                        self.cancel()
                        raise InferenceCancelled("Inference canceled")
                    if not line:
                        continue
                    event = _workflow_event(line)
                    on_event(event)
                    if (
                        event["type"] == "segment"
                        and event.get("operation") == "transcription"
                    ):
                        transcription_segments.append(
                            TranscriptionSegment.from_mapping(
                                event.get("segment") or {}
                            )
                        )
                    elif event["type"] == "task_result":
                        self._require_success(event, "Remote task failed")
                        if event.get("operation") == "diarization":
                            diarization_segments = [
                                _diarization_segment(segment)
                                for segment in event.get("segments") or []
                            ]
                        elif event.get("operation") == "transcription":
                            transcription_info = TranscriptionInfo.from_mapping(
                                event.get("info") or {}
                            )
                    elif event["type"] == "result":
                        final_result = self._require_success(
                            event, "Remote workflow failed"
                        )
            finally:
                response.close()
            if final_result is None:
                raise InferenceWorkerError("Remote workflow ended without a result.")
            return InferenceWorkflowResult(
                transcription_info=transcription_info,
                transcription_segments=tuple(transcription_segments),
                diarization_segments=tuple(diarization_segments),
            )
        except InferenceCancelled:
            raise
        except InferenceWorkerError:
            raise
        except (OSError, requests.RequestException, ValueError) as error:
            if self._cancel_event.is_set() or is_cancelled():
                raise InferenceCancelled("Inference canceled") from error
            raise InferenceWorkerError(f"Remote workflow failed: {error}") from error
        finally:
            with self._active_lock:
                self._active_response = None
                self._active_job_id = None
                self._active_job_token = None

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._active_lock:
            response = self._active_response
            job_id = self._active_job_id
            job_token = self._active_job_token
        if response is not None:
            response.close()
        if job_id:
            try:
                requests.delete(
                    (
                        f"{self.profile.url}/v1/audio/jobs/{job_id}"
                        if job_token
                        else f"{self.profile.url}/v1/jobs/{job_id}"
                    ),
                    headers=(
                        self._job_headers(job_token)
                        if job_token else self._headers()
                    ),
                    timeout=_CANCEL_TIMEOUT,
                )
            except requests.RequestException:
                pass

    def close(self) -> None:
        self.cancel()
        self._session.close()

    def _fetch_models(self) -> tuple[tuple[ModelDescriptor, ...], str, frozenset[str]]:
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
        raw_features = payload.get("features") or []
        if not isinstance(raw_features, list) or not all(
            isinstance(feature, str) for feature in raw_features
        ):
            raise InferenceWorkerError("Remote server features must be a list of strings.")
        return (
            tuple(models),
            str(payload.get("server_version") or "unknown"),
            frozenset(raw_features),
        )

    def _stream_audio_request(
        self,
        endpoint: str,
        audio_path: str,
        fields: dict[str, str],
        is_cancelled: Callable[[], bool],
    ) -> Iterable[dict]:
        self._cancel_event.clear()
        path = Path(audio_path)
        if path.suffix.lower() != ".flac":
            raise InferenceWorkerError(
                "Remote inference requires a locally prepared FLAC audio file."
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
                for line in response.iter_lines(chunk_size=1):
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

    def _job_headers(self, token: str) -> dict[str, str]:
        return {**self._headers(), "X-noScribe-Job-Token": token}

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


def _required_response_string(value: dict, key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise InferenceWorkerError(f"Remote response has no {key!r}.")
    return item


def _workflow_event(line: bytes) -> dict:
    try:
        event = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceWorkerError(f"Invalid event from remote backend: {error}") from error
    if not isinstance(event, dict) or event.get("type") not in {
        "log", "status", "progress", "segment", "task_started",
        "task_result", "result",
    }:
        raise InferenceWorkerError("Invalid workflow event from remote backend.")
    if event["type"] in {"task_result", "result"} and "ok" not in event:
        raise InferenceWorkerError("Remote result event has no success flag.")
    return event
