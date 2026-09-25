"""Small development server for exercising noScribe's remote backend client."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_TOKEN = "development-key"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


class DummyInferenceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, token: str, delay: float):
        super().__init__(address, DummyInferenceHandler)
        self.token = token
        self.delay = delay
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()


class DummyInferenceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: DummyInferenceServer

    def do_GET(self) -> None:
        if not self._authenticate():
            return
        if self.path == "/v1/models":
            self._send_json(200, {
                "protocol_version": 1,
                "server_version": "dummy-1",
                "features": ["queued_workflows"],
                "data": [
                    {
                        "id": "dummy-transcription",
                        "name": "Dummy transcription",
                        "engine": "dummy",
                        "capabilities": ["transcription"],
                    },
                    {
                        "id": "dummy-diarization",
                        "name": "Dummy diarization",
                        "engine": "dummy",
                        "capabilities": ["diarization"],
                    },
                ],
            })
            return
        prefix = "/v1/audio/jobs/"
        if self.path.startswith(prefix):
            job_id = self.path[len(prefix):]
            with self.server.jobs_lock:
                job = self.server.jobs.get(job_id)
            if job is None or not self._job_token_matches(job):
                self._send_json(404, {"error": "Unknown job"})
                return
            self._send_json(200, {
                "job_id": job_id,
                "state": "ready_for_upload",
                "position": 0,
            })
            return
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._authenticate():
            return
        if self.path == "/v1/audio/jobs":
            self._reserve_workflow()
            return
        workflow_prefix = "/v1/audio/jobs/"
        if self.path.startswith(workflow_prefix) and self.path.endswith("/audio"):
            job_id = self.path[len(workflow_prefix):-len("/audio")]
            self._run_workflow(job_id)
            return
        if self.path not in {
            "/v1/audio/transcriptions",
            "/v1/audio/diarizations",
        }:
            self._send_json(404, {"error": "Not found"})
            return
        try:
            fields, uploaded_file = self._read_multipart()
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        if fields.get("response_format") != "noscribe_jsonl":
            self._send_json(400, {"error": "response_format must be noscribe_jsonl"})
            return
        if not uploaded_file[0].lower().endswith(".flac"):
            self._send_json(400, {"error": "The uploaded audio must be FLAC"})
            return

        job_id = f"job-{uuid.uuid4().hex}"
        cancel_event = threading.Event()
        with self.server.jobs_lock:
            self.server.jobs[job_id] = cancel_event
        try:
            events = (
                self._transcription_events(fields)
                if self.path == "/v1/audio/transcriptions"
                else self._diarization_events(fields)
            )
            self._send_event_stream(job_id, cancel_event, events)
        finally:
            with self.server.jobs_lock:
                self.server.jobs.pop(job_id, None)

    def do_DELETE(self) -> None:
        if not self._authenticate():
            return
        prefix = (
            "/v1/audio/jobs/"
            if self.path.startswith("/v1/audio/jobs/")
            else "/v1/jobs/"
        )
        if not self.path.startswith(prefix):
            self._send_json(404, {"error": "Not found"})
            return
        job_id = self.path[len(prefix):]
        with self.server.jobs_lock:
            job = self.server.jobs.get(job_id)
        if job is None:
            self._send_json(404, {"error": "Unknown job"})
            return
        if prefix == "/v1/audio/jobs/" and not self._job_token_matches(job):
            self._send_json(404, {"error": "Unknown job"})
            return
        cancel_event = job["cancel"] if isinstance(job, dict) else job
        cancel_event.set()
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _reserve_workflow(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length <= 0 or length > 64 * 1024:
                raise ValueError("Invalid reservation size")
            value = json.loads(self.rfile.read(length))
            tasks = value["tasks"]
            audio = value["audio"]
            if not isinstance(tasks, list) or not tasks:
                raise ValueError("A workflow requires tasks")
            if any(
                not isinstance(task, dict)
                or task.get("type") not in {"transcription", "diarization"}
                for task in tasks
            ):
                raise ValueError("Unsupported workflow task")
            size = int(audio["size"])
            if size <= 0 or size > MAX_UPLOAD_BYTES:
                raise ValueError("Invalid audio size")
            if not str(audio.get("filename", "")).lower().endswith(".flac"):
                raise ValueError("The uploaded audio must be FLAC")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
            return
        job_id = f"job-{uuid.uuid4().hex}"
        job_token = uuid.uuid4().hex + uuid.uuid4().hex
        with self.server.jobs_lock:
            self.server.jobs[job_id] = {
                "token": job_token,
                "tasks": tasks,
                "size": size,
                "cancel": threading.Event(),
            }
        self._send_json(202, {
            "job_id": job_id,
            "job_token": job_token,
            "state": "ready_for_upload",
            "position": 0,
        })

    def _run_workflow(self, job_id: str) -> None:
        with self.server.jobs_lock:
            job = self.server.jobs.get(job_id)
        if job is None or not self._job_token_matches(job):
            self._send_json(404, {"error": "Unknown job"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length != job["size"]:
            self._send_json(400, {"error": "Upload size mismatch"})
            return
        uploaded = self.rfile.read(length)
        if len(uploaded) != length:
            self._send_json(400, {"error": "Incomplete upload"})
            return

        events = []
        for index, task in enumerate(job["tasks"]):
            operation = task.get("type")
            options = task.get("options") or {}
            events.append({
                "type": "task_started",
                "task_index": index,
                "operation": operation,
            })
            if operation == "transcription":
                for event in self._transcription_events(options):
                    if event["type"] == "result":
                        event = {
                            "type": "task_result",
                            "ok": event["ok"],
                            "info": event.get("info") or {},
                        }
                    event.update(task_index=index, operation=operation)
                    events.append(event)
            elif operation == "diarization":
                for event in self._diarization_events(options):
                    if event["type"] == "result":
                        event = {
                            "type": "task_result",
                            "ok": event["ok"],
                            "segments": event.get("segments") or [],
                        }
                    event.update(task_index=index, operation=operation)
                    events.append(event)
            else:
                self._send_json(400, {"error": "Unsupported task"})
                return
        events.append({"type": "result", "ok": True})
        try:
            self._send_event_stream(job_id, job["cancel"], events)
        finally:
            with self.server.jobs_lock:
                self.server.jobs.pop(job_id, None)

    def _job_token_matches(self, job: dict[str, Any]) -> bool:
        return self.headers.get("X-noScribe-Job-Token") == job["token"]

    def _authenticate(self) -> bool:
        expected = f"Bearer {self.server.token}"
        if self.headers.get("Authorization") == expected:
            return True
        self._send_json(401, {"error": "Invalid or missing API key"})
        return False

    def _read_multipart(self) -> tuple[dict[str, str], tuple[str, bytes]]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise ValueError("Expected a multipart/form-data request")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as error:
            raise ValueError("Missing or invalid Content-Length") from error
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("Upload size is invalid or exceeds 512 MiB")
        body = self.rfile.read(length)
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("ascii")
            + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
        )
        if not message.is_multipart():
            raise ValueError("Invalid multipart body")

        fields: dict[str, str] = {}
        uploaded_file: tuple[str, bytes] | None = None
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            if name == "file" and filename:
                uploaded_file = (filename, payload)
            elif filename is None:
                fields[name] = payload.decode(part.get_content_charset() or "utf-8")
        if uploaded_file is None:
            raise ValueError("Multipart request has no audio file")
        if not uploaded_file[1]:
            raise ValueError("Uploaded audio file is empty")
        if not fields.get("model"):
            raise ValueError("Multipart request has no model")
        return fields, uploaded_file

    def _send_event_stream(
        self,
        job_id: str,
        cancel_event: threading.Event,
        events: list[dict[str, Any]],
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-noScribe-Job-ID", job_id)
        self.end_headers()
        self.close_connection = True
        try:
            for event in events:
                if cancel_event.wait(self.server.delay):
                    event = {
                        "type": "result",
                        "ok": False,
                        "error": "Dummy job canceled",
                    }
                    self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
                    self.wfile.flush()
                    return
                self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            cancel_event.set()

    @staticmethod
    def _transcription_events(fields: dict[str, str]) -> list[dict[str, Any]]:
        language = fields.get("language") or "en"
        return [
            {"type": "progress", "pct": 10, "detail": "dummy-start"},
            {
                "type": "segment",
                "segment": {
                    "start": 0.0,
                    "end": 2.5,
                    "text": " This is a simulated transcription.",
                    "words": [
                        {"word": "This", "start": 0.0, "end": 0.4, "prob": 1.0},
                        {"word": "is", "start": 0.4, "end": 0.6, "prob": 1.0},
                    ],
                },
            },
            {"type": "progress", "pct": 65, "detail": "dummy-middle"},
            {
                "type": "segment",
                "segment": {
                    "start": 2.5,
                    "end": 5.0,
                    "text": " It came from the noScribe dummy server.",
                },
            },
            {
                "type": "result",
                "ok": True,
                "info": {
                    "duration": 5.0,
                    "language": language,
                    "language_probability": 1.0,
                    "sample_rate": 48000,
                },
            },
        ]

    @staticmethod
    def _diarization_events(_fields: dict[str, str]) -> list[dict[str, Any]]:
        return [
            {"type": "progress", "step": "segmentation", "pct": 50},
            {"type": "progress", "step": "embeddings", "pct": 100},
            {
                "type": "result",
                "ok": True,
                "duration": 5.0,
                "segments": [
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.7},
                    {"speaker": "SPEAKER_01", "start": 2.3, "end": 5.0},
                ],
            },
        ]

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_string: str, *args) -> None:
        print(f"{self.client_address[0]} - {format_string % args}")


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str = DEFAULT_TOKEN,
    delay: float = 0.35,
) -> DummyInferenceServer:
    return DummyInferenceServer((host, port), token=token, delay=max(delay, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Delay in seconds between streamed events.",
    )
    args = parser.parse_args()
    server = create_server("127.0.0.1", args.port, args.token, args.delay)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"noScribe dummy server listening on {url}")
    print("Use this development profile:")
    print(
        "\n".join([
            "schema_version: 1",
            "id: dummy-server",
            "name: Dummy-Server",
            "driver: noscribe-http-v1",
            "enabled: true",
            f"url: {url}",
            f"api_key: {args.token}",
        ])
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dummy server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
