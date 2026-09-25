import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from noScribe.inference import DiarizationRequest, TranscriptionRequest
from noScribe.inference import InferenceWorkerError
from noScribe.models import ModelRef
from noScribe.plugins.factory import register_remote_profiles
from noScribe.plugins.registry import BackendRegistry
from noScribe.plugins.remote_http import RemoteHttpPlugin
from noScribe.plugins.remote_profiles import RemoteBackendProfile


@contextmanager
def _remote_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            received.append(("GET", self.path, self.headers, b""))
            if self.path != "/v1/models":
                self.send_error(404)
                return
            self._send_json({
                "protocol_version": 1,
                "server_version": "0.1-test",
                "data": [
                    {
                        "id": "precise",
                        "name": "precise",
                        "engine": "faster-whisper",
                        "capabilities": ["transcription"],
                    },
                    {
                        "id": "speakers",
                        "name": "Pyannote",
                        "engine": "pyannote",
                        "capabilities": ["diarization"],
                    },
                    {"id": "llm", "name": "Local LLM"},
                ],
            })

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            received.append(("POST", self.path, self.headers, body))
            if self.path == "/v1/audio/transcriptions":
                events = [
                    {"type": "progress", "pct": 25, "detail": "upload"},
                    {
                        "type": "segment",
                        "segment": {"start": 0.5, "end": 1.5, "text": "Hello"},
                    },
                    {
                        "type": "result",
                        "ok": True,
                        "info": {"duration": 2.0, "language": "en"},
                    },
                ]
            elif self.path == "/v1/audio/diarizations":
                events = [
                    {"type": "progress", "step": "segmentation", "pct": 50},
                    {
                        "type": "result",
                        "ok": True,
                        "segments": [
                            {"speaker": "SPEAKER_00", "start": 0.5, "end": 1.75}
                        ],
                    },
                ]
            else:
                self.send_error(404)
                return
            payload = b"".join(
                json.dumps(event).encode("utf-8") + b"\n" for event in events
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-noScribe-Job-ID", "job-123")
            self.end_headers()
            self.wfile.write(payload)

        def do_DELETE(self):
            received.append(("DELETE", self.path, self.headers, b""))
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send_json(self, value):
            payload = json.dumps(value).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _profile(url, **overrides):
    values = {
        "id": "ifs-server",
        "name": "IfS-Server",
        "driver": "noscribe-http-v1",
        "url": url,
        "api_key": "test-key",
        "enabled": True,
    }
    values.update(overrides)
    return RemoteBackendProfile(**values)


def test_remote_http_plugin_loads_models_and_streams_transcription(tmp_path):
    audio_path = tmp_path / "audio.opus"
    audio_path.write_bytes(b"OggS-test-audio")

    with _remote_server() as (url, received):
        plugin = RemoteHttpPlugin(_profile(url))
        segments = []
        progress = []
        info = plugin.transcribe(
            TranscriptionRequest(
                str(audio_path),
                ModelRef("ifs-server", "precise"),
                language="en",
            ),
            on_segment=segments.append,
            on_progress=lambda percent, detail: progress.append((percent, detail)),
        )
        plugin.close()

    assert plugin.manifest.name == "IfS-Server"
    assert [str(model.ref) for model in plugin.list_models()] == [
        "ifs-server:precise",
        "ifs-server:speakers",
    ]
    assert segments[0].text == "Hello"
    assert progress == [(25.0, "upload")]
    assert info.language == "en"
    assert all(
        request[2].get("Authorization") == "Bearer test-key"
        for request in received
    )
    post_body = next(item[3] for item in received if item[0] == "POST")
    assert b'name="model"' in post_body
    assert b"precise" in post_body
    assert b'name="file"; filename="audio.opus"' in post_body


def test_remote_http_plugin_maps_diarization_seconds_to_milliseconds(tmp_path):
    audio_path = tmp_path / "audio.opus"
    audio_path.write_bytes(b"OggS-test-audio")

    with _remote_server() as (url, _received):
        plugin = RemoteHttpPlugin(_profile(url))
        progress = []
        segments = plugin.diarize(
            DiarizationRequest(
                str(audio_path),
                ModelRef("ifs-server", "speakers"),
                num_speakers=2,
            ),
            on_progress=lambda step, percent: progress.append((step, percent)),
        )
        plugin.close()

    assert progress == [("segmentation", 50)]
    assert segments[0].start_ms == 500
    assert segments[0].end_ms == 1750
    assert segments[0].label == "SPEAKER_00"


def test_remote_http_plugin_refuses_uncompressed_transport_file(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"not-uploaded")

    with _remote_server() as (url, received):
        plugin = RemoteHttpPlugin(_profile(url))
        try:
            plugin.transcribe(
                TranscriptionRequest(
                    str(audio_path),
                    ModelRef("ifs-server", "precise"),
                ),
                on_segment=lambda _segment: None,
            )
        except InferenceWorkerError as error:
            assert "Opus" in str(error)
        else:
            raise AssertionError("WAV upload was not rejected")
        finally:
            plugin.close()

    assert not any(item[0] == "POST" for item in received)


def test_remote_registration_skips_disabled_and_isolates_unknown_drivers():
    registry = BackendRegistry()
    profiles = (
        _profile("https://disabled.example.org", id="disabled", enabled=False),
        _profile(
            "https://unknown.example.org",
            id="unknown",
            driver="future-driver",
        ),
    )

    errors = register_remote_profiles(registry, profiles)

    assert registry.list_plugins() == ()
    assert len(errors) == 1
    assert errors[0].profile.id == "unknown"
    assert "Unsupported" in errors[0].message
