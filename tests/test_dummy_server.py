import threading

import requests

from noScribe.inference import DiarizationRequest, TranscriptionRequest
from noScribe.models import ModelRef
from noScribe.plugins.remote_http import RemoteHttpPlugin
from noScribe.plugins.remote_profiles import RemoteBackendProfile
from tools.noscribe_dummy_server import create_server


def test_dummy_server_supports_the_remote_client_end_to_end(tmp_path):
    server = create_server(port=0, token="test-key", delay=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    profile = RemoteBackendProfile(
        id="dummy-server",
        name="Dummy-Server",
        driver="noscribe-http-v1",
        url=url,
        api_key="test-key",
    )
    audio_path = tmp_path / "audio.flac"
    audio_path.write_bytes(b"fLaC-dummy-audio")

    try:
        unauthorized = requests.get(f"{url}/v1/models", timeout=2)
        assert unauthorized.status_code == 401

        plugin = RemoteHttpPlugin(profile)
        transcription = []
        info = plugin.transcribe(
            TranscriptionRequest(
                str(audio_path),
                ModelRef("dummy-server", "dummy-transcription"),
                language="de",
            ),
            on_segment=transcription.append,
        )
        diarization = plugin.diarize(
            DiarizationRequest(
                str(audio_path),
                ModelRef("dummy-server", "dummy-diarization"),
            )
        )
        plugin.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert info.language == "de"
    assert [segment.text.strip() for segment in transcription] == [
        "This is a simulated transcription.",
        "It came from the noScribe dummy server.",
    ]
    assert [segment.label for segment in diarization] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert server.jobs == {}
