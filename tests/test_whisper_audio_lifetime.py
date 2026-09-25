"""Keep language-detection audio out of transcription's peak memory usage."""
from queue import Queue
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from noScribe.whisper_mp_worker import whisper_proc_entrypoint


@pytest.mark.parametrize(
    "language_name, language_code, is_multilingual, expected_language, expected_decodes",
    [
        ("Auto", None, True, "de", 1),
        ("German", "de", True, "de", 0),
        ("Multilingual", None, True, None, 0),
        ("Auto", None, False, "en", 0),
    ],
)
def test_audio_released_before_transcription(
    monkeypatch, tmp_path, language_name, language_code, is_multilingual,
    expected_language, expected_decodes,
):
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    decoded = []
    transcriptions = []

    def decode(path, sampling_rate):
        assert path == str(audio_path)
        assert sampling_rate == 16000
        audio = np.zeros(16000, dtype=np.float32)
        decoded.append(weakref.ref(audio))
        return audio

    class Model:
        model = SimpleNamespace(is_multilingual=is_multilingual)
        feature_extractor = SimpleNamespace(sampling_rate=16000)

        def detect_language(self, audio, **kwargs):
            assert audio is decoded[-1]()
            return "de", 0.99, []

        def transcribe(self, path, **kwargs):
            assert path == str(audio_path)
            assert all(ref() is None for ref in decoded)
            assert kwargs["language"] == expected_language
            assert kwargs["multilingual"] == (language_name == "Multilingual")
            transcriptions.append(path)
            return iter([]), SimpleNamespace(duration=1.0, language=expected_language)

    monkeypatch.setattr("faster_whisper.WhisperModel", lambda *a, **kw: Model())
    monkeypatch.setattr("faster_whisper.audio.decode_audio", decode)
    messages = Queue()
    whisper_proc_entrypoint(
        {
            "audio_path": str(audio_path),
            "model_path": str(tmp_path),
            "device": "cpu",
            "language": (
                language_code if language_name not in ("Auto", "Multilingual") else None
            ),
            "multilingual": language_name == "Multilingual",
        },
        messages,
    )
    all_messages = list(messages.queue)
    results = [msg for msg in all_messages if msg["type"] == "result"]
    statuses = [msg for msg in all_messages if msg["type"] == "status"]
    assert len(results) == 1
    assert results[0]["ok"], results[0]
    assert results[0]["info"]["duration"] == 1.0
    assert len(decoded) == expected_decodes
    assert transcriptions == [str(audio_path)]
    status_ids = [status["message_id"] for status in statuses]
    assert "vad" in status_ids
    assert "start_transcription" in status_ids
    assert ("language_detect" in status_ids) == (expected_decodes == 1)
