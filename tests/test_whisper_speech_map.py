"""With speaker detection, Whisper takes its speech map from the diarization.

Silero dropped a quiet voice before Whisper could hear it; the diarization's
turns keep it (the measurement is above _speech_map_from in
noScribe/whisper_mp_worker.py). The swap works through the one name
faster-whisper imports into its transcribe module, so the guard that matters
most is that it still looks the map up there: a faster-whisper release that
stops doing so would leave the feature silently dead.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from noScribe import whisper_mp_worker as w

SR = 16000


def test_turns_become_padded_chunks_with_short_gaps_closed():
    turns = [[1.0, 2.0], [2.3, 3.0], [5.0, 6.0]]   # 0.3 s gap closed, 2 s gap kept
    got = w.speech_map(turns, 10 * SR)
    pad = int(w.SPEECH_MAP_PAD_S * SR)
    assert got == [{'start': SR - pad, 'end': 3 * SR + pad}, {'start': 5 * SR - pad, 'end': 6 * SR + pad}]


def test_chunks_stay_inside_the_audio_and_empty_turns_vanish():
    got = w.speech_map([[-1.0, 0.5], [3.0, 3.0], [9.8, 12.0]], 10 * SR)
    assert got == [{'start': 0, 'end': int(0.55 * SR)}, {'start': int(9.75 * SR), 'end': 10 * SR}]


@pytest.mark.parametrize("min_gap, chunks", [(None, 2), (1.0, 1)])
def test_the_gap_to_close_is_the_constant(monkeypatch, min_gap, chunks):
    turns = [[1.0, 2.0], [2.7, 3.0]]   # 0.6 s apart, 0.5 s after padding
    if min_gap is not None:
        monkeypatch.setattr(w, "SPEECH_MAP_MIN_GAP_S", min_gap)
    assert len(w.speech_map(turns, 10 * SR)) == chunks


ft = pytest.importorskip("faster_whisper.transcribe")


def test_faster_whisper_still_asks_its_module_for_the_speech_map():
    """transcribe() and detect_language() must look get_speech_timestamps up as a
    module global -- the name _speech_map_from replaces."""
    assert hasattr(ft, "get_speech_timestamps")
    for fn in (ft.WhisperModel.transcribe, ft.WhisperModel.detect_language):
        assert "get_speech_timestamps" in fn.__code__.co_names, fn.__qualname__


def test_the_swap_answers_with_the_turns_and_is_undone():
    silero = ft.get_speech_timestamps
    audio = np.zeros(10 * SR, np.float32)
    with w._speech_map_from([[1.0, 2.0]]) as mapped:
        assert mapped
        assert ft.get_speech_timestamps(audio, None) == w.speech_map([[1.0, 2.0]], len(audio))
    assert ft.get_speech_timestamps is silero


def test_the_swap_is_undone_when_the_call_inside_fails():
    silero = ft.get_speech_timestamps
    with pytest.raises(RuntimeError):
        with w._speech_map_from([[1.0, 2.0]]):
            raise RuntimeError("transcription failed")
    assert ft.get_speech_timestamps is silero


@pytest.mark.parametrize("turns", [None, []])
def test_without_turns_silero_stays(turns):
    silero = ft.get_speech_timestamps
    with w._speech_map_from(turns) as mapped:
        assert not mapped and ft.get_speech_timestamps is silero


def test_main_hands_the_diarization_to_the_whisper_worker(tmp_path, monkeypatch):
    """The worker's arguments, taken where main.py starts the child process."""
    pytest.importorskip("tkinter")
    import queue
    import noScribe.main as m
    monkeypatch.setattr(m, "config", {})
    seen = []

    class FakeProcess:
        def __init__(self, target, args):
            self.args, self.exitcode = args, 0

        def start(self):
            seen.append(self.args[0])
            self.args[1].put({"type": "result", "ok": True, "info": {"duration": 1.0}})

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(m.mp, "get_context", lambda method: SimpleNamespace(Queue=queue.Queue, Process=FakeProcess))
    app = m.HeadlessApp()
    job = m.create_transcription_job(audio_file=str(tmp_path / "a.wav"), transcript_file=str(tmp_path / "a.html"),
                                     speaker_detection="auto", cli_mode=True)
    app._run_whisper_subprocess_stream("x.wav", job, lambda seg: None,
                                       [{"start": 1500, "end": 4200, "label": "SPEAKER_00"}])
    app._run_whisper_subprocess_stream("x.wav", job, lambda seg: None)
    assert [a["speech_turns"] for a in seen] == [[[1.5, 4.2]], None]


def test_the_worker_transcribes_inside_the_swap(tmp_path, monkeypatch):
    """The wiring itself: whisper_proc_entrypoint must hand faster-whisper the turns'
    map while transcribe() runs, and while detect_language() runs for 'Auto' --
    a stand-in model asks for it in both (the pattern of test_whisper_audio_lifetime)."""
    from queue import Queue
    asked = []
    ask = lambda kwargs: ft.get_speech_timestamps(np.zeros(10 * SR, np.float32), kwargs.get("vad_parameters"))

    class Model:
        model = SimpleNamespace(is_multilingual=True)
        feature_extractor = SimpleNamespace(sampling_rate=SR)

        def detect_language(self, audio, **kwargs):
            asked.append(("detect", ask(kwargs)))
            return "de", 1.0, []

        def transcribe(self, path, **kwargs):
            asked.append(ask(kwargs))
            return iter([]), SimpleNamespace(duration=10.0, language="de")

    monkeypatch.setattr("faster_whisper.WhisperModel", lambda *a, **kw: Model())
    monkeypatch.setattr("faster_whisper.audio.decode_audio", lambda path, sampling_rate: np.zeros(SR, np.float32))
    (tmp_path / "a.wav").touch()
    want = w.speech_map([[1.0, 2.0]], 10 * SR)
    for language, expected in (("German", [want]), ("Auto", [("detect", want), want])):
        asked.clear()
        messages = Queue()
        w.whisper_proc_entrypoint({
            "whisper_model": SimpleNamespace(path=tmp_path), "device": "cpu", "audio_path": str(tmp_path / "a.wav"),
            "language_name": language, "language_code": "de", "speech_turns": [[1.0, 2.0]]}, messages)
        results = [m for m in list(messages.queue) if m["type"] == "result"]
        assert len(results) == 1 and results[0]["ok"], results
        assert asked == expected, language


@pytest.mark.parametrize("detection, handed", [("auto", True), ("none", False)])
def test_the_diarization_reaches_whisper(tmp_path, monkeypatch, detection, handed):
    """Through the pipeline: with speaker detection the turns go to Whisper, without
    it there are none (and diarization is None rather than unbound)."""
    pytest.importorskip("tkinter")
    import soundfile as sf
    import noScribe.main as m
    monkeypatch.setattr(m, "config_dir", str(tmp_path))
    monkeypatch.setattr(m, "config", {})
    audio = tmp_path / "a.wav"
    sf.write(audio, np.zeros(SR * 3, dtype=np.int16), SR)
    app = m.HeadlessApp()
    turns = [{"start": 0, "end": 2000, "label": "SPEAKER_00"}]
    monkeypatch.setattr(app, "_run_diarize_subprocess", lambda *a, **k: turns)
    seen = []
    monkeypatch.setattr(app, "_run_whisper_subprocess_stream",
                        lambda *args: seen.append(args) or SimpleNamespace(duration=3.0))
    job = m.create_transcription_job(audio_file=str(audio), transcript_file=str(tmp_path / "a.html"),
                                     speaker_detection=detection, cli_mode=True)
    app._process_single_job(job)
    assert seen, "Whisper was not started"
    assert seen[0][3] == (turns if handed else None)
