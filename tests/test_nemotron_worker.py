"""noScribe.nemotron_mp_worker without the model: the turns it hands to main.py,
the features it builds, and when `auto` may choose it."""
from pathlib import Path

import numpy as np
import pytest

from noScribe import nemotron_mp_worker as nw


def test_turns_are_noscribe_turns_sorted_by_start():
    """main.py's overlap assignment stops scanning at the first turn that starts
    after a segment, so the turns must come sorted -- and a turn of no length
    would only be noise in the log."""
    got = nw.turns([{'Start': 3.2, 'End': 4.0, 'Speaker': 1}, {'Start': 0.0, 'End': 2.5, 'Speaker': 0},
                    {'Start': 5.0, 'End': 5.0, 'Speaker': 1}])
    assert got == [{'start': 0, 'end': 2500, 'label': 'SPEAKER_00'},
                   {'start': 3200, 'end': 4000, 'label': 'SPEAKER_01'}]


def test_features_in_chunks_equal_one_pass():
    """The worker computes the spectrogram in chunks through the processor's streaming
    calls, which must not change a single value at the sizes it uses, nor the mask that leaves the last
    frame out, whatever the chunk size and length -- including the last valid frame,
    which the streamed chunks leave out and a separate call has to supply."""
    module = pytest.importorskip(
        'transformers.models.nemotron3_diarization.processing_nemotron3_diarization')
    from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
        NemotronAsrStreamingFeatureExtractor)
    fe = NemotronAsrStreamingFeatureExtractor(
        feature_size=128, hop_length=160, n_fft=512, win_length=400, preemphasis=0.97, sampling_rate=16000)
    processor = module.Nemotron3DiarizationProcessor(fe)
    rng = np.random.default_rng(0)
    for length in (16000 * 3, 16000 * 3 + 77, 16000 * 3 + 160, 16000 * 3 + 350):  # the last: under one window left
        audio = (rng.standard_normal(length) * 0.05).astype(np.float32)
        whole = processor(audio, sampling_rate=16000)
        for chunk_frames in (8, 56, 800, 80_000):
            got = nw.features(processor, audio, 16000, chunk_frames=chunk_frames)
            assert got.input_features.shape == whole.input_features.shape
            if chunk_frames >= 56:
                assert (got.input_features == whole.input_features).all(), (length, chunk_frames)
            else:
                # Pieces of 8 frames, far below the worker's 60,000, differ in the
                # last float32 bit under numpy 2.4 on Linux (at most 9.5e-7, 296 of
                # 301 frames; numpy 2.5 and macOS give exactly the one pass). They
                # are here for the seams they make, not for the arithmetic.
                np.testing.assert_allclose(got.input_features.numpy(), whole.input_features.numpy(),
                                           rtol=0, atol=2e-6, err_msg=f"{(length, chunk_frames)}")
            assert (got.attention_mask == whole.attention_mask.long()).all()


def test_auto_needs_the_model_in_transformers_librosa_and_its_weights_on_disk(monkeypatch, tmp_path):
    """main.py picks the engine in the GUI process, which must not import transformers
    or librosa for it, and `auto` must never start a download. Without librosa
    transformers cannot build the processor (NameError in its feature extractor),
    and the worker has no fallback, so `auto` has to stay with pyannote."""
    import importlib.util
    import subprocess
    import sys
    import types
    import huggingface_hub
    code = ('import sys; from noScribe.nemotron_mp_worker import available; available(); '
            'assert "transformers" not in sys.modules and "librosa" not in sys.modules')
    subprocess.run([sys.executable, '-c', code], check=True, cwd=Path(__file__).resolve().parent.parent)

    (tmp_path / 'models' / 'nemotron3_diarization').mkdir(parents=True)
    knows = types.SimpleNamespace(origin=str(tmp_path / '__init__.py'))
    older = types.SimpleNamespace(origin=str(tmp_path / 'models' / '__init__.py'))
    librosa = types.SimpleNamespace(origin='/site-packages/librosa/__init__.py')
    monkeypatch.setattr(nw, 'model_source', lambda: nw.MODEL_REPO)
    whole = {'config.json', 'processor_config.json', 'model.safetensors'}
    for spec, lib, cached, expected in ((knows, librosa, whole, True), (knows, librosa, set(), False),
                                        (knows, librosa, {'model.safetensors'}, False), (knows, None, whole, False),
                                        (older, librosa, whole, False), (None, librosa, whole, False)):
        specs = {'transformers': spec, 'librosa': lib}
        monkeypatch.setattr(importlib.util, 'find_spec', lambda name, specs=specs: specs[name])
        monkeypatch.setattr(huggingface_hub, 'try_to_load_from_cache',
                            lambda repo, name, cached=cached: f'/cache/{name}' if name in cached else None)
        assert nw.available() is expected, (spec, lib, cached)


@pytest.mark.parametrize("inherited", [None, "0", "false"])
def test_hub_telemetry_is_off_before_transformers_is_imported(tmp_path, inherited):
    """huggingface_hub reads HF_HUB_DISABLE_TELEMETRY once, at import, so the
    worker has to set it before transformers pulls the hub in. A stand-in
    transformers reports the value it was imported under through the worker's
    own error path (the pattern of test_pyannote_font_cache)."""
    import os
    import subprocess
    import sys
    import textwrap
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text("def set_num_threads(n): pass\n")
    (tmp_path / "transformers").mkdir()
    (tmp_path / "transformers" / "__init__.py").write_text(
        'import os\nraise RuntimeError("telemetry=%r" % os.environ.get("HF_HUB_DISABLE_TELEMETRY"))\n')
    probe = textwrap.dedent("""
        import queue
        from noScribe.nemotron_mp_worker import nemotron_proc_entrypoint
        q = queue.Queue()
        nemotron_proc_entrypoint({"audio_path": "unused.wav"}, q)
        while True:
            message = q.get_nowait()
            if message["type"] == "result":
                print(message["error"])
                break
    """)
    repo = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), repo])}
    env.pop("HF_HUB_DISABLE_TELEMETRY", None)
    if inherited is not None:
        env["HF_HUB_DISABLE_TELEMETRY"] = inherited
    proc = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "telemetry='1'" in proc.stdout, proc.stdout


def test_progress_is_repeated_while_the_model_runs(monkeypatch):
    """The model's one call reports nothing for minutes on a long recording; a
    server's stream that stays silent that long is cut by a proxy's idle timeout."""
    import threading
    monkeypatch.setattr(nw, "KEEPALIVE_S", 0.02)
    sent, enough = [], threading.Event()

    class Q:
        def put(self, message):
            sent.append(message["pct"])
            if len(sent) >= 6:
                enough.set()

    with nw.Progress(Q()) as progress:
        progress(5.7)
        assert enough.wait(5)
    assert sent[0] == 0 and sent[1] == 5 and set(sent[2:]) == {5}
    count = len(sent)
    import time
    time.sleep(0.1)
    assert len(sent) == count  # it stops with the block


def test_the_worker_runs_through_the_plugin(tmp_path):
    """The real worker and model, spawned by the shared local backend, on six
    seconds of tone: the result passes the protocol, its probabilities decode at
    the model's frame rate, and the progress ends at 100."""
    if not nw.available():
        pytest.skip("Nemotron's weights are not on this machine")
    import soundfile as sf
    from noScribe.inference import DiarizationRequest, LocalInferenceBackend
    from noScribe.models import ModelRef
    from noScribe.plugins.local_nemotron import LocalNemotronPlugin

    rate = 16000
    t = np.arange(rate * 6) / rate
    tone = np.where(t < 3, np.sin(2 * np.pi * 180 * t), np.sin(2 * np.pi * 260 * t)) * 0.3
    sf.write(tmp_path / "a.wav", (tone * 32767).astype(np.int16), rate)
    progress = []
    plugin = LocalNemotronPlugin(LocalInferenceBackend())
    result = plugin.diarize(DiarizationRequest(str(tmp_path / "a.wav"), ModelRef("local-nemotron", "default")),
                            on_progress=lambda step, pct: progress.append((step, pct)))
    rows = nw.decode_probabilities(result.extensions[nw.EXTENSION_KEY]["probabilities"])
    assert rows.shape[1] == 8 and abs(rows.shape[0] - 600) <= 2
    assert progress[-1] == (nw.PROGRESS_STEP, 100)
    assert all(segment.end_ms > segment.start_ms for segment in result)
