"""The log-Mel clamp floor from a percentile instead of the maximum.

mlx-voxtral clamps the log-Mel at `log_max - 8` with log_max taken over the
whole input, so one loud cell -- a door slam -- raises the floor for the whole
pass and costs a measured +1.52 WER points at 28 dB
(docs/voxtral-mel-clamp-floor.md). The engine therefore replaces the processor's
feature extractor with one whose floor comes from a percentile of the
spectrogram (MEL_FLOOR_PERCENTILE).

Two things are pinned. First, that the replacement changes *only* the floor:
with pct=100 its output must be bit-identical to the library's own path, on
loud, quiet and silent input -- the proof that the spectrogram underneath is
the reference one, and the guard that catches the library's mel path drifting
away from our copy of it. Second, the mechanism: a transient that lifts the
maximum floor by more than 20 dB moves the percentile floor by a fraction of a
dB. Not by nothing -- the transient's own cells displace the top of the
distribution a little -- which is why the model-level test at the end asserts
a small difference, not identical text.

The numpy tests run everywhere; the library comparisons need mlx-voxtral, and
the last test the local model build.
"""
import importlib.resources as impres

import numpy as np
import pytest

from noScribe.voxtral_engine import (MEL_FLOOR_PERCENTILE, MEL_FLOOR_RANGE,
                                     SAMPLE_RATE, _PercentileFloorFeatures,
                                     _Voxtral, _local_copy, clamp_log_mel)

_MODEL = _local_copy("voxtral-mini-8bit")


def _reference_clamp(x):
    """mlx-voxtral's own clamp and affinity, spelled out in float32."""
    x = np.asarray(x, dtype=np.float32)
    floor = x.max() - np.float32(MEL_FLOOR_RANGE)
    return (np.maximum(x, floor) + np.float32(4.0)) / np.float32(4.0)


def _floor_db(features):
    """The clamp floor a feature block was built with, in dB up to a constant:
    the floor is the smallest value in the block (every block has cells on it
    -- the lowest mel filters of the 128-band Slaney bank are empty, and the
    30 s padding is silent), x4 undoes the affinity's scale (its offset
    cancels in the differences this is used for), x10 turns log10 into dB."""
    return float(np.asarray(features).min()) * 40.0


def _with_block(clean, at=0.45, seconds=0.1):
    """The dose instrument of the clamp-floor measurements: a full-scale block
    dropped into a quiet recording."""
    spiked = clean.copy()
    pos = int(at * len(spiked))
    spiked[pos:pos + int(seconds * SAMPLE_RATE)] = 1.0
    return spiked


def _raw_spectrogram(seed, top, seconds=30):
    """A log-Mel-shaped block whose maximum lands at `top`."""
    rng = np.random.default_rng(seed)
    x = rng.normal(-4.0, 2.0, size=(128, 100 * seconds)).astype(np.float32)
    return x - x.max() + np.float32(top)


# --------------------------------------------------------------------------- #
# numpy only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("top", [2.0, 0.4, -1.3, -6.0])
def test_percentile_100_is_the_reference_clamp(top):
    """With the percentile at 100 the function is the library's formula --
    bit for bit, in float32, whichever range the maximum falls in."""
    x = _raw_spectrogram(0, top)
    out = clamp_log_mel(x, 100)
    assert out.dtype == np.float32
    assert np.array_equal(out, _reference_clamp(x))


def test_the_percentile_floor_never_sits_above_the_reference_floor():
    """The one direction the change can go: cells the reference keeps are kept
    unchanged, cells it clamps may only come out lower. Lowering the floor
    exposes noise detail the model ignores; raising it destroys quiet speech."""
    x = _raw_spectrogram(1, 1.5)
    ref = _reference_clamp(x)
    out = clamp_log_mel(x)
    kept = x >= x.max() - np.float32(MEL_FLOOR_RANGE)
    assert np.array_equal(out[kept], ref[kept])
    assert np.all(out <= ref)
    assert np.any(out < ref)


def test_the_cap_bounds_the_floor_but_never_raises_it_past_the_reference():
    """With `cap`, the statistic never sits more than cap below the maximum:
    on a spectrogram whose 99th percentile is far below the top, the capped
    floor lands exactly at max - cap - 8, and at pct=100 the cap is inert
    (the percentile is already the maximum), so the bit-identity control
    covers the capped path too."""
    x = _raw_spectrogram(3, 1.5)
    assert np.array_equal(clamp_log_mel(x, 100, cap=np.float32(2.0)),
                          _reference_clamp(x))
    uncapped_top = np.float32(np.percentile(x, MEL_FLOOR_PERCENTILE))
    assert uncapped_top < x.max() - np.float32(2.0)   # the cap has work to do
    capped = clamp_log_mel(x, MEL_FLOOR_PERCENTILE, cap=np.float32(2.0))
    expected_floor = (x.max() - np.float32(2.0) - np.float32(MEL_FLOOR_RANGE)
                      + np.float32(4.0)) / np.float32(4.0)
    assert capped.min() == expected_floor
    assert np.all(capped >= clamp_log_mel(x, MEL_FLOOR_PERCENTILE))
    assert np.all(capped <= _reference_clamp(x))


def test_a_burst_of_loud_cells_moves_the_reference_floor_but_not_the_percentile():
    """A 0.1 s transient is a couple of hundred cells among 1.5 million on a
    two-minute pass. The maximum floor follows it all the way; the percentile
    floor moves by the rank displacement those cells cause at the top of the
    distribution -- a fraction of a dB at 99, where the top percent of a
    120 s pass is 15 000 cells. (At 99.9 the same burst would move it by
    0.8 dB here and by 5 dB on a single 30 s block; that is why 99.)"""
    clean = _raw_spectrogram(2, 1.5, seconds=120)
    spiked = clean.copy()
    spiked[:20, 5400:5410] = np.float32(4.5)     # 30 dB over the maximum, low bins, 10 frames
    rise_max = _floor_db(clamp_log_mel(spiked, 100)) - _floor_db(clamp_log_mel(clean, 100))
    rise_pct = _floor_db(clamp_log_mel(spiked)) - _floor_db(clamp_log_mel(clean))
    assert rise_max > 25.0
    assert 0.0 <= rise_pct < 0.5


# --------------------------------------------------------------------------- #
# against the library
# --------------------------------------------------------------------------- #
def _speech_like(seconds, level, seed=0):
    """Harmonics under a syllable-rate envelope plus a little noise -- enough
    structure for the mel cells to spread like speech does, at any level."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    x = sum(np.sin(2 * np.pi * f * t + k)
            for k, f in enumerate((140, 280, 560, 1100, 2300))) * env
    x += 0.05 * rng.standard_normal(len(t))
    return (x / np.abs(x).max() * level).astype(np.float32)


def _library_features(audio):
    from mlx_voxtral.audio_processing import VoxtralFeatureExtractor
    out = VoxtralFeatureExtractor()(audio, sampling_rate=SAMPLE_RATE,
                                    return_tensors="mlx")
    return np.array(out["input_features"])


@pytest.mark.parametrize("audio", [
    pytest.param(_speech_like(61.0, 0.9), id="full-scale, three chunks with padding"),
    pytest.param(_speech_like(20.0, 0.06), id="quiet: log_max in [-2, 0)"),
    pytest.param(_speech_like(8.0, 0.001), id="-60 dB"),
    pytest.param(np.zeros(5 * SAMPLE_RATE, np.float32), id="silence"),
])
def test_pct_100_is_bit_identical_to_the_library(audio):
    """The acceptance control from the work order: at percentile 100 the
    engine's extractor must reproduce mlx-voxtral's features bit for bit. The
    quiet case is the one the library's `global_max` lever cannot pass."""
    pytest.importorskip("mlx_voxtral")
    ours = np.array(_PercentileFloorFeatures(100)(audio)["input_features"])
    ref = _library_features(audio)
    assert ours.shape == ref.shape
    assert ours.dtype == ref.dtype == np.float32
    assert np.array_equal(ours, ref)


def test_a_transient_barely_moves_the_percentile_floor_on_real_features():
    """A full-scale 0.1 s block in a quiet recording, the dose instrument of
    the clamp-floor write-up: the library's floor follows it by tens of dB,
    the engine's by a fraction of one."""
    pytest.importorskip("mlx_voxtral")
    clean = _speech_like(60.0, 0.1)
    spiked = _with_block(clean)
    ex = _PercentileFloorFeatures()
    rise_max = _floor_db(_library_features(spiked)) - _floor_db(_library_features(clean))
    rise_pct = (_floor_db(ex(spiked)["input_features"])
                - _floor_db(ex(clean)["input_features"]))
    assert rise_max > 20.0
    # The block also *replaces* a tenth of a second of speech cells, so the
    # percentile may move a hair either way -- hence the absolute bound.
    assert abs(rise_pct) < 1.0


def test_the_floor_does_not_depend_on_the_padding():
    """The block is zero-padded to a 30 s multiple and the padding cells sit
    at the -10 minimum, so the percentile has to be taken over the input's own
    frames: a 10 s clip in a 30 s block must get the floor of its own
    spectrogram, not the one the block's percentile would give (several dB
    lower, and 18 dB lower on a 5 s clip)."""
    pytest.importorskip("mlx_voxtral")
    import mlx.core as mx
    from mlx_voxtral.audio_processing import log_mel_spectrogram
    clip = _speech_like(10.0, 0.3)
    # The unpadded spectrogram through the library's no-clamp lever (its
    # affinity inverted; lossy in the last bits, which a dB check ignores).
    raw = np.array(log_mel_spectrogram(mx.array(clip), global_max=-1e6)) * 4.0 - 4.0
    own_floor = (np.percentile(raw, MEL_FLOOR_PERCENTILE) - MEL_FLOOR_RANGE) * 10.0
    block = np.pad(raw, ((0, 0), (0, 3000 - raw.shape[1])), constant_values=-10.0)
    block_floor = (np.percentile(block, MEL_FLOOR_PERCENTILE) - MEL_FLOOR_RANGE) * 10.0
    assert own_floor - block_floor > 1.0          # the two answers differ
    got = _floor_db(_PercentileFloorFeatures()(clip)["input_features"]) - 40.0
    assert abs(got - own_floor) < 0.01


def test_the_engine_installs_the_percentile_floor(monkeypatch):
    """_Voxtral.__init__ swaps the processor's extractor, so the whole engine
    -- every pass, every retry rung, the head probe -- sees the same features.
    Driven with stubs: no weights, no model, just the constructor."""
    mlx_voxtral = pytest.importorskip("mlx_voxtral")
    from types import SimpleNamespace

    model = SimpleNamespace(language_model=None, lm_head=None)
    proc = SimpleNamespace(feature_extractor="library", _special_token_ids=None)
    monkeypatch.setattr(mlx_voxtral, "load_voxtral_model",
                        lambda repo, dtype=None: (model, None))
    monkeypatch.setattr(mlx_voxtral.VoxtralProcessor, "from_pretrained",
                        staticmethod(lambda repo: proc))
    vox = _Voxtral("stub")
    assert isinstance(vox.proc.feature_extractor, _PercentileFloorFeatures)
    assert vox.proc.feature_extractor.pct == MEL_FLOOR_PERCENTILE


# --------------------------------------------------------------------------- #
# at the transcript, on the local model
# --------------------------------------------------------------------------- #
def _word_edits(a, b):
    """Words touched by the edit between two transcripts (Levenshtein on
    words, counting substitutions, insertions and deletions)."""
    a, b = a.split(), b.split()
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


@pytest.mark.skipif(_MODEL is None,
                    reason="local models/voxtral-mini-8bit not present (opt-in)")
def test_a_transient_costs_the_transcript_far_less_under_the_percentile_floor(tmp_path):
    """The knock case from the work order at the transcript: two minutes of
    the bundled interview, attenuated to a quiet recording, once clean and
    once with a full-scale 0.1 s block at 45 %. Under the library floor the
    block rewrites words all over the passage; under the engine's floor the
    residual shift of a fraction of a dB may still flip a word or two, and
    the test allows exactly that -- a handful of words, and fewer than the
    library loses -- rather than pretending the text is identical."""
    pytest.importorskip("mlx_voxtral")
    sf = pytest.importorskip("soundfile")
    from mlx_voxtral.audio_processing import VoxtralFeatureExtractor
    from noScribe.audio.convert import ToWav

    # The pipeline's own conversion to 16 kHz mono, as in test_pyannote_waveform_load.
    wav = tmp_path / "interview.wav"
    with ToWav(impres.files("tests") / "data" / "interview.mp3", wav) as towav:
        towav.stop_after(132_000)
        while towav.convert():
            pass
    audio, sr = sf.read(str(wav), dtype="float32")
    assert sr == SAMPLE_RATE
    clean = audio[12 * SAMPLE_RATE:132 * SAMPLE_RATE]
    clean = (clean / np.abs(clean).max() * 0.1).astype(np.float32)   # -20 dB
    spiked = _with_block(clean)

    vox = _Voxtral(_MODEL)
    engine_floor = vox.proc.feature_extractor

    def edits(extractor):
        vox.proc.feature_extractor = extractor
        a = vox.transcribe_array(clean, "de")
        b = vox.transcribe_array(spiked, "de")
        return _word_edits(a, b), len(a.split())

    edits_pct, n_words = edits(engine_floor)
    edits_max, _ = edits(VoxtralFeatureExtractor())
    assert n_words > 200
    assert edits_pct < edits_max, (edits_pct, edits_max)
    assert edits_pct <= 0.02 * n_words, (edits_pct, n_words)
