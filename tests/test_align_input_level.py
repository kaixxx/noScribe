"""The aligner sees each window at zero mean and unit variance, whatever the level.

Found on a voice 17 dB below its neighbour: fed the raw signal, the aligner
stamped its words 10-20 s early, into the louder speaker's time. Mechanism and
measurement are in the comment in _Aligner._forward_windows.

The model here is a stand-in whose logits are a plain function of its input,
so any level that reaches it shows up in the emissions.
"""
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from noScribe import voxtral_engine as ve  # noqa: E402
from noScribe.voxtral_engine import EMISSION_WINDOW_SEC, SAMPLE_RATE, _Aligner  # noqa: E402

FRAME = 320  # samples per emission frame, as wav2vec2's 20 ms stride
WIN, MIN_WIN = int(EMISSION_WINDOW_SEC * SAMPLE_RATE), int(0.025 * SAMPLE_RATE)  # as _emission passes them


class _LevelSensitiveModel:
    """logits[t] = [mean, rms, max] of frame t -- no normalisation of its own."""

    def __init__(self):
        self.seen = []

    def __call__(self, batch):
        x = batch[0]
        self.seen.append(x.clone())
        frames = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
        logits = torch.stack([frames.mean(1), frames.pow(2).mean(1).sqrt(), frames.max(1).values], 1)
        return types.SimpleNamespace(logits=logits.unsqueeze(0))


def _aligner():
    al = object.__new__(_Aligner)
    al._torch = torch
    al.device = "cpu"
    al.model = _LevelSensitiveModel()
    return al


def _speech_like(seconds, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    envelope = 0.5 + 0.5 * np.sin(np.linspace(0, 40 * np.pi, n)) ** 2
    return (rng.standard_normal(n) * envelope * 0.1).astype(np.float32)


@pytest.mark.parametrize("gain", [0.14, 0.3, 3.0])  # 0.14 = the 17 dB of the finding
def test_emissions_do_not_depend_on_the_recording_level(gain):
    """Equal up to the extractor's 1e-7 epsilon, which at 17 dB down is ~0.1 % of
    the variance (~2e-3 in these emissions); without the normalisation the gap is
    ~0.4, so the tolerance still tells the two apart."""
    audio = _speech_like(45)  # three windows at 20 s, the last one short
    ref = _aligner()._forward_windows(torch.from_numpy(audio), WIN, MIN_WIN)
    got = _aligner()._forward_windows(torch.from_numpy(audio * gain), WIN, MIN_WIN)
    assert len(ref) == len(got) == 3
    for a, b in zip(ref, got):
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-2)


def _extractor_norm(x):
    """Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm without padding, in numpy."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / np.sqrt(x.var() + 1e-7)


def test_every_window_reaches_the_model_normalised_on_its_own():
    """Per window, not per call: a loud window must not set the scale of a quiet one,
    a DC offset is removed, and a very quiet window keeps the extractor's epsilon."""
    loud, quiet = _speech_like(20, seed=1), _speech_like(20, seed=2) * 0.005 + 0.3
    al = _aligner()
    al._forward_windows(torch.from_numpy(np.concatenate([loud, quiet])), WIN, MIN_WIN)
    assert len(al.model.seen) == 2
    for x, src in zip(al.model.seen, (loud, quiet)):
        np.testing.assert_allclose(x.numpy(), _extractor_norm(src), atol=2e-4)


def test_matches_the_feature_extractor_itself():
    transformers = pytest.importorskip("transformers")
    x = _speech_like(7, seed=3) * 0.02 + 0.01
    al = _aligner()
    al._forward_windows(torch.from_numpy(x), WIN, MIN_WIN)
    want = transformers.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm([x], None)[0]
    np.testing.assert_allclose(al.model.seen[0].numpy(), want, atol=2e-4)


def test_a_short_rest_takes_the_scale_of_the_window_that_ends_with_it():
    """The rest after the last full window must not be scaled on its own: 0.3 s of
    room tone after speech would otherwise reach the model at unit variance."""
    speech = _speech_like(20, seed=4)
    tone = (np.random.default_rng(5).standard_normal(int(0.3 * SAMPLE_RATE)) * 1e-3).astype(np.float32)
    x = np.concatenate([speech, tone])
    al = _aligner()
    al._forward_windows(torch.from_numpy(x), WIN, MIN_WIN)
    assert len(al.model.seen) == 2
    ref = x[len(x) - WIN:]
    want = (tone - ref.astype(np.float64).mean()) / np.sqrt(ref.astype(np.float64).var() + 1e-7)
    np.testing.assert_allclose(al.model.seen[1].numpy(), want, atol=2e-4)
    assert float(al.model.seen[1].std()) < 0.05  # stays room tone, not speech


def test_silence_does_not_blow_up():
    """An all-zero window has no variance; the epsilon keeps it finite."""
    al = _aligner()
    out = al._forward_windows(torch.zeros(5 * SAMPLE_RATE), WIN, MIN_WIN)
    assert np.isfinite(out[0].numpy()).all()


# --------------------------------------------------------------------------- #
# The level-matched second alignment (_AlignerPool.align_words)
# --------------------------------------------------------------------------- #


def test_level_matching_survives_digital_silence():
    """The measurement's envelope (scipy's running filter) dipped below zero after a
    loud stretch; its root was NaN, the percentile floor NaN, and with it the whole
    pass -- every word then landed in the last second. Exact zeros must stay finite."""
    x = np.concatenate([_speech_like(10, seed=6), np.zeros(8 * SAMPLE_RATE, np.float32), _speech_like(10, seed=7) * 0.1])
    y = ve._level_matched(x)
    assert np.isfinite(y).all()
    assert y.dtype == np.float32 and len(y) == len(x)


@pytest.mark.parametrize("pause_sec, lifted", [(2, False), (10, True)])
def test_level_matching_lifts_the_quiet_voice_and_a_pause_only_past_the_floor(pause_sec, lifted):
    """The floor is the envelope's LEVEL_FLOOR_PERCENTILE percentile: a pause under
    that share of the pass stays a pause, a longer one is lifted with the rest --
    which is what the coverage check in _AlignerPool.align_words guards against."""
    loud, quiet = _speech_like(10, seed=8), _speech_like(10, seed=9) * 0.1   # 20 dB apart
    room = (np.random.default_rng(10).standard_normal(pause_sec * SAMPLE_RATE) * 1e-4).astype(np.float32)
    y = ve._level_matched(np.concatenate([loud, room, quiet]))
    rms = lambda a: float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))
    n, p = 10 * SAMPLE_RATE, pause_sec * SAMPLE_RATE
    a, b, c = y[:n], y[n + SAMPLE_RATE // 2:n + p - SAMPLE_RATE // 2], y[n + p:]
    assert 0.7 < rms(c) / rms(a) < 1.4   # the quiet voice now as loud as the loud one
    assert (rms(b) > 0.3 * rms(a)) == lifted


def test_level_matching_lifts_no_more_than_its_cap():
    """Over 30 % near-silence puts the percentile floor at the dither's own level,
    ~100 dB down: uncapped, the dither would come out as loud as the speech. An
    all-zero pass has nothing to level and comes back as it was."""
    speech = _speech_like(5, seed=13)
    dither = (np.random.default_rng(14).standard_normal(8 * SAMPLE_RATE) * 1e-6).astype(np.float32)
    y = ve._level_matched(np.concatenate([speech, dither]))
    assert np.isfinite(y).all()
    assert np.abs(y[-4 * SAMPLE_RATE:]).max() < 0.2   # capped: -100 dB + 60 dB; uncapped peaks near 4
    silent = np.zeros(4 * SAMPLE_RATE, np.float32)
    assert np.array_equal(ve._level_matched(silent), silent)


def test_level_matching_moves_its_mean_as_the_measured_filter_did():
    """Agrees with the scipy envelope the measurement used, edges included."""
    ndimage = pytest.importorskip("scipy.ndimage")
    x = np.concatenate([_speech_like(3, seed=11), _speech_like(3, seed=12) * 0.05])
    env = np.sqrt(np.maximum(ndimage.uniform_filter1d(np.square(x, dtype=np.float64),
                                                      int(ve.LEVEL_ENVELOPE_SEC * SAMPLE_RATE)), 0))
    want = x / np.maximum(env, np.percentile(env, ve.LEVEL_FLOOR_PERCENTILE))
    np.testing.assert_allclose(ve._level_matched(x), want, rtol=1e-4, atol=1e-6)


def test_speech_coverage_counts_speech_with_a_word_nearby():
    turns = [(0.0, 10.0, "A"), (20.0, 30.0, "B")]
    words = [{"start": 100.0 + t, "end": 100.0 + t + 0.3} for t in np.arange(0, 10, 0.8)]  # t_offset 100
    assert abs(ve._speech_coverage(words, turns, 100.0) - 0.5) < 0.03
    assert ve._speech_coverage([], turns, 100.0) == 0.0


class _FakeAligner:
    """Spreads the words evenly over `first` seconds of the window on the first call
    and over `second` seconds on the second (level-matched) one."""

    def __init__(self, first=20.0, second=40.0, second_prob=0.9):
        self.calls, self.spans, self.second_prob = [], (first, second), second_prob

    def align_words(self, words, window, t_offset=0.0):
        self.calls.append(window)
        span = self.spans[min(len(self.calls), 2) - 1]
        step = span / len(words)
        prob = 0.9 if len(self.calls) == 1 else self.second_prob
        return [{"word": w, "start": t_offset + i * step, "end": t_offset + i * step + 0.3, "prob": prob}
                for i, w in enumerate(words)]


def _pool(fake):
    pool = ve._AlignerPool("en", None)
    pool.aligner_for = lambda text, remember=True: fake
    return pool


WORDS = [f"w{i}" for i in range(60)]
WINDOW = np.zeros(40 * SAMPLE_RATE, np.float32)
TWO = [(0.0, 20.0, "A"), (20.0, 40.0, "B")]


def test_quiet_speakers_empty_turn_is_filled_by_the_level_matched_alignment():
    fake = _FakeAligner()
    got = _pool(fake).align_words(WORDS, WINDOW, 5.0, turns=TWO)
    assert len(fake.calls) == 2
    assert max(w["end"] for w in got) > 5.0 + 30.0   # B's turn has words now


@pytest.mark.parametrize("turns", [None, [(0.0, 20.0, "A"), (20.0, 40.0, "A")]])
def test_without_two_diarized_speakers_one_alignment_is_all(turns):
    fake = _FakeAligner()
    _pool(fake).align_words(WORDS, WINDOW, 5.0, turns=turns)
    assert len(fake.calls) == 1


def test_no_second_alignment_where_it_could_not_win():
    """Coverage >= 1 - margin already: the second could never add the margin."""
    fake = _FakeAligner()
    _pool(fake).align_words(WORDS, WINDOW, 5.0, turns=[(0.0, 20.0, "A"), (19.5, 20.0, "B")])
    assert len(fake.calls) == 1


@pytest.mark.parametrize("margin, taken", [(0.05, True), (0.4, False)])
def test_the_second_alignment_is_taken_only_past_the_margin(monkeypatch, margin, taken):
    """First covers ~31 % of the speech, second ~61 %: +30 points clears the default
    margin and not one of 40 -- and 31 % is below 1 - 0.4, so it does run both."""
    monkeypatch.setattr(ve, "LEVEL_COVERAGE_MARGIN", margin)
    fake = _FakeAligner(first=12.0, second=24.0)
    got = _pool(fake).align_words(WORDS, WINDOW, 5.0, turns=TWO)
    assert len(fake.calls) == 2
    assert (max(w["end"] for w in got) > 5.0 + 20.0) == taken


def test_a_second_alignment_that_covers_no_more_is_not_taken():
    fake = _FakeAligner(first=20.0, second=20.0)
    got = _pool(fake).align_words(WORDS, WINDOW, 5.0, turns=TWO)
    assert len(fake.calls) == 2 and max(w["end"] for w in got) < 5.0 + 21.0


def test_a_second_alignment_spread_by_the_fallback_is_not_taken():
    """Evenly spread words (prob 0.0, a window the DP could not align) cover the
    speech well without being aligned at all; they must not win on coverage."""
    fake = _FakeAligner(second_prob=0.0)
    got = _pool(fake).align_words(WORDS, WINDOW, 5.0, turns=TWO)
    assert len(fake.calls) == 2 and max(w["end"] for w in got) < 5.0 + 21.0
