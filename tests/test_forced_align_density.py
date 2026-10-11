"""When forced_align can run at all -- and what happens when it cannot.

The DP demands `n_frames >= len(targets) + n_repeats` (as torchaudio did before
it), where n_repeats counts the pairs of immediately adjacent identical tokens
(a CTC path has to place a blank between two identical labels). The old
pre-check left only ~5.3% headroom, but German text has 4-30% such pairs -- the
windows in between got through, forced_align raised, and the `except` silently
turned that into evenly spread timestamps for the whole window.
"""
import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from noScribe import ctc_align  # noqa: E402
from noScribe.voxtral_engine import (  # noqa: E402
    SAMPLE_RATE,
    _Aligner,
)

VOCAB_SIZE = 30


def _stub_aligner():
    """Aligner without weights: real split/check logic, faked emissions.

    The emission yields exactly as many frames as _predict_frames predicts, so
    that routing and the leaf check see the same number in the test.
    """
    al = object.__new__(_Aligner)
    al.vocab = {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
    al.blank = 0
    al.delim = None
    gen = torch.Generator().manual_seed(0)
    al._emission = lambda audio: torch.log_softmax(
        torch.rand((al._predict_frames(len(audio)), VOCAB_SIZE), generator=gen),
        dim=-1).numpy()
    return al


# --------------------------------------------------------------------------- #
# The condition itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tokens", [
    [7, 18, 18, 11, 25],           # one pair
    [7, 18, 18, 11, 25, 25],       # two pairs
    [4, 4, 4, 9, 9],               # a triple run counts as two pairs
    [3, 1, 4, 1, 5],               # no repetition
])
def test_repeat_count_matches_what_the_dp_demands(tokens):
    """Measured against the DP itself: the smallest T at which forced_align
    no longer raises is exactly len(tokens) + _adjacent_repeats."""
    need = _Aligner._adjacent_repeats(tokens) + len(tokens)
    smallest = None
    for T in range(len(tokens), len(tokens) + 12):
        emission = np.zeros((T, VOCAB_SIZE), dtype=np.float32)
        try:
            ctc_align.forced_align(emission, tokens, blank=0)
            smallest = T
            break
        except ValueError:
            continue
    assert smallest == need


def test_a_window_in_the_repeat_band_does_not_degrade_silently(caplog):
    """Exactly the case the old 0.95 bound let through: barely enough frames
    for the tokens, but not for the repeats. No exception may escape -- and if
    an even spread is all that is left, it has to be in the log, otherwise a
    broken window cannot be told from a good one. Until 2026-08 this stood
    only in this docstring: the leaf check spread silently, only the `except`
    path next to it warned."""
    al = _stub_aligner()
    words = ["aabb"] * 40                      # every word brings two pairs along
    audio = np.zeros(int(3.5 * SAMPLE_RATE), dtype=np.float32)
    tokens, _ = al._tokenize(words)
    frames = al._predict_frames(len(audio))
    assert len(tokens) <= frames, "test setup: the tokens must just about fit"
    assert len(tokens) + al._adjacent_repeats(tokens) > frames, \
        "test setup: only the repeats may push it over"

    with caplog.at_level(logging.WARNING, logger="noScribe.voxtral_engine"):
        out = al.align_words(words, audio)      # must not raise
    assert len(out) == len(words)
    assert all(w["end"] >= w["start"] for w in out)
    # After splitting, some halves manage a real alignment; at least one leaf
    # stays too dense and is spread -- and has to say so.
    assert any(w["prob"] == 0.0 for w in out), "test setup: no leaf was spread"
    spread_lines = [r for r in caplog.records if "evenly spread" in r.getMessage()]
    assert spread_lines, "spread, but nothing in the log"
    assert "repeats need more frames" in spread_lines[0].getMessage()


def test_prediction_matches_the_emission_it_replaces():
    """The split decision runs on predicted frames, so that a window which is
    going to be split anyway does not pay for a discarded wav2vec2 pass.
    Prediction and real emission therefore have to agree."""
    al = _stub_aligner()
    for seconds in (0.5, 2.5, 20.0, 20.5, 47.3):
        audio = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
        assert al._emission(audio).shape[0] == al._predict_frames(len(audio))


# --------------------------------------------------------------------------- #
# Partial coverage: the salvage prefix
# --------------------------------------------------------------------------- #
def test_a_partial_prefix_has_its_own_entry_point():
    """The split cuts the audio by the words' character share -- admissible
    only when the words fill the window. A salvage prefix does not; for that
    there is `align_prefix` (see test_salvage_prefix_alignment). What is
    pinned here is that `align_words` does not accept this case in the first
    place, instead of silently computing it wrong."""
    import inspect
    assert "words_span_audio" not in inspect.signature(_Aligner.align_words).parameters
    assert hasattr(_Aligner, "align_prefix")


def test_a_full_window_is_still_split_as_before():
    """Counter-check: full coverage (the normal case) still splits."""
    al = _stub_aligner()
    words = ["abcde"] * 2000
    audio = np.zeros(1100 * SAMPLE_RATE, dtype=np.float32)
    out = al.align_words(words, audio)
    assert len(out) == len(words)
    assert any(w["prob"] != 0.0 for w in out), "nothing at all was aligned"


def test_dense_recursion_is_bounded():
    """Halving does not lower the density: the audio is cut at the same
    character share as the words, so tokens/frames is the same in both halves
    and the check fires again on every level. Unchecked, this ran to full
    depth and pushed 10-13 times the chunk through wav2vec2, only to end in
    leaves that spread evenly anyway."""
    al = _stub_aligner()
    seen = []
    inner = al._emission
    al._emission = lambda audio: (seen.append(len(audio)), inner(audio))[1]

    words = ["abcdefghij"] * 400                # very dense for short audio
    audio = np.zeros(int(4.0 * SAMPLE_RATE), dtype=np.float32)
    out = al.align_words(words, audio)

    assert len(out) == len(words)
    # 2 levels of density split => at most 2^2 leaves, plus their emissions.
    assert len(seen) <= 2 ** _Aligner.MAX_DENSE_SPLIT_DEPTH, len(seen)
    assert sum(seen) <= 3 * len(audio), "too much audio computed more than once"
