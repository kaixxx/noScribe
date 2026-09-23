"""`prob` on a word is a probability, not a log probability.

`voxtral_engine`'s module docstring promises the segment shape it streams is
compatible with `whisper_mp_worker`'s, and that worker puts faster-whisper's
`word.probability` -- a value in [0, 1] -- into this field. The forced aligner
gets its scores from `noScribe.ctc_align.merge_tokens` (torchaudio's before it),
which returns log probabilities (<= 0, measured -11.96 .. -0.0014), so the two
paths disagreed on what the field meant.

0.0 has a second job here: it is the sentinel for a word that was spread evenly
or interpolated rather than actually aligned, and the salvage guard tests
exactly that. Converting with exp() keeps the sentinel intact and, as a bonus,
stops a genuine score of exactly 0.0 from colliding with it -- it now maps to
1.0.

No `importorskip` here on purpose: `voxtral_engine` imports stdlib only at
module level and these tests never load a model, so they run on the Linux CI
too -- which has no `transformers` and would otherwise skip the file that pins
this contract.
"""
import math

import pytest

from noScribe.voxtral_engine import _Aligner


class _Span:
    def __init__(self, token, start, end, score):
        self.token, self.start, self.end, self.score = token, start, end, score


def _aligner():
    """An _Aligner without loading a model -- only the stamp maths is used.

    Same shape as the stubs in test_forced_align_cap.py and
    test_salvage_prefix_alignment.py.
    """
    al = object.__new__(_Aligner)
    al.blank = 0
    return al


def test_log_scores_become_probabilities():
    # Two words, one token each, with log-probabilities the aligner really emits.
    spans = [_Span(5, 0, 10, math.log(0.5)), _Span(7, 10, 20, math.log(0.25))]
    out = _aligner()._stamps_from_spans(spans, ["a", "b"], [0, 1], fps=100.0,
                                        t_start=0.0, t_end=1.0)
    assert out is not None
    assert [w["prob"] for w in out] == [pytest.approx(0.5), pytest.approx(0.25)]


def test_a_perfect_score_does_not_look_like_the_unaligned_sentinel():
    """log p == 0.0 means certainty; it must not read as "never aligned"."""
    spans = [_Span(5, 0, 10, 0.0)]
    out = _aligner()._stamps_from_spans(spans, ["a"], [0], fps=100.0,
                                        t_start=0.0, t_end=1.0)
    assert out[0]["prob"] == pytest.approx(1.0)


def test_spread_words_keep_the_zero_sentinel():
    """_spread marks words it could not align with prob 0.0; that must stay."""
    out = _aligner()._spread(["a", "b"], [0.0] * 1600, 0.0)
    assert [w["prob"] for w in out] == [0.0, 0.0]
