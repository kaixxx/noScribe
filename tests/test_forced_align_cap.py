"""Oversized alignment windows are split before the DP ever sees them.

FORCED_ALIGN_MAX_CELLS is a memory and latency budget for the numpy DP; its
comment in voxtral_engine has the figures and the history. This test drives
align_words with a fake emission (no wav2vec2 download) over a case above the
cap and asserts every actual DP invocation stays under it while every word is
still timestamped.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from noScribe import ctc_align  # noqa: E402
from noScribe.voxtral_engine import (  # noqa: E402
    _Aligner,
    FORCED_ALIGN_MAX_CELLS,
    SAMPLE_RATE,
)

VOCAB_SIZE = 30  # incl. blank


def _stub_aligner():
    al = object.__new__(_Aligner)
    al.vocab = {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
    al.blank = 0
    al.delim = None
    gen = torch.Generator().manual_seed(0)
    # 50 fps at 16 kHz, random log-probs -- shape is all that matters here.
    al._emission = lambda audio: torch.log_softmax(
        torch.rand((len(audio) // 320, VOCAB_SIZE), generator=gen), dim=-1
    ).numpy()
    return al


def test_oversized_window_splits_below_cap(monkeypatch):
    al = _stub_aligner()

    calls = []
    real_fa = ctc_align.forced_align

    def checked_fa(log_probs, targets, blank=0):
        calls.append(_Aligner._dp_cells(len(targets), log_probs.shape[0]))
        return real_fa(log_probs, targets, blank=blank)

    monkeypatch.setattr(ctc_align, "forced_align", checked_fa)

    # 1100 s -> 55_000 frames; 2000 x 5-char words -> 10_000 tokens
    # -> 55_000 * 20_001 = 1.10e9 cells, just above the 2**30 cap.
    words = ["abcde"] * 2000
    audio = np.zeros(1100 * SAMPLE_RATE, dtype=np.float32)
    n_frames = len(audio) // 320
    assert n_frames * (2 * len(words) * 5 + 1) > FORCED_ALIGN_MAX_CELLS

    out = al.align_words(words, audio)

    assert calls, "forced_align was never reached"
    assert all(c <= FORCED_ALIGN_MAX_CELLS for c in calls)
    assert len(calls) >= 2  # the window was actually split
    assert len(out) == len(words)
    assert all(w["end"] >= w["start"] for w in out)
    starts = [w["start"] for w in out]
    assert starts == sorted(starts)
