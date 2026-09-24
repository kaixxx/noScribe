"""The audio time at which a clean prefix ends -- determined reliably.

Keeping the prefix in the loop ladder needs exactly one number: where in the
audio the clean part stops. Two things stand in the way, both measured on real
emissions (300 s of German, ground truth from the greedy decode of the same
aligner, so known to the frame):

1. **Pure alignment cannot stop early.** A CTC path has to occupy every frame,
   and parking on blank across 215 s of real speech costs more (-21905) than
   dragging the last prefix words across it. So the most probable path does
   exactly that: an 85 s prefix ended at 299.98 s of a 300 s window -- with an
   immaculate score (-0.000), which is why no downstream guard could ever have
   seen it. Nor is it an arbitrary tie-break: the shifted path wins by 802
   nat. The remedy is the star token from the MMS/torchaudio recipe for
   partial transcripts.
2. **A large window breaks FORCED_ALIGN_MAX_CELLS.** The answer `align_words`
   gives to that -- halve the words, cut the audio by their character share --
   is inadmissible for a prefix (measured: resume point 152 s too late, that
   speech fell out of the transcript). `align_prefix` therefore cuts the WORDS
   and gives each piece the entire remaining audio.

This left the rung not only limited (1500 s window: prefixes up to ~440 s), it
also cut in the wrong place in the very band where it fired.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from noScribe import ctc_align  # noqa: E402

from noScribe import voxtral_engine as v  # noqa: E402
from noScribe.voxtral_engine import (  # noqa: E402
    FORCED_ALIGN_MAX_CELLS,
    SAMPLE_RATE,
    _Aligner,
)

VOCAB_SIZE = 30
FPS = 50.0                      # 16 kHz / 320 samples per frame


def _stub_aligner(emission_frames):
    """Aligner without weights: real selection/chain logic, staged emission."""
    al = object.__new__(_Aligner)
    al.vocab = {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
    al.blank = 0
    al.delim = None
    al._emission = lambda audio: emission_frames
    return al


_ALPHA = "abcdefghijklmnopqrstuvwxy"      # 'z' is reserved for the foreign material


def _words(n):
    """`n` words, as different from one another as possible.

    Important for the proof: if the wording repeats, the alignment has several
    equally good paths and the "true" time is not determined at all -- the
    test then measures nothing but the arbitrariness of the tie-break.
    """
    return [_ALPHA[i % 25] + _ALPHA[(i // 25) % 25] + _ALPHA[(i * 7) % 25]
            + _ALPHA[(i * 11 + 3) % 25] + _ALPHA[(i * 17 + 5) % 25]
            for i in range(n)]


def _planted(al, words, n_frames, frames_per_char=2):
    """Emission in which `words` really stand at the start of the audio -- and
    in which speech CONTINUES afterwards.

    The second part is the decisive one: in the real case, what lies behind
    the prefix is not silence but the rest of the window. Parking on blank is
    expensive there, and that is exactly why pure alignment drags the last
    prefix words across the remaining audio (recomputed on real emissions:
    the shifted path wins by 802 nat, that is no arbitrariness of the
    tie-break). Returns: (emission, true end of the last word).
    """
    path = [al.blank] * n_frames
    f = int(1.0 * FPS)                       # a little lead-in
    for w in words:
        for ch in w:
            for _ in range(frames_per_char):
                path[f] = al.vocab[ch]
                f += 1
            path[f] = al.blank               # blank between identical labels
            f += 1
        f += 2                               # word gap
    true_end = (f - 3) / FPS
    # Remaining audio: continuous speech, no blank -- otherwise not covering it
    # costs nothing and the defect never occurs in the first place. And varied,
    # because that is exactly where the shifted path draws its advantage from:
    # for every word it finds matching letters somewhere further back.
    for n, g in enumerate(range(f + int(1.0 * FPS), n_frames)):
        path[g] = al.vocab[_ALPHA[(n // frames_per_char) % len(_ALPHA)]]
    em = torch.full((n_frames, VOCAB_SIZE), -12.0)
    em[torch.arange(n_frames), torch.tensor(path)] = 0.0
    return torch.log_softmax(em, dim=-1).numpy(), true_end


def _audio(n_frames):
    return np.zeros(int(n_frames * 320), dtype=np.float32)


def _count_cells(monkeypatch):
    cells = []
    real = ctc_align.forced_align

    def counting(log_probs, targets, blank=0):
        cells.append(_Aligner._dp_cells(len(targets), log_probs.shape[0]))
        return real(log_probs, targets, blank=blank)

    monkeypatch.setattr(ctc_align, "forced_align", counting)
    return cells


# --------------------------------------------------------------------------- #
# The core: a prefix that breaks a single forced_align computation
# --------------------------------------------------------------------------- #
def test_a_prefix_too_big_for_one_call_still_gets_a_real_end_time(monkeypatch):
    """Until now this case ended in `_spread` (prob 0.0) and the rung refused
    -- a 1500 s window let only prefixes up to ~440 s through. Now the chain
    has to deliver a real time, and the right one at that."""
    n_frames = 6000                                     # 120 s
    words = _words(100)
    al = _stub_aligner(None)
    emission, true_end = _planted(al, words, n_frames)
    al._emission = lambda audio: emission

    # Set the cell limit so that the whole prefix cannot be aligned in one go
    # -- the same state as a 900 s prefix in a 1500 s window, only small
    # enough for a test.
    tokens, _ = al._tokenize(words)
    cap = n_frames * (2 * len(tokens) + 1) // 3
    monkeypatch.setattr(v, "FORCED_ALIGN_MAX_CELLS", cap)
    cells = _count_cells(monkeypatch)

    out = al.align_prefix(words, _audio(n_frames))

    assert len(cells) >= 2, "the prefix fitted into one computation after all"
    assert all(c <= cap for c in cells), cells
    assert len(out) == len(words)
    # This is the number the cut is made from.
    assert out[-1]["prob"] != 0.0, "the prefix end was not really aligned"
    assert abs(out[-1]["end"] - true_end) < 0.5, (out[-1]["end"], true_end)
    # ...and the prefix stays at the front: it must not run into the foreign material.
    assert out[-1]["end"] < n_frames / FPS * 0.6
    starts = [w["start"] for w in out]
    assert starts == sorted(starts)
    assert all(w["end"] >= w["start"] for w in out)


def test_the_audio_is_never_cut_by_character_share(monkeypatch):
    """The actual defect: it is the WORDS that get split, never the audio by
    character share. Every piece therefore has to run against the entire
    remaining audio -- recognisable by the frame count of every computation
    matching the remainder, not a fraction estimated from the text length."""
    n_frames = 6000
    words = _words(100)
    al = _stub_aligner(None)
    emission, true_end = _planted(al, words, n_frames)
    al._emission = lambda audio: emission

    tokens, _ = al._tokenize(words)
    monkeypatch.setattr(v, "FORCED_ALIGN_MAX_CELLS",
                        n_frames * (2 * len(tokens) + 1) // 3)

    seen = []
    real = ctc_align.forced_align
    monkeypatch.setattr(
        ctc_align, "forced_align",
        lambda e, t, blank=0: (seen.append(e.shape[0]), real(e, t, blank=blank))[1])

    al.align_prefix(words, _audio(n_frames))

    assert seen[0] == n_frames, "the first piece did not see the whole window"
    # Every following piece starts at the end of the previous one and keeps
    # everything after it: the remaining length shrinks but always ends at the
    # window end.
    assert seen == sorted(seen, reverse=True)
    # The remainder is real remaining audio, not the prefix's estimated
    # character share: after the last word there is still foreign material in
    # the window.
    assert seen[-1] > (n_frames - true_end * FPS) * 0.8


def test_a_small_window_is_still_one_single_alignment(monkeypatch):
    """Counter-check: where one computation is enough, nothing may be chopped
    up -- the chain is the exception, not the new normal."""
    n_frames = 3000
    words = _words(30)
    al = _stub_aligner(None)
    emission, true_end = _planted(al, words, n_frames)
    al._emission = lambda audio: emission
    cells = _count_cells(monkeypatch)

    out = al.align_prefix(words, _audio(n_frames))

    assert len(cells) == 1 and cells[0] <= FORCED_ALIGN_MAX_CELLS
    assert abs(out[-1]["end"] - true_end) < 0.5
    assert out[-1]["prob"] != 0.0


# --------------------------------------------------------------------------- #
# What has to happen when a piece cannot really be aligned
# --------------------------------------------------------------------------- #
def test_one_failed_piece_invalidates_the_whole_chain(monkeypatch):
    """A chain is only as good as its weakest link: had a piece boundary been
    guessed, all following pieces would be off -- and would come back with
    immaculate scores, i.e. undetectably. A failed piece therefore has to
    throw the whole result onto evenly spread times (prob 0.0), which
    `_salvage_prefix` by construction does not cut on."""
    n_frames = 6000
    words = _words(100)
    al = _stub_aligner(None)
    emission, _ = _planted(al, words, n_frames)
    al._emission = lambda audio: emission

    tokens, _ = al._tokenize(words)
    monkeypatch.setattr(v, "FORCED_ALIGN_MAX_CELLS",
                        n_frames * (2 * len(tokens) + 1) // 3)

    calls = []
    real = ctc_align.forced_align

    def flaky(log_probs, targets, blank=0):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated failure in the second piece")
        return real(log_probs, targets, blank=blank)

    monkeypatch.setattr(ctc_align, "forced_align", flaky)

    out = al.align_prefix(words, _audio(n_frames))

    assert len(out) == len(words)
    assert all(w["prob"] == 0.0 for w in out), \
        "a partial result of the chain was reported as a real time"
    # ...and _salvage_prefix refuses on it.
    text = " ".join(words[:len(words) // 2]) + ". " + "dass das, " * 40
    got, why = v._salvage_prefix(text, _audio(n_frames),
                                 lambda w, a: al.align_prefix(w, a))
    assert got is None and why


def test_the_ladder_asks_for_a_prefix_alignment(monkeypatch):
    """Wiring: the rung has to be given `align_prefix`. `align_words` with
    `words_span_audio=False` visibly degrades to `_spread` on large windows --
    exactly the ceiling that is meant to fall here."""
    from noScribe.voxtral_engine import _AlignerPool

    used = []

    class _Stub:
        def align_prefix(self, words, audio, t_offset=0.0):
            used.append("align_prefix")
            return [{"word": w, "start": i, "end": i + 1, "prob": -0.1}
                    for i, w in enumerate(words)]

        def align_words(self, words, audio, t_offset=0.0, words_span_audio=True):
            used.append("align_words")
            return []

    pool = _AlignerPool("de", None)
    pool.aligner_for = lambda text, remember=True: _Stub()
    out = pool.align_for_salvage(["Guten", "Morgen."], "AUDIO")

    assert used == ["align_prefix"], used
    assert out[-1]["end"] == 2
