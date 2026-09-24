"""CTC forced alignment in numpy -- the Voxtral engine's word-timestamp DP.

Replaces ``torchaudio.functional.forced_align`` and ``merge_tokens``, keeping
their semantics (unbatched): ``forced_align`` returns one label per frame and
that label's log-probability per frame; ``merge_tokens`` collapses the frames
into spans with an *exclusive* end and the mean score.

Why this exists rather than the C++ kernel: torchaudio 2.9-2.11 index the DP
buffer in 32 bits and segfault past ~2**31 cells (pytorch/audio#4208) -- numpy
indexes in 64 bits, so the worst case is slow, not dead; and on an exact tie
between the advance-one and advance-two candidates that kernel takes the worse
value (pytorch/audio#4221), this one the larger, so it is a correct Viterbi.
Everywhere else it is frame-for-frame identical to torchaudio (see
tests/test_forced_align_stability.py; the tie cases are in tests/test_ctc_align.py).

Cost and memory: ~1.35x the C++ kernel, and one byte per DP cell for the
backtrace (as in torchaudio), so the cell budgets in ``voxtral_engine`` still
apply. The measurements, every optimisation tried, and the benches that
reproduce them: ``docs/viterbi-numpy-brief.md``, ``docs/scripts/viterbi_bench*.py``.
"""
from typing import List, NamedTuple

import numpy as np

# Same floor as torchaudio. -inf is safe because nothing in the loop subtracts;
# the only operations are comparisons, maxima and additions of finite values.
NEG = np.float32(-np.inf)


class TokenSpan(NamedTuple):
    """One run of a non-blank label: ``[start, end)`` in frames, mean score."""
    token: int
    start: int
    end: int
    score: float


def forced_align(log_probs, targets, blank=0):
    """Viterbi alignment of ``targets`` to ``log_probs`` (``[T, vocab]``).

    Returns ``(path, scores)``: the label chosen at each of the ``T`` frames and
    its log-probability there. Raises ``ValueError`` where torchaudio does --
    empty targets, a blank outside the vocabulary or inside the targets, or
    fewer frames than ``len(targets) + repeats`` (a CTC path needs a blank
    between two identical labels). The engine pre-checks the frame budget and
    cannot produce the other cases; the raise is what keeps its
    ``except -> warn -> spread`` path honest if one ever slips through.

    The DP runs in float32 whatever the input dtype -- measured, and worth a
    third of the runtime -- so ``scores`` is float32.
    """
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1                                   # blank-interleaved states
    if L == 0:
        raise ValueError("targets must not be empty")
    if not 0 <= blank < log_probs.shape[1]:
        raise ValueError(f"blank index {blank} is outside the vocabulary "
                         f"({log_probs.shape[1]} entries)")
    repeats = adjacent_repeats(targets)
    if T < L + repeats:
        raise ValueError(f"targets length is too long for CTC: {L} tokens + "
                         f"{repeats} repeats need {L + repeats} frames, got {T}")
    # A target state carrying the blank label would collapse away in
    # merge_tokens: its word loses its span and every later word slides onto
    # the wrong token -- a quietly wrong transcript instead of the caller's
    # log-and-spread fallback. So reject it, as torchaudio does.
    if np.any(targets == blank):
        raise ValueError("targets should not contain the blank index "
                         f"({blank}); found {int(np.count_nonzero(targets == blank))}")

    # Advance-by-two penalty: 0 on token states whose preceding token differs
    # (skipping the blank between them is legal), -inf everywhere else. Added
    # to the shifted alpha it sinks the illegal lanes in one contiguous add;
    # fancy indexing here was the single most expensive line of the loop.
    pen = np.full(N, NEG, dtype=np.float32)
    pen[(np.flatnonzero(targets[1:] != targets[:-1]) + 1) * 2 + 1] = 0.0

    # Every buffer is allocated once; the loop allocates only the L-element
    # gather of the token emissions (measured: cheaper than any out= form). The backtrace
    # table is uint8 on purpose (T x N bytes -- 146 MB on a 300 s chunk; int64
    # would be 1.2 GB) and np.empty because every row t >= 1 is fully written.
    alpha = np.full(N, NEG, dtype=np.float32)
    nxt = np.full(N, NEG, dtype=np.float32)
    two = np.full(N, NEG, dtype=np.float32)         # two[:2] stays -inf
    c1 = np.zeros(N, dtype=bool)                    # c1[0] stays False
    c2 = np.zeros(N, dtype=bool)
    c1v, c2v = c1.view(np.uint8), c2.view(np.uint8)
    bt = np.empty((T, N), dtype=np.uint8)
    bt[0] = 0

    alpha[0] = log_probs[0, blank]
    alpha[1] = log_probs[0, targets[0]]

    # torchaudio's tie order, per cell with predecessors x0 (stay), x1 (advance
    # one), x2 (advance two): 2 only if strictly best, else 1 only if strictly
    # better than 0, else 0. Its extra `x1 > x2` clause is the defect above
    # and is deliberately not reproduced.
    for t in range(1, T):
        np.greater(alpha[:-1], alpha[1:], out=c1[1:])          # advance-one beats stay
        np.maximum(alpha[1:], alpha[:-1], out=nxt[1:])
        nxt[0] = alpha[0]
        np.add(alpha[:-2], pen[2:], out=two[2:])               # advance-two lane
        np.greater(two, nxt, out=c2)                           # ... beats both
        np.maximum(nxt, two, out=nxt)
        row = bt[t]                                            # 2 where the skip won, else c1
        np.add(c2v, c2v, out=row)                              # (masked writes scale with
        np.maximum(row, c1v, out=row)                          #  mask density; these do not)
        lp = log_probs[t]
        nxt[0::2] += lp[blank]                                 # blank states: one scalar
        nxt[1::2] += lp[targets]                               # token states: gather of L
        alpha, nxt = nxt, alpha

    # torchaudio ends on the final blank only if it is strictly better.
    s = N - 1 if alpha[N - 1] > alpha[N - 2] else N - 2
    path = np.empty(T, dtype=np.int64)
    scores = np.empty(T, dtype=np.float32)
    # Label per state: blank on the even states, the target on the odd ones.
    # np.zeros here would be a silent bug for any blank index other than 0.
    ext = np.full(N, blank, dtype=np.int64)
    ext[1::2] = targets
    for t in range(T - 1, -1, -1):                 # ~T scalar steps, <20 ms
        tok = ext[s]
        path[t] = tok
        scores[t] = log_probs[t, tok]
        s -= int(bt[t, s])   # int(): numpy would type `s -= uint8` as uint8 and overflow at N > 255
    return path, scores


def adjacent_repeats(targets) -> int:
    """Pairs of identical neighbouring targets. Each needs one extra frame, since
    a CTC path has to put a blank between two identical labels -- so a window
    can only be aligned when ``frames >= len(targets) + adjacent_repeats``."""
    targets = np.asarray(targets)
    return int(np.count_nonzero(targets[1:] == targets[:-1]))


def merge_tokens(path, scores, blank=0) -> List[TokenSpan]:
    """Runs of equal non-blank labels as spans with an exclusive end.

    A token on frames 0, 1, 2 comes back as ``start=0, end=3``: ``end - start``
    is the duration and ``end / fps`` the time after the last frame. The engine
    relies on that; do not add a ``+1``.
    """
    path = np.asarray(path)
    scores = np.asarray(scores)
    if len(path) == 0:
        return []
    cut = np.flatnonzero(path[1:] != path[:-1]) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(path)]))
    return [TokenSpan(int(path[s]), int(s), int(e), float(scores[s:e].mean()))
            for s, e in zip(starts, ends) if path[s] != blank]
