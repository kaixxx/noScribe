"""Round 3: v6 = v4 with -inf floor and arithmetic backtrace row; v6s = +signbit; v7 = v6 + band.

`final` is the product module, noScribe/ctc_align.py -- v6 plus validation -- so the
bench keeps checking what actually ships against torchaudio and the recorded reference.
"""
import sys, time
import numpy as np
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import viterbi_bench as vb, viterbi_bench2 as vb2
from noScribe import ctc_align
from viterbi_bench import fa_v0, fa_torch, random_cases, tie_cases
from viterbi_bench2 import fa_v4, make_chunk, bench

NEG = np.float32(-np.inf)


def forced_align_np(log_probs, targets, blank=0, signbit=False, band=False):
    """CTC Viterbi forced alignment, torchaudio.functional.forced_align semantics."""
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1
    repeats = int(np.count_nonzero(targets[1:] == targets[:-1]))
    if L == 0:
        raise ValueError("targets must not be empty")
    if T < L + repeats:
        raise ValueError(f"targets length is too long for CTC: {L} tokens + {repeats} "
                         f"repeats need {L + repeats} frames, got {T}")
    # Advance-by-two penalty: 0 on token states whose predecessor token differs
    # (a skip over the blank between them is legal), -inf everywhere else.
    pen = np.full(N, NEG, dtype=np.float32)
    pen[(np.flatnonzero(targets[1:] != targets[:-1]) + 1) * 2 + 1] = 0.0

    alpha = np.full(N, NEG, dtype=np.float32)
    nxt = np.full(N, NEG, dtype=np.float32)
    two = np.full(N, NEG, dtype=np.float32)
    c1 = np.zeros(N, dtype=bool)
    c2 = np.zeros(N, dtype=bool)
    c1v, c2v = c1.view(np.uint8), c2.view(np.uint8)
    d = np.empty(N, dtype=np.float32)
    bt = np.empty((T, N), dtype=np.uint8)
    bt[0] = 0

    alpha[0] = log_probs[0, blank]
    alpha[1] = log_probs[0, targets[0]]

    for t in range(1, T):
        if band:
            lo = max(0, 2 * (L - (T - t))); hi = min(N, 2 * t + 2)
        else:
            lo, hi = 0, N
        a1 = max(lo, 1); a2 = max(lo, 2)
        # stay vs advance-by-one
        if signbit:
            np.subtract(alpha[a1:hi], alpha[a1 - 1:hi - 1], out=d[a1:hi]); np.signbit(d[a1:hi], out=c1[a1:hi])
        else:
            np.greater(alpha[a1 - 1:hi - 1], alpha[a1:hi], out=c1[a1:hi])
        np.maximum(alpha[a1:hi], alpha[a1 - 1:hi - 1], out=nxt[a1:hi])
        if lo == 0:
            nxt[0] = alpha[0]
        # advance-by-two
        np.add(alpha[a2 - 2:hi - 2], pen[a2:hi], out=two[a2:hi])
        if signbit:
            np.subtract(nxt[a2:hi], two[a2:hi], out=d[a2:hi]); np.signbit(d[a2:hi], out=c2[a2:hi])
        else:
            np.greater(two[a2:hi], nxt[a2:hi], out=c2[a2:hi])
        np.maximum(nxt[a2:hi], two[a2:hi], out=nxt[a2:hi])
        # backpointer row: 2 where the skip won, else 1 where advance-one won, else 0
        row = bt[t]
        np.add(c2v[lo:hi], c2v[lo:hi], out=row[lo:hi])
        np.maximum(row[lo:hi], c1v[lo:hi], out=row[lo:hi])
        # emissions: blanks share one scalar, token states gather their label
        lp = log_probs[t]
        if band:
            nxt[lo:hi][lo % 2::2] += lp[blank]
            j0, j1 = (lo + 1) // 2, hi // 2
            nxt[2 * j0 + 1:2 * j1:2] += lp[targets[j0:j1]]
        else:
            nxt[0::2] += lp[blank]
            nxt[1::2] += lp[targets]
        alpha, nxt = nxt, alpha

    # torchaudio: the final blank only if strictly better than the last token
    s = N - 1 if alpha[N - 1] > alpha[N - 2] else N - 2
    path = np.empty(T, dtype=np.int64)
    scores = np.empty(T, dtype=np.float32)
    ext = np.zeros(N, dtype=np.int64)
    ext[1::2] = targets
    for t in range(T - 1, -1, -1):
        tok = ext[s]
        path[t] = tok
        scores[t] = log_probs[t, tok]
        s -= int(bt[t, s])      # int(): numpy would type `s -= uint8` as uint8
    return path, scores


def fa_v6(lp, tg, blank=0): return forced_align_np(lp, tg, blank)
def fa_v6s(lp, tg, blank=0): return forced_align_np(lp, tg, blank, signbit=True)
def fa_v7(lp, tg, blank=0): return forced_align_np(lp, tg, blank, band=True)
def fa_v7s(lp, tg, blank=0): return forced_align_np(lp, tg, blank, band=True, signbit=True)

CANDS = {"v0": fa_v0, "v4": fa_v4, "v6": fa_v6, "v6s": fa_v6s, "v7": fa_v7, "v7s": fa_v7s,
         "final": ctc_align.forced_align}

if __name__ == "__main__":
    vb.NEG = NEG; vb2.NEG = NEG
    which = sys.argv[1:] or list(CANDS)
    with np.errstate(all='raise'):
        for name in which:
            vb2.check(name, CANDS[name])
            vb.check_blanks(name, CANDS[name])
    for peaky in (True, False):
        lp, tg = make_chunk(14985, 38, 4886, peaky)
        p_ref, _ = fa_torch(lp, tg)
        print(f"\nchunk peaky={peaky}: torch={bench(fa_torch, lp, tg):.0f} ms")
        for name in which:
            p, _ = CANDS[name](lp, tg)
            print(f"  {name:4s}: {bench(CANDS[name], lp, tg):6.0f} ms   identical={np.array_equal(p, p_ref)}")
    # a short window too: 20 s, typical align_prefix piece is longer, but dispatch share grows
    lp, tg = make_chunk(1000, 38, 330, True, seed=3)
    p_ref, _ = fa_torch(lp, tg)
    print(f"\nsmall T=1000 L=330: torch={bench(fa_torch, lp, tg, 20):.1f} ms")
    for name in which:
        p, _ = CANDS[name](lp, tg)
        print(f"  {name:4s}: {bench(CANDS[name], lp, tg, 20):6.1f} ms   identical={np.array_equal(p, p_ref)}")
