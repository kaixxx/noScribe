"""Candidates for the numpy CTC Viterbi, checked against torchaudio and timed.

V0  = the reference from docs/viterbi-numpy-brief.md, verbatim.
V1  = interleaved layout, three local tweaks (no `two` lane, uint8 compare, take(out=)).
V2  = split-lane layout (blank lane / token lane), contiguous half-width ops.
V2s = V2 but writing into strided views of one interleaved backtrace table.
V3  = V2 + torchaudio's reachable-band windowing.
"""
import sys, time
import numpy as np
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_TESTS = _os.path.join(_HERE, '..', '..', 'tests')
import torch
import torchaudio.functional as F

NEG = np.float32(-3.4e38)


# ---------------------------------------------------------------- V0 (brief)
def fa_v0(log_probs, targets, blank=0):
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1
    skip_idx = (np.flatnonzero(targets[1:] != targets[:-1]) + 1) * 2 + 1
    src_idx = skip_idx - 2

    alpha = np.full(N, NEG, dtype=np.float32)
    nxt = np.empty(N, dtype=np.float32)
    two = np.full(N, NEG, dtype=np.float32)
    c1 = np.zeros(N, dtype=bool)
    c2 = np.empty(N, dtype=bool)
    bt = np.zeros((T, N), dtype=np.uint8)

    alpha[0] = log_probs[0, blank]
    if N > 1:
        alpha[1] = log_probs[0, targets[0]]

    for t in range(1, T):
        np.greater(alpha[:-1], alpha[1:], out=c1[1:])
        np.maximum(alpha[1:], alpha[:-1], out=nxt[1:])
        nxt[0] = alpha[0]
        two[skip_idx] = alpha[src_idx]
        np.greater(two, nxt, out=c2)
        np.maximum(nxt, two, out=nxt)
        row = bt[t]
        np.copyto(row, c1.view(np.uint8))
        np.copyto(row, 2, where=c2)
        lp = log_probs[t]
        nxt[0::2] += lp[blank]
        nxt[1::2] += lp[targets]
        alpha, nxt = nxt, alpha

    s = N - 1 if alpha[N - 1] >= alpha[N - 2] else N - 2
    path = np.empty(T, dtype=np.int64)
    scores = np.empty(T, dtype=np.float32)
    ext = np.zeros(N, dtype=np.int64)
    ext[1::2] = targets
    for t in range(T - 1, -1, -1):
        tok = ext[s]
        path[t] = tok
        scores[t] = log_probs[t, tok]
        s -= int(bt[t, s])
    return path, scores


def _final_state(last_tok, last_blank, N):
    # torchaudio: alphas[S-1] > alphas[S-2] ? S-1 : S-2  (tie -> the token state)
    return N - 1 if last_blank > last_tok else N - 2


def _backtrace(log_probs, targets, bt, s):
    T = log_probs.shape[0]
    N = bt.shape[1]
    path = np.empty(T, dtype=np.int64)
    scores = np.empty(T, dtype=np.float32)
    ext = np.zeros(N, dtype=np.int64)
    ext[1::2] = targets
    for t in range(T - 1, -1, -1):
        tok = ext[s]
        path[t] = tok
        scores[t] = log_probs[t, tok]
        s -= int(bt[t, s])
    return path, scores


# ---------------------------------------------------------------- V1
def fa_v1(log_probs, targets, blank=0):
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1
    legal = np.zeros(N, dtype=bool)                  # advance-by-two target states
    legal[(np.flatnonzero(targets[1:] != targets[:-1]) + 1) * 2 + 1] = True

    alpha = np.full(N, NEG, dtype=np.float32)
    nxt = np.empty(N, dtype=np.float32)
    c2 = np.zeros(N, dtype=bool)
    g = np.empty(L, dtype=np.float32)
    bt = np.zeros((T, N), dtype=np.uint8)

    alpha[0] = log_probs[0, blank]
    if N > 1:
        alpha[1] = log_probs[0, targets[0]]

    for t in range(1, T):
        row = bt[t]
        np.greater(alpha[:-1], alpha[1:], out=row[1:])         # 1 where advance-one wins
        np.maximum(alpha[1:], alpha[:-1], out=nxt[1:])
        nxt[0] = alpha[0]
        np.greater(alpha[:-2], nxt[2:], out=c2[2:])            # advance-two candidate
        np.logical_and(c2, legal, out=c2)
        np.copyto(nxt[2:], alpha[:-2], where=c2[2:])
        np.copyto(row, 2, where=c2)
        lp = log_probs[t]
        nxt[0::2] += lp[blank]
        np.take(lp, targets, out=g)
        nxt[1::2] += g
        alpha, nxt = nxt, alpha

    s = _final_state(alpha[N - 2], alpha[N - 1], N)
    return _backtrace(log_probs, targets, bt, s)


# ---------------------------------------------------------------- V2 (split lanes)
def fa_v2(log_probs, targets, blank=0):
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1
    legal = np.zeros(L, dtype=bool)
    legal[1:] = targets[1:] != targets[:-1]

    B = np.full(L + 1, NEG, dtype=np.float32)    # blank states, B[j] before token j
    K = np.full(L, NEG, dtype=np.float32)        # token states
    Bn = np.empty(L + 1, dtype=np.float32)
    Kn = np.empty(L, dtype=np.float32)
    c2 = np.zeros(L, dtype=bool)
    g = np.empty(L, dtype=np.float32)
    btB = np.zeros((T, L + 1), dtype=np.uint8)
    btK = np.zeros((T, L), dtype=np.uint8)

    B[0] = log_probs[0, blank]
    if L:
        K[0] = log_probs[0, targets[0]]

    for t in range(1, T):
        rB = btB[t]
        rK = btK[t]
        np.greater(K, B[1:], out=rB[1:])
        np.maximum(B[1:], K, out=Bn[1:])
        Bn[0] = B[0]
        np.greater(B[:L], K, out=rK)
        np.maximum(K, B[:L], out=Kn)
        np.greater(K[:-1], Kn[1:], out=c2[1:])
        np.logical_and(c2, legal, out=c2)
        np.copyto(Kn[1:], K[:-1], where=c2[1:])
        np.copyto(rK, 2, where=c2)
        lp = log_probs[t]
        Bn += lp[blank]
        np.take(lp, targets, out=g)
        Kn += g
        B, Bn = Bn, B
        K, Kn = Kn, K

    s = _final_state(K[L - 1] if L else NEG, B[L], N)
    path = np.empty(T, dtype=np.int64)
    scores = np.empty(T, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        if s & 1:
            k = s >> 1
            tok = targets[k]
            step = btK[t, k]
        else:
            tok = blank
            step = btB[t, s >> 1]
        path[t] = tok
        scores[t] = log_probs[t, tok]
        s -= int(step)
    return path, scores


# ---------------------------------------------------------------- V2s (strided bt views)
def fa_v2s(log_probs, targets, blank=0):
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1
    legal = np.zeros(L, dtype=bool)
    legal[1:] = targets[1:] != targets[:-1]

    B = np.full(L + 1, NEG, dtype=np.float32)
    K = np.full(L, NEG, dtype=np.float32)
    Bn = np.empty(L + 1, dtype=np.float32)
    Kn = np.empty(L, dtype=np.float32)
    c2 = np.zeros(L, dtype=bool)
    g = np.empty(L, dtype=np.float32)
    bt = np.zeros((T, N), dtype=np.uint8)
    btB = bt[:, 0::2]
    btK = bt[:, 1::2]

    B[0] = log_probs[0, blank]
    if L:
        K[0] = log_probs[0, targets[0]]

    for t in range(1, T):
        rB = btB[t]
        rK = btK[t]
        np.greater(K, B[1:], out=rB[1:])
        np.maximum(B[1:], K, out=Bn[1:])
        Bn[0] = B[0]
        np.greater(B[:L], K, out=rK)
        np.maximum(K, B[:L], out=Kn)
        np.greater(K[:-1], Kn[1:], out=c2[1:])
        np.logical_and(c2, legal, out=c2)
        np.copyto(Kn[1:], K[:-1], where=c2[1:])
        np.copyto(rK, 2, where=c2)
        lp = log_probs[t]
        Bn += lp[blank]
        np.take(lp, targets, out=g)
        Kn += g
        B, Bn = Bn, B
        K, Kn = Kn, K

    s = _final_state(K[L - 1] if L else NEG, B[L], N)
    return _backtrace(log_probs, targets, bt, s)


# ---------------------------------------------------------------- V3 (V2 + band)
def fa_v3(log_probs, targets, blank=0):
    """V2 restricted to states reachable from the start and able to reach the
    end. Cells outside the band stay NEG, exactly as in V2, so the answer is
    the same; only the work shrinks. In token-index terms the band at frame t
    (0-based) is tokens [lo, hi) with hi = min(L, t+1) reachable from the start
    and lo = max(0, L - (T - t)) able to reach the end (repeats ignored: that
    makes the band slightly wider than torchaudio's, never narrower)."""
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1
    legal = np.zeros(L, dtype=bool)
    legal[1:] = targets[1:] != targets[:-1]

    B = np.full(L + 1, NEG, dtype=np.float32)
    K = np.full(L, NEG, dtype=np.float32)
    Bn = np.full(L + 1, NEG, dtype=np.float32)
    Kn = np.full(L, NEG, dtype=np.float32)
    c2 = np.zeros(L, dtype=bool)
    g = np.empty(L, dtype=np.float32)
    btB = np.zeros((T, L + 1), dtype=np.uint8)
    btK = np.zeros((T, L), dtype=np.uint8)

    B[0] = log_probs[0, blank]
    if L:
        K[0] = log_probs[0, targets[0]]

    for t in range(1, T):
        lo = max(0, L - (T - t))           # first token index still able to finish
        hi = min(L, t + 1)                 # one past the last token reachable by now
        # blank lane j in [lo, hi]  (B[hi] is the blank after token hi-1)
        # token lane k in [lo, hi)
        rB = btB[t]
        rK = btK[t]
        Kw = K[lo:hi]
        np.greater(Kw, B[lo + 1:hi + 1], out=rB[lo + 1:hi + 1])
        np.maximum(B[lo + 1:hi + 1], Kw, out=Bn[lo + 1:hi + 1])
        Bn[lo] = B[lo]
        Knw = Kn[lo:hi]
        np.greater(B[lo:hi], Kw, out=rK[lo:hi])
        np.maximum(Kw, B[lo:hi], out=Knw)
        if hi - lo > 1:
            np.greater(K[lo:hi - 1], Knw[1:], out=c2[lo + 1:hi])
            np.logical_and(c2[lo + 1:hi], legal[lo + 1:hi], out=c2[lo + 1:hi])
            np.copyto(Knw[1:], K[lo:hi - 1], where=c2[lo + 1:hi])
            np.copyto(rK[lo + 1:hi], 2, where=c2[lo + 1:hi])
        lp = log_probs[t]
        Bn[lo:hi + 1] += lp[blank]
        np.take(lp, targets[lo:hi], out=g[lo:hi])
        Knw += g[lo:hi]
        if lo:                              # cells that just left the band must not leak
            Bn[lo - 1] = NEG
            Kn[lo - 1] = NEG
        B, Bn = Bn, B
        K, Kn = Kn, K

    s = _final_state(K[L - 1] if L else NEG, B[L], N)
    path = np.empty(T, dtype=np.int64)
    scores = np.empty(T, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        if s & 1:
            k = s >> 1
            tok = targets[k]
            step = btK[t, k]
        else:
            tok = blank
            step = btB[t, s >> 1]
        path[t] = tok
        scores[t] = log_probs[t, tok]
        s -= int(step)
    return path, scores


CANDIDATES = {"v0": fa_v0, "v1": fa_v1, "v2": fa_v2, "v2s": fa_v2s, "v3": fa_v3}


# ---------------------------------------------------------------- oracle
def fa_torch(log_probs, targets, blank=0):
    lp = torch.from_numpy(np.ascontiguousarray(log_probs, dtype=np.float32)).unsqueeze(0)
    tg = torch.tensor(np.asarray(targets), dtype=torch.int32).unsqueeze(0)
    p, s = F.forced_align(lp, tg, blank=blank)
    return p[0].numpy().astype(np.int64), s[0].numpy()


def random_cases(n, rng, C=38, tight_every=4):
    for i in range(n):
        L = int(rng.integers(1, 120))
        tg = rng.integers(1, C, size=L)
        if i % 3 == 0 and L > 3:
            tg[1::2] = tg[0:-1:2][: len(tg[1::2])]
        R = int(np.sum(tg[1:] == tg[:-1]))
        T = int(L + R + (0 if i % tight_every == 0 else rng.integers(0, 300)))
        lp = rng.standard_normal((T, C)).astype(np.float32)
        lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
        yield lp, tg


def tie_cases():
    # end-of-sequence tie: final blank vs final token have equal alpha
    lp = np.full((2, 3), np.log(1 / 3), dtype=np.float32)
    yield lp, np.array([1])
    # ties everywhere: uniform emissions, many tokens, generous T
    for T, L in [(10, 3), (40, 7), (25, 12)]:
        lp = np.full((T, 5), np.log(0.2), dtype=np.float32)
        yield lp, np.array([1, 2, 2, 1, 3, 3, 3, 4, 1, 2, 3, 4][:L])
    # integer-valued emissions -> exact ties in sums
    rng = np.random.default_rng(7)
    for _ in range(30):
        L = int(rng.integers(1, 30))
        tg = rng.integers(1, 4, size=L)
        R = int(np.sum(tg[1:] == tg[:-1]))
        T = int(L + R + rng.integers(0, 40))
        lp = -rng.integers(0, 3, size=(T, 4)).astype(np.float32)
        yield lp, tg


def blank_cases(n=200, rng=None):
    """Cases with a non-zero blank index.

    Every other generator here leaves blank at 0, and so does the recorded
    reference, so a DP that hard-codes 0 as the blank label passes all 277 of
    them and still mislabels every blank frame in production. torchaudio's own
    test suite uses blank=5; this closes the same gap.
    """
    rng = rng or np.random.default_rng(99)
    C = 6
    for blank in (0, 3, 5):
        for _ in range(n // 3):
            T = int(rng.integers(4, 14))
            L = int(rng.integers(1, 4))
            tg = np.array([x for x in rng.integers(0, C, size=L) if x != blank])
            if len(tg) == 0:
                continue
            R = int(np.sum(tg[1:] == tg[:-1]))
            if T < len(tg) + R:
                continue
            logits = rng.standard_normal((T, C)).astype(np.float32)
            logits[:, blank] += 2.0          # blanks must actually reach the path
            lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
            yield lp.astype(np.float32), tg, blank


def check_blanks(name, fn):
    """`fn` must match torchaudio for blank indices other than 0."""
    bad = n = 0
    for lp, tg, blank in blank_cases():
        n += 1
        p_ref, s_ref = fa_torch(lp, tg, blank=blank)
        p, s = fn(lp, tg, blank=blank)
        if not (np.array_equal(p, p_ref) and np.allclose(s, s_ref, atol=1e-5)):
            bad += 1
    print(f"{name}: {n - bad}/{n} identical with non-zero blanks")
    return bad == 0


def check(name, fn):
    bad = 0
    n = 0
    rng = np.random.default_rng(1234)
    for lp, tg in list(random_cases(200, rng)) + list(tie_cases()):
        n += 1
        p_ref, s_ref = fa_torch(lp, tg)
        p, s = fn(lp, tg)
        if not (np.array_equal(p, p_ref) and np.allclose(s, s_ref, atol=1e-6)):
            bad += 1
            if bad <= 3:
                print(f"  {name}: divergence on T={lp.shape[0]} L={len(tg)}")
    # recorded reference
    sys.path.insert(0, _TESTS)
    import test_forced_align_stability as st
    ref = np.load(st.REF_PATH)                    # inputs are stored in the file
    for idx in range(st.N_CASES):
        n += 1
        lp, tg = ref[f"lp_{idx}"], ref[f"targets_{idx}"]
        p, s = fn(lp, tg)
        if not np.array_equal(ref[f"paths_{idx}"][0], p):
            bad += 1
            print(f"  {name}: recorded case {idx} diverged")
    print(f"{name}: {n - bad}/{n} identical to torchaudio")
    return bad == 0


def bench(name, fn, lp, tg, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(lp, tg)
        best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    which = sys.argv[1:] or list(CANDIDATES)
    for name in which:
        check(name, CANDIDATES[name])

    # the 300 s chunk from the brief: T=14985, C=38, L=4886
    rng = np.random.default_rng(0)
    T, C, L = 14985, 38, 4886
    tg = rng.integers(1, C, size=L)
    rep = rng.random(L - 1) < 0.04                   # ~4 % doubled letters
    tg[1:][rep] = tg[:-1][rep]
    lp = rng.standard_normal((T, C)).astype(np.float32)
    lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    p_ref, _ = fa_torch(lp, tg)
    print(f"\nchunk: T={T} L={L} N={2*L+1} cells={T*(2*L+1)/1e6:.0f}M  "
          f"torch={bench('torch', fa_torch, lp, tg)*1000:.0f} ms")
    for name in which:
        fn = CANDIDATES[name]
        p, _ = fn(lp, tg)
        ok = np.array_equal(p, p_ref)
        print(f"  {name:4s}: {bench(name, fn, lp, tg)*1000:6.0f} ms   identical={ok}")
