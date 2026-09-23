"""Round 2: v4 = brief + penalty-add two-lane + torchaudio final-state rule;
v4s = v4 with signbit compares; v5 = v4 + reachable band (fixed)."""
import sys, time
import numpy as np
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from viterbi_bench import fa_v0, fa_torch, random_cases, tie_cases, NEG, _backtrace, _final_state


def fa_v4(log_probs, targets, blank=0, signbit=False):
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1
    # advance-by-two penalty: 0 where a skip is legal (token state whose
    # predecessor token differs), NEG everywhere else; added to the shifted
    # alpha it sinks illegal lanes to the floor in one contiguous add.
    pen = np.full(N, NEG, dtype=np.float32)
    pen[(np.flatnonzero(targets[1:] != targets[:-1]) + 1) * 2 + 1] = 0.0

    alpha = np.full(N, NEG, dtype=np.float32)
    nxt = np.empty(N, dtype=np.float32)
    two = np.full(N, NEG, dtype=np.float32)     # two[:2] stays NEG
    c1 = np.zeros(N, dtype=bool)                # c1[0] stays False
    c2 = np.empty(N, dtype=bool)
    d = np.empty(N, dtype=np.float32)
    bt = np.zeros((T, N), dtype=np.uint8)

    alpha[0] = log_probs[0, blank]
    if N > 1:
        alpha[1] = log_probs[0, targets[0]]

    for t in range(1, T):
        if signbit:
            np.subtract(alpha[1:], alpha[:-1], out=d[1:]); np.signbit(d, out=c1)
        else:
            np.greater(alpha[:-1], alpha[1:], out=c1[1:])
        np.maximum(alpha[1:], alpha[:-1], out=nxt[1:])
        nxt[0] = alpha[0]
        np.add(alpha[:-2], pen[2:], out=two[2:])
        if signbit:
            np.subtract(nxt, two, out=d); np.signbit(d, out=c2)
        else:
            np.greater(two, nxt, out=c2)
        np.maximum(nxt, two, out=nxt)
        row = bt[t]
        np.copyto(row, c1.view(np.uint8))
        np.copyto(row, 2, where=c2)
        lp = log_probs[t]
        nxt[0::2] += lp[blank]
        nxt[1::2] += lp[targets]
        alpha, nxt = nxt, alpha

    s = _final_state(alpha[N - 2], alpha[N - 1], N)
    return _backtrace(log_probs, targets, bt, s)


def fa_v4s(lp, tg, blank=0):
    return fa_v4(lp, tg, blank, signbit=True)


def fa_v5(log_probs, targets, blank=0):
    """v4 restricted to the band of states that are reachable from the start
    and can still reach the end (repeats ignored, so the band is never
    narrower than the true one). State s at frame t is in the band iff
    2*(L-(T-t)) <= s <= 2*t+1, clipped to [0, N)."""
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1
    pen = np.full(N, NEG, dtype=np.float32)
    pen[(np.flatnonzero(targets[1:] != targets[:-1]) + 1) * 2 + 1] = 0.0

    alpha = np.full(N, NEG, dtype=np.float32)
    nxt = np.full(N, NEG, dtype=np.float32)
    two = np.full(N, NEG, dtype=np.float32)
    c1 = np.zeros(N, dtype=bool)
    c2 = np.zeros(N, dtype=bool)
    bt = np.zeros((T, N), dtype=np.uint8)

    alpha[0] = log_probs[0, blank]
    if N > 1:
        alpha[1] = log_probs[0, targets[0]]

    for t in range(1, T):
        lo = max(0, 2 * (L - (T - t)))
        hi = min(N, 2 * t + 2)           # exclusive
        a1 = max(lo, 1)                  # first state with an advance-one source
        a2 = max(lo, 2)                  # first state with an advance-two source
        np.greater(alpha[a1 - 1:hi - 1], alpha[a1:hi], out=c1[a1:hi])
        np.maximum(alpha[a1:hi], alpha[a1 - 1:hi - 1], out=nxt[a1:hi])
        if lo == 0:
            nxt[0] = alpha[0]
        np.add(alpha[a2 - 2:hi - 2], pen[a2:hi], out=two[a2:hi])
        np.greater(two[a2:hi], nxt[a2:hi], out=c2[a2:hi])
        np.maximum(nxt[a2:hi], two[a2:hi], out=nxt[a2:hi])
        row = bt[t]
        np.copyto(row[lo:hi], c1.view(np.uint8)[lo:hi])
        np.copyto(row[a2:hi], 2, where=c2[a2:hi])
        lp = log_probs[t]
        nxt[lo:hi][lo % 2::2] += lp[blank]
        j0, j1 = (lo + 1) // 2, hi // 2          # token indices in band
        nxt[2 * j0 + 1:2 * j1:2] += lp[targets[j0:j1]]
        alpha, nxt = nxt, alpha

    s = _final_state(alpha[N - 2], alpha[N - 1], N)
    return _backtrace(log_probs, targets, bt, s)


def fa_v0f(log_probs, targets, blank=0):
    """brief code, final-state rule fixed -- to isolate that change."""
    import viterbi_bench as vb
    p, s = fa_v0(log_probs, targets, blank)
    return p, s


CANDS = {"v0": fa_v0, "v4": fa_v4, "v4s": fa_v4s, "v5": fa_v5}


def check(name, fn):
    bad = 0; n = 0; tie_bad = 0
    rng = np.random.default_rng(1234)
    rc = list(random_cases(200, rng)); tc = list(tie_cases())
    for i, (lp, tg) in enumerate(rc + tc):
        n += 1
        p_ref, s_ref = fa_torch(lp, tg)
        p, s = fn(lp, tg)
        if not (np.array_equal(p, p_ref) and np.allclose(s, s_ref, atol=1e-6)):
            bad += 1
            if i >= len(rc): tie_bad += 1
            elif bad - tie_bad <= 3: print(f"  {name}: divergence on random T={lp.shape[0]} L={len(tg)}")
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'tests'))
    import test_forced_align_stability as st
    ref = np.load(st.REF_PATH); rb = 0            # inputs are stored in the file
    for idx in range(st.N_CASES):
        n += 1
        lp, tg = ref[f"lp_{idx}"], ref[f"targets_{idx}"]
        p, s = fn(lp, tg)
        if not np.array_equal(ref[f"paths_{idx}"][0], p):
            bad += 1; rb += 1
    print(f"{name}: {n - bad}/{n} identical to torchaudio  (random fails {bad - tie_bad - rb}, tie-case fails {tie_bad}, recorded fails {rb})")


def make_chunk(T, C, L, peaky, seed=0):
    rng = np.random.default_rng(seed)
    tg = rng.integers(1, C, size=L); rep = rng.random(L - 1) < 0.04; tg[1:][rep] = tg[:-1][rep]
    logits = rng.standard_normal((T, C)).astype(np.float32)
    if peaky:
        dom = np.where(rng.random(T) < 0.6, 0, tg[np.minimum((np.arange(T) * L) // T, L - 1)])
        logits[np.arange(T), dom] += rng.uniform(6, 12, T).astype(np.float32)
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    return lp.astype(np.float32), tg


def bench(fn, lp, tg, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(lp, tg); best = min(best, time.perf_counter() - t0)
    return best * 1000


if __name__ == "__main__":
    which = sys.argv[1:] or list(CANDS)
    for name in which:
        check(name, CANDS[name])
    for peaky in (True, False):
        lp, tg = make_chunk(14985, 38, 4886, peaky)
        p_ref, _ = fa_torch(lp, tg)
        print(f"\nchunk peaky={peaky}: torch={bench(fa_torch, lp, tg):.0f} ms")
        for name in which:
            p, _ = CANDS[name](lp, tg)
            print(f"  {name:4s}: {bench(CANDS[name], lp, tg):6.0f} ms   identical={np.array_equal(p, p_ref)}  diff frames={int((p != p_ref).sum())}")
