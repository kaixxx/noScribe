"""Independent oracles for noScribe.ctc_align, the numpy forced-alignment DP.

The recorded torchaudio reference in test_forced_align_stability.py is the
acceptance test; it has two blind spots that this file closes, both found the
hard way while writing the DP:

* every recorded case uses ``blank=0``, so a DP that hard-codes 0 as the blank
  label passes all 43 of them and mislabels every blank frame in production
  with any other blank index -- here the blank varies;
* random emissions never tie exactly, so the tie order is never exercised --
  here the log-probs are small integers, which makes ties common, and the
  oracle is brute force over every path, which has no tie order at all.

Neither oracle needs torch, so this runs on the Linux CI without the stack.
"""
import itertools

import numpy as np
import pytest

from noScribe.ctc_align import TokenSpan, adjacent_repeats, forced_align, merge_tokens


def _collapse(path, blank):
    out, prev = [], None
    for x in path:
        if x != blank and x != prev:
            out.append(int(x))
        prev = x
    return out


def _best_valid_score(lp, targets, blank):
    """Highest total log-prob over *every* frame labelling that spells the
    targets -- exhaustive, so it is an oracle rather than another Viterbi."""
    T, C = lp.shape
    best = None
    for cand in itertools.product(range(C), repeat=T):
        if _collapse(cand, blank) == list(targets):
            s = sum(float(lp[t, c]) for t, c in enumerate(cand))
            best = s if best is None or s > best else best
    return best


def _scalar_viterbi(lp, targets, blank):
    """The textbook DP in plain Python ints and floats: no uint8 backtrace, no
    vectorisation, nothing shared with the implementation under test except
    the tie order (2 only if strictly best, 1 only if strictly better than 0)."""
    T, L = lp.shape[0], len(targets)
    S = 2 * L + 1
    ninf = float("-inf")
    alpha = [ninf] * S
    alpha[0], alpha[1] = float(lp[0, blank]), float(lp[0, targets[0]])
    back = [[0] * S for _ in range(T)]
    for t in range(1, T):
        nxt = [ninf] * S
        nxt[0] = alpha[0] + float(lp[t, blank])
        for i in range(1, S):
            x0, x1, x2 = alpha[i], alpha[i - 1], ninf
            if i % 2 and i > 1 and targets[i // 2] != targets[i // 2 - 1]:
                x2 = alpha[i - 2]
            if x2 > x1 and x2 > x0:
                r, back[t][i] = x2, 2
            elif x1 > x0:
                r, back[t][i] = x1, 1
            else:
                r, back[t][i] = x0, 0
            lab = blank if i % 2 == 0 else targets[i // 2]
            nxt[i] = r + float(lp[t, lab])
        alpha = nxt
    s = S - 1 if alpha[S - 1] > alpha[S - 2] else S - 2
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = blank if s % 2 == 0 else int(targets[s // 2])
        s -= back[t][s]
    return path


def _tiny_cases(blank, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        C = int(rng.integers(3, 6))
        T = int(rng.integers(2, 6))
        L = int(rng.integers(1, 4))
        tg = [int(x) for x in rng.integers(0, C, size=L) if x != blank]
        if not tg or blank >= C:
            continue
        if T < len(tg) + adjacent_repeats(tg):
            continue
        # small integers: exact ties everywhere, which is the point
        lp = -rng.integers(0, 3, size=(T, C)).astype(np.float32)
        out.append((lp, tg))
    return out


@pytest.mark.parametrize("blank", [0, 2, 4])
def test_returns_a_most_likely_valid_path_on_every_tiny_case(blank):
    for lp, tg in _tiny_cases(blank, n=120, seed=blank):
        path, scores = forced_align(lp, tg, blank=blank)
        assert _collapse(path, blank) == tg, (path, tg)
        assert np.array_equal(scores, lp[np.arange(len(path)), path])
        assert float(scores.sum()) == pytest.approx(_best_valid_score(lp, tg, blank), abs=1e-5)


def test_the_torchaudio_tie_defect_is_not_reproduced():
    """pytorch/audio#4221: on this input torchaudio 2.11 returns [1, 2, 2] with
    score -3.0 because its chain falls through to the worst predecessor when
    the advance-one and advance-two candidates tie. The optimum is -1.0."""
    lp = np.array([[0., -1., -1.], [0., -1., -2.], [-2., 0., 0.]], dtype=np.float32)
    path, scores = forced_align(lp, [1, 2], blank=0)
    assert _collapse(path, 0) == [1, 2]
    assert float(scores.sum()) == -1.0


@pytest.mark.parametrize("blank", [0, 7])
def test_matches_a_scalar_viterbi_past_the_uint8_range(blank):
    """N = 2L+1 > 255 exercises the backtrace cursor: written as `s -= bt[t, s]`
    it overflows as uint8 and was invisible on every small case."""
    rng = np.random.default_rng(7)
    T, C, L = 400, 12, 160
    tg = [int(x) for x in rng.integers(0, C, size=L) if x != blank]
    logits = rng.standard_normal((T, C)).astype(np.float32)
    logits[:, blank] += 1.5                     # blanks must actually reach the path
    lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    assert 2 * len(tg) + 1 > 255
    path, scores = forced_align(lp, tg, blank=blank)
    assert path.tolist() == _scalar_viterbi(lp, tg, blank)
    assert _collapse(path, blank) == tg


def test_merge_tokens_has_an_exclusive_end_and_a_mean_score():
    path = [0, 1, 1, 0, 2, 2, 2]
    scores = np.array([-9, -1, -3, -9, -2, -4, -6], dtype=np.float32)
    assert merge_tokens(path, scores) == [
        TokenSpan(1, 1, 3, -2.0),               # frames 1, 2 -> end 3, not 2
        TokenSpan(2, 4, 7, -4.0),
    ]
    # a blank index other than 0 is filtered, and 0 is then an ordinary label
    assert merge_tokens([3, 0, 0, 3], np.zeros(4, np.float32), blank=3) == [TokenSpan(0, 1, 3, 0.0)]
    assert merge_tokens([0, 0], [0.0, 0.0]) == []
    assert merge_tokens([], []) == []              # torchaudio returned [] here too


def test_rejects_what_torchaudio_rejects():
    lp = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        forced_align(lp, [])
    with pytest.raises(ValueError, match="too long"):
        forced_align(lp, [1, 1, 1])             # 3 tokens + 2 repeats need 5 frames
    with pytest.raises(ValueError, match="blank"):
        forced_align(lp, [1, 0, 2], blank=0)
    with pytest.raises(ValueError, match="outside"):
        forced_align(lp, [1, 2], blank=-1)      # would silently index the last column
    with pytest.raises(ValueError, match="outside"):
        forced_align(lp, [1, 2], blank=4)
    forced_align(lp, [1, 2, 3])                 # and the tight fit itself is fine
    assert adjacent_repeats([1, 1, 1, 2, 2]) == 3 and adjacent_repeats([7]) == 0


def test_accepts_float64_and_non_contiguous_emissions():
    rng = np.random.default_rng(3)
    lp = rng.standard_normal((50, 6)).astype(np.float64)
    tg = [1, 2, 2, 5]
    path, scores = forced_align(lp, tg)
    path_f, _ = forced_align(np.asfortranarray(lp), tg)
    assert np.array_equal(path, path_f)
    assert scores.dtype == np.float32        # the DP works in float32 by design

