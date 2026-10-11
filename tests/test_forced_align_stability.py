"""Acceptance test for the Voxtral aligner's DP against a recorded reference.

The word timestamps come from ``noScribe.ctc_align`` (``forced_align`` +
``merge_tokens``), a numpy replacement for the torchaudio kernel of the same
names. ``tests/data/forced_align_ref.npz`` holds 43 cases -- random emissions
incl. heavy repeats, tight T == L+R fits, single-token targets -- with the
inputs (``lp_i``, ``targets_i``) and torchaudio's answer to them (``paths_i``,
``scores_i``, ``spani_i``, ``spanf_i``), recorded from torchaudio 2.8 on
2026-07-22 and verified identical on 2.11. The numpy DP must reproduce every
path and span bit-for-bit; that is what makes it a drop-in rather than a
rewrite.

The inputs are stored, not regenerated, so this file needs neither torch nor
torchaudio and runs wherever ``tests/test_ctc_align.py`` does. Only
``--regenerate`` needs both, because the reference is meant to stay the
*other* implementation's answer:

    python tests/test_forced_align_stability.py --regenerate

What the reference cannot see, tests/test_ctc_align.py covers: blank indices
other than 0, and exact ties, where the numpy DP is deliberately *better*
than torchaudio (pytorch/audio#4221).
"""
from pathlib import Path

import numpy as np
import pytest

from noScribe import ctc_align

REF_PATH = Path(__file__).parent / "data" / "forced_align_ref.npz"
N_CASES = 43


@pytest.fixture(scope="module")
def ref():
    return np.load(REF_PATH)


@pytest.mark.parametrize("idx", range(N_CASES))
def test_forced_align_matches_recorded_reference(ref, idx):
    lp, targets = ref[f"lp_{idx}"], ref[f"targets_{idx}"]
    paths, scores = ctc_align.forced_align(lp, targets, blank=0)
    spans = ctc_align.merge_tokens(paths, scores)
    # The integer outputs decide the word timestamps -- they must be
    # bit-identical everywhere (verified across torchaudio 2.8/2.11 and
    # macOS arm64 / Linux x86_64). The reference keeps torchaudio's batch axis.
    assert np.array_equal(ref[f"paths_{idx}"][0], paths), "alignment path changed"
    assert np.array_equal(ref[f"spani_{idx}"],
                          [[s.token, s.start, s.end] for s in spans]), "token spans changed"
    # Float scores may differ in the last ulp with the platform's reduction
    # order, so a tight tolerance instead of equality.
    assert np.allclose(ref[f"scores_{idx}"][0], scores, rtol=0, atol=1e-5), "frame scores drifted"
    assert np.allclose(ref[f"spanf_{idx}"], [s.score for s in spans], rtol=0, atol=1e-5), \
        "span scores drifted"


def test_reference_is_complete(ref):
    """Every case carries inputs and all four recorded outputs, and the inputs
    are consistent with each other -- a half-regenerated file must not pass."""
    for idx in range(N_CASES):
        lp, targets = ref[f"lp_{idx}"], ref[f"targets_{idx}"]
        assert lp.dtype == np.float32 and lp.ndim == 2
        assert ref[f"paths_{idx}"].shape == (1, lp.shape[0])
        assert targets.min() >= 1 and targets.max() < lp.shape[1]
        assert len(ref[f"spani_{idx}"]) == len(ref[f"spanf_{idx}"]) >= 1


# --------------------------------------------------------------------------- #
# Recording the reference: torch + torchaudio, and only here
# --------------------------------------------------------------------------- #
def _cases():
    """Deterministic case specs: (T, C, targets) -- the set the file was
    recorded from. Changing it changes every key, so only with --regenerate."""
    rng = np.random.default_rng(42)
    specs = []
    for i in range(40):
        C = int(rng.integers(5, 60))           # vocab incl. blank
        L = int(rng.integers(1, 80))           # target length
        tg = rng.integers(1, C, size=L)
        if i % 3 == 0 and L > 3:               # force heavy repeats sometimes
            tg[1::2] = tg[0:-1:2][: len(tg[1::2])]
        R = int(np.sum(tg[1:] == tg[:-1]))
        T = int(L + R + rng.integers(0, 200))  # from tightest possible upward
        specs.append((T, C, tg))
    specs.append((1, 5, np.array([2])))        # single frame, single token
    specs.append((500, 40, np.array([7])))     # long audio, one token
    specs.append((7, 6, np.array([3, 3, 3, 3])))  # T == L+R exactly
    assert len(specs) == N_CASES
    return specs


def _record(T, C, tg):
    """torchaudio's answer to a seeded random emission, as stored in the file."""
    import torch
    import torchaudio.functional as F

    g = torch.Generator().manual_seed(hash((T, C, len(tg))) % (2**31))
    lp = torch.randn((1, T, C), generator=g).log_softmax(-1)
    targets = torch.tensor(tg, dtype=torch.int32).unsqueeze(0)
    paths, scores = F.forced_align(lp, targets, blank=0)
    spans = F.merge_tokens(paths[0], scores[0])
    return {
        "lp": lp[0].numpy().astype(np.float32),
        "targets": np.asarray(tg, dtype=np.int64),
        "paths": paths.numpy(),
        "scores": scores.numpy(),
        "spani": np.array([[s.token, s.start, s.end] for s in spans], dtype=np.int64),
        "spanf": np.array([s.score for s in spans], dtype=np.float64),
    }


if __name__ == "__main__":
    import sys

    if "--regenerate" not in sys.argv:
        sys.exit("run via pytest, or pass --regenerate to rewrite the reference")
    out = {}
    for i, spec in enumerate(_cases()):
        for key, arr in _record(*spec).items():
            out[f"{key}_{i}"] = arr
    REF_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(REF_PATH, **out)
    import torchaudio

    print(f"wrote {REF_PATH} from torchaudio {torchaudio.__version__}")
