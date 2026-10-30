"""Is denoising before the encoder worth it?

The obvious worry: denoisers are optimised for the human ear and leave
artefacts an ASR model has never seen in training. The obvious hope: a noisy
recording becomes more legible. Both are measurable if one adds the noise
oneself and knows the truth.

FLEURS is clean, so noise is added under control (white noise at a chosen
SNR) and every denoising chain is then held against "doing nothing at all".

    python docs/scripts/fleurs_noise.py <n> <build> <snr_db> <cond> [cond ...]
    python docs/scripts/fleurs_noise.py 150 models/voxtral-mini-8bit 10 raw afftdn anlmdn
"""
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer
from audio_filters import apply_filter, CHAINS
import fleurs_gain
from fleurs_gain import load_clips, paired_bootstrap

SR = 16000
CHAINS = dict(CHAINS)
CHAINS.setdefault("anlmdn", "anlmdn=s=0.001:p=0.002:r=0.006")


def noisy(clip, snr_db, rng):
    """Add white noise at the requested SNR, measured over the clip."""
    p = float((clip.astype(np.float64) ** 2).mean())
    n = rng.standard_normal(len(clip)) * np.sqrt(p / 10 ** (snr_db / 10))
    return np.ascontiguousarray(clip + n, dtype=np.float32)


def render(x, cond):
    if cond == "raw":
        return np.ascontiguousarray(x, dtype=np.float32)
    return apply_filter(x, CHAINS[cond])


def main():
    n, build, snr = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
    conds = sys.argv[4:] or ["raw"]
    fleurs_gain.N = n
    clips, refs = load_clips()
    rng = np.random.default_rng(20260728)
    noised = [noisy(c, snr, rng) for c in clips]
    sec = sum(len(c) for c in clips) / SR
    print(f"# FLEURS de_de test, {len(clips)} recordings, {sec/60:.1f} min, "
          f"white noise at {snr:.0f} dB SNR")
    print(f"# Build: {build}\n")
    print(f"{'condition':14s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")

    import mlx.core as mx
    from noScribe.voxtral_engine import _Voxtral
    vox = _Voxtral(build)

    store = {}
    for cond in ["clean"] + conds:
        rows = []
        t0 = time.time()
        src = clips if cond == "clean" else noised
        for clip, ref_text in zip(src, refs):
            a = np.ascontiguousarray(clip, np.float32) if cond == "clean" \
                else render(clip, cond)
            hyp = vox.transcribe_array(a, "de",
                                       max_new_tokens=int(len(a) / SR * 20) + 512)
            r, h = norm(ref_text), norm(hyp)
            rc, hc = "".join(r), "".join(h)
            rows.append(dict(w=wer(r, h)[0], ref_w=len(r),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        W = sum(r["w"] for r in rows) / max(1, sum(r["ref_w"] for r in rows)) * 100
        C = sum(r["c"] for r in rows) / max(1, sum(r["ref_c"] for r in rows)) * 100
        print(f"{cond:14s} {W:6.2f}% {C:6.2f}% {sec/el:6.2f}x", flush=True)
        store[cond] = rows
        json.dump(store, open("/tmp/fleurs_noise.json", "w"))
        mx.clear_cache()

    print("\nAgainst 'raw' (noisy, untreated), paired over recordings:")
    base = store["raw"]
    for cond in [c for c in store if c != "raw"]:
        b = store[cond]
        lo_w, hi_w = paired_bootstrap(base, b, "w")
        lo_c, hi_c = paired_bootstrap(base, b, "c")
        dW = (sum(x['w'] for x in base) - sum(x['w'] for x in b)) / sum(x['ref_w'] for x in base) * 100
        dC = (sum(x['c'] for x in base) - sum(x['c'] for x in b)) / sum(x['ref_c'] for x in base) * 100
        print(f"  raw - {cond:12s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]"
              f"{'' if lo_w <= 0 <= hi_w else '  *'}   "
              f"dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]"
              f"{'' if lo_c <= 0 <= hi_c else '  *'}")
    print("\n* = interval excludes 0")


if __name__ == "__main__":
    main()
