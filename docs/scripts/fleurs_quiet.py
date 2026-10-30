"""Does a quiet passage survive? A test with ground truth for the failure mode
that the podcast passage only showed once.

On the hand-corrected podcast excerpt the whole CER difference between a loud
and a quiet rendering came down to ONE recovered aside -- a parenthetical spoken
markedly quieter than the surrounding speech. That is an anecdote. This builds
the same situation deliberately, hundreds of times, out of FLEURS utterances
whose transcript is known:

    LOUD utterance | 0.5 s silence | QUIET utterance (-N dB) | 0.5 s silence

and reports, per condition, how much of the LOUD and of the QUIET part came back.
Because the two halves are scored separately, a condition that rescues quiet
speech at the cost of loud speech cannot hide behind a flat overall CER.

Conditions are gains ("-12"), named filter chains from audio_filters.CHAINS
("speechnorm"), or "raw".

With `--noise <dB>` an even noise bed is laid underneath, that many dB below
the loud half. Without it the attenuated half keeps its signal-to-noise ratio,
which is never the case in real material -- see the warning in build_pairs.

    python docs/scripts/fleurs_quiet.py <n_pairs> <build> <drop_db> [--noise <dB>] <cond> [cond ...]
    python docs/scripts/fleurs_quiet.py 90 models/voxtral-mini-8bit 25 raw -12 speechnorm
    python docs/scripts/fleurs_quiet.py 90 models/voxtral-mini-8bit 25 --noise 45 raw dynaudnorm
"""
import collections
import json
import pathlib
import random
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer
from audio_filters import apply_filter, CHAINS

SR = 16000
GAP = int(0.5 * SR)
RESAMPLES = 10000


def build_pairs(clips, refs, drop_db, noise_db=None, seed=20260728):
    """LOUD | gap | QUIET | gap, plus the two reference word lists.

    `noise_db` lays an even noise floor over the finished pair, that many dB
    below the RMS of the loud half. Without it the test is unrealistic in
    exactly the way that decides the question: attenuating a clean recording
    preserves its signal-to-noise ratio, a genuinely quiet voice in the same
    room does not. A leveller can undo the attenuation, but it lifts the noise
    with it and cannot give back an SNR that was never there.
    """
    out = []
    rng = np.random.default_rng(seed)
    g = 10 ** (-abs(drop_db) / 20)
    for i in range(0, len(clips) - 1, 2):
        a, b = clips[i], clips[i + 1]
        x = np.concatenate([a, np.zeros(GAP, np.float32),
                            (b * g).astype(np.float32), np.zeros(GAP, np.float32)])
        if noise_db is not None:
            loud_rms = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
            x = x + rng.standard_normal(len(x)) * loud_rms * 10 ** (-abs(noise_db) / 20)
        out.append((np.ascontiguousarray(x, np.float32),
                    norm(refs[i]), norm(refs[i + 1])))
    return out


def recall(ref_words, hyp_words):
    """Fraction of the reference words present in the hypothesis, with multiplicity.

    Deliberately not an edit distance: the question here is whether the passage
    reached the transcript at all, not how it was spelled.
    """
    have = collections.Counter(hyp_words)
    got = 0
    for w, k in collections.Counter(ref_words).items():
        got += min(k, have.get(w, 0))
    return got, len(ref_words)


def render(x, cond):
    if cond == "raw":
        return x
    if cond in CHAINS:
        return apply_filter(x, CHAINS[cond])
    return (x * 10 ** (float(cond) / 20)).astype(np.float32)


def paired(a, b, num, den):
    rng = random.Random(20260728)
    n = len(a)
    diffs = []
    for _ in range(RESAMPLES):
        s = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i][num] for i in s) / max(1, sum(a[i][den] for i in s))
        db = sum(b[i][num] for i in s) / max(1, sum(b[i][den] for i in s))
        diffs.append((db - da) * 100)
    diffs.sort()
    return diffs[int(0.025 * RESAMPLES)], diffs[int(0.975 * RESAMPLES)]


def main():
    n_pairs, build, drop = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
    args = sys.argv[4:]
    noise = None
    if "--noise" in args:
        k = args.index("--noise")
        noise = float(args[k + 1])
        args = args[:k] + args[k + 2:]
    conds = args or ["raw"]

    import fleurs_gain
    fleurs_gain.N = n_pairs * 2
    clips, refs = fleurs_gain.load_clips()
    pairs = build_pairs(clips, refs, drop, noise)
    sec = sum(len(p[0]) for p in pairs) / SR
    print(f"# FLEURS de_de, {len(pairs)} pairs, {sec/60:.1f} min, "
          f"quiet half {drop:.0f} dB below the loud one"
          + (f", noise bed {noise:.0f} dB below the loud one, "
             f"the quiet half thus sits {noise-drop:+.0f} dB above it"
             if noise else ""))
    print(f"# Build: {build}\n")
    print(f"{'condition':14s} {'peak':>6s} {'loud recall':>12s} {'quiet recall':>13s} "
          f"{'WER':>7s} {'CER':>7s} {'Speed':>7s}")

    import mlx.core as mx
    from noScribe.voxtral_engine import _Voxtral
    vox = _Voxtral(build)

    store = {}
    for cond in conds:
        rows = []
        peak = 0.0
        t0 = time.time()
        for x, ref_a, ref_b in pairs:
            a = np.ascontiguousarray(render(x, cond), dtype=np.float32)
            peak = max(peak, float(np.abs(a).max()))
            hyp = norm(vox.transcribe_array(
                a, "de", max_new_tokens=int(len(a) / SR * 20) + 512))
            ga, na = recall(ref_a, hyp)
            gb, nb = recall(ref_b, hyp)
            ref = ref_a + ref_b
            rc, hc = "".join(ref), "".join(hyp)
            rows.append(dict(la=ga, na=na, lb=gb, nb=nb,
                             w=wer(ref, hyp)[0], ref_w=len(ref),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        s = lambda k: sum(r[k] for r in rows)
        print(f"{cond:14s} {peak:6.3f} {s('la')/max(1,s('na'))*100:11.1f}% "
              f"{s('lb')/max(1,s('nb'))*100:12.1f}% "
              f"{s('w')/max(1,s('ref_w'))*100:6.2f}% {s('c')/max(1,s('ref_c'))*100:6.2f}% "
              f"{sec/el:6.2f}x", flush=True)
        store[cond] = rows
        json.dump(store, open("/tmp/fleurs_quiet.json", "w"))
        mx.clear_cache()

    if len(conds) > 1:
        print("\nAgainst the first condition, paired over pairs "
              "(95% interval, positive = better):")
        base = store[conds[0]]
        for cond in conds[1:]:
            lo_b, hi_b = paired(base, store[cond], "lb", "nb")
            lo_a, hi_a = paired(base, store[cond], "la", "na")
            d = lambda k, n: (sum(r[k] for r in store[cond]) / max(1, sum(r[n] for r in store[cond]))
                              - sum(r[k] for r in base) / max(1, sum(r[n] for r in base))) * 100
            print(f"  {conds[0]} -> {cond:12s} quiet {d('lb','nb'):+6.2f} "
                  f"[{lo_b:+.2f}, {hi_b:+.2f}]{'' if lo_b <= 0 <= hi_b else '  *'}   "
                  f"loud {d('la','na'):+6.2f} "
                  f"[{lo_a:+.2f}, {hi_a:+.2f}]{'' if lo_a <= 0 <= hi_a else '  *'}")
        print("\n* = interval excludes 0")


if __name__ == "__main__":
    main()
