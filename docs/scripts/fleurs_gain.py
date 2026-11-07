"""Does the absolute input level change what Voxtral hears? FLEURS edition.

The hand-corrected podcast reference is 422 words -- section 8 of
docs/voxtral-audio-preprocessing.md shows its bootstrap interval is far too
wide to settle a level effect. FLEURS gives hundreds of independent utterances,
so a paired test over utterances has real power.

The clips are already 16 kHz mono float, so applying a gain isolates the level
question completely: no resampler, no integer stage, no downmix in the way.
Per-utterance error counts are written to /tmp/fleurs_gain.json for the paired
bootstrap in the same file.

A gain spec ending in "c" clips the result at +-1.0 and rounds it through int16,
i.e. it simulates a recording that arrives too hot for the production path. That
is how the cost of 3 dB headroom gets measured: drive both variants equally hot,
let one clip and give the other the headroom, and compare.

    python docs/scripts/fleurs_gain.py <n_samples> <build> <gain_db> [gain_db ...]
    python docs/scripts/fleurs_gain.py 150 models/voxtral-mini-8bit 0 -12 -24
    python docs/scripts/fleurs_gain.py 150 models/voxtral-mini-8bit 5.3c 2.3c
"""
import io
import json
import pathlib
import random
import sys
import time

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer

# Module level so the other measurement scripts can import load_clips() and
# paired_bootstrap() without this file reading their command line.
N = 100
RESAMPLES = 10000


def apply_spec(wav, spec):
    """"-12" -> gain in dB. "n1.24" -> scale this clip to that peak. A trailing
    "c" clips at +-1.0 and rounds through int16, i.e. simulates a recording that
    arrives too hot for the production path."""
    clip = spec.endswith("c")
    body = spec[:-1] if clip else spec
    if body.startswith("n"):
        a = wav / max(1e-9, float(np.abs(wav).max())) * float(body[1:])
    else:
        a = wav * 10 ** (float(body) / 20)
    if clip:
        a = (np.rint(np.clip(a, -1.0, 1.0) * 32768).clip(-32768, 32767)
             .astype(np.int16).astype(np.float32) / 32768)
    return np.ascontiguousarray(a, dtype=np.float32)


def load_clips():
    from datasets import load_dataset, Audio
    ds = load_dataset("google/fleurs", "de_de", split=f"test[:{N}]")
    ds = ds.cast_column("audio", Audio(decode=False))
    clips, refs = [], []
    for row in ds:
        a = row["audio"]
        data = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == 16000, sr
        clips.append(np.ascontiguousarray(wav))
        refs.append(row["transcription"])
    return clips, refs


def paired_bootstrap(a, b, key):
    """95% interval of the difference in the pooled rate, resampling utterances."""
    n = len(a)
    idx = list(range(n))
    diffs = []
    rng = random.Random(20260728)
    for _ in range(RESAMPLES):
        s = [rng.choice(idx) for _ in range(n)]
        ea = sum(a[i][key] for i in s); eb = sum(b[i][key] for i in s)
        ref = sum(a[i]["ref_" + key[0]] for i in s) or 1
        diffs.append((ea - eb) / ref * 100)
    diffs.sort()
    return diffs[int(0.025 * RESAMPLES)], diffs[int(0.975 * RESAMPLES)]


def main():
    global N
    N = int(sys.argv[1])
    BUILD = sys.argv[2]
    GAINS = sys.argv[3:] or ["0"]
    clips, refs = load_clips()
    sec = sum(len(c) for c in clips) / 16000
    print(f"# FLEURS de_de test, {len(clips)} recordings, {sec/60:.1f} min")
    print(f"# Build: {BUILD}\n")
    print(f"{'Gain':>7s} {'peak':>6s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")

    import mlx.core as mx
    from noScribe.voxtral_engine import _Voxtral
    vox = _Voxtral(BUILD)

    per_gain = {}
    for g_db in GAINS:
        rows = []
        t0 = time.time()
        peak = 0.0
        for wav, ref_text in zip(clips, refs):
            a = apply_spec(wav, g_db)
            peak = max(peak, float(np.abs(a).max()))
            hyp = vox.transcribe_array(
                a, "de", max_new_tokens=int(len(a) / 16000 * 20) + 512)
            r, h = norm(ref_text), norm(hyp)
            rc, hc = "".join(r), "".join(h)
            rows.append(dict(w=wer(r, h)[0], ref_w=len(r),
                             c=wer(rc, hc)[0], ref_c=len(rc), hyp=hyp))
        el = time.time() - t0
        W = sum(r["w"] for r in rows) / max(1, sum(r["ref_w"] for r in rows)) * 100
        C = sum(r["c"] for r in rows) / max(1, sum(r["ref_c"] for r in rows)) * 100
        print(f"{str(g_db):>7s} {peak:6.3f} {W:6.2f}% {C:6.2f}% {sec/el:6.2f}x", flush=True)
        per_gain[g_db] = rows
        json.dump(per_gain, open("/tmp/fleurs_gain.json", "w"))
        mx.clear_cache()

    if len(GAINS) > 1:
        print("\nPaired differences over recordings "
              "(95% interval; if it contains 0, the difference is not demonstrable):")
        base = GAINS[0]
        for g_db in GAINS[1:]:
            a, b = per_gain[base], per_gain[g_db]
            lo_w, hi_w = paired_bootstrap(a, b, "w")
            lo_c, hi_c = paired_bootstrap(a, b, "c")
            star_w = "" if lo_w <= 0 <= hi_w else "  *"
            star_c = "" if lo_c <= 0 <= hi_c else "  *"
            print(f"  {base} dB - {g_db} dB   "
                  f"dWER {(sum(x['w'] for x in a)-sum(x['w'] for x in b))/sum(x['ref_w'] for x in a)*100:+.2f} "
                  f"[{lo_w:+.2f}, {hi_w:+.2f}]{star_w}   "
                  f"dCER {(sum(x['c'] for x in a)-sum(x['c'] for x in b))/sum(x['ref_c'] for x in a)*100:+.2f} "
                  f"[{lo_c:+.2f}, {hi_c:+.2f}]{star_c}")
        print("\n* = interval excludes 0")


if __name__ == "__main__":
    main()
