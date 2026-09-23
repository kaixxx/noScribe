"""Does it matter WHERE in the 30-s window the speech lies?

The feature extractor splits every input into 30-second windows and pads the
last one with zeros (`mlx_voxtral/audio_processing.py`, `process_audio_chunk`).
Our chunker cuts at speech pauses, not on the 30-s grid -- an utterance can
therefore sit right on a window boundary and is then seen by two encoder
windows.

(Up to mlx-voxtral 0.0.5 every window additionally had its own `log_max`
floor; since 0.0.6 normalisation runs over the whole file. The split encoder
view of the utterance remains, and the null result below still holds -- a
repeat would now measure only the position effect, no longer the floor.)

If the position on the grid made a difference, that would be a free win: the
chunker would merely have to align its cuts to the 30-s grid as well. If not,
the question is settled.

Measured by prepending k seconds of silence to every FLEURS recording. At
roughly 12 s length that puts it once at the start of a window, once in the
middle, once across the boundary.

    python docs/scripts/fleurs_window.py <n> <build> <offset_s> [offset_s ...]
    python docs/scripts/fleurs_window.py 180 models/voxtral-mini-8bit 0 6 14 22 27
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
import fleurs_gain
from fleurs_gain import load_clips, paired_bootstrap

SR = 16000


def main():
    n, build = int(sys.argv[1]), sys.argv[2]
    offsets = sys.argv[3:] or ["0"]
    fleurs_gain.N = n
    clips, refs = load_clips()
    lens = np.array([len(c) / SR for c in clips])
    sec = lens.sum()
    print(f"# FLEURS de_de test, {len(clips)} recordings, {sec/60:.1f} min")
    print(f"# Length of the recordings: median {np.median(lens):.1f}s, "
          f"p5 {np.percentile(lens,5):.1f}s, p95 {np.percentile(lens,95):.1f}s")
    print(f"# Build: {build}\n")
    print(f"{'Offset':>8s} {'straddling':>13s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")

    import mlx.core as mx
    from noScribe.voxtral_engine import _Voxtral
    vox = _Voxtral(build)

    store = {}
    for off in offsets:
        pad = int(float(off) * SR)
        rows = []
        straddle = 0
        t0 = time.time()
        for clip, ref_text in zip(clips, refs):
            a = np.concatenate([np.zeros(pad, np.float32), clip]).astype(np.float32)
            if pad and (pad // (30 * SR)) != ((pad + len(clip)) // (30 * SR)):
                straddle += 1
            hyp = vox.transcribe_array(a, "de",
                                       max_new_tokens=int(len(a) / SR * 20) + 512)
            r, h = norm(ref_text), norm(hyp)
            rc, hc = "".join(r), "".join(h)
            rows.append(dict(w=wer(r, h)[0], ref_w=len(r),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        W = sum(r["w"] for r in rows) / max(1, sum(r["ref_w"] for r in rows)) * 100
        C = sum(r["c"] for r in rows) / max(1, sum(r["ref_c"] for r in rows)) * 100
        print(f"{off:>7s}s {straddle:12d} {W:6.2f}% {C:6.2f}% {sec/el:6.2f}x", flush=True)
        store[off] = rows
        json.dump(store, open("/tmp/fleurs_window.json", "w"))
        mx.clear_cache()

    if len(offsets) > 1:
        print("\nAgainst offset 0, paired over recordings "
              "(95% interval; if it contains 0, the difference is not demonstrable):")
        base = store[offsets[0]]
        for off in offsets[1:]:
            b = store[off]
            lo_w, hi_w = paired_bootstrap(base, b, "w")
            lo_c, hi_c = paired_bootstrap(base, b, "c")
            dW = (sum(x['w'] for x in base) - sum(x['w'] for x in b)) / sum(x['ref_w'] for x in base) * 100
            dC = (sum(x['c'] for x in base) - sum(x['c'] for x in b)) / sum(x['ref_c'] for x in base) * 100
            print(f"  0s - {off:>3s}s   dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]"
                  f"{'' if lo_w <= 0 <= hi_w else '  *'}   "
                  f"dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]"
                  f"{'' if lo_c <= 0 <= hi_c else '  *'}")
        print("\n* = interval excludes 0")


if __name__ == "__main__":
    main()
