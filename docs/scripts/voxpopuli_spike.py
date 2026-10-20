"""Round 2: the transient damage on many streams instead of one passage.

The dose-response of the clamp floor (`docs/voxtral-audio-preprocessing.md`,
`docs/voxtral-mel-clamp-floor.md`) initially rested on a single hand-corrected
passage. Here it runs over ten VoxPopuli streams with gold transcripts, each
once clean and once with an inserted transient, under both floors.

The question is not "which floor is better" -- `voxpopuli_floor.py` measures
that on clean material -- but: **what does a transient cost under each
floor.** The comparison is therefore always WITHIN a scheme (clean against
bang under the same floor), which also cancels out any bias of a reference
transcript.

The bang is placed relative to the speech peak of the stream, not absolutely,
so that all streams see the same dose.

    python docs/scripts/voxpopuli_spike.py [n_streams] [seconds] [dB_above_speech] [floors]

`VOX_BUILD=models/voxtral-small-4bit` measures another build than the 3B one.
With exactly two floor arms the run also pairs the floors against each other,
on clean audio and with the bang.

`floors` is the arm list in the short form of `voxpopuli_floor.floor_arm`
(`max,99.9,99c20`); clean is compared against bang per arm, so a max arm is
not mandatory. The max floor comes explicitly from the library, the percentile
floors from the production code (`_PercentileFloorFeatures`), because since it
went in the extractor on `vox.proc` is already the percentile path.
"""
import os
import pathlib
import sys
import time

import numpy as np
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer                                    # noqa: E402
from fleurs_stream import streams, paired, SR                # noqa: E402
from noScribe.voxtral_engine import _Voxtral                 # noqa: E402
from voxpopuli_floor import floor_arm, load_clips            # noqa: E402


def pair(x, db_over, at=0.45, ms=100):
    """(clean, with bang) at a guaranteed dose `db_over` above the speech peak.

    The bang is **not** amplified beyond the speech peak; instead the audio is
    attenuated and the impulse set to full scale. The detour is necessary
    because real material is already driven to 1.0: an upscaled impulse gets
    clipped to 1.0 there and the dose silently fails to arrive. That is exactly
    what happened on a first run -- the requested +12 dB arrived as a median
    5.2 dB floor rise, two streams saw 0.1 dB, and the result looked like a
    clean null effect.

    Both return values carry the same attenuation, so the comparison is paired.
    """
    a = np.array(x, dtype=np.float32, copy=True)
    a *= (10.0 ** (-db_over / 20.0)) / max(float(np.abs(a).max()), 1e-9)
    sp = a.copy()
    n = int(ms * SR / 1000)
    pos = int(at * len(a))
    sp[pos:pos + n] = 1.0
    return a, sp


def floor_rise_db(clean, spiked):
    """By how much does the impulse lift the clamp floor? The control that the
    intervention takes effect at all."""
    from mel_outlier_gap import raw_log_mel
    return (raw_log_mel(spiked).max() - raw_log_mel(clean).max()) * 10.0


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    db_over = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    specs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["max", "99.9"]

    clips, refs = load_clips(int(n_streams * sec / 9) + 60)
    st = streams(clips, refs, n_streams, sec)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} streams from VoxPopuli de (gold), ~{sec:.0f}s each, "
          f"{total/60:.1f} min total")
    print(f"# Transient: {db_over:+.0f} dB above the speech peak, 100 ms, at 45 %")

    vox = _Voxtral(os.environ.get("VOX_BUILD") or str(REPO / "models" / "voxtral-mini-8bit"))
    arms = [floor_arm(s) for s in specs]

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    # Does the dose arrive? Without this control the run may measure nothing.
    rises = [floor_rise_db(*pair(x, db_over)) for x, _ in st]
    print(f"# Control: floor rise median {np.median(rises):.1f} dB "
          f"(min {min(rises):.1f}, max {max(rises):.1f})")
    assert np.median(rises) > 0.7 * db_over, (
        f"Dose does not arrive: median {np.median(rises):.1f} dB instead of ~{db_over}")

    def score(ex, spike):
        vox.proc.feature_extractor = ex
        rows, t0 = [], time.time()
        for x, words in st:
            a = pair(x, db_over)[1 if spike else 0]
            hyp = norm(vox.transcribe_array(
                a, "de", max_new_tokens=int(len(a) / SR * 20) + 512))
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        mx.clear_cache()
        return rows, time.time() - t0

    print(f"\n{'floor':16s} {'audio':10s} {'WER':>7s} {'CER':>7s}")
    store = {}
    for name, ex in arms:
        for au, spike in (("clean", False), ("with bang", True)):
            rows, el = score(ex, spike)
            store[(name, au)] = rows
            print(f"{name:16s} {au:10s} {rate(rows,'w','ref_w'):6.2f}% "
                  f"{rate(rows,'c','ref_c'):6.2f}%", flush=True)

    print("\nCost of the transient, paired per floor "
          "(95 %; if it contains 0, not demonstrable):")
    for name, _ in arms:
        a, b = store[(name, "clean")], store[(name, "with bang")]
        lo_w, hi_w = paired(a, b, "w", "ref_w")
        lo_c, hi_c = paired(a, b, "c", "ref_c")
        dW = rate(b, "w", "ref_w") - rate(a, "w", "ref_w")
        dC = rate(b, "c", "ref_c") - rate(a, "c", "ref_c")
        star = " *" if (lo_w > 0) == (hi_w > 0) else ""
        print(f"  {name:16s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star}"
              f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]")
    print("  (positive = the bang hurts)")
    if len(arms) == 2:
        (n0, _), (n1, _) = arms
        print(f"\n{n1} against {n0}, paired (negative = {n1} better):")
        for au in ("clean", "with bang"):
            a, b = store[(n0, au)], store[(n1, au)]
            lo_w, hi_w = paired(a, b, "w", "ref_w")
            lo_c, hi_c = paired(a, b, "c", "ref_c")
            dW = rate(b, "w", "ref_w") - rate(a, "w", "ref_w")
            dC = rate(b, "c", "ref_c") - rate(a, "c", "ref_c")
            print(f"  {au:10s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]"
                  f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]")


if __name__ == "__main__":
    main()
