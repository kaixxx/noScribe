"""Does the percentile floor cost anything when a pass is mostly silent?

`voxpopuli_floor.py` measures dense speech: the streams consist almost
entirely of speech, and there the 99th percentile lies 19.5-26.8 dB below the
maximum (median 22; on the edited interview only 15.6-17.8). In a pass that
consists mostly of pauses or lead-in -- the 60-s head probe on a recording
that starts with silence, a window over a long pause --, the top percent of
the cells is still speech, but the percentile slides downwards within the
speech cells, and the floor sits lower than in any dense stream. Whether the
model hears differently there is what this run measures: the same gold
transcripts, the same clips, but room noise between the clips instead of
0.4 s of silence, until the speech share is at `share`. Both floors on
bit-identical audio, paired.

    python docs/scripts/voxpopuli_sparse.py [n_streams] [seconds] [share] [floors]

`floors` in the short form of `voxpopuli_floor.floor_arm` (`max,99,99c20`);
the first arm is the baseline of the paired evaluation.

`share` is the speech share of the stream (0.5 = half speech, half noise).
The room noise is white noise at -60 dBFS -- the level of a quiet room in a
recording that is driven to 1.0.
"""
import pathlib
import sys

import numpy as np
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer                                    # noqa: E402
from fleurs_stream import paired, SR                         # noqa: E402
from noScribe.voxtral_engine import _Voxtral                 # noqa: E402
from voxpopuli_floor import floor_arm, load_clips            # noqa: E402
from mel_outlier_gap import raw_log_mel                      # noqa: E402

ROOM_DB = -60.0


def sparse_streams(clips, refs, n_streams, sec, share, seed=20260823):
    """Streams of `sec` seconds in which speech makes up `share` of the time
    and the rest is room noise, spread evenly between the clips."""
    rng = np.random.default_rng(seed)
    out, i = [], 0
    want_speech = int(sec * share * SR)
    while len(out) < n_streams and i < len(clips):
        parts, words, total = [], [], 0
        while total < want_speech and i < len(clips):
            parts.append(clips[i])
            words += norm(refs[i])
            total += len(clips[i])
            i += 1
        if total < want_speech * 0.8 or not words:
            continue
        gap = int((sec * SR - total) / (len(parts) + 1))
        room = lambda: (rng.standard_normal(gap) * 10 ** (ROOM_DB / 20)).astype(np.float32)
        audio = [room()]
        for p in parts:
            audio += [p, room()]
        out.append((np.concatenate(audio).astype(np.float32), words))
    return out


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    specs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["max", "99"]

    clips, refs = load_clips(int(n_streams * sec * share / 9) + 60)
    st = sparse_streams(clips, refs, n_streams, sec, share)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} streams from VoxPopuli de (gold), ~{sec:.0f}s each, "
          f"speech share {share:.0%}, {total/60:.1f} min total")

    # The dose: how far below the maximum does the percentile lie?
    for spec in specs:
        if spec == "max":
            continue
        p = float(spec.split("c")[0])
        gaps = [(raw_log_mel(x).max() - np.percentile(raw_log_mel(x), p)) * 10
                for x, _ in st]
        print(f"# Gap max -> percentile {p:g}: median {np.median(gaps):.1f} dB "
              f"(min {min(gaps):.1f}, max {max(gaps):.1f})")

    vox = _Voxtral(str(REPO / "models" / "voxtral-mini-8bit"))
    arms = [floor_arm(s) for s in specs]

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    print(f"\n{'floor':18s} {'WER':>7s} {'CER':>7s}")
    store = {}
    for name, ex in arms:
        vox.proc.feature_extractor = ex
        rows = []
        for x, words in st:
            hyp = norm(vox.transcribe_array(
                x, "de", max_new_tokens=int(len(x) / SR * 20) + 512))
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        print(f"{name:18s} {rate(rows,'w','ref_w'):6.2f}% {rate(rows,'c','ref_c'):6.2f}%",
              flush=True)
        store[name] = rows
        mx.clear_cache()

    a = store[arms[0][0]]  # first arm is the baseline
    print(f"\nPaired against {arms[0][0]} (95%; if it contains 0, not demonstrable):")
    for name, _ in arms[1:]:
        b = store[name]
        lo_w, hi_w = paired(a, b, "w", "ref_w")
        lo_c, hi_c = paired(a, b, "c", "ref_c")
        dW = rate(b, "w", "ref_w") - rate(a, "w", "ref_w")
        dC = rate(b, "c", "ref_c") - rate(a, "c", "ref_c")
        star = " *" if (lo_w > 0) == (hi_w > 0) else ""
        print(f"  {name:18s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star}"
              f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]")
    print("  (positive = percentile worse)")


if __name__ == "__main__":
    main()
