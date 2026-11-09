"""Mel-domain comparison of the pre-processing variants, offset separated out.

A gain change moves every log-mel feature by the same amount -- it is a pure
additive offset, not a distortion (section 2 of docs/voxtral-audio-preprocessing.md).
Reporting a raw mean absolute difference therefore makes a level change look
catastrophic next to a real signal defect. So every difference here is split:

  offset  the best-fit constant -- recoverable in principle, a knob
  shape   what is left after removing it -- the part no gain can undo

Reference is the float/soxr rendering at the source's own level: the most
faithful representation of the recording we can put in front of the encoder.

    python docs/scripts/preproc_mel.py <source-audio> <t0> <t1> [variant ...]
"""
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from preproc_variants import build, SR
from mlx_voxtral.audio_processing import process_audio_chunk

WINDOW = 30 * SR


def mel(a):
    """Log-mel over the whole clip, one 30 s window at a time.

    This was how the model saw it up to mlx-voxtral 0.0.5. Since 0.0.6 the
    spectrogram is computed over the whole audio and split afterwards, so the
    per-window clamp floor this reproduces is gone -- and the variants that
    move the file-wide maximum (p1-headroom3, p3-peak1) are exactly the ones
    that would read differently. `process_audio_chunk` still exists, so a
    rerun succeeds and silently measures the retired path.
    """
    out = []
    for i in range(0, len(a), WINDOW):
        out.append(np.array(process_audio_chunk(a[i:i + WINDOW])))
    return np.concatenate(out, axis=0)


def main():
    src, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    ids = sys.argv[4:] or ["p0-ist", "p1-headroom3", "p2-soxr", "p3-peak1",
                           "p4-lufs23", "p5-lufs16", "hot-clip", "hot-headroom"]
    if "p2-soxr" not in ids:
        ids = ids + ["p2-soxr"]
    v = build(src, t0, t1, ids)
    ref = mel(v["p2-soxr"])

    print(f"# Reference: p2-soxr (float, soxr VHQ, original level) on "
          f"{pathlib.Path(src).name} {t0:.0f}-{t1:.0f}s")
    print(f"# Feature value range of the reference: {ref.min():+.2f} .. {ref.max():+.2f}\n")
    print(f"{'variant':16s} {'Offset':>8s} {'Shape RMS':>10s} {'Shape p99.9':>12s} "
          f"{'Shape max':>10s} {'Bins>111':>9s}")
    for vid in ids:
        d = mel(v[vid]) - ref
        off = float(d.mean())
        s = d - off
        top = np.abs(s[:, 112:]).mean() if s.shape[1] >= 128 else float("nan")
        print(f"{vid:16s} {off:+8.4f} {np.sqrt((s ** 2).mean()):10.2e} "
              f"{np.quantile(np.abs(s), 0.999):12.2e} {np.abs(s).max():10.4f} "
              f"{top:9.2e}")


if __name__ == "__main__":
    main()
