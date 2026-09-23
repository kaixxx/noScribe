"""Find a known passage in another rendering of the same recording.

When a recording exists in a raw and in a processed version, a hand-corrected
reference points at only one of them. The other cannot be computed: a leveller
such as a commercial mastering service cuts silences, so the time axis is not
shifted linearly but compressed.

The search therefore runs on the loudness envelope, and start and end
separately, so that a compression inside the passage becomes visible. The
envelope survives levelling well enough: the pattern of speech and pause
remains even when the levels are equalised.

Besides the location, the correlation at the optimum and the second-best
correlation are printed. If those lie close together, the find is not
trustworthy.

    python docs/scripts/locate_passage.py <passage.wav> <file> [file ...]
"""
import pathlib
import sys

import numpy as np
import soundfile as sf
import soxr

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from audio_audit import decode_float

SR = 16000
HOP = 800          # 50 ms envelope
EDGE_SEC = 20      # this much of the start and of the end is used for searching


def envelope(x):
    e = np.abs(x[:len(x) // HOP * HOP]).reshape(-1, HOP).mean(axis=1)
    return (e - e.mean()) / (e.std() + 1e-9)


def find(hay_env, needle):
    n = envelope(needle)
    c = np.correlate(hay_env, n, "valid") / len(n)
    k = int(np.argmax(c))
    # second-best value outside the neighbourhood of the hit
    mask = np.ones(len(c), bool)
    mask[max(0, k - len(n)):k + len(n)] = False
    second = float(c[mask].max()) if mask.any() else 0.0
    return k * HOP / SR, float(c[k]), second


def main():
    passage, _ = sf.read(sys.argv[1], dtype="float32")
    print(f"# Passage {len(passage)/SR:.1f}s from {sys.argv[1]}")
    print(f"{'file':44s} {'start':>9s} {'end':>9s} {'length':>8s} "
          f"{'corr':>6s} {'2nd':>6s}")
    for path in sys.argv[2:]:
        x, sr = decode_float(path)
        hay = envelope(soxr.resample(x, sr, SR, quality="HQ").astype(np.float32))
        t_a, c_a, s_a = find(hay, passage[:SR * EDGE_SEC])
        t_b, c_b, s_b = find(hay, passage[-SR * EDGE_SEC:])
        end = t_b + EDGE_SEC
        print(f"{pathlib.Path(path).name[:44]:44s} {t_a:8.2f}s {end:8.2f}s "
              f"{end-t_a:7.2f}s {min(c_a,c_b):6.3f} {max(s_a,s_b):6.3f}")


if __name__ == "__main__":
    main()
