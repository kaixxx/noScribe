"""Find the passage in a long recording that is worth correcting.

A hand-corrected reference costs real work, so the passage should be able to
answer the question it is built for. For the leveller question in
docs/voxtral-audio-preprocessing.md that means: a lot of speech, and within
it a large gap between loud and quiet spots. That is exactly what is searched
for here -- not the "nicest" spot.

Windows of fixed length are rated by

  speech share   share of the seconds above the estimated noise floor
  spread         p90 - p10 of the per-second level over the speech seconds

The spread as a percentile distance, not as max-min: a single door slam must
not decide the selection.

The converted WAV is cached under /tmp/preproc_cache.

    python docs/scripts/find_passage.py <file> [length_s] [count]
"""
import pathlib
import sys

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
SR = 16000
CACHE = pathlib.Path("/tmp/preproc_cache")


def production_wav(src):
    """The file through the real converter, once, and cached."""
    from noScribe.audio.convert import ToWav
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / (pathlib.Path(src).stem.replace(" ", "_") + "_p0.wav")
    if not out.exists():
        with ToWav(pathlib.Path(src), out, force=True) as c:
            while c.convert():
                pass
    return out


def per_second_db(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, sr
    n = len(x) // SR
    r = x[:n * SR].reshape(n, SR)
    return 20 * np.log10(np.sqrt((r.astype(np.float64) ** 2).mean(axis=1)) + 1e-12)


def main():
    src = sys.argv[1]
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    topn = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    wav = production_wav(src)
    db = per_second_db(wav)
    # Estimate the noise floor from the lowest tenth, speech 12 dB above it.
    floor = np.percentile(db, 10)
    thr = floor + 12
    speech = db > thr
    print(f"# {pathlib.Path(src).name}: {len(db)/60:.1f} min, "
          f"noise floor ~{floor:.1f} dB, speech threshold {thr:.1f} dB, "
          f"speech share overall {speech.mean()*100:.0f}%")

    rows = []
    for t in range(0, len(db) - win, 10):
        seg, m = db[t:t + win], speech[t:t + win]
        if m.mean() < 0.7 or m.sum() < 30:
            continue
        s = seg[m]
        rows.append((float(np.percentile(s, 90) - np.percentile(s, 10)), t,
                     float(m.mean()), float(np.percentile(s, 90)),
                     float(np.percentile(s, 10))))
    rows.sort(reverse=True)

    # Candidates at least one window length apart, so the list does not show
    # the same spot ten times.
    picked = []
    for r in rows:
        if all(abs(r[1] - p[1]) >= win for p in picked):
            picked.append(r)
        if len(picked) >= topn:
            break

    print(f"\n{'from':>8s} {'to':>8s} {'spread':>10s} {'speech':>8s} "
          f"{'loud p90':>9s} {'quiet p10':>10s}")
    for spread, t, frac, hi, lo in picked:
        print(f"{t:8d} {t+win:8d} {spread:9.1f} dB {frac*100:7.0f}% "
              f"{hi:8.1f} dB {lo:9.1f} dB")

    if rows:
        allspread = np.array([r[0] for r in rows])
        print(f"\n# Spread over all {len(rows)} candidate windows: "
              f"median {np.median(allspread):.1f} dB, p90 {np.percentile(allspread,90):.1f} dB, "
              f"max {allspread.max():.1f} dB")


if __name__ == "__main__":
    main()
