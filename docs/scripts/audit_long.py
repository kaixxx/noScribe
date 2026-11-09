"""Streaming audit for long recordings, and a per-minute level profile.

audio_audit.py holds the whole file in memory, which a five-hour Zoom recording
will not tolerate. This one streams: overall peak / DC / clipping counts, plus a
per-minute loudness profile, because on unmastered material the interesting
question is not the average level but how far the quiet minutes sit below the
loud ones. That spread is what a single global gain cannot fix.

    python docs/scripts/audit_long.py <file> [minutes_per_bucket]
"""
import pathlib
import sys

import numpy as np
import av
import soxr
import torch
import torchaudio

SR = 16000


def stream_mono(path, chunk_sec=60):
    """Yield (t0, mono float64 at 16 kHz) minute by minute, downmixed (L+R)/2."""
    c = av.open(str(path))
    st = c.streams.audio[0]
    layout = "stereo" if st.channels > 1 else "mono"
    res = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=st.rate)
    buf, t0, need = [], 0.0, st.rate * chunk_sec
    for frame in c.decode(st):
        for fr in res.resample(frame):
            a = fr.to_ndarray()
            buf.append(a.mean(axis=0) if a.shape[0] > 1 else a[0])
        if sum(len(b) for b in buf) >= need:
            x = np.concatenate(buf).astype(np.float64)
            yield t0, x[:need], st.rate
            buf = [x[need:]]
            t0 += chunk_sec
    if buf:
        x = np.concatenate(buf).astype(np.float64)
        if len(x):
            yield t0, x, st.rate
    c.close()


def lufs(x, sr):
    if len(x) < sr * 0.5:
        return float("nan")
    w = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    try:
        return float(torchaudio.functional.loudness(w, sr))
    except Exception:
        return float("nan")


def main():
    path = sys.argv[1]
    bucket = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    peak = 0.0
    over = n = 0
    dc = 0.0
    prof = []
    for t0, x, rate in stream_mono(path, chunk_sec=int(60 * bucket)):
        peak = max(peak, float(np.abs(x).max()))
        over += int(np.sum(np.abs(x) > 1.0))
        dc += float(x.sum())
        n += len(x)
        prof.append((t0, lufs(x, rate), float(np.abs(x).max()),
                     20 * np.log10(float(np.sqrt((x ** 2).mean())) + 1e-30)))
    print(f"# {pathlib.Path(path).name}: {n / rate / 60:.1f} min")
    print(f"#   Peak {peak:.4f}   Samples>1.0 {over}   DC {dc / max(1, n):+.2e}")
    L = np.array([p[1] for p in prof], dtype=float)
    L = L[np.isfinite(L)]
    if len(L):
        q = np.percentile(L, [5, 25, 50, 75, 95])
        print(f"#   LUFS pro {bucket:g} min: Median {q[2]:.1f}, "
              f"p5 {q[0]:.1f}, p95 {q[4]:.1f}, Spanne p95-p5 {q[4]-q[0]:.1f} dB")
    print(f"\n{'t/min':>7s} {'LUFS':>7s} {'peak':>7s} {'RMS dB':>8s}")
    for t0, lo, pk, rms in prof:
        print(f"{t0/60:7.0f} {lo:7.1f} {pk:7.3f} {rms:8.1f}")


if __name__ == "__main__":
    main()
