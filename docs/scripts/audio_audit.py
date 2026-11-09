"""Audit what the real source files actually look like before we touch them.

Answers, per file: how hot is it, does it clip, does it clip only *after*
resampling, is there DC, how loud is it in LUFS. These are the numbers that
decide whether headroom or normalisation is worth anything.

    python docs/scripts/audio_audit.py [file ...]        # default: Audiotest/*
"""
import sys, glob, pathlib
import numpy as np
import av
import soxr
import torch
import torchaudio

REPO = pathlib.Path(__file__).resolve().parents[2]
SR = 16000


def decode_float(path, max_sec=None):
    """Decode to mono float64 at the file's native rate, without any int step.

    Mono is (L+R)/2 -- the same downmix noScribe/audio/convert.py performs, so
    the numbers below describe the signal our pipeline actually sees.
    """
    c = av.open(str(path))
    st = c.streams.audio[0]
    layout = "stereo" if st.channels > 1 else "mono"
    res = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=st.rate)
    buf = []
    for frame in c.decode(st):
        for fr in res.resample(frame):
            a = fr.to_ndarray()
            buf.append(a.mean(axis=0) if a.shape[0] > 1 else a[0])
        if max_sec and sum(len(b) for b in buf) > st.rate * max_sec:
            break
    rate = st.rate
    c.close()
    return np.concatenate(buf).astype(np.float64), rate


def true_peak(x, sr, oversample=4):
    """Inter-sample peak, the level a reconstruction filter would actually see."""
    up = soxr.resample(x, sr, sr * oversample, quality="VHQ")
    return float(np.abs(up).max())


def lufs(x, sr):
    w = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    try:
        return float(torchaudio.functional.loudness(w, sr))
    except Exception:
        return float("nan")


def audit(path):
    x, sr = decode_float(path)
    y = soxr.resample(x, sr, SR, quality="VHQ")          # ideal resample, float
    n = len(x)
    return dict(
        name=pathlib.Path(path).name,
        sr=sr,
        sec=n / sr,
        dc=float(x.mean()),
        rms_db=20 * np.log10(float(np.sqrt((x ** 2).mean())) + 1e-30),
        peak=float(np.abs(x).max()),
        tpeak=true_peak(x, sr),
        over=int(np.sum(np.abs(x) > 1.0)),
        peak16k=float(np.abs(y).max()),
        over16k=int(np.sum(np.abs(y) > 1.0)),
        lufs=lufs(x, sr),
        hf=10 * np.log10(hf_share(x, sr) + 1e-30),
    )


def hf_share(x, sr):
    """Energy above 8 kHz as a fraction of the total -- what 16 kHz discards."""
    if sr <= 16000:
        return 0.0
    n = 1 << 16
    acc = tot = 0.0
    fr = np.fft.rfftfreq(n, 1 / sr)
    m = fr >= 8000
    for h in range(0, min(len(x) - n, sr * 300), n):
        S = np.abs(np.fft.rfft(x[h:h + n] * np.hanning(n))) ** 2
        acc += S[m].sum(); tot += S.sum()
    return acc / tot if tot else 0.0


def main(paths):
    print(f"{'file':16s} {'sr':>6s} {'sec':>7s} {'DC':>9s} {'RMS':>7s} "
          f"{'peak':>6s} {'true':>6s} {'>1.0':>7s} {'pk@16k':>7s} {'>1@16k':>7s} "
          f"{'LUFS':>7s} {'>8kHz':>7s}")
    for p in paths:
        try:
            r = audit(p)
        except Exception as exc:
            print(f"{pathlib.Path(p).name[:16]:16s} -- {exc}")
            continue
        print(f"{r['name'][:16]:16s} {r['sr']:6d} {r['sec']:7.1f} {r['dc']:+9.2e} "
              f"{r['rms_db']:7.1f} {r['peak']:6.3f} {r['tpeak']:6.3f} {r['over']:7d} "
              f"{r['peak16k']:7.3f} {r['over16k']:7d} {r['lufs']:7.1f} {r['hf']:7.1f}")


if __name__ == "__main__":
    args = sys.argv[1:] or (sorted(glob.glob(str(REPO / "Audiotest" / "*.m4a")))
                            + sorted(glob.glob(str(REPO / "Audiotest2" / "*.m4a")))
                            + [str(REPO / "Audiotest2" / "referenz" / "hart_780-900.wav")])
    main(args)
