"""Build the candidate pre-processing paths for one window of a source file.

Every variant below produces 16 kHz mono float32 of the SAME audio window, so
the hand-corrected reference text applies to all of them unchanged. The point
of separating this from the scoring script is that the variants are also useful
on their own -- for mel-domain comparisons that need no model run.

Variant ids:
  p0-ist          production path: PyAV -> pcm_s16le/16k/mono, read back /32768
  p1-headroom3    production path with -3 dB applied before the int16 step
  p2-soxr         decode to float, soxr VHQ to 16k, no integer stage at all
  p3-peak1        p2, peak-normalised to -1 dBFS
  p4-lufs23       p2, loudness-normalised to -23 LUFS (EBU R128 / BS.1770)
  p5-lufs16       p2, loudness-normalised to -16 LUFS
  g<N>            p2 attenuated by N dB (pure level ladder, no clipping)
  hot-clip        p2 driven to peak 1.24 (the hottest real file) and clipped
  hot-headroom    same drive, -3 dB instead of clipping -- nothing is clipped
"""
import pathlib
import numpy as np
import av
import soxr
import soundfile as sf

SR = 16000
MARGIN_SEC = 10.0        # decoded either side, so resampler edges fall outside


def _decode_native(path, t0=None, t1=None):
    """Mono float64 at the native rate, downmixed as (L+R)/2 like convert.py.

    Returns (samples, rate, offset_sec) with offset_sec == 0: the decode always
    starts at the top of the file and stops once it has enough.

    It deliberately does NOT seek. Seeking and anchoring the buffer on the first
    decoded frame's pts was tried and is wrong: AAC carries an encoder delay
    (2112 samples at 44.1 kHz), which the pts after a seek does not account for.
    On an .m4a that shifted the window by exactly that much and quietly changed
    every number measured through this path -- on an .mp4 of the same content it
    did not, so the bug hid until two runs of the same measurement disagreed.
    Reading from the top is slow on a multi-hour file and correct on every file.
    """
    c = av.open(str(path))
    st = c.streams.audio[0]
    layout = "stereo" if st.channels > 1 else "mono"
    res = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=st.rate)
    buf, have = [], 0
    want = None if t1 is None else int(st.rate * (t1 + MARGIN_SEC))
    keep_from = 0 if t0 is None else int(st.rate * max(0.0, t0 - 2 * MARGIN_SEC))
    for frame in c.decode(st):
        for fr in res.resample(frame):
            a = fr.to_ndarray()
            m = a.mean(axis=0) if a.shape[0] > 1 else a[0]
            # Everything before the window is counted but not kept: otherwise
            # a five-hour recording puts hours in memory to get two minutes.
            # The counter remains the time base.
            if have + len(m) > keep_from:
                buf.append(m[max(0, keep_from - have):].astype(np.float32))
            have += len(m)
        if want and have > want:
            break
    rate = st.rate
    c.close()
    x = np.concatenate(buf).astype(np.float64) if buf else np.zeros(0)
    return x, rate, keep_from / rate


def _window(x, rate, t0, t1, base=0.0, margin=True):
    """Fenster aus einem Puffer, der bei `base` Sekunden beginnt."""
    m = MARGIN_SEC if margin else 0
    lo = int(max(0, (t0 - m - base) * rate))
    hi = int(min(len(x), (t1 + m - base) * rate))
    return x[lo:hi], base + lo / rate


def production_path(src, cache_dir):
    """Run the real noScribe converter on the whole file, once, and cache it."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from noScribe.audio.convert import ToWav
    cache = pathlib.Path(cache_dir) / (pathlib.Path(src).stem + "_p0.wav")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with ToWav(pathlib.Path(src), cache, force=True) as c:
            while c.convert():
                pass
    y, sr = sf.read(cache, dtype="float32")
    assert sr == SR, sr
    return y


def loudness(x, sr=SR):
    import torch, torchaudio
    w = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0)
    return float(torchaudio.functional.loudness(w, sr))


def to_int16_and_back(x):
    """Exactly what pcm_s16le + libsndfile's float32 read do to the samples."""
    return (np.rint(np.asarray(x) * 32768).clip(-32768, 32767)
            .astype(np.int16).astype(np.float32) / 32768)


def build(src, t0, t1, ids, cache_dir="/tmp/preproc_cache"):
    """Return {id: float32 array at 16 kHz} for the window [t0, t1)."""
    out = {}
    need_native = any(i != "p0-ist" for i in ids)

    if "p0-ist" in ids or "p1-headroom3" in ids:
        full = production_path(src, cache_dir)
        seg = full[int(t0 * SR):int(t1 * SR)]
        if "p0-ist" in ids:
            out["p0-ist"] = seg.astype(np.float32)

    if need_native or "p1-headroom3" in ids:
        x, rate, base = _decode_native(src, t0, t1)
        w, w_t0 = _window(x, rate, t0, t1, base)
        y = soxr.resample(w, rate, SR, quality="VHQ")
        lo = int(round((t0 - w_t0) * SR))
        base = y[lo:lo + int((t1 - t0) * SR)].astype(np.float32)

    if "p1-headroom3" in ids:
        # -3 dB has to be applied BEFORE the integer stage to mean anything,
        # so this variant reproduces the production chain rather than reusing
        # its output: same resampler as p0 would be ideal, but swr is not
        # reachable with a gain in between, and section 1 of the write-up
        # shows the two resamplers differ only above 7 kHz.
        out["p1-headroom3"] = to_int16_and_back(base * (10 ** (-3 / 20)))

    if "p2-soxr" in ids:
        out["p2-soxr"] = base
    if "p3-peak1" in ids:
        out["p3-peak1"] = (base * (10 ** (-1 / 20) / max(1e-9, np.abs(base).max()))
                           ).astype(np.float32)
    for vid, target in (("p4-lufs23", -23.0), ("p5-lufs16", -16.0)):
        if vid in ids:
            g = 10 ** ((target - loudness(base)) / 20)
            out[vid] = (base * g).astype(np.float32)
    for i in ids:
        if i.startswith("g") and i[1:].lstrip("-").isdigit():
            out[i] = (base * 10 ** (float(i[1:]) / 20)).astype(np.float32)
        elif i.startswith("f-"):
            # f-<chain>: a named libavfilter chain from audio_filters, applied to
            # the faithful float rendering. This is how a leveller gets measured
            # on real material rather than on constructed pairs.
            from audio_filters import apply_filter, CHAINS
            out[i] = apply_filter(base, CHAINS[i[2:]])
    if "hot-clip" in ids or "hot-headroom" in ids:
        hot = base / max(1e-9, float(np.abs(base).max())) * 1.24
        if "hot-clip" in ids:
            out["hot-clip"] = to_int16_and_back(np.clip(hot, -1.0, 1.0))
        if "hot-headroom" in ids:
            out["hot-headroom"] = to_int16_and_back(hot * (10 ** (-3 / 20)))
    return out
