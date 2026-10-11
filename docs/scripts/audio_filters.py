"""Run a libavfilter chain over a 16 kHz mono float array.

Everything noScribe already depends on ships these filters, so a leveller or a
denoiser in front of the encoder costs no new dependency -- only the question of
whether it helps, which the scripts next to this one measure.

    y = apply_filter(x, "speechnorm=e=12.5:r=0.0001:l=1")
"""
from fractions import Fraction

import numpy as np
import av

SR = 16000

# Named chains, so the measurement scripts and the write-up refer to the same
# settings. Defaults are the filters' own unless a reason is given.
CHAINS = {
    "raw":         None,
    # speechnorm's r is the per-sample rise rate; the default 0.0001 needs many
    # seconds to lift a quiet passage, which is exactly the case under test, so
    # a faster rise is used here.
    "speechnorm":  "speechnorm=e=12.5:r=0.002:l=1",
    "dynaudnorm":  "dynaudnorm=f=150:g=9:p=0.9:m=20",
    # loudnorm runs its internals at 192 kHz, so it has to be brought back.
    "loudnorm16":  "loudnorm=I=-16:TP=-1.5:LRA=7,aresample=16000",
    "acompressor": "acompressor=threshold=0.05:ratio=4:attack=20:release=250:makeup=2",
    "highpass60":  "highpass=f=60:poles=2",
    "afftdn":      "afftdn=nr=12:nf=-40",
    "arnndn":      None,      # needs a model file; left unset on purpose
}


def apply_filter(x, chain, sr=SR):
    """Return the filtered signal as float32, same sample rate."""
    if not chain:
        return np.ascontiguousarray(x, dtype=np.float32)
    graph = av.filter.Graph()
    src = graph.add_abuffer(format="fltp", sample_rate=sr, layout="mono",
                            time_base=Fraction(1, sr))
    prev = src
    for part in chain.split(","):
        name, _, args = part.partition("=")
        node = graph.add(name.strip(), args.strip() or None)
        prev.link_to(node)
        prev = node
    sink = graph.add("abuffersink")
    prev.link_to(sink)
    graph.configure()

    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(np.asarray(x, dtype=np.float32)[np.newaxis, :]),
        format="fltp", layout="mono")
    frame.sample_rate = sr
    frame.time_base = Fraction(1, sr)
    frame.pts = 0
    graph.push(frame)
    graph.push(None)

    out = []
    while True:
        try:
            out.append(graph.pull().to_ndarray().ravel())
        except (av.error.EOFError, av.error.BlockingIOError):
            break
    return (np.concatenate(out).astype(np.float32) if out
            else np.zeros(0, dtype=np.float32))


if __name__ == "__main__":
    # Smoke test: a two-level signal shows what each chain does to the quiet part.
    t = np.arange(SR * 4) / SR
    loud = 0.5 * np.sin(2 * np.pi * 300 * t[:SR * 2])
    quiet = 0.5 * 10 ** (-25 / 20) * np.sin(2 * np.pi * 300 * t[:SR * 2])
    x = np.concatenate([loud, quiet]).astype(np.float32)
    print(f"{'chain':14s} {'len':>7s} {'laut':>8s} {'leise':>8s} {'Abstand':>9s}")
    for name, chain in CHAINS.items():
        if name == "arnndn":
            continue
        y = apply_filter(x, chain)
        if len(y) < len(x) // 2:
            print(f"{name:14s} {len(y):7d}  (leer)")
            continue
        a = np.abs(y[:SR * 2]).max(); b = np.abs(y[SR * 2:]).max()
        print(f"{name:14s} {len(y):7d} {20*np.log10(a+1e-12):8.1f} "
              f"{20*np.log10(b+1e-12):8.1f} {20*np.log10(a/(b+1e-12)):9.1f} dB")
