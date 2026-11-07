"""Does the per-block normalisation of the log-Mel cost anything? On long streams.

HISTORICAL: since mlx-voxtral 0.0.6 `mel-ref` is the built-in path; both
variants are identical and the control assertion below therefore fails. For a
repeat, `stock` would have to be the 0.0.5 version of
`_process_audio_array_with_chunking`. Result: docs/voxtral-mel-clamp-floor.md §8.

Up to 0.0.5 mlx-voxtral computed the log-Mel per 30-s block and normalised
every block against its own maximum. Voxtral specifies one spectrogram over
the whole input that is split only afterwards (arXiv:2507.13264 §2.1) -- the
clamp sits at `log_max - 8`, so a block with its own maximum lands on a
different floor.

Two hand-corrected passages do not settle this: on `hart_780-900` both
versions are word-identical (block-level span only 0.12 log10), on
`zoom_9890-10190` the difference looks significant -- but that reference is
biased by 0.25 CER points in favour of the build whose draft it was corrected
from. The same trap as with the chunker (see
docs/voxtral-audio-preprocessing.md §5).

So many long streams: FLEURS recordings chained into roughly `SEC` seconds
each, transcript known. Both versions see **bit-identical audio**, so the
comparison is paired.

Setup and bootstrap come from fleurs_stream.py; here two feature paths are
compared instead of two window states.

    python docs/scripts/mel_stream.py [n_streams] [build]
"""
import json
import pathlib
import sys
import time

import numpy as np
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer
from fleurs_stream import streams, paired, SR
import fleurs_gain
from noScribe.voxtral_engine import _Voxtral
from mlx_voxtral.audio_processing import (  # noqa: E402
    log_mel_spectrogram, VoxtralFeatureExtractor, N_SAMPLES, N_MELS, N_FRAMES)

SEC = 300.0


class RefExtractor:
    """One spectrogram over the whole padded audio, split afterwards."""

    def __call__(self, audio, sampling_rate=SR, return_tensors="mlx", **kw):
        a = np.asarray(audio, dtype=np.float32)
        pad = (-len(a)) % N_SAMPLES
        if pad:
            a = np.pad(a, (0, pad))
        mel, _ = log_mel_spectrogram(mx.array(a))
        return {"input_features":
                mel.reshape(N_MELS, len(a) // N_SAMPLES, N_FRAMES).transpose(1, 0, 2)}


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    build = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "models" / "voxtral-mini-8bit")

    fleurs_gain.N = int(n_streams * SEC / 11) + 40   # ~11 s per recording
    st = streams(*fleurs_gain.load_clips(), n_streams, SEC)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} streams from FLEURS de_de, ~{SEC:.0f}s each, {total/60:.1f} min total")
    print(f"# Build: {pathlib.Path(build).name}, language 'de' pinned, greedy\n", flush=True)

    vox = _Voxtral(build)
    # The library path explicitly: since the percentile floor went in,
    # vox.proc no longer carries the stock extractor.
    stock, ref_ex = VoxtralFeatureExtractor(), RefExtractor()

    # Does the swap take effect at all? Otherwise the run measures the same
    # thing twice -- for an hour, and the result would look like a clean null
    # effect.
    probe = st[0][0]
    d = np.abs(np.array(ref_ex(probe)["input_features"])
               - np.array(stock(probe, sampling_rate=SR,
                                return_tensors="mlx")["input_features"]))
    assert d.max() > 0.1, f"Feature paths barely differ: {d.max()}"
    print(f"# Control: feature max|diff| on stream 0 = {d.max():.4f}\n")
    print(f"{'variant':10s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    store = {}
    for name, extractor in (("mel-ist", stock), ("mel-ref", ref_ex)):
        vox.proc.feature_extractor = extractor
        rows, t0 = [], time.time()
        for x, words in st:
            a = np.ascontiguousarray(x, dtype=np.float32)
            hyp = norm(vox.transcribe_array(
                a, "de", max_new_tokens=int(len(a) / SR * 20) + 512))
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        print(f"{name:10s} {rate(rows, 'w', 'ref_w'):6.2f}% "
              f"{rate(rows, 'c', 'ref_c'):6.2f}% {total/el:6.2f}x", flush=True)
        store[name] = rows
        json.dump(store, open("/tmp/mel_stream.json", "w"))
        mx.clear_cache()

    lo_w, hi_w = paired(store["mel-ist"], store["mel-ref"], "w", "ref_w")
    lo_c, hi_c = paired(store["mel-ist"], store["mel-ref"], "c", "ref_c")
    dW = rate(store["mel-ref"], "w", "ref_w") - rate(store["mel-ist"], "w", "ref_w")
    dC = rate(store["mel-ref"], "c", "ref_c") - rate(store["mel-ist"], "c", "ref_c")

    def star(lo, hi):
        return " *" if (lo > 0) == (hi > 0) else ""

    print("\nPaired difference mel-ref - mel-ist (95%; if it contains 0, not demonstrable):")
    print(f"  dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star(lo_w, hi_w)}"
          f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]{star(lo_c, hi_c)}")
    print("  (positive = mel-ref worse)")
    if len(st) < 8:
        # The bootstrap draws streams with replacement: with a handful the
        # interval shrinks to the scatter of a few points and sets a star that
        # means nothing. With a single stream it is even zero-width.
        print(f"  WARNING: only {len(st)} stream(s) -- the interval does not hold. "
              f"Use 18 to decide.")


if __name__ == "__main__":
    main()
