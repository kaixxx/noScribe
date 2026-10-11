"""Measure the same speech from different sources against one reference.

When a recording exists in a raw and in a processed version, that is a natural
experiment: same words, different pre-processing, one hand-corrected reference
for both. Only the time axes do not line up -- a leveller such as a commercial
mastering service cuts silences, so the passage sits somewhere else in each
version. Hence every arm is given its own window here.

An arm is `label=file@from:to` and optionally `+chain`, where the chain is a
name from audio_filters.CHAINS:

    raw=recording.m4a@737:858.6
    raw_dyn=recording.m4a@737:858.6+dynaudnorm
    levelled=recording-proc.m4a@725.65:845.65

The window is found with docs/scripts/locate_passage.py. Transcripts land in
/tmp/wer_<label>.txt for bootstrap_cer.py and adjudicate.py.

    python docs/scripts/pair_cer.py <reference.txt> <build> <arm> [arm ...]
"""
import pathlib
import sys
import time

import numpy as np
import soundfile as sf
import soxr

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer, OVERLAP
from audio_filters import apply_filter, CHAINS
from audio_audit import decode_float

SR = 16000


def parse(spec):
    label, _, rest = spec.partition("=")
    rest, _, chain = rest.partition("+")
    path, _, window = rest.partition("@")
    t0, _, t1 = window.partition(":")
    return label, path, float(t0), float(t1), (chain or None)


def render(path, t0, t1, chain, cache):
    """Window as 16 kHz mono float32, modelled on the real production path:
    (L+R)/2, then to 16 kHz. The resampler here is soxr instead of swr --
    measured, that makes no difference (docs/voxtral-audio-preprocessing.md
    §2), and it keeps the window boundaries exact."""
    if path not in cache:
        x, sr = decode_float(path)
        cache[path] = soxr.resample(x, sr, SR, quality="VHQ").astype(np.float32)
    y = cache[path][int(t0 * SR):int(t1 * SR)]
    return np.ascontiguousarray(apply_filter(y, CHAINS[chain]) if chain else y,
                                dtype=np.float32)


def main():
    ref_path, build = sys.argv[1], sys.argv[2]
    specs = [parse(s) for s in sys.argv[3:]]

    raw = open(ref_path, encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    ref_chars = "".join(ref)
    print(f"# {ref_path}: {len(ref)} words / {len(ref_chars)} chars")
    print(f"# Build: {build}\n")
    print(f"{'Arm':16s} {'sec':>6s} {'peak':>6s} {'LUFS':>7s} {'WER':>7s} {'CER':>7s} "
          f"{'Sub':>4s} {'Del':>4s} {'Ins':>4s} {'W':>5s}  Overlap")

    from noScribe.voxtral_engine import _Voxtral
    from preproc_variants import loudness
    vox = _Voxtral(build)
    cache = {}
    for label, path, t0, t1, chain in specs:
        a = render(path, t0, t1, chain, cache)
        dur = len(a) / SR
        text = vox.transcribe_array(a, "de", max_new_tokens=int(dur * 20) + 512)
        hyp = norm(text)
        err, sub, dele, ins = wer(ref, hyp)
        cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
        got = sum(1 for w in overlap_words if w in hyp) if overlap_words else 0
        print(f"{label:16s} {dur:6.1f} {np.abs(a).max():6.3f} {loudness(a):7.1f} "
              f"{err/len(ref)*100:6.2f}% {cer*100:6.2f}% {sub:4d} {dele:4d} {ins:4d} "
              f"{len(hyp):5d}  {got}/{len(overlap_words)}", flush=True)
        open(f"/tmp/wer_{label}.txt", "w").write(text)


if __name__ == "__main__":
    main()
