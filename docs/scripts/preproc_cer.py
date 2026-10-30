"""Hold the build fixed, vary the audio pre-processing, score against the reference.

docs/scripts/wer.py answers "which build hears better" on a fixed audio file.
This answers the other half: whether the way we hand the audio over costs
anything. Same reference text, same normalisation, same metrics -- transcripts
land in /tmp/wer_<variant>.txt so docs/scripts/bootstrap_cer.py can put a
confidence interval on any pair without changes.

    python docs/scripts/preproc_cer.py <reference.txt> <source-audio> <t0> <t1> \
        <build> <variant> [variant ...]

    python docs/scripts/preproc_cer.py Audiotest2/referenz/hart_780-900_REFERENZ.txt \
        "Audiotest2/Mona Podcast 323.m4a" 780 900 models/voxtral-mini-8bit \
        p0-ist p2-soxr g-6 g-12 g-18 g-24
"""
import pathlib
import sys
import time

import numpy as np
import soundfile as sf
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer, OVERLAP
from preproc_variants import build, SR, loudness
from noScribe.voxtral_engine import _Voxtral


def main():
    ref_path, src, t0, t1, build_path = sys.argv[1:6]
    t0, t1 = float(t0), float(t1)
    ids = sys.argv[6:]

    raw = open(ref_path, encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    ref_chars = "".join(ref)

    variants = build(src, t0, t1, ids)
    print(f"# {ref_path}: {len(ref)} words / {len(ref_chars)} chars, "
          f"{t1 - t0:.0f}s from {pathlib.Path(src).name}")
    print(f"# Build: {build_path}\n")
    print(f"{'variant':16s} {'peak':>6s} {'LUFS':>7s} {'WER':>7s} {'CER':>7s} "
          f"{'Sub':>4s} {'Del':>4s} {'Ins':>4s} {'W':>5s} {'Speed':>6s}  Overlap")

    vox = _Voxtral(build_path)
    dur = t1 - t0
    for vid in ids:
        a = variants[vid]
        mx.reset_peak_memory()
        t = time.time()
        text = vox.transcribe_array(a, "de", max_new_tokens=int(dur * 20) + 512)
        el = time.time() - t
        hyp = norm(text)
        err, sub, dele, ins = wer(ref, hyp)
        cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
        got = sum(1 for w in overlap_words if w in hyp) if overlap_words else 0
        print(f"{vid:16s} {np.abs(a).max():6.3f} {loudness(a):7.1f} "
              f"{err / len(ref) * 100:6.2f}% {cer * 100:6.2f}% {sub:4d} {dele:4d} "
              f"{ins:4d} {len(hyp):5d} {dur / el:5.2f}x  {got}/{len(overlap_words)}",
              flush=True)
        open(f"/tmp/wer_{vid}.txt", "w").write(text)
        sf.write(f"/tmp/audio_{vid}.wav", a, SR)


if __name__ == "__main__":
    main()
