"""FLEURS German: the external yardstick.

FLEURS is the benchmark the Voxtral technical report uses (arXiv:2507.13264,
German WER 3.38 for Small / 3.54 for Mini), so numbers measured here can be
held against a published figure instead of only against each other.

Caveat worth keeping in mind: FLEURS is read-aloud, clean, single-speaker
audio. Published work finds quantisation damage several times larger on hard
material than on clean test sets, so a FLEURS delta UNDERSTATES what a build
costs on real interview audio -- which is why the hand-corrected podcast
reference is measured alongside it.

    python docs/scripts/fleurs.py <n_samples> <build> [build ...]
"""
import io, re, sys, time, unicodedata
import numpy as np

import pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]  # docs/scripts/x.py -> repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'docs' / 'scripts'))
from wer import norm, wer

N = int(sys.argv[1])
BUILDS = sys.argv[2:]

import soundfile as sf
from datasets import load_dataset, Audio
print(f"# FLEURS de_de test, first {N} recordings", flush=True)
# decode=False: datasets 5.x routes decoding through torchcodec, whose bundled
# ffmpeg bindings do not find Homebrew's libav*. The files are plain wav, so
# soundfile reads them directly and the dependency disappears.
ds = load_dataset("google/fleurs", "de_de", split=f"test[:{N}]")
ds = ds.cast_column("audio", Audio(decode=False))

clips, refs = [], []
for row in ds:
    a = row["audio"]
    data = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"unexpected sample rate {sr}")
    clips.append(np.ascontiguousarray(wav))
    refs.append(row["transcription"])

total_sec = sum(len(c) for c in clips) / 16000
print(f"# {len(clips)} recordings, {total_sec/60:.1f} min of audio\n")
print(f"{'Build':30s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")

import mlx.core as mx
from noScribe.voxtral_engine import _Voxtral



def transcriber(path):
    """A build path -> a callable(wav)->text. "whisper:<name>" selects
    faster-whisper with noScribe's own settings, so both engines can be scored
    on the same clips with the same metric."""
    if path.startswith("whisper:"):
        from faster_whisper import WhisperModel
        m = WhisperModel(f"models/{path.split(':', 1)[1]}", device="cpu",
                         compute_type="int8")

        def run(wav):
            segs, _ = m.transcribe(wav, language="de", beam_size=5)
            return " ".join(s.text.strip() for s in segs)
        return m, run
    vox = _Voxtral(path)
    return vox, lambda wav: vox.transcribe_array(
        wav, "de", max_new_tokens=int(len(wav) / 16000 * 20) + 512)


for path in BUILDS:
    handle, run = transcriber(path)
    errs = refw = cerr = refc = 0
    t0 = time.time()
    for wav, ref_text in zip(clips, refs):
        hyp = run(wav)
        r, h = norm(ref_text), norm(hyp)
        errs += wer(r, h)[0]; refw += len(r)
        rc, hc = "".join(r), "".join(h)
        cerr += wer(rc, hc)[0]; refc += len(rc)
    el = time.time() - t0
    print(f"{path.split('/')[-1]:30s} {errs/refw*100:6.2f}% {cerr/refc*100:6.2f}% "
          f"{total_sec/el:6.2f}x", flush=True)
    del handle
    mx.clear_cache()
