"""Parakeet-TDT against the same two yardsticks as the Voxtral builds.

Scores `nvidia/parakeet-tdt-0.6b-v3` (via parakeet-mlx) with the *identical*
normalisation and edit distance as docs/scripts/wer.py, so the numbers can be
put next to the Voxtral table in docs/voxtral-quantisation.md without a
metric caveat.

    python docs/scripts/parakeet_wer.py ref <reference.txt> <audio.wav>
    python docs/scripts/parakeet_wer.py fleurs <n_samples>

Parakeet is a transducer: no language argument, no prompt, no max_new_tokens.
It emits punctuation and capitalisation itself, and the normalisation strips
both -- same as for Voxtral.
"""
import pathlib, re, sys, tempfile, time

import mlx.core as mx
import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Reuse norm/wer/OVERLAP from wer.py verbatim -- the metric must not drift
# between scripts. Cut the file before main() and drop the Voxtral import, so
# scoring Parakeet does not pull in the MLX Voxtral stack.
_src = open(REPO / "docs/scripts/wer.py", encoding="utf-8").read()
_src = _src.split("def main()")[0].replace(
    "from noScribe.voxtral_engine import _Voxtral", "")
_ns = {"__file__": str(REPO / "docs/scripts/wer.py")}
exec(_src, _ns)
norm, wer, OVERLAP = _ns["norm"], _ns["wer"], _ns["OVERLAP"]

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def load_model():
    from parakeet_mlx import from_pretrained
    return from_pretrained(MODEL)


def transcribe(model, audio, sr, beam=0):
    """parakeet-mlx reads from a file, so arrays go through a temp wav."""
    kw = {}
    if beam:
        from parakeet_mlx import Beam, DecodingConfig
        kw["decoding_config"] = DecodingConfig(decoding=Beam(beam_size=beam))
    with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
        sf.write(fh.name, audio, sr)
        return model.transcribe(fh.name, **kw)


def score_reference(ref_path, wav_path, beam=0):
    raw = open(ref_path, encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    ref_chars = "".join(ref)
    audio, sr = sf.read(wav_path, dtype="float32")
    dur = len(audio) / sr

    model = load_model()
    mx.reset_peak_memory()
    t0 = time.time()
    result = transcribe(model, audio, sr, beam)
    el = time.time() - t0

    text = result.text
    hyp = norm(text)
    err, sub, dele, ins = wer(ref, hyp)
    cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
    got = sum(1 for w in overlap_words if w in hyp) if overlap_words else 0

    n_tok = sum(len(s.tokens) for s in result.sentences)
    print(f"# {pathlib.Path(wav_path).name}: reference {len(ref)} words / "
          f"{len(ref_chars)} chars (+{len(overlap_words)} overlapping), {dur:.0f}s")
    print(f"{'Build':30s} {'WER':>7s} {'CER':>7s} {'Sub':>5s} {'Del':>5s} "
          f"{'Ins':>5s} {'Speed':>8s} {'Peak':>7s}  Overlap")
    label = f"parakeet-tdt-0.6b-v3{'-beam%d' % beam if beam else ''}"
    print(f"{label:30s} {err/len(ref)*100:6.2f}% {cer*100:6.2f}% "
          f"{sub:5d} {dele:5d} {ins:5d} {dur/el:7.2f}x "
          f"{mx.get_peak_memory()/1e9:6.1f}G  {got}/{len(overlap_words)}")
    print(f"# {len(result.sentences)} sentences, {n_tok} timestamped tokens")
    out = pathlib.Path(f"/tmp/parakeet_{pathlib.Path(wav_path).stem}{'_beam' if beam else ''}.txt")
    out.write_text(text, encoding="utf-8")
    print(f"# transcript -> {out}")


def score_fleurs(n):
    import io
    from datasets import load_dataset, Audio
    ds = load_dataset("google/fleurs", "de_de", split=f"test[:{n}]")
    # decode=False for the same reason as fleurs.py: datasets 5.x routes
    # decoding through torchcodec, whose bundled ffmpeg bindings do not find
    # Homebrew's libav*. The files are plain wav, so soundfile reads them.
    ds = ds.cast_column("audio", Audio(decode=False))

    model = load_model()
    refw = hypw = errs = refc = cerr = 0
    secs = el = 0.0
    for row in ds:
        a = row["audio"]
        data = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio)
        secs += len(audio) / sr
        t0 = time.time()
        text = transcribe(model, audio, sr).text
        el += time.time() - t0
        ref, hyp = norm(row["transcription"]), norm(text)
        errs += wer(ref, hyp)[0]
        refw += len(ref)
        hypw += len(hyp)
        rc = "".join(ref)
        cerr += wer(rc, "".join(hyp))[0]
        refc += len(rc)
    print(f"# FLEURS de_de test[:{n}] -- {refw} reference words, {secs:.0f}s audio")
    print(f"{'Build':30s} {'WER':>7s} {'CER':>7s} {'Speed':>8s}")
    print(f"{'parakeet-tdt-0.6b-v3':30s} {errs/refw*100:6.2f}% "
          f"{cerr/refc*100:6.2f}% {secs/el:7.2f}x")


if __name__ == "__main__":
    if sys.argv[1] == "ref":
        score_reference(sys.argv[2], sys.argv[3],
                        beam=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif sys.argv[1] == "fleurs":
        score_fleurs(int(sys.argv[2]))
    else:
        sys.exit(__doc__)
