"""transcribe.cpp's Voxtral against the same yardsticks as our MLX build.

[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) is a GGML
speech-to-text library that carries Voxtral alongside fifteen other families,
with Metal, CUDA, Vulkan and HIP backends. That makes it the only route we
know to the model this engine ships on hardware MLX cannot reach, so the
question is narrow and empirical: does its Voxtral transcribe our German
references as well as `voxtral-mini-8bit` does?

The reason to doubt it is [issue #82](https://github.com/handy-computer/transcribe.cpp/issues/82):
Voxtral's Tekken tokenizer is not implemented there and the loader falls back
to qwen2 pretokenization, with the reporter observing German word-level
garbles ("Publikum" -> "Pubikom"). When we ported Voxtral we had to verify the
prompt token stream against `mistral_common` exactly before the quality
numbers held, so an approximate tokenizer is the first thing to measure, not
the last.

Scored with the *identical* normalisation and edit distance as
docs/scripts/wer.py, so the numbers sit next to the tables in
docs/voxtral-quantisation.md without a metric caveat. The engine is a CLI
binary rather than a Python library, so it is driven through subprocess and
the audio must already be 16 kHz mono WAV (both references are).

    python docs/scripts/engines/transcribe_cpp_wer.py \\
        <transcribe-cli> <model.gguf> <reference.txt> <audio.wav> [--lang de]
"""
import pathlib
import re
import subprocess
import sys
import tempfile
import time

import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Reuse norm/wer/OVERLAP from wer.py verbatim -- the metric must not drift
# between scripts. Same excerpt trick as parakeet_wer.py: cut the file before
# main() and drop the Voxtral import, so scoring does not pull in MLX.
_src = open(REPO / "docs/scripts/wer.py", encoding="utf-8").read()
_src = _src.split("def main()")[0].replace(
    "from noScribe.voxtral_engine import _Voxtral", "")
_ns = {"__file__": str(REPO / "docs/scripts/wer.py")}
exec(_src, _ns)
norm, wer, OVERLAP = _ns["norm"], _ns["wer"], _ns["OVERLAP"]


def transcribe(cli, gguf, wav, lang=None):
    """Returns (text, seconds).

    The transcript is collected through `-o`, not from stdout. `-q` silences
    the *library* log but the CLI still prints its own banner (audio, model,
    backend) and a `realtime:` footer around a `text:` line — scoring stdout
    charges all of that as insertions (76 of them on a 422-word passage, which
    is what this function looked like on its first run).
    """
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as fh:
        out_path = fh.name
    cmd = [str(cli), "-m", str(gguf), "-q", "-o", out_path]
    if lang:
        cmd += ["--language", lang]
    cmd.append(str(wav))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    el = time.time() - t0
    if r.returncode != 0:
        sys.exit(f"transcribe-cli failed ({r.returncode}):\n{r.stderr[-2000:]}")
    text = pathlib.Path(out_path).read_text(encoding="utf-8").strip()
    pathlib.Path(out_path).unlink(missing_ok=True)
    if not text:
        sys.exit("transcribe-cli produced an empty transcript")
    return text, el


def score(cli, gguf, ref_path, wav_path, lang=None):
    raw = open(ref_path, encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    ref_chars = "".join(ref)
    info = sf.info(str(wav_path))
    if info.samplerate != 16000 or info.channels != 1:
        sys.exit(f"{wav_path}: transcribe.cpp needs 16 kHz mono, got "
                 f"{info.samplerate} Hz / {info.channels} ch")

    text, el = transcribe(cli, gguf, wav_path, lang)
    hyp = norm(text)
    err, sub, dele, ins = wer(ref, hyp)
    cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
    got = sum(1 for w in overlap_words if w in hyp) if overlap_words else 0

    print(f"# {pathlib.Path(wav_path).name}: reference {len(ref)} words / "
          f"{len(ref_chars)} chars (+{len(overlap_words)} overlapping), "
          f"{info.duration:.0f}s")
    print(f"{'Build':34s} {'WER':>7s} {'CER':>7s} {'Sub':>5s} {'Del':>5s} "
          f"{'Ins':>5s} {'Speed':>8s}  Overlap")
    label = f"transcribe.cpp {pathlib.Path(gguf).stem.split('-')[-1]}"
    if lang:
        label += f" (--language {lang})"
    print(f"{label:34s} {err/len(ref)*100:6.2f}% {cer*100:6.2f}% "
          f"{sub:5d} {dele:5d} {ins:5d} {info.duration/el:7.2f}x "
          f"  {got}/{len(overlap_words)}")

    out = pathlib.Path("/tmp") / f"transcribecpp_{pathlib.Path(wav_path).stem}.txt"
    out.write_text(text, encoding="utf-8")
    print(f"# transcript -> {out}")
    return hyp, ref


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lang = None
    for a in sys.argv[1:]:
        if a.startswith("--lang"):
            lang = a.split("=", 1)[1] if "=" in a else None
    if "--lang" in sys.argv:
        lang = sys.argv[sys.argv.index("--lang") + 1]
    if len(args) < 4:
        sys.exit(__doc__.split("    python")[1].strip())
    score(args[0], args[1], args[2], args[3], lang)


if __name__ == "__main__":
    main()
