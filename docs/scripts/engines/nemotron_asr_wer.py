"""Nemotron 3.5 ASR against the same yardsticks as the Voxtral builds.

Scores `nvidia/nemotron-3.5-asr-streaming-0.6b` (MLX conversion
`mlx-community/nemotron-3.5-asr-streaming-0.6b`, run via mlx-audio's
`nemotron_asr` model) with the *identical* normalisation and edit distance as
docs/scripts/wer.py. Copied from docs/scripts/engines/parakeet_wer.py; only
model loading and transcription differ, plus two read-outs on the hypothesis
(comma density, English function words) that do not touch the metric.

    python docs/scripts/engines/nemotron_asr_wer.py ref <reference.txt> <audio.wav> [arm]
    python docs/scripts/engines/nemotron_asr_wer.py fleurs <n_samples> [arm]

arm = <language>[@<right_context>], e.g. `de-DE` (default), `auto`, `en-US`,
`de-DE@6`. The right context picks the streaming chunk: chunk = right + 1
encoder frames of 80 ms, so @13 = 1120 ms (the default, most look-ahead),
@6 = 560 ms, @3 = 320 ms, @0 = 80 ms -- the four this checkpoint lists.

Nemotron is a cache-aware streaming transducer: the language is a one-hot
prompt concatenated to every encoder frame, and mlx-audio's generate() runs
the streaming encoder (caches, no recompute), token-identical to the offline
chunked-limited pass. It emits punctuation and capitalisation; language tags
(`<de-DE>`) are special tokens and dropped by the tokenizer, not here.
"""
import os, pathlib, re, sys, time

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

MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
# Transcripts go to a temp directory outside the repository, as parakeet_wer.py's
# /tmp did: the private references may carry real names.
OUT = pathlib.Path(os.environ.get("NEMOTRON_ASR_OUT", "/tmp/nemotron_asr"))

# English function words and fillers that are NOT also German words -- the
# code-switching read-out from docs/other-asr-engines.md (Parakeet: und->and,
# wenn->when, es ist->it is, gut->good, ja->yeah). "was", "so", "man", "in",
# "die", "also", "okay" and the like are left out because they are German too.
ENGLISH = set("""and when it is good yeah yes the you of to that this what with
but not are have has if or my your they he she just well because then there
here we our be been were would could should can't don't it's i'm that's
right really very like know think""".split())


def parse_arm(arm):
    lang, _, right = (arm or "de-DE").partition("@")
    return lang, (int(right) if right else 13)


def load_model():
    from mlx_audio.stt import load
    return load(MODEL)


def transcribe(model, audio, sr, arm="de-DE"):
    """mlx-audio takes a 16 kHz mono array directly."""
    assert sr == 16000, sr
    lang, right = parse_arm(arm)
    return model.generate(mx.array(audio), language=lang,
                          att_context_size=[56, right])


def commas_per_100(text, n_words):
    return text.count(",") / max(1, n_words) * 100


def english_tokens(hyp):
    return [w for w in hyp if w in ENGLISH]


def score_reference(ref_path, wav_path, arm="de-DE"):
    raw = open(ref_path, encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    ref_chars = "".join(ref)
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr

    model = load_model()
    transcribe(model, audio[: sr * 5], sr, arm)  # warm-up, not timed
    mx.reset_peak_memory()
    t0 = time.time()
    result = transcribe(model, audio, sr, arm)
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
    label = f"nemotron-3.5-asr {arm}"
    print(f"{label:30s} {err/len(ref)*100:6.2f}% {cer*100:6.2f}% "
          f"{sub:5d} {dele:5d} {ins:5d} {dur/el:7.2f}x "
          f"{mx.get_peak_memory()/1e9:6.1f}G  {got}/{len(overlap_words)}")
    en = english_tokens(hyp)
    en_ref = english_tokens(ref)
    print(f"# {len(hyp)} hyp words, commas/100w {commas_per_100(text, len(hyp)):.2f} "
          f"(reference {commas_per_100(OVERLAP.sub(' ', raw), len(ref)):.2f})")
    print(f"# English function words: {len(en)} in hyp {en}; "
          f"{len(en_ref)} in reference {en_ref}")
    print(f"# {len(result.sentences)} sentences, {n_tok} timestamped tokens")
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{pathlib.Path(wav_path).stem}_{arm.replace('@', '_')}.txt"
    out.write_text(text, encoding="utf-8")
    words = OUT / f"{pathlib.Path(wav_path).stem}_{arm.replace('@', '_')}.tokens.tsv"
    words.write_text("".join(f"{t.start:.2f}\t{t.text}\n"
                             for s in result.sentences for t in s.tokens),
                     encoding="utf-8")
    print(f"# transcript -> {out}")


def score_fleurs(n, arm="de-DE"):
    import io
    from datasets import load_dataset, Audio
    ds = load_dataset("google/fleurs", "de_de", split=f"test[:{n}]")
    # decode=False for the same reason as fleurs.py: datasets 5.x routes
    # decoding through torchcodec, whose bundled ffmpeg bindings do not find
    # Homebrew's libav*. The files are plain wav, so soundfile reads them.
    ds = ds.cast_column("audio", Audio(decode=False))

    model = load_model()
    refw = hypw = errs = refc = cerr = commas = 0
    S = D = I = 0
    en = []
    secs = el = 0.0
    lines = []
    for row in ds:
        a = row["audio"]
        data = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio)
        secs += len(audio) / sr
        t0 = time.time()
        text = transcribe(model, audio, sr, arm).text
        el += time.time() - t0
        ref, hyp = norm(row["transcription"]), norm(text)
        e, s, d, i = wer(ref, hyp)
        errs += e; S += s; D += d; I += i
        refw += len(ref)
        hypw += len(hyp)
        commas += text.count(",")
        en += english_tokens(hyp)
        rc = "".join(ref)
        cerr += wer(rc, "".join(hyp))[0]
        refc += len(rc)
        lines.append(f"{row['id']}\t{text}")
    print(f"# FLEURS de_de test[:{n}] -- {refw} reference words, {secs:.0f}s audio")
    print(f"{'Build':30s} {'WER':>7s} {'CER':>7s} {'Speed':>8s}")
    print(f"{'nemotron-3.5-asr ' + arm:30s} {errs/refw*100:6.2f}% "
          f"{cerr/refc*100:6.2f}% {secs/el:7.2f}x")
    print(f"# sub {S} del {D} ins {I}; commas/100w {commas/max(1, hypw)*100:.2f}; "
          f"English function words {len(en)} {en}")
    OUT.mkdir(exist_ok=True)
    (OUT / f"fleurs{n}_{arm.replace('@', '_')}.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "ref":
        score_reference(sys.argv[2], sys.argv[3],
                        arm=sys.argv[4] if len(sys.argv) > 4 else "de-DE")
    elif sys.argv[1] == "fleurs":
        score_fleurs(int(sys.argv[2]),
                     arm=sys.argv[3] if len(sys.argv) > 3 else "de-DE")
    else:
        sys.exit(__doc__)
