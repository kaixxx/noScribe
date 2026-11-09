"""Qwen3-ASR against the same yardsticks as the Voxtral builds.

`Qwen/Qwen3-ASR-1.7B-hf` is the Voxtral principle at half the size: an audio
encoder in front of a Qwen3-Omni language model. Unlike Parakeet it takes an
explicit language, and unlike Voxtral it takes a free-form context prompt --
both are measured here as separate arms.

Scored with the *identical* normalisation and edit distance as
docs/scripts/wer.py, so the numbers go straight next to the Voxtral table.

    python docs/scripts/qwen_asr_wer.py ref <reference.txt> <audio.wav> [arm]
    python docs/scripts/qwen_asr_wer.py fleurs <n_samples> [arm]

arm: "de" (forced German, the default), "auto" (language=None),
     "vocab:<term,term,...>" (forced German plus a context prompt, passed as a
     system message -- `apply_transcription_request`'s documented `prompt=`
     argument does not exist in this transformers build and is silently
     dropped), or "chunkN" (forced German, the audio cut into ~N-second
     windows at the quietest point near each boundary).
"""
import pathlib, sys, tempfile, time

import numpy as np
import soundfile as sf
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Reuse norm/wer/OVERLAP from wer.py verbatim -- the metric must not drift
# between scripts. Cut before main() and drop the Voxtral import.
_src = open(REPO / "docs/scripts/wer.py", encoding="utf-8").read()
_src = _src.split("def main()")[0].replace(
    "from noScribe.voxtral_engine import _Voxtral", "")
_ns = {"__file__": str(REPO / "docs/scripts/wer.py")}
exec(_src, _ns)
norm, wer, OVERLAP = _ns["norm"], _ns["wer"], _ns["OVERLAP"]

MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def parse_arm(arm):
    """arm string -> (language, prompt, chunk_sec, label)."""
    if arm == "auto":
        return None, None, 0, "qwen3-asr-1.7b (auto)"
    if arm.startswith("vocab:"):
        terms = arm.split(":", 1)[1]
        return "German", f"Vocabulary: {terms}.", 0, "qwen3-asr-1.7b (de+vocab)"
    if arm.startswith("chunk"):
        n = int(arm[5:])
        return "German", None, n, f"qwen3-asr-1.7b (de, {n}s chunks)"
    return "German", None, 0, "qwen3-asr-1.7b (de)"


def cut_points(audio, sr, chunk_sec):
    """Window boundaries snapped to the quietest 100 ms within +/-3 s."""
    bounds, pos = [0], 0
    while len(audio) - pos > chunk_sec * sr * 1.5:
        target = pos + chunk_sec * sr
        lo, hi = max(pos + sr, target - 3 * sr), min(len(audio), target + 3 * sr)
        win = int(0.1 * sr)
        seg = np.abs(audio[lo:hi])
        energy = np.convolve(seg, np.ones(win) / win, mode="valid")
        pos = lo + int(np.argmin(energy)) + win // 2
        bounds.append(pos)
    bounds.append(len(audio))
    return bounds


def load():
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    return proc, model


def _one(proc, model, audio, sr, language, prompt, dur):
    """The processor reads files, so arrays go through a temp wav."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
        sf.write(fh.name, audio, sr)
        if prompt:
            # `prompt=` is documented on apply_transcription_request but absent
            # from this build's signature, so it lands in **kwargs and is
            # dropped with a warning. The template concatenates every system
            # message, so the context goes in next to the language instead.
            conv = [{"role": "system",
                     "content": [{"type": "text",
                                  "text": f"{language}\n{prompt}"}]},
                    {"role": "user",
                     "content": [{"type": "audio", "audio": fh.name}]}]
            inputs = proc.apply_chat_template(conv, tokenize=True,
                                              add_generation_prompt=True,
                                              return_dict=True)
        else:
            kw = {"language": language} if language else {}
            inputs = proc.apply_transcription_request(audio=fh.name, **kw)
    inputs = inputs.to(model.device, model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(dur * 20) + 512,
                             do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return proc.decode(gen, return_format="transcription_only")[0]


def transcribe(proc, model, audio, sr, language, prompt, dur, chunk_sec=0):
    if not chunk_sec or len(audio) <= chunk_sec * sr * 1.5:
        return _one(proc, model, audio, sr, language, prompt, dur)
    bounds = cut_points(audio, sr, chunk_sec)
    parts = []
    for a, b in zip(bounds, bounds[1:]):
        seg = audio[a:b]
        parts.append(_one(proc, model, seg, sr, language, prompt,
                          len(seg) / sr))
    return " ".join(p.strip() for p in parts)


def score_reference(ref_path, wav_path, arm):
    language, prompt, chunk_sec, label = parse_arm(arm)
    raw = open(ref_path, encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    ref_chars = "".join(ref)
    audio, sr = sf.read(wav_path, dtype="float32")
    dur = len(audio) / sr

    proc, model = load()
    t0 = time.time()
    text = transcribe(proc, model, audio, sr, language, prompt, dur, chunk_sec)
    el = time.time() - t0

    hyp = norm(text)
    err, sub, dele, ins = wer(ref, hyp)
    cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
    got = sum(1 for w in overlap_words if w in hyp) if overlap_words else 0

    print(f"# {pathlib.Path(wav_path).name} on {DEVICE}: reference {len(ref)} "
          f"words / {len(ref_chars)} chars (+{len(overlap_words)} overlapping), "
          f"{dur:.0f}s")
    print(f"{'Build':30s} {'WER':>7s} {'CER':>7s} {'Sub':>5s} {'Del':>5s} "
          f"{'Ins':>5s} {'Speed':>8s}  Overlap")
    print(f"{label:30s} {err/len(ref)*100:6.2f}% {cer*100:6.2f}% {sub:5d} "
          f"{dele:5d} {ins:5d} {dur/el:7.2f}x  {got}/{len(overlap_words)}")
    stem = pathlib.Path(wav_path).stem
    out = pathlib.Path(f"/tmp/qwen_{stem}_{arm.split(':')[0]}.txt")
    out.write_text(text, encoding="utf-8")
    print(f"# transcript -> {out}")


def score_fleurs(n, arm):
    import io
    from datasets import load_dataset, Audio
    language, prompt, chunk_sec, label = parse_arm(arm)
    ds = load_dataset("google/fleurs", "de_de", split=f"test[:{n}]")
    # decode=False for the same reason as fleurs.py: datasets 5.x routes
    # decoding through torchcodec, whose bundled ffmpeg bindings do not find
    # Homebrew's libav*. The files are plain wav, so soundfile reads them.
    ds = ds.cast_column("audio", Audio(decode=False))

    proc, model = load()
    refw = errs = refc = cerr = 0
    secs = el = 0.0
    for row in ds:
        a = row["audio"]
        data = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.ascontiguousarray(audio)
        dur = len(audio) / sr
        secs += dur
        t0 = time.time()
        text = transcribe(proc, model, audio, sr, language, prompt, dur, chunk_sec)
        el += time.time() - t0
        ref, hyp = norm(row["transcription"]), norm(text)
        errs += wer(ref, hyp)[0]
        refw += len(ref)
        rc = "".join(ref)
        cerr += wer(rc, "".join(hyp))[0]
        refc += len(rc)
    print(f"# FLEURS de_de test[:{n}] on {DEVICE} -- {refw} reference words, "
          f"{secs:.0f}s audio")
    print(f"{'Build':30s} {'WER':>7s} {'CER':>7s} {'Speed':>8s}")
    print(f"{label:30s} {errs/refw*100:6.2f}% {cerr/refc*100:6.2f}% "
          f"{secs/el:7.2f}x")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ref":
        score_reference(sys.argv[2], sys.argv[3],
                        sys.argv[4] if len(sys.argv) > 4 else "de")
    elif mode == "fleurs":
        score_fleurs(int(sys.argv[2]),
                     sys.argv[3] if len(sys.argv) > 3 else "de")
    else:
        sys.exit(__doc__)
