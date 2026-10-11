"""VibeVoice-ASR against the same yardsticks as the Voxtral builds.

`microsoft/VibeVoice-ASR-HF` is not another engine behind the same seam: it does
ASR, diarization and timestamping in one pass and emits speaker-attributed
segments directly, i.e. noScribe's whole pipeline in one model. Scored here with
the *identical* normalisation and edit distance as docs/scripts/wer.py, plus the
things only this model can be asked -- how many speakers it found, and whether
it punctuates.

    python docs/scripts/vibevoice_wer.py ref <reference.txt> <audio.wav> [arm]
    python docs/scripts/vibevoice_wer.py fleurs <n_samples> [arm]

arm: "plain" (default) or "vocab:<term,term,...>" (context prompt).
"""
import json, pathlib, re, sys, tempfile, time

import numpy as np
import soundfile as sf
import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Reuse norm/wer/OVERLAP from wer.py verbatim -- the metric must not drift.
_src = open(REPO / "docs/scripts/wer.py", encoding="utf-8").read()
_src = _src.split("def main()")[0].replace(
    "from noScribe.voxtral_engine import _Voxtral", "")
_ns = {"__file__": str(REPO / "docs/scripts/wer.py")}
exec(_src, _ns)
norm, wer, OVERLAP = _ns["norm"], _ns["wer"], _ns["OVERLAP"]

TAG = re.compile(r"\[[^\]]{0,40}\]")
MODEL = "microsoft/VibeVoice-ASR-HF"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    # bf16 is not optional here: fp32 would be ~33 GB of weights.
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    return proc, model


def transcribe(proc, model, audio, sr, prompt, dur):
    """Returns (plain text, parsed segments)."""
    kw = {"prompt": prompt} if prompt else {}
    with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
        sf.write(fh.name, audio, sr)
        inputs = proc.apply_transcription_request(audio=fh.name, **kw)
    inputs = inputs.to(model.device, model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(dur * 30) + 1024,
                             do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    text = proc.decode(gen, return_format="transcription_only")[0]
    # The model annotates non-speech as bracketed tags -- "[Silence]",
    # "[Human Sounds]". They are metadata, not transcription, but `norm()`
    # would score them as inserted words, so they come out before scoring.
    # This is output cleanup, not a change to the metric.
    text = TAG.sub(" ", text).strip()
    try:
        parsed = proc.decode(gen, return_format="parsed")[0]
    except Exception as exc:                      # malformed structured output
        parsed = [{"_parse_error": str(exc)}]
    return text, parsed


def describe_speakers(parsed, dur):
    """What only this model can be asked: who, and for how long."""
    if not parsed or "_parse_error" in parsed[0]:
        return "parse failed"
    spk = {}
    for seg in parsed:
        try:
            spk[seg["Speaker"]] = spk.get(seg["Speaker"], 0.0) + (
                float(seg["End"]) - float(seg["Start"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not spk:
        return f"{len(parsed)} segments, no usable speaker fields"
    share = ", ".join(f"S{k} {v/max(dur, 1e-9)*100:.0f}%"
                      for k, v in sorted(spk.items()))
    return f"{len(spk)} speakers over {len(parsed)} segments ({share})"


def score_reference(ref_path, wav_path, arm):
    prompt = arm.split(":", 1)[1] if arm.startswith("vocab:") else None
    label = "vibevoice-asr (+vocab)" if prompt else "vibevoice-asr"
    raw = open(ref_path, encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    ref_chars = "".join(ref)
    audio, sr = sf.read(wav_path, dtype="float32")
    dur = len(audio) / sr

    proc, model = load()
    t0 = time.time()
    text, parsed = transcribe(proc, model, audio, sr, prompt, dur)
    el = time.time() - t0

    hyp = norm(text)
    err, sub, dele, ins = wer(ref, hyp)
    cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
    got = sum(1 for w in overlap_words if w in hyp) if overlap_words else 0
    commas = text.count(",") / max(1, len(text.split())) * 100

    print(f"# {pathlib.Path(wav_path).name} on {DEVICE}: reference {len(ref)} "
          f"words / {len(ref_chars)} chars (+{len(overlap_words)} overlapping), "
          f"{dur:.0f}s")
    print(f"{'Build':26s} {'WER':>7s} {'CER':>7s} {'Sub':>5s} {'Del':>5s} "
          f"{'Ins':>5s} {'Speed':>7s} {'Kommas':>7s}  Overlap")
    print(f"{label:26s} {err/len(ref)*100:6.2f}% {cer*100:6.2f}% {sub:5d} "
          f"{dele:5d} {ins:5d} {dur/el:6.2f}x {commas:6.2f}  "
          f"{got}/{len(overlap_words)}")
    print(f"# diarization: {describe_speakers(parsed, dur)}")
    stem = pathlib.Path(wav_path).stem
    pathlib.Path(f"/tmp/vibevoice_{stem}.txt").write_text(text, encoding="utf-8")
    pathlib.Path(f"/tmp/vibevoice_{stem}.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"# transcript -> /tmp/vibevoice_{stem}.txt (+ .json)")


def score_fleurs(n, arm):
    import io
    from datasets import load_dataset, Audio
    prompt = arm.split(":", 1)[1] if arm.startswith("vocab:") else None
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
        text, _ = transcribe(proc, model, audio, sr, prompt, dur)
        el += time.time() - t0
        ref, hyp = norm(row["transcription"]), norm(text)
        errs += wer(ref, hyp)[0]
        refw += len(ref)
        rc = "".join(ref)
        cerr += wer(rc, "".join(hyp))[0]
        refc += len(rc)
    print(f"# FLEURS de_de test[:{n}] on {DEVICE} -- {refw} reference words, "
          f"{secs:.0f}s audio")
    print(f"{'Build':26s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")
    print(f"{'vibevoice-asr':26s} {errs/refw*100:6.2f}% {cerr/refc*100:6.2f}% "
          f"{secs/el:6.2f}x")


if __name__ == "__main__":
    mode = sys.argv[1]
    arm = sys.argv[-1] if sys.argv[-1].startswith("vocab:") else "plain"
    if mode == "ref":
        score_reference(sys.argv[2], sys.argv[3], arm)
    elif mode == "fleurs":
        score_fleurs(int(sys.argv[2]), arm)
    else:
        sys.exit(__doc__)
