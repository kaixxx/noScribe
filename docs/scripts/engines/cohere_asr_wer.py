"""Cohere Transcribe against the same yardsticks as the Voxtral builds.

`CohereLabs/cohere-transcribe-03-2026` is ~2B parameters at Apache-2.0 and the
best open-weight model on the Open ASR Leaderboard's English long-form tab.
Its decoder prompt carries explicit toggles -- punctuation, ITN, timestamps and
diarization -- so those are arms here rather than guesses.

Scored with the *identical* normalisation and edit distance as
docs/scripts/wer.py.

    python docs/scripts/cohere_asr_wer.py tokens
    python docs/scripts/cohere_asr_wer.py ref <reference.txt> <audio.wav> [arm]
    python docs/scripts/cohere_asr_wer.py fleurs <n_samples> [arm]

arm: "plain" (default), "nopnc", "itn", "timestamp", "diarize".
`tokens` prints which decoder-prompt toggles the tokenizer actually knows,
because the "on" variants are inferred from the "off" ones in the processor.
"""
import pathlib, re, sys, time

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

MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TAG = re.compile(r"<\|[^|]{0,30}\|>")
# This checkpoint opens the first window with a header of its own -- once per
# file, not per window, and it survives anything put in the context slot. It is
# an artefact of the prompt format, not a transcription, so it comes off before
# scoring. Leaving it in costs 0.7 WER points on the 422-word passage.
HEADER = re.compile(r"^\s*Input transcript[^:]{0,40}:\s*")

# The processor builds the decoder prompt from a fixed token list; each arm
# swaps one "off" token for its "on" counterpart.
SWAP = {
    "nopnc":     ("<|pnc|>",         "<|nopnc|>"),
    "itn":       ("<|noitn|>",       "<|itn|>"),
    "timestamp": ("<|notimestamp|>", "<|timestamp|>"),
    "diarize":   ("<|nodiarize|>",   "<|diarize|>"),
}


def load():
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    return proc, model


def show_tokens():
    """Which toggles does the tokenizer actually know?"""
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(MODEL).tokenizer
    unk = tok.convert_tokens_to_ids(tok.unk_token)
    for name, (off, on) in SWAP.items():
        ids = tok.convert_tokens_to_ids([off, on])
        state = ["known" if i not in (None, unk) else "UNKNOWN" for i in ids]
        print(f"{name:10s} {off:18s} {state[0]:8s}   {on:16s} {state[1]}")


def flip_toggle(proc, decoder_input_ids, arm):
    """Swap one decoder-prompt token in place.

    The processor windows long audio itself and returns one prompt row per
    window (`audio_chunk_index` says how to put them back together), so the
    prompt must be patched rather than rebuilt -- a hand-built single row would
    not match the encoder's batch.
    """
    if arm not in SWAP or arm == "nopnc":
        return decoder_input_ids
    off, on = SWAP[arm]
    off_id, on_id = proc.tokenizer.convert_tokens_to_ids([off, on])
    unk = proc.tokenizer.convert_tokens_to_ids(proc.tokenizer.unk_token)
    if on_id in (None, unk):
        raise SystemExit(f"tokenizer does not know {on} -- run `tokens`")
    ids = decoder_input_ids.clone()
    ids[ids == off_id] = on_id
    return ids


def transcribe(proc, model, audio, sr, arm, dur):
    inputs = proc(audio=audio, language="de", sampling_rate=sr,
                  punctuation=(arm != "nopnc"))
    chunk_index = inputs.pop("audio_chunk_index")
    inputs["decoder_input_ids"] = flip_toggle(
        proc, inputs["decoder_input_ids"], arm)
    inputs = inputs.to(model.device)
    for k, v in inputs.items():                   # keep ids integral
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs[k] = v.to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(dur * 20) + 512,
                             do_sample=False)
    # The reassembly zips `audio_chunk_index` against a *list* of texts, so the
    # windows have to come from batch_decode -- proc.decode() forwards to the
    # tokenizer's single-sequence decode and would zip the chunk index against
    # the characters of one joined string.
    raw = proc.batch_decode(out, skip_special_tokens=False)
    clean = proc.batch_decode(out, skip_special_tokens=True)
    text = proc._reassemble_chunk_texts(clean, chunk_index, " ")[0]
    rawtext = proc._reassemble_chunk_texts(raw, chunk_index, " ")[0]
    # Structured arms interleave control tokens; keep the raw form for
    # inspection and strip them for scoring.
    return HEADER.sub("", TAG.sub(" ", text).strip()), rawtext


def score_reference(ref_path, wav_path, arm):
    raw = open(ref_path, encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    ref_chars = "".join(ref)
    audio, sr = sf.read(wav_path, dtype="float32")
    dur = len(audio) / sr

    proc, model = load()
    t0 = time.time()
    text, rawtext = transcribe(proc, model, audio, sr, arm, dur)
    el = time.time() - t0

    hyp = norm(text)
    err, sub, dele, ins = wer(ref, hyp)
    cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
    got = sum(1 for w in overlap_words if w in hyp) if overlap_words else 0
    commas = text.count(",") / max(1, len(text.split())) * 100

    print(f"# {pathlib.Path(wav_path).name} on {DEVICE}: reference {len(ref)} "
          f"words / {len(ref_chars)} chars (+{len(overlap_words)} overlapping), "
          f"{dur:.0f}s")
    print(f"{'Build':30s} {'WER':>7s} {'CER':>7s} {'Sub':>5s} {'Del':>5s} "
          f"{'Ins':>5s} {'Speed':>7s} {'Kommas':>7s}  Overlap")
    print(f"{'cohere-transcribe (' + arm + ')':30s} {err/len(ref)*100:6.2f}% "
          f"{cer*100:6.2f}% {sub:5d} {dele:5d} {ins:5d} {dur/el:6.2f}x "
          f"{commas:6.2f}  {got}/{len(overlap_words)}")
    stem = pathlib.Path(wav_path).stem
    pathlib.Path(f"/tmp/cohere_{stem}_{arm}.txt").write_text(text, encoding="utf-8")
    pathlib.Path(f"/tmp/cohere_{stem}_{arm}_raw.txt").write_text(rawtext, encoding="utf-8")
    print(f"# transcript -> /tmp/cohere_{stem}_{arm}.txt (+ _raw.txt)")


def score_fleurs(n, arm):
    import io
    from datasets import load_dataset, Audio
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
        text, _ = transcribe(proc, model, audio, sr, arm, dur)
        el += time.time() - t0
        ref, hyp = norm(row["transcription"]), norm(text)
        errs += wer(ref, hyp)[0]
        refw += len(ref)
        rc = "".join(ref)
        cerr += wer(rc, "".join(hyp))[0]
        refc += len(rc)
    print(f"# FLEURS de_de test[:{n}] on {DEVICE} -- {refw} reference words, "
          f"{secs:.0f}s audio")
    print(f"{'Build':30s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")
    print(f"{'cohere-transcribe (' + arm + ')':30s} {errs/refw*100:6.2f}% "
          f"{cerr/refc*100:6.2f}% {secs/el:6.2f}x")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tokens":
        show_tokens()
    elif mode == "ref":
        score_reference(sys.argv[2], sys.argv[3],
                        sys.argv[4] if len(sys.argv) > 4 else "plain")
    elif mode == "fleurs":
        score_fleurs(int(sys.argv[2]),
                     sys.argv[3] if len(sys.argv) > 3 else "plain")
    else:
        sys.exit(__doc__)
