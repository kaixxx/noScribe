# Scoring other people's ASR engines

Six throwaway-shaped scripts that turned out to be worth keeping, because the
question "is there something better than Voxtral for German interviews yet?"
keeps coming back and rebuilding the harness each time is the expensive part.

## What they have in common

Each script scores one foreign engine on the same three yardsticks the Voxtral
builds are measured against, and reuses `norm()` and `wer()` from
`../wer.py` **verbatim** — pulled in by cutting that file before `main()` and
dropping its Voxtral import, so a comparison never silently changes the metric:

| yardstick | what it is |
|---|---|
| `Audiotest2/referenz/hart_780-900` | 422 words, 2 min, crosstalk and product names |
| `Audiotest2/referenz/zoom_9890-10190` | 859 words, 5 min, video call |
| FLEURS German, 100 recordings | the external, clean, read-aloud yardstick |

The audio is private and lives outside the repository.

The results and what they mean are in
[`../../other-asr-engines.md`](../../other-asr-engines.md). The short version,
confirmed six times over now: **FLEURS ranks
these models in an order that real German conversation does not.** Use these
scripts to find candidates, never to decide.

## The scripts

| script | model | needs |
|---|---|---|
| `parakeet_wer.py` | `nvidia/parakeet-tdt-0.6b-v3` | `pip install parakeet-mlx` |
| `nemotron_asr_wer.py` | `nvidia/nemotron-3.5-asr-streaming-0.6b` | mlx-audio from git main, in a throwaway venv |
| `qwen_asr_wer.py` | `Qwen/Qwen3-ASR-1.7B-hf` | nothing |
| `vibevoice_wer.py` | `microsoft/VibeVoice-ASR-HF` | nothing |
| `cohere_asr_wer.py` | `CohereLabs/cohere-transcribe-03-2026` | nothing (gated repo — accept on the model page first) |
| `transcribe_cpp_wer.py` | Voxtral Q8_0 GGUF via transcribe.cpp | the `transcribe` CLI, built from source |

"nothing" means transformers already supports the architecture natively, so the
venv does not have to be touched at all. `parakeet-mlx` is the one exception: it
adds only `dacite` on top of what is installed and its `mlx>=0.22.1` floor is
satisfied by the pinned 0.32.1, so it can be installed and removed without
disturbing the Voxtral stack. Verify with `pip freeze` before and after — it
should come back byte-identical.

**None of these belong in `environments/`.** They are for answering a question,
not for running noScribe.

## Usage

```bash
venv/bin/python3 docs/scripts/engines/<script> ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav [arm]
venv/bin/python3 docs/scripts/engines/<script> fleurs 100 [arm]
```

The `arm` argument differs per engine, because the interesting knob does:
Parakeet takes a beam width, Nemotron takes `language[@right_context]`, Qwen
takes `de` / `auto` / `chunkN` / `vocab:…`, Cohere takes its decoder-prompt toggles (`plain`, `nopnc`, `itn`, `timestamp`,
`diarize`) and has a `tokens` mode that reports which of them the tokenizer
actually knows. VibeVoice needs no arm; it emits speaker, time and text anyway.

## Adding a seventh

Copy the closest script and keep three things:

1. **The metric import stays byte-identical.** If a model needs its output
   cleaned before scoring — VibeVoice's `[Silence]` tags, Cohere's control
   tokens — strip it in the transcribe step and say so in a comment. That is
   output cleanup; changing `norm()` would be metric drift.
2. **Report punctuation.** Comma density per 100 words caught a real defect in
   two of four engines. Whisper sits at 10.62, Voxtral at 10.87.
3. **Do not trust a single speed number on MPS.** The same clip measured 2.04x
   cold and 4.45x warm in one session.
