# Other ASR engines, measured against Voxtral

Is anything else better for German interview audio yet? Six engines have been
measured against the shipped `voxtral-mini-8bit` build, and none has replaced
it. This file is the record, so the same candidate is not re-evaluated from
scratch every time it trends. Which *Voxtral* build to ship is a different
question and lives in [voxtral-quantisation.md](voxtral-quantisation.md).

## How these numbers are made

One harness, one metric, for every engine: the scripts under
`docs/scripts/engines/` reuse `norm` and `wer` from `docs/scripts/wer.py`
unchanged, so a row here is comparable with a row there.
`docs/scripts/engines/README.md` has the shape they share and how to add one.

Three yardsticks, deliberately unequal:

* **Hard passage** — 422 words, two minutes, hand-corrected: overlapping speech,
  crosstalk, brand names, a dialect speaker. The hardest thing here.
* **Second reference** — 859 words, five minutes, a video call, hand-corrected:
  real conversation, cleaner recording.
* **FLEURS German** — 100 read-aloud recordings, 25 minutes, public benchmark.

The reason for all three is the finding that keeps repeating below: **a model
can top the read-aloud benchmark and be unusable for interview work.** Speeds
measured on MPS are indicative only — the same clip has measured 2.04x cold and
4.45x warm in one session.

> **A note on the second reference.** It was produced by hand-correcting this
> engine's own draft, so scores on it flatter Voxtral by construction. The
> figure below (1.98 % / 1.09 %) is the honest one, re-scored 2026-08-22 under
> mlx-voxtral 0.0.6. An earlier **0.81 % / 0.64 %** circulated and should not
> be quoted: it was the residual distance to the very draft the reference was
> corrected from — two substitutions on 859 words of conversational audio was
> implausibly good, and that was the tell. Comparisons keep their direction
> either way, with a smaller multiplier.

## The three tables

**Hard passage, 422 words.** Ranked by word error.

| Engine | WER | CER | Sub | Del | Ins | Speed | Commas/100w |
|---|---:|---:|---:|---:|---:|---:|---:|
| **voxtral-mini-8bit** | **4.27 %** | **3.39 %** | 10 | 8 | 0 | 6.8–7.2x | 10.87 |
| transcribe.cpp Q8_0, `--language de` | 4.98 % | 3.83 % | 10 | 10 | 1 | 7.11x | — |
| transcribe.cpp Q8_0, auto | 5.45 % | 3.83 % | 11 | 10 | 2 | 7.10x | — |
| whisper-fast | 8.06 % | 3.34 % | 22 | 4 | 8 | 2.43x | 10.62 |
| cohere-transcribe | 10.43 % | 3.69 % | 25 | 4 | 15 | **16.58x** | 11.74 |
| qwen3-asr-1.7b, 60 s chunks | 10.90 % | 4.33 % | 29 | 5 | 12 | — | 8.73 |
| qwen3-asr-1.7b, one pass | 11.14 % | 4.08 % | 31 | 4 | 12 | — | 0 |
| qwen3-asr-1.7b, auto language | 11.85 % | 4.33 % | 32 | 5 | 13 | — | — |
| vibevoice-asr | 13.03 % | 6.98 % | 34 | 2 | 19 | 0.66x | **11.34** |
| parakeet-tdt-0.6b-v3, beam 5 | 14.69 % | 6.49 % | 49 | 6 | 7 | 9.13x | — |
| parakeet-tdt-0.6b-v3, greedy | 18.01 % | 9.88 % | 49 | 20 | 7 | 26.47x | — |

**Second reference, 859 words.**

| Engine | WER | CER | Sub | Del | Ins | Speed |
|---|---:|---:|---:|---:|---:|---:|
| **voxtral-mini-8bit** | **1.98 %** | **1.09 %** | 12 | 4 | 1 | 7.97x |
| transcribe.cpp Q8_0, `--language de` | 2.10 % | 1.33 % | 12 | 5 | 1 | 8.25x |
| parakeet-tdt-0.6b-v3, beam 5 | 8.50 % | 5.18 % | 34 | 15 | 24 | 10.30x |
| cohere-transcribe | 9.20 % | 6.12 % | — | — | — | **27.93x** |
| parakeet-tdt-0.6b-v3, greedy | 10.59 % | 6.81 % | 36 | 14 | 41 | 41.75x |
| qwen3-asr-1.7b, 60 s chunks | 10.83 % | 6.69 % | 36 | 16 | 41 | — |
| qwen3-asr-1.7b, one pass | 12.11 % | 7.01 % | 43 | 17 | 44 | — |
| vibevoice-asr | 12.34 % | 7.58 % | 35 | 10 | 61 | 0.87x |

**FLEURS German, 100 recordings.** The ranking inverts almost completely.

| Engine | WER | CER | Speed |
|---|---:|---:|---:|
| **qwen3-asr-1.7b** | **3.96 %** | **1.23 %** | — |
| whisper-precise | 4.05 % | 1.40 % | 4.11x |
| cohere-transcribe | 4.55 % | 1.85 % | 13.51x |
| voxtral-mini-8bit | 4.81 % | 1.44 % | 7.64x |
| parakeet-tdt-0.6b-v3, greedy | 4.81 % | 2.15 % | **44.45x** |
| vibevoice-asr | 8.26 % | 5.84 % | — |

Read the three together and the pattern is the whole point: on FLEURS the field
is separated by tenths of a point and Voxtral is mid-table; on real conversation
the same field spreads over an order of magnitude and the order reverses.

---

## transcribe.cpp's Voxtral — on par, and that is the finding

[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) is a GGML
speech-to-text library (MIT) carrying Voxtral alongside fifteen other families,
with Metal, CUDA, Vulkan and HIP backends. It is the only route we know to
*this* model on hardware MLX cannot reach — which is the whole reason to measure
it, since the engine is otherwise Apple-Silicon-only. Built from source (Metal),
`Voxtral-Mini-3B-2507-Q8_0.gguf` from the project's own GGUF repo, with our own
build re-run the same day so both columns come from one machine and one pin.

**Identical substitution counts on both passages** (10 and 12), the difference
sitting in one or two extra deletions — inside the ±1.8 to ±2.5 CER points a
passage this size can resolve. The two engines are indistinguishable on this
material, at the same speed.

**The expected failure did not appear.** [Issue #82](https://github.com/handy-computer/transcribe.cpp/issues/82)
reports Voxtral's Tekken tokenizer as unimplemented there, with the loader
falling back to qwen2 pretokenization and producing German word-level garbles
("Publikum" → "Pubikom"). That is the defect this measurement was designed to
catch, and on 1281 words of German it did not show: a near-miss scan over every
produced word absent from the reference turns up three pairs, all ordinary
mishearings (`geworden` for `geboren`, `schokopourridge` for `schokoporridge`)
rather than dropped-letter garbles. The issue may be real on other material or
other quants; it is not visible here, at Q8_0, on this audio.

So the quality objection to a cross-platform Voxtral does not survive contact
with the measurement. What remains is engine-level, not model-level, and is
recorded nowhere else, so here it is: transcribe.cpp advertises
`TRANSCRIBE_TIMESTAMPS_NONE` for Voxtral; there is no repetition/loop defence on
its causal-LM path (the only compression-ratio gate in that tree is Whisper's
2.4, which we measured as too coarse); and a C API cannot hand us the logits
processor the loop breaker rides on.

## Parakeet-TDT — rejected

`nvidia/parakeet-tdt-0.6b-v3`: a FastConformer encoder with a Token-and-Duration
Transducer decoder, 0.6B parameters, CC-BY-4.0, 25 European languages including
German, and — unlike anything else here — distributed as MLX, ONNX, CoreML and
GGUF, so it would run on Windows and Linux CPUs too. A transducer also cannot do
several things this engine defends against: it emits tokens bound to audio
frames, so repetition loops, whole-chunk language drift and a dropped head are
structurally impossible, and word timestamps fall out of the predicted durations
instead of needing a CTC aligner. Measured via `parakeet-mlx` 0.5.2 at bf16.

**On FLEURS it ties Voxtral to the second decimal, and on conversation it is
four to five times behind.** That is the benchmark's whole worth here.

**The failure has a name: uncontrolled code-switching.** v3 is multilingual with
no language conditioning — no language token, so `transcribe()` has no
`language` argument. When the acoustics get ambiguous it writes German function
words as their English homophones: *und* as "and", *wenn* as "when", *es ist* as
"it is", *gut* as "good", *ja* as "yeah". Counted: **14 English function-word
tokens in 407 words** on the hard passage, **0 in 882 words** on the cleaner
second reference. It is triggered by audio quality, and no input suppresses it.

The error *profile* is the second problem: 49 substitutions against Voxtral's 10
on the same passage. Parakeet replaces where Voxtral omits, and a wrong word
survives proof-reading in a way a missing one does not — the same argument that
decided against `whisper-precise`. Beam search (width 5) is worth having if the
model is ever revisited: it cuts deletions from 20 to 6 and roughly 3 WER
points, for two thirds of the throughput.

## Qwen3-ASR-1.7B — wins the benchmark, loses the job

`Qwen/Qwen3-ASR-1.7B-hf` is the Voxtral principle at half the size: an audio
encoder in front of a Qwen3-Omni language model, Apache-2.0, 30 languages. Two
things make it easier to try than anything else here — **transformers supports it
natively** (`AutoModelForMultimodalLM`, no new dependency in this venv) — and it
takes an explicit language as well as a free-form context prompt.

On FLEURS German it is the best model this project has measured, ahead of
Whisper and of the shipped Voxtral build in both words and characters. On the
two conversational references it is two to six times behind. The cause is not
code-switching — there is none, and forcing the language buys only 0.7 WER
points over auto-detect. It simply hears this material less well.

Three findings that outlive the verdict:

**Punctuation collapses on long input, and chunking fixes it.** Fed the whole
120 s or 300 s clip, the model returns text with *zero* commas and *zero* full
stops. Cut into ~60 s windows it punctuates normally — 8.73 and 10.33 commas per
100 words, against Whisper's 10.62 and Voxtral's 10.87. A length sweep puts the
usable band at roughly 30–90 s. The card advertises long audio; for German prose
output it does not hold, and any engine built on this model would have to chunk
far more aggressively than Voxtral does.

**The context prompt is a real hotword mechanism — the thing Voxtral lacks.**
With `Vocabulary: …` in the system message, "Vita Flor" becomes "VitaFlor",
"Son Vita" becomes "Sonvita" and a mangled compound comes back correct, on a
controlled 30 s clip with no other change to the text. Over the full passage it
costs nothing in word error (10.90 % either way) and 0.05 points of character
error. This is exactly what `voxtral_corrections.yml` exists to work around, and
it is the one capability that would argue for the model.

**But it is paid for in punctuation:** the same vocabulary hint halves comma
density, 8.73 to 4.28 per 100 words. A term list behaves as a decode
perturbation here too, just a cheaper one than in Voxtral — the same phenomenon
as the `repetition_penalty` default, and a reminder to measure punctuation
whenever a decode-level knob is turned.

Two implementation notes for anyone who picks this up. The `prompt=` argument
documented on `apply_transcription_request` **does not exist** in transformers
5.15.0.dev0 — it lands in `**kwargs` and is dropped with a warning; the context
has to go into a system message next to the language, which is what the chat
template concatenates anyway. And an MLX port would be the honest place to
measure throughput.

## VibeVoice-ASR — the architecture works, the recognition does not

`microsoft/VibeVoice-ASR-HF` is not another engine behind the same seam. It does
ASR, diarization and timestamping in **one pass** and emits speaker-attributed
segments directly — noScribe's whole pipeline collapsed into one model. MIT,
16.7 GB in bf16, transformers-native, with 4-bit and 8-bit MLX ports from
mlx-community.

FLEURS is arguably the wrong test for it — its design point is an hour of
multi-speaker audio, not a 15-second clip, and one of the 100 clips came back as
nothing but a `[Silence]` tag. Take the 8.26 % as a lower bound rather than a
verdict.

**The diarization, however, is the real thing.** Frame by frame against
noScribe's own pyannote pipeline on the same two clips, 10 ms resolution, best
speaker permutation:

| | speakers | segments | timeline labelled | agreement with pyannote |
|---|---:|---:|---:|---:|
| hard passage | 2 vs 2 | 16 vs 22 | 100 % vs 96.4 % | **98.2 %** |
| second reference | 2 vs 2 | 19 vs 52 | 97.1 % vs 93.3 % | **97.5 %** |

Same speaker count, and near-total agreement on who is speaking — from a model
that produced the transcript in the same forward pass. The segmentation is much
coarser (16 and 19 turns against 22 and 52), a problem for subtitle cues but not
for speaker attribution. It also punctuates better than anything else measured
here (11.34 and 12.57 commas per 100 words) and labels non-speech explicitly
(`[Silence]`, `[Human Sounds]`); those tags are stripped before scoring, and
leaving them in costs 3.3 WER points on FLEURS.

**Not adopted, and the reason is only recognition.** Three times Voxtral's word
error on the hard passage, six times on the second reference, and 0.66–0.87x
realtime on MPS — slower than the two-stage pipeline it would replace, in which
pyannote is cheap and Voxtral runs at 6.6x. But the architecture is validated,
and that is worth writing down: **a single model really can deliver speaker,
time and text at pyannote-grade diarization quality.** The thing to watch is a
model of this shape that hears German conversation as well as Voxtral does.
Nothing here suggests that is far off.

## Cohere Transcribe — the best challenger so far, still not close enough

`CohereLabs/cohere-transcribe-03-2026`, from the leaderboard below: ~2B
parameters, Apache-2.0, 3.9 GB, transformers-native, best open-weight model on
the leaderboard's English long-form tab. Measured at bf16 on MPS, language
forced to `de`.

**On the hard passage its character error is within noise of Voxtral's** — 3.69
against 3.39, where this project's noise floor is ~0.15 points and a single
build's word-error interval is ±3. By the measure that tracks what was *heard*
rather than how it was spelled, a 2B model at 16x realtime is level with the 3B
Voxtral build at 6.8x, on the hardest audio here. That is the best result any
challenger has produced.

**The second reference decides it anyway:** 9.20 % against 1.98 %, character
error 6.12 against 1.09 — a factor of five, far outside anything the error bars
cover, on the larger of the two references.

Three limitations from its own model card, all of which matter for an engine:

* **No timestamps, no diarization.** The tokenizer knows `<|timestamp|>` and
  `<|diarize|>` and the decoder prompt accepts them — leftovers of the training
  format. Setting them changes nothing useful (the diarize arm just truncates,
  307 words against 426), and the card says the model does not feature either.
  Word timestamps would still need the CTC aligner.
* **No language detection**, and explicitly inconsistent on code-switched audio.
  noScribe's "Auto" would have to be resolved before the engine is called.
* **It hallucinates on silence** and wants a VAD or noise gate in front. Visible
  here: the first window opens with a header of its own, `Input transcript
  corrected:`, once per file, regardless of the context slot. Stripped before
  scoring; leaving it in costs 0.7 WER points.

Not adopted. But it is the first challenger where the gap is about a specific
weakness rather than the whole model, and its ecosystem is the broadest of
anything here — transformers, vLLM, mlx-audio, ONNX, GGUF, a Rust port and a
WebGPU demo. Worth re-measuring when Cohere ships a successor.

*Practical note:* the repo is gated (click-through) and its Xet transfer fails
with `Unable to parse string as hex hash value`. `HF_HUB_DISABLE_XET=1` in front
of the download falls back to plain HTTP and works.

## Voxtral-Mini-4B-Realtime — ruled out on the vendor's own numbers

The streaming **Voxtral-Mini-4B-Realtime** model (Awni Hannun's
[voxmlx](https://github.com/awni/voxmlx) runs it with a bounded rotating KV
cache) was considered as a low-memory option, then ruled out on Mistral's own
published figures: on German FLEURS it scores 6.19 % WER at its 480 ms setting
and 4.15 % even at 2.4 s delay — worse than the offline Voxtral Mini 3B
(3.54 %), a smaller model. The causal/streaming architecture trades look-ahead
for latency, and on hard conversational audio the gap would only widen. Its
advantages — sub-500 ms latency, bounded memory — are irrelevant to offline file
transcription.

**`voxtral-mini-2602` ("Transcribe 2")** is ruled out for a different reason:
still API-only, and therefore unusable for confidential interviews.

## What the Open ASR Leaderboard says (German tab, checked 2026-08-20)

The leaderboard has a German tab fed by `hf-audio/multilingual_evals`
(`multilingual_de.csv`) — an independent check on everything above. Ranked by
FLEURS German WER, Common Voice alongside:

| Model | FLEURS | MCV | RTFx |
|---|---:|---:|---:|
| microsoft/azure-speech-06-2026 *(API)* | 1.93 | 1.88 | — |
| elevenlabs/scribe_v2 *(API)* | 2.30 | 2.19 | — |
| assemblyai/universal-3-pro *(API)* | 2.42 | 2.76 | — |
| reson8/resonant-1 *(API)* | 2.56 | 3.01 | — |
| **mistralai/Voxtral-Small-24B-2507** | **2.61** | 3.19 | 83 |
| openai/whisper-large-v3 | 3.20 | 4.79 | 328 |
| CohereLabs/cohere-transcribe-03-2026 | 3.33 | **2.87** | 607 |
| Qwen/Qwen3-ASR-1.7B-hf | 3.35 | 4.60 | 369 |
| nvidia/canary-1b-v2 | 3.43 | 4.69 | 1308 |
| **mistralai/Voxtral-Mini-3B-2507** | **3.64** | 5.29 | 150 |
| microsoft/Phi-4-multimodal-instruct | 3.99 | 4.25 | 123 |
| nvidia/parakeet-tdt-0.6b-v3 | 4.16 | 4.07 | 3363 |
| microsoft/VibeVoice-ASR-HF | 7.44 | 20.97 | 114 |

**Voxtral-Small is the best open-weight model on German here.** Everything above
it reports no RTFx, which on this leaderboard marks a proprietary API. The engine
this project settled on is not a compromise pick; it is the top of the open field
for this language.

It also cross-checks the harness, though not perfectly. The broad ordering on
FLEURS matches ours — Qwen3-ASR ahead of Voxtral-Mini, VibeVoice far behind —
but Parakeet is one place *behind* Voxtral-Mini there while our own run has the
two tied at 4.81 %, and the absolute values differ by more than normalisation
alone would explain: Voxtral-Mini 3.64 against our 4.81, Cohere 3.33 against our
4.55, both more than a point apart, while Qwen (0.61) and Parakeet (0.65) sit
close. Our 4-bit Voxtral-Small scores 2.82 against the leaderboard's 2.61 for
the unquantised model, so the quantisation costs about 0.2 points on clean
audio, which is the same story the build sweeps tell.

**`CohereLabs/cohere-transcribe-03-2026` was the one candidate the table added
that we had not seen** — best Common Voice German of any open model listed at
2.87, ahead of Voxtral-Small's 3.19, and best open model outright on the English
long-form tab. It has since been measured; see above, and note how little the
leaderboard predicted: two Common Voice points ahead of Voxtral-Small, and a
factor of five behind Voxtral-Mini on a real conversation. The long-form tab is
**English only**, so it says nothing about German conversation — the gap this
project keeps running into has no public benchmark at all.

## Surveyed, not measured

Scaling up within Parakeet's own family does not help: `parakeet-tdt-1.1b` and
`canary-qwen-2.5b` are English-only. `nvidia/canary-1b-v2` is the real step up —
25 languages, CC-BY-4.0, 4.40 % against Parakeet's 5.04 % on NVIDIA's own FLEURS
German figure — but it is an attention encoder-decoder rather than a transducer,
so it gives back the structural guarantees that made the family interesting, and
0.6 FLEURS points say nothing about conversational audio.
`OpenMOSS-Team/MOSS-Transcribe-Diarize` does VibeVoice's joint trick and is more
popular, but supports only Chinese and English.

Inside mlx-audio, three German-capable ASR models are unmeasured: `canary`,
`granite_speech` and `nemotron_asr`. The rest of its `stt/models/` carry no
German at all, and `mega_asr` is a router over Qwen3-ASR rather than a model.
After six comparisons the pattern is stable enough to predict the outcome:
these models tie on FLEURS and lose on real conversation.

## Reproducing

```bash
# Parakeet (needs `pip install parakeet-mlx`, which is NOT in the requirements --
# it pulls only dacite on top of what is already installed, and mlx 0.32.1
# satisfies its floor, so it can be added and removed without disturbing the
# pinned Voxtral stack)
python docs/scripts/engines/parakeet_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav 5      # trailing 5 = beam width
python docs/scripts/engines/parakeet_wer.py fleurs 100

# Qwen3-ASR (no extra dependency -- transformers 5.13+ has it)
python docs/scripts/engines/qwen_asr_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav chunk60
python docs/scripts/engines/qwen_asr_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav "vocab:VitaFlor, Sonvita"
python docs/scripts/engines/qwen_asr_wer.py fleurs 100

# VibeVoice-ASR: same yardsticks, plus its own diarization
python docs/scripts/engines/vibevoice_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav
python docs/scripts/engines/vibevoice_wer.py fleurs 100

# Cohere Transcribe (gated repo -- accept on the model page first; its Xet
# transfer is broken, so HF_HUB_DISABLE_XET=1 for the download)
python docs/scripts/engines/cohere_asr_wer.py tokens
python docs/scripts/engines/cohere_asr_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav plain
python docs/scripts/engines/cohere_asr_wer.py fleurs 100

# transcribe.cpp (build it first; audio must already be 16 kHz mono WAV)
python docs/scripts/engines/transcribe_cpp_wer.py \
    <transcribe-cli> <Voxtral-Mini-3B-2507-Q8_0.gguf> \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav --lang de
```

Start `venv/bin/python3`, not `python`.
