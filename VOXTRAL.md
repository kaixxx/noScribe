# Voxtral transcription engine (Apple Silicon)

An optional alternative to faster-whisper, using Mistral's **Voxtral** models via
[`mlx-voxtral`](https://pypi.org/project/mlx-voxtral/) on the Apple Silicon GPU.

On German / Swiss-German interview and podcast audio it is, in our tests,
**more accurate and more readable than Whisper** — it gets technical terms right
where Whisper mis-hears them (e.g. *Wortfindungsstörungen*, not
*Gottfindungsstörungen*), spells consistently, and produces fluent, readable
sentences instead of literal disfluent strings. With the 3B model it runs
several times **faster than realtime**; the 24B build is slower than the
recording (see the table below). Left to itself it falls into repetition loops
on hard audio more often than Whisper does, which is why the engine detects a
loop and repairs the pass instead of shipping the damage.

## Install (macOS, Apple Silicon)

```bash
pip install -r environments/requirements_voxtral_macOS_arm64.txt
```

The models `voxtral-mini-8bit` and `voxtral-small-4bit` then appear in the
model dropdown, each with the RAM it needs. They are downloaded on first use.

### A note on the pinned dependencies

`mlx`, `mlx-lm` and `mlx-voxtral` are pinned to exact versions.
[`mlx-voxtral`](https://github.com/mzbac/mlx.voxtral) went untouched between
2025-08 and 2026-08, so the pin started life as a deliberate freeze; the author
has since relicensed it to plain MIT and merged all four fixes this engine had
reported, so it is a normal version constraint again. `mlx-lm` is pinned because
decoding drives `generate_step` directly. `tests/test_voxtral_pin.py` checks that
the installed versions are the pinned ones.

The pin is `0.0.6`. One of those fixes moves this engine's output — the log-Mel
is now computed over the whole audio rather than per 30-second chunk, changing
the encoder input on most frames — and it was measured over 92 minutes of paired
audio to cost nothing at the transcript (dWER +0.02 [−0.19, +0.24]); see
[docs/voxtral-mel-clamp-floor.md](docs/voxtral-mel-clamp-floor.md), section 8.

The engine also swaps the processor's feature extractor for its own
(`_PercentileFloorFeatures`): the library clamps the log-Mel at `log_max − 8`
with the maximum taken over the whole input, so one loud transient — a door
slam — raises the floor for the entire pass and costs a measured +1.52 WER
points at 28 dB; the engine takes the floor from a percentile of the
spectrogram instead, which removes that and costs nothing on clean material.
The spectrogram itself is the library's (built from its own window, STFT and
filter bank), and `tests/test_mel_floor.py` pins it bit for bit to the
library's path at percentile 100. The same document has the measurements.

The coupling to `mlx-voxtral` is small and listed, so the pin is a known
liability with a measured exit route: five library calls (`load_voxtral_model`,
`VoxtralProcessor`, `apply_transcrition_request`, `model.generate` for the
repetition-penalty rung, `proc.decode`), the four audio primitives behind the
feature extractor, three `mlx-lm` imports for the decode loop, and a handful
of attribute reaches. Everything else in `voxtral_engine.py` — chunking, loop
detection, the temperature ladder, forced alignment, prefix salvage — is this
project's own and does not depend on which package loads the weights. If the
pin ever breaks against a newer MLX, the replacement is
[`mlx-audio`](https://github.com/Blaizzy/mlx-audio);
[docs/migration-mlx-audio.md](docs/migration-mlx-audio.md) records what was
verified about it (it does not load the published quantised builds, and fails
silently doing so; it carried the same stop-token defect this engine already
works around, fixed by this project's pull request on 2026-09-02 and not in a
release as of 0.5.1) and the steps and acceptance criteria for the move.

A frozen (PyInstaller) build does not bundle the MLX stack, so a packaged app
shows no Voxtral models; `is_available()` checks for the packages and the
feature stays inert without them. Voxtral is for running from source on an
Apple Silicon Mac.

## Language & word-timestamp quality

Word timestamps come from a CTC forced aligner: a wav2vec2 model produces
per-frame emissions (transformers, on the GPU), and a numpy Viterbi in
`noScribe/ctc_align.py` turns them into word boundaries — torchaudio's
`forced_align` kernel is no longer used, see
[docs/viterbi-numpy-brief.md](docs/viterbi-numpy-brief.md) for why. The
aligner model is chosen **per chunk from the transcribed text itself**: when one language dominates the
chunk (function-word analysis; non-Latin scripts are recognised directly),
the char-native model for that language is used -- so "Auto" gets the same
alignment quality as an explicit language choice. Mixed speech with a clear
majority language (e.g. German with English phrases) uses the majority
model, which also anchors the minority-language words; only text without a
dominant language falls back to the romanised multilingual aligner
(MMS-300M, 1130 languages). If an explicitly selected language contradicts
what the transcript looks like, a warning is logged.

## Which model

Two builds, both quantised on Apple Silicon and published so they download on
first use. They appear in the model menu only on Apple Silicon Macs — MLX cannot
run anywhere else — and Whisper (`precise`) stays the default because it runs on
every platform. The menu shows what each build needs, because picking one that
does not fit does not fail loudly: the machine starts swapping and the run stops
making progress. Builds that cannot fit are refused before a run starts.

| Build | Size | Needs | Notes |
|---|---:|---:|---|
| `voxtral-mini-8bit` (3B) | 6 GB | ~13 GB | **recommended** — runs on a 16 GB Mac, faster than realtime, reproduces the bf16 transcript at ~4.5× the speed |
| `voxtral-small-4bit` (24B) | 15 GB | ~23 GB | for clean, read-aloud audio and material heavy with names; ~2× realtime, runs on a 32 GB Mac |

**Which of the two?** It depends on the recording. On clean, read-aloud speech
the 24B model is clearly better (FLEURS German 2.8 % against 4.9 % word error).
On conversation — four hand-corrected passages of interview, podcast and video
call — Mini was as good or better on every one, and the 24B model looped where
Mini did not. For interviews and podcasts, pick Mini: it is also four times
faster and runs on a 16 GB Mac. The measurements, including the comparison
against Whisper, are in
[docs/voxtral-quantisation.md](docs/voxtral-quantisation.md).

Both builds keep the **audio encoder in bf16** and quantise the language model
and `lm_head` — Mini to 8 bit, Small to 4. The encoder runs once per pass, so its
precision costs no speed, but compressing it measurably costs accuracy on
difficult audio. `lm_head` runs once per generated token and is left quantised
for that reason. Small ships its body at 4 bit so that it fits in 32 GB; a 6-bit body measured no
better, and an 8-bit one (~34 GB) was not measured.

They download on first use from Hugging Face
([mini](https://huggingface.co/MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder),
[small](https://huggingface.co/MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder));
the weights are Apache-2.0. Other bit widths (4/5/6-bit, or a bf16 encoder on a
lower-bit body) can be made locally in a few seconds with
`tools/quantize_voxtral.py` — see the script's header and
[docs/voxtral-quantisation.md](docs/voxtral-quantisation.md).

## How it works

Voxtral produces clean text but no timestamps, so noScribe uses two paths:

- **Fast path** – plain text only. Used only for `.txt` output without
  speakers, timestamps or pauses — ideal for the "just give me a clean
  transcript" case (e.g. course summaries). Its segment times are
  approximations, which is why formats that embed audio-sync anchors never
  use it.
- **Long path** – word timestamps are recovered by CTC forced alignment
  against a language-matched wav2vec2 model (the WhisperX approach) and
  grouped into subtitle-sized cues. Used automatically for `.html` (its
  anchors drive the editor's click-to-play audio sync), `.vtt` subtitles,
  visible timestamps, speaker detection and pause marking.

Note: Voxtral has no prompt/hotword hook, so the "Disfluencies" option cannot
steer it (a log line says so). Voxtral Mini naturally smooths most fillers;
Voxtral Small stays closer to the exact wording.

## Long audio & memory

Voxtral transcribes each pass in a single `generate()` call whose peak memory
grows roughly linearly with the pass length (flash attention + a KV cache, *not*
O(T²)). Files up to one pass long therefore go through in one piece; longer ones
are split.

A pass is capped at **10 minutes**, whatever the machine could hold. That is the
longest Voxtral has been *measured* on: the widely quoted 30/40 minutes is a
capacity calculation (12.5 Hz frame rate against a 32k context), while the
paper's own long-form ASR protocol segments one-hour earnings calls "into
shorter, 10 minute variants" ([arXiv:2507.13264](https://arxiv.org/abs/2507.13264)).
Past that this project has recorded two distinct failures — whole passes coming
back translated, and passes returning without their opening — so the cap is a
refusal to run the model twice as far out as anyone has measured it, not a fix
for either (both are near-ties that a slightly shorter window would not dodge).

Below the cap the per-pass length is chosen from installed RAM so the estimated
generate peak stays within physical memory (compute on swapped-out MLX buffers
would thrash and never finish). The one-off model-load spike is allowed to swap
— it frees before transcription starts. The sizing, a little above the measured
peaks: mini ≈ 6.5 GB + ~0.4 GB/min, small ≈ 16.5 GB + ~0.4 GB/min. Rough
per-pass lengths:

| RAM | mini-8bit | small-4bit |
|----:|:---------:|:----------:|
| 16 GB | ~7 min | won't run (refused) |
| 24 GB | 10 min | ~1 min |
| 32 GB+ | 10 min | 10 min |

When a file is longer than one pass it is split into **equal, pause-aligned
passes**: each cut is snapped to a real speaker pause found in a wide window
(searching backward, since a shorter pass is always memory-safe), and a short
lead-in overlap is carried across the seam and de-duplicated by timestamp — so a
pass never splits a word and boundaries are effectively lossless.

The first pass is additionally checked for a dropped opening: a window
occasionally returns without its first seconds of speech, silently, so a short
head of the same audio is decoded and whatever is missing is spliced back. Later
passes are not checked: on the long path they carry a lead-in overlap that the
previous pass already transcribed, and probing every pass was measured at ~10 %
of each decode. (The fast path has no overlap, so there a later pass is simply
unguarded.) When the check finds something, the log says so.

This is not rare enough to skip: on a raw Zoom recording, 5 of 64 windows cut at
300 s and 600 s came back missing their opening, once losing 18 words of fluent
speech. It depends on the recording — read-aloud benchmark audio never shows it.
Measurements in [docs/voxtral-benchmarks.md](docs/voxtral-benchmarks.md), §5.

To pin the length yourself, set `voxtral_chunk_sec:` (seconds) in `config.yml`
(`0` = automatic). If other apps need RAM, raise `voxtral_ram_reserve_gb:`
instead — that is the knob the automatic length is computed against. Raising it past 10 minutes
is refused — see the cap above.

## Correcting brand / product / programme names

Voxtral has no hotword support, so it mis-hears proper names. Maintain a simple
find/replace list at:

```
<config dir>/voxtral_corrections.yml
```

(macOS: `~/Library/Application Support/noScribe/voxtral_corrections.yml`)

Speaker names entered in the app are corrected in the same step, but only
where the model wrote a name as it sounds and spelled it another way ("Mohna"
or "Mona", "Steffy" or "Steffi"). A word that sounds different ("Muna") may be
someone else and is left alone — put it in the list above if it is not — and
so is any word the macOS dictionary of the transcript's language knows: "Mohn"
next to a speaker called Mon, or a real name spelled differently ("Marcus" next
to "Markus"), which may be someone else. Where the system has no dictionary for
the language, or one that accepts every word, names are left as the model wrote
them.

```yaml
- to: VitaFlor
  from: [vitaflor, "vita flor", "flor-öl", "flor-öle"]
- to: Sonvita
  from: [sonvida, sonvieda]
```

Matches are whole-word and case-insensitive. The file is created empty (with
commented examples) on first use — add your recurring brand, product and
programme names (e.g. from earlier podcast transcripts).

Why this and not a prompt: the transcription request has exactly one text slot
(`lang:xx`), and putting terms there was measured to act as a decode
perturbation rather than a vocabulary hint — on three clips it fixed one term,
ignored another and broke a third that the plain request had got right.
Voxtral's chat mode *does* use a term list, but it is not a verbatim
transcriber: one of four clips came back as a 12-word answer instead of an
84-word transcript, and another wrote the instruction into the text. A term
Voxtral does not know is therefore fixed after the fact, here.

## Author

The Voxtral integration for noScribe (engine, forced alignment, quantised
model builds) was created by **[Markus Kämmerer](https://markus-kaemmerer.de)**
· [Instagram @markuskaemmerer](https://www.instagram.com/markuskaemmerer/).

It was written with [Claude Code](https://claude.com/claude-code): the design
decisions, the measurements and what to make of them are the author's; Claude
wrote and reviewed much of the code and the write-ups under that direction.
