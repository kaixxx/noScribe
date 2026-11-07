# Voxtral: which build, and why

Everything here was measured on an M1 Max (32 GB) with `mlx-voxtral` 0.0.4.
The scripts are in `docs/scripts/`, the builds are made with
`tools/quantize_voxtral.py`.

(Two things have changed under these numbers since. The pin moved to 0.0.6,
whose log-Mel change was measured at the transcript and found neutral; and the
engine now clamps the log-Mel at a percentile of the spectrogram rather than its
maximum (`voxtral-mel-clamp-floor.md`). The clamp moves single passages by a
few words either way, so **the absolute figures in the older tables below are
max-floor figures** — the shipped Mini build scores 5.45 % / 3.29 % on the hard
passage today, not 4.27 % / 3.39 % — while the comparisons between builds hold:
the 2026-09 re-run, which measured every build under the percentile floor,
reorders none of them. Where a table comes from that re-run it says so.)

## The short answer

Ship **`voxtral-mini-8bit`**: the 3B weights at 8 bit with the **audio encoder
left in bf16**. On hard German interview audio it reproduces the unquantised
transcript word for word at 4.5x the speed, in about 7 GB for a two-minute pass
(`MEM_MODEL` meters the rest; see *Decode path and memory*). It is at its
optimum: more bits change nothing, fewer cost accuracy.

Next to it, **`voxtral-small-4bit`**: the 24B weights at 4 bit, encoder again in
bf16, 15 GB, ~19 GB peak for a ten-minute pass, so it runs on a 32 GB Mac. It
replaced an 8-bit build that needs ~34 GB. A 6-bit body measured no better than
the 4-bit one; an 8-bit body has not been measured against it (chapter
*Running the 24B model on 32 GB*). It is the better model for clean audio (FLEURS 2.78 % against 4.89 %
WER) and **not** for conversation, where Mini is as good or better and Small
loops more often. Both are worked out below.

## How the measurements are made

Two yardsticks, because they disagree — and the disagreement is the finding.

**A hand-corrected passage of real podcast audio** (`Audiotest2/referenz/`):
two minutes, 422 words, chosen for what makes transcription hard — brand names,
foreign words, two speakers talking over each other. Corrected by ear against
the recording.

**FLEURS German** (`docs/scripts/fleurs.py`): the benchmark the Voxtral report
uses (arXiv:2507.13264), so the numbers can be held against a published figure.
Read-aloud, clean, one speaker.

Both are scored with the same metric (`docs/scripts/wer.py`):

- **WER** — word error rate, the usual measure.
- **CER** — character error rate on the text with all spaces removed. German
  lets the writer choose between "Floröl" and "Flor Öl", or "Hokuspokus"
  and "Hocus Pocus"; identical speech, different typing. WER charges those as
  errors, CER cannot see them. **Where WER and CER disagree, the difference was
  orthographic; where both move together, the model misheard.**

The reference marks overlapping speech as `//text//`. A transcript that drops
those words is not wrong — the model is asked for the dominant voice — so they
are counted separately rather than as errors.

Generation is deterministic (`temperature=0.0`): the same build on the same
audio produces byte-identical output across runs. Differences between builds
are real, not run-to-run noise.

## What each part of the model is worth

A Voxtral build has three parts that can carry different precision:

| Part | Size | Runs | Precision matters for |
|---|---|---|---|
| audio encoder + projector | ~0.6B in **both** model sizes | once per pass | accuracy, not speed |
| `lm_head` | 0.4B (3B model) / 0.7B (24B) | once per **generated token** | speed, not accuracy |
| language model body | the rest | per token | memory, mostly |

That the encoder is *the same size in both models* is why it is cheap to
protect in the 24B build and merely cheap in the 3B one.

### The encoder wants 8 bit (measured, isolated)

24B at 4 bit, `lm_head` and language model held fixed, only the encoder varied:

| Encoder | WER | CER |
|---|---:|---:|
| 4 bit | 10.19 % | 4.87 % |
| 6 bit | 9.72 % | 4.28 % |
| **8 bit** | **8.29 %** | **2.46 %** |
| bf16 | 8.29 % | 2.41 % |

Monotone, and the step from 6 to 8 bit halves the character error rate. WER and
CER move together, so these are genuine mishearings. **Above 8 bit nothing more
is gained** — bf16 matches 8 bit, and the 0.05-point CER difference between them
is one character out of 2034.

This matters because the 4-bit 24B build used before this sweep carried its
encoder at 6 bit, and the locally-converted 6-bit build had it at 6 bit too. (The
`voxtral-small-4bit` shipped today keeps it in bf16.)

**On the 24B model the threshold is 8 bit. On the 3B model there is no
threshold in this range at all** — see "Does the 3B encoder need the bits?"
below. The two models do not behave the same way, and the 24B result must not
be carried over to the 3B one.

### `lm_head` wants to stay quantised

Same experiment, `lm_head` varied instead:

| `lm_head` | WER (24B) | CER (24B) | WER (3B) | CER (3B) | Speed (3B) |
|---|---:|---:|---:|---:|---:|
| 4 bit | 7.82 % | 2.11 % | — | — | — |
| 6 bit | 8.29 % | 2.41 % | 4.98 % | 3.83 % | 6.74x |
| 8 bit | 8.29 % | 2.46 % | 4.27 % | 3.39 % | 6.72x |
| bf16 | 8.29 % | 2.46 % | 4.27 % | 3.39 % | **4.93x** |

Leaving it dense buys nothing and costs 27 % throughput, because it runs on
every generated token. 8 bit it is.

**On the 24B model this column is noise, and it runs backwards.** More bits
score slightly *worse* (2.11 → 2.41 → 2.46), and the whole spread is seven
characters out of 2034. Under greedy decoding a single token choice flips and
the text diverges from there on. So `lm_head` precision is simply irrelevant on
the 24B model; pick 4 bit for the speed. On the 3B model the 6→8 step is nine
characters plus two extra deletions, marginally above that noise floor, and 8
bit is free anyway.

### The resulting 3B build

| Build | WER | CER | Speed | Peak (120 s) |
|---|---:|---:|---:|---:|
| 8 bit uniform | 4.74 % | **3.34 %** | 6.68x | 7.2 GB |
| **8 bit, encoder bf16** | **4.27 %** | 3.39 % | **6.60x** | 7.7 GB |
| bf16 | 4.27 % | 3.39 % | 1.46x | 13.2 GB |

The recommended build is byte-identical to bf16 on the reference passage. The
encoder costs 0.5 GB and no measurable speed, because it runs once per pass.

**Above 8 bit there is nothing to gain, and no useful step between.** bf16,
fp16 and 8 bit produce the same text on a 150 s excerpt (414 words each), and
fp16 is neither faster nor more accurate than bf16 (1.84x against 1.86x). A
16-bit integer format does not exist in MLX and would be pointless if it did:
quantised, a 256x256 block needs 128 KB plus 8 KB of scales, i.e. more than
fp16's 128 KB. The whole speed jump is bf16 to *any* quantisation (1.5x to
7.3x); between 8 and 4 bit lie 12 % speed and the quality cliff below.

The 2026-09 re-run (percentile floor) repeated the bf16-against-8-bit question on
both hand-corrected passages. On the hard one the two transcripts are
byte-identical; on the five-minute video call (844 words) they differ in three
places — one spelling (a compound written with and without its hyphen), one word, one
doubled filler word the bf16 build leaves out — none of them recognisably the
better reading, paired difference dWER +0.12 [−0.23, +0.58]. The 8-bit body
*is* the unquantised model for practical purposes.

**But the two metrics disagree about whether it is worth anything.** The uniform
build differs from the recommended one by exactly two insertions — same
substitutions, same deletions — and its CER is a hair *lower*. Two insertions at
roughly zero character cost means two compound words were written apart
("Flor Öl" for "Floröl"). By the rule stated at the top of this document
that is orthography, not recognition. FLEURS says the same thing on 100
recordings: 4.81 % for both, CER 1.40 % uniform against 1.44 % with the bf16
encoder.

Neither scoring run, though, has the resolution to settle what the encoder's
precision is worth. The next chapter takes that question to 26 619 words.

### Below 8 bit: affine and float 4-bit formats

Same re-run, 3B model, encoder bf16 and `lm_head` 8 bit throughout, only the
body's format varied (✱ = paired interval against 8 bit excludes zero):

| 3B body | hard: WER / CER | video call: WER / CER | Speed | Peak |
|---|---:|---:|---:|---:|
| **8 bit (shipped)** | 5.45 / 3.29 | 2.68 / 1.68 | 7.0–7.9x | 7–8 GB |
| bf16 | 5.45 / 3.29 | 2.56 / 1.70 | 1.5–1.9x | 13 GB |
| 4 bit affine | 8.06 / **6.05** ✱ | 3.49 ✱ / 2.07 | 8.9–9.2x | 5–6 GB |
| `mxfp4` | 6.64 / 3.15 | 3.84 ✱ / 2.49 ✱ | 8.8–9.2x | 5–6 GB |
| `nvfp4` | 7.11 / 3.24 | 4.19 ✱ / 2.64 ✱ | 8.6–9.4x | 5–6 GB |

Every 4-bit body loses measurably, for ~20 % speed and 2 GB. The two float
formats MLX offers alongside affine quantisation are not a way round it: on the
video call both are worse than affine 4 bit, on the hard passage they hear as
well as 8 bit but insert more. No Apple GPU computes in FP4 natively; MLX
unpacks both formats to fp16 in its kernels on every M chip, so they cost the
same memory bandwidth as affine 4 bit (4.25–4.5 bits per weight) and buy no
speed on an M1. On the 24B model `mxfp4` again scored below affine and `nvfp4`
broke the model outright — an endless "ist, ist, ist" on both passages; MLX's
`nvfp4` takes an optional per-tensor global scale, which `mlx_lm`'s
quantisation does not set. Data-driven quantisation (AWQ, GPTQ, DWQ in
`mlx_lm`) could at best lift a 4-bit Mini to its 8-bit level; since 8 bit
already runs on a 16 GB Mac, it was not pursued.

## Does the 3B encoder need the bits?

Three builds, identical except for the audio encoder — language model and
`lm_head` at 8 bit throughout, only `audio_tower` and `projector` vary, verified
against the quant configs (212 `language_model` tensors and `lm_head` at 8 bit in
all three; `audio_tower` has 192 tensors, the projector 2). Scored
on the hand-corrected passage, with bootstrap intervals over 10 000 resamples
(`docs/scripts/bootstrap_cer.py`):

| Encoder | WER | CER | paired difference vs bf16 |
|---|---:|---:|---|
| bf16 (shipped) | 4.27 % | 3.39 % | — |
| 8 bit | 4.74 % | 3.34 % | dWER −0.47 [−1.73, +0.50], dCER +0.05 [−0.14, +0.29] |
| **6 bit** | **4.03 %** | **3.29 %** | dWER +0.24 [+0.00, +0.74], dCER +0.10 [+0.00, +0.30] |

**No paired interval excludes zero, and the 6-bit encoder is nominally the best
of the three.** The 24B model's clean 8-bit threshold does not transfer.

These intervals are also what the passage's resolution actually is, and it is
worth writing down: a single build's WER is 4.27 % **[1.90, 7.43]**, i.e. ±3
points. Comparing two builds paired on the same blocks is far sharper (±0.3),
because the shared difficulty of the material cancels — but a lone score from
this passage carries an error bar three points wide, and half-point differences
between builds are below what it can see.

422 words therefore cannot settle the encoder question, so the same question
went to 26 619 words of hard German — the Swiss-German interview, a podcast, and
the pathological window — with no reference at all
(`docs/scripts/encoder_diff.py`). Both builds transcribe identical fixed
windows; every difference between the two transcripts is classified by character
similarity, on the theory that a build which only *spells* differently produces
near-identical spans:

| vs the bf16 build | 8-bit encoder | 6-bit encoder |
|---|---:|---:|
| differing spans (unique) | 104 (88) | 139 (109) |
| of those acoustic (different word / omission) | 58 = 56.9 % | 77 = 56.2 % |
| — of which omissions rather than another word | 45 | 63 |
| acoustic per 1000 words | 2.18 | 2.89 |
| total words against bf16 | +7 | +27 |
| omission balance | −8 words | +8 words |

**The differences are not merely orthographic.** Roughly 57 % of them are a
different word heard, or a word one build emitted and the other did not — and
the divergence scales with the bit reduction, 104 spans at 8 bit against 139 at
6. The encoder's precision does something acoustic.

**Which direction it cuts is not established.** The diff is symmetric: bf16 is
the reference point, not the truth. Omissions come out balanced both ways (−8
and +8 words), the lower-precision builds emit slightly *more* text rather than
dropping speech, and where ground truth does exist — the hand-corrected passage
— the 6-bit build scored best. Individual cases point both ways: the 8-bit
encoder produced "buttercroissants" where the bf16 one produced "buttercrosons".

**Hence the shipped build keeps bf16.** Not because it measurably hears better —
it does not — but because the encoder's precision demonstrably changes what is
heard while the direction is unknown, and 0.5 GB is a cheap price for holding
the highest precision until that is resolved.

**And it stays unresolved, deliberately.** Resolving it would mean listening to
the ~109 localised disagreements and judging each by ear. That is the cheapest
route there is — far cheaper than a second hand-corrected reference passage —
and it is still an evening of someone's attention for a question whose answer
changes nothing: bf16 is kept either way, because it is the safe side of an
open question and costs 0.5 GB. Do not re-propose it.

## Clean audio and hard audio rank the models differently

Read by word error rate the ranking inverts between the two sets — the 24B model
wins decisively on clean audio (2.82 % against 4.81 % on FLEURS), the 3B model
on the hard passage (4.27 % against 7.82 %, that 24B figure being the best 24B
configuration found: encoder bf16, `lm_head` 4 bit). Both tables are below,
under *Against Whisper*; this is the one place they have to be read together.

That inversion is an artefact of the word metric — a statement about spelling
read as one about hearing. The next chapter separates the two.

**Weight of evidence:** the FLEURS numbers rest on 25 minutes and 100
recordings; the podcast numbers on a single two-minute passage of 422 words and
2034 characters. One character is 0.049 CER points, one word 0.237 WER points,
and the bootstrap above puts a single build's error bar at ±3 WER points. Only
the paired comparisons on this passage are sharp enough to carry weight.

## 24B against 3B: one hears better, the other writes better

Sweeping encoder (4/6/8/bf16) and `lm_head` (4/6/8/bf16) puts every 24B
configuration between **7.8 % and 10.2 %** WER on the hard passage, against
4.27 % for the 3B build. Read that way there is a wall, and the sweep says
nothing about where it comes from: seventeen builds, no gradient.

Ranked by CER the same seventeen builds separate cleanly into two groups, and
the 24B model is the better half:

| Rank | Build | CER | WER | Char. errors | Word errors |
|---|---|---:|---:|---:|---:|
| 1–2 | small, encoder bf16, `lm_head` 4 bit | **2.11 %** | 7.82 % | 43 | 33 |
| 3–4 | small, encoder ≥ 8 bit, `lm_head` 6 bit | 2.41 % | 8.29 % | 49 | 35 |
| 5–7 | small, encoder ≥ 8 bit, `lm_head` 8/bf16 | 2.46 % | 8.29 % | 50 | 35 |
| 8 | mini 8 bit uniform | 3.34 % | 4.74 % | 68 | 20 |
| 9–13 | **mini 8 bit, encoder bf16** (shipped), dense, bf16 | 3.39 % | 4.27 % | 69 | 18 |
| 14 | mini 8 bit, `lm_head` 6 bit | 3.83 % | 4.98 % | 78 | 21 |
| 15–17 | small, encoder < 8 bit | 4.03–4.87 % | 9.48–10.19 % | 82–99 | 40–43 |

**Every 24B build with an 8-bit-or-better encoder beats every 3B build**, with a
gap of 0.88 CER points between the two groups as ranked here — far outside the
noise floor. The
24B model also has the lowest CER of anything measured here, Whisper included.

The clearest single statistic is **characters per word error**: 3.8 for the 3B
build, 1.3 for the 24B one. The 24B model's errors are on average a third the
size of the 3B model's. It misspells; the 3B model mishears.

Note which way that measurement is biased. CER runs on space-free text, so the
24B model's eight insertions — the verbatim false starts and filler the
hand-corrected reference smooths away — are charged against it at full character
length. It wins anyway.

In short: **the 24B model hears this passage distinctly better, and the 3B model
writes it down better.** Choosing the 3B build is right on readability and on
6.6x versus 1.8x realtime — not on recognition accuracy.

What the sweep establishes about precision is narrow: **on the 24B model the
audio encoder is the only part whose bit width matters, and its threshold is 8
bit.** By CER the encoder row is cleanly monotone (4.87 → 4.28 → 2.46 → 2.41);
the 6→8 step is 37 characters, the 8→bf16 step is one. Every other column in the
sweep is noise. None of this carries over to the 3B model, which shows no such
threshold.

The 8-bit 24B build needs ~27 GB of weights and runs at 0.80x realtime — slower than the
recording. Not usable on 32 GB. The 4-bit build is a different matter: see
"Running the 24B model on 32 GB" below.

### The re-run on four passages narrows "hears better" to clean audio

Everything above rests on the one hard passage. The 2026-09 re-run put the
shipped 4-bit 24B build (`lm_head` 4 bit) against the 3B build on four passages
of German conversation and on FLEURS, under both log-Mel floors (WER / CER in %):

| Passage | Mini, max floor | Small, max floor | Mini, percentile | Small, percentile |
|---|---:|---:|---:|---:|
| hard podcast, 120 s | 4.27 / 3.39 | 7.82 / **2.26** | 5.45 / 3.29 | 8.77 / 3.88 |
| video call, 300 s | 1.98 / 1.09 | 4.66 / 3.68 | 2.68 / 1.68 | 3.49 / 2.59 |
| podcast A, 360 s | 3.63 / 1.88 | 15.41 / 13.02 | 2.81 / 1.66 | 7.62 / 5.35 |
| podcast B, 360 s | 3.53 / 1.95 | 5.64 / 3.03 | 3.53 / 1.95 | 5.44 / 2.69 |
| FLEURS de, 100 recordings | 4.81 / 1.44 | 2.99 / 0.88 | 4.89 / 1.50 | **2.78 / 0.75** |

The hard passage under the max floor is the only conversational cell where the
24B build has the lower character error — the finding this chapter was built
on. On the other three passages it is behind under either floor, and on
podcast A it loops: it writes one fifteen-word sentence twice (percentile
floor) or repeats itself for 122 inserted words (max floor). On a further
podcast window, decoded raw, it ran into "Jetzt. Jetzt. Jetzt. …" after three
minutes under both floors and never came back, while the 3B build transcribed
the same ten minutes in full. Through the engine the loop is caught. With
word timestamps the prefix salvage kept the first 103 s and re-decoded the
rest (1970 words, 7.3 minutes for ten of audio); without them the ladder
retried at temperature 0.2, then split the pass twice, down to 150-s pieces
(1972 words, sixteen minutes). The 3B build wrote 1974 words. The 24B model loops more
readily on this material than the 3B one, and the loops cost it its speed.

Two cautions on the table. The video call's reference was corrected from the 3B
build's own draft and flatters it; the two podcast passages were corrected by
hand for the speaker study, and how neutral their text is toward either model
was not recorded. Neither caution explains a looped sentence.

So the distinction the chapter title draws holds for **clean** audio, where the
24B build is clearly ahead (FLEURS, above, and the published 2.61 % for the
unquantised model). For conversation it is not the better choice.

## Against Whisper

Same passage, same metric, noScribe's own Whisper settings:

| Model | WER | CER | Sub | Del | **Ins** | Speed |
|---|---:|---:|---:|---:|---:|---:|
| **voxtral-mini-8bit** | **4.27 %** | 3.39 % | 10 | 8 | **0** | 6.79x |
| voxtral-small-4bit (best) | 7.82 % | 2.11 % | 20 | 5 | 8 | 1.77x |
| whisper-fast | 8.06 % | 3.34 % | 22 | 4 | 8 | 2.43x |
| whisper-precise | 14.22 % | 8.75 % | 22 | 4 | **34** | 2.65x |

The error *profiles* differ more than the totals:

- **Voxtral omits.** Eight deletions, zero insertions. It dropped a whole
  parenthetical ("bitte höre das jetzt in Anführungszeichen") but invented
  nothing.
- **Whisper invents.** `precise` transcribed that parenthetical correctly, and
  got "Tohuwabohu" and "Hokuspokus" right where Voxtral failed — then
  hallucinated 27 consecutive words of nonsense at the end of the clip,
  including a product that does not exist ("Omega 2"). Its loop detector fired
  at every temperature step.

For interview and podcast work this favours Voxtral: an omission is visible
when proof-reading, a fluent hallucination is not. Note also that the larger
Whisper model is the *worse* one here — `precise` loops where `fast` does not.

Two more differences show up over a whole recording rather than a passage.
Voxtral **punctuates more densely** — 10.51 commas per 100 words against
Whisper's 9.76 on the 20-minute podcast, which is what makes its transcripts
read better even where the word error is comparable — and it **gets brand and
programme names right where Whisper does not**: on one product name, Whisper
wrote none of its three occurrences correctly, the 3B build three of four and
the 24B build four of four. One name in one recording is a direction, not a
rate; the general fix for names is the correction list (`VOXTRAL.md`), because
no decoding parameter mends them.

The CER column ranks these four differently from the WER column, and the gap
between `whisper-fast` and the 24B Voxtral build is where it matters most: near
identical word error (8.06 % vs 7.82 %), but 3.34 % against 2.11 % in
characters. Those two transcripts are not of comparable quality; one of them is
mostly misspelling what it heard correctly.

### The same four models on clean audio

| FLEURS German (100 recordings, 25 min) | WER | CER | Speed |
|---|---:|---:|---:|
| voxtral-small-4bit | **2.82 %** | 0.76 % | 2.03x |
| whisper-precise | 4.05 % | 1.40 % | 4.11x |
| whisper-fast | 4.05 % | 1.40 % | 4.01x |
| voxtral-mini-8bit | 4.81 % | 1.44 % | 7.64x |

The interesting question the two tables answer together is **how far each model
falls when the audio gets hard**. By word error the 3B model appears to get
*better* on hard audio, which is not a thing that happens — a good sign the word
metric is measuring something other than recognition. By character error the
picture is ordinary:

Divide the hard-passage CER by the clean one, model by model: 2.4x for
`voxtral-mini-8bit`, 2.8x for `voxtral-small-4bit`, 2.4x for `whisper-fast` —
and **6.3x for `whisper-precise`**.

Hard audio costs every sane model roughly 2.5x in character error, and the
ranking from the clean set survives into the hard one. The outlier is
`whisper-precise`, which does not degrade but breaks: its 6.3x is hallucination,
a different failure altogether. That — not a reversal of model quality — is the
finding that should drive the engine choice for interview work.

(The two Whisper builds scoring identically is not a measurement error: they
are different models — 819 MB and 1618 MB — but produce byte-identical output
on all 15 clips checked. On read-aloud speech both saturate; the larger model
can only show a difference on difficult audio, where its failure mode is
hallucination.)

## Reproducing

```bash
# the recommended build
python tools/quantize_voxtral.py mistralai/Voxtral-Mini-3B-2507 \
    models/voxtral-mini-8bit 8 64 dense-encoder --lm-head-bits 8

# score it (also leaves the transcripts in /tmp/wer_<build>.txt)
python docs/scripts/wer.py Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav models/voxtral-mini-8bit
python docs/scripts/fleurs.py 100 models/voxtral-mini-8bit whisper:precise

# the 24B source is mistralai/Voxtral-Small-24B-2507: 11 shards, 48.5 GB
# (its duplicate consolidated.safetensors is skipped)

# the encoder variants the next two commands compare against
python tools/quantize_voxtral.py mistralai/Voxtral-Mini-3B-2507 \
    models/voxtral-mini-8bit-enc<N> 8 64 dense-encoder \
    --lm-head-bits 8 --encoder-bits <N>

# how much of a build-to-build difference the passage can actually resolve
python docs/scripts/bootstrap_cer.py Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    voxtral-mini-8bit voxtral-mini-8bit-enc6

# compare two builds over hours of audio, no reference needed
python docs/scripts/encoder_diff.py models/voxtral-mini-8bit \
    models/voxtral-mini-8bit-enc6 --audio <file> [file ...]
```

Scoring a rival engine is a different exercise with its own harness; the
commands are in [other-asr-engines.md](other-asr-engines.md), the scripts
under [`docs/scripts/engines/`](scripts/engines/README.md).

Each build is ~20 seconds to make and 5–21 GB on disk. On macOS, remember that
hourly Time Machine snapshots keep deleted builds alive: reclaim with
`tmutil thinlocalsnapshots / <bytes> 4`.

`encoder_diff.py` writes its raw event list outside the repository
(`NOSCRIBE_MESS_DIR`, otherwise the temp directory): it quotes the audio
verbatim, and the audio is real interview material. Only the aggregate counts
belong in `docs/`.

## Decode path and memory (what moves peak, what doesn't)

Decoding runs through Apple's maintained `mlx_lm.generate_step` (a thin adapter
bridges our LM), which chunks the audio prompt instead of one forward. That is
byte-identical to the reference decode but ~7% faster and ~18% lower peak on long
passes. Four memory levers were then measured and mostly rejected:

- **Peak is set during *prefill*** (processing the audio prompt), not generation:
  prefill+5 tokens already peaks at 7.7 GB vs 8.7 GB for a full 600 s pass. The
  floor is model weights + full KV cache, both irreducible for a bidirectional,
  full-context model.
- **`prefill_step_size` is not a lever** — 512/1024/2048 give an identical peak.
- **KV-cache quantization makes it *worse*, not better.** Because our "context" is
  a huge audio prompt, quantizing the cache adds a dequant transient during
  prefill attention that scales with prompt length and exceeds the
  generation-phase saving (16-bit slope 4.3 MB/s vs 8-bit 5.7 MB/s). It wins only
  for short-prompt/long-generation (chat), the opposite of transcription.
- **KV-cache *size* cannot unlock the 8-bit 24B model on 32 GB either.**
  small-8bit is 25 GB of weights; after weights + headroom there is ~0 GB left
  for the cache on 32 GB. The weights are the wall. What *does* fit is
  small-4bit — see the next chapter.

Net: `MEM_MODEL["mini8"]` was recalibrated to the generate_step path. The
entry, the 180–1200 s sweep behind it and the reason for its safety margin are
in the comment above `MEM_MODEL` in `noScribe/voxtral_engine.py` — the one place
that has to stay right, since the auto-chunker reads it.

## Running the 24B model on 32 GB

The 8-bit 24B build is out of reach on 32 GB, and one with a 6-bit body fits
only with short passes (22.5 GB peak, below). The 4-bit one fits with room to
spare, and it is the configuration the measurements point to anyway, so it is
the 24B build this app ships: **`voxtral-small-4bit`**, a 4-bit language model and
`lm_head` with a bf16 audio encoder.

The bit sweep, read by CER, says the encoder's precision matters up to 8 bit
and `lm_head`'s does not. It says nothing about the language-model body, which
it held at 4 bit throughout. That was measured separately in 2026-09: the same
build with a **6-bit body** (encoder bf16, `lm_head` 4 bit, 20.9 GB), both
decoding identical 30-s windows of the four hand-corrected passages, paired:

| | 4-bit body | 6-bit body | 4 − 6, paired |
|---|---:|---:|---|
| hard podcast | 7.58 / 2.41 | 7.58 / 2.46 | dWER +0.00 [−0.71, +0.68] |
| video call | 6.75 / 5.53 | 4.77 / 3.38 | dWER +1.98 [−0.81, +6.63] |
| podcast A | 4.99 / 2.49 | 4.99 / 2.41 | dWER +0.00 [−0.45, +0.45] |
| podcast B | 5.73 / 2.93 | 6.02 / 3.13 | dWER −0.29 [−1.23, +0.58] |
| all four pooled | | | dWER +0.41 [−0.41, +1.69], dCER +0.49 [−0.31, +1.80] |
| FLEURS de, 100 | 2.78 / 0.75 | 2.32 / 0.78 | (CER unmoved) |
| speed / peak | 2.2x / 16.6 GB | 1.7x / 22.5 GB | |

No difference is demonstrable. The video-call gap is one event, a 17-word
phrase the 4-bit build invented at a window edge; FLEURS moves the word metric
but not the character one, i.e. spelling. So the body shows no gradient from
4 to 6 bit, and the 8-bit body was not measured: it needs ~34 GB for a long
pass (short windows would fit), and with no step from 4 to 6 there was no
reason to expect one from 6 to 8.
`lm_head` at 4 bit is
not worse than at 8 in the 2026-09 re-run either (hard 8.77 / 3.88 against
9.00 / 3.98, video call 3.49 / 2.59 against 4.42 / 3.31, neither difference
demonstrable), and it is the smaller, faster choice. The build weighs 15 GB,
4.82 bits per weight on average.

Its peak memory was calibrated on the `generate_step` path in fresh processes:
16.65 / 17.25 / 18.53 / 19.99 / 21.36 GB at 180 / 300 / 600 / 900 / 1200 s of a
podcast, 17.35 / 18.99 / 22.54 GB at 300 / 600 / 1200 s of an interview — a
line of 15.71 GB + 0.0051 GB/s. The old one-shot-prefill entry (15.2 + 0.017)
over-predicted a ten-minute pass by 7 GB. The new `MEM_MODEL["small"]` entry and
its margin are explained in the comment above it. What `_auto_chunk_sec()` now
allows:

| RAM | mini-8bit | small-4bit |
|---|---:|---:|
| 16 GB | ~7 min | refused |
| 24 GB | 10 min | ~70 s (up to ~210 s pinned) |
| 32 GB | 10 min | **10 min** (the 600 s cap binds) |

So the 24B model runs on a 32 GB Mac in ten-minute passes at ~2x realtime — a
four-hour recording in about two hours. Before choosing it, know that it is the
better model for clean audio and not for conversation, where it loops more
readily (chapter above).

One more observation from those overnight runs, not yet explained: three 24B
runs between 02:40 and 03:50 died with a Metal `GPU Timeout Error` while
3B runs in the same hour finished, but at half their usual speed. The same 24B
decodes went through before and after that window. Something else was holding
the GPU; the 24B model, whose steps take longer, is the one that ran into the
watchdog.

Reproduce the build with:

```bash
python tools/quantize_voxtral.py mistralai/Voxtral-Small-24B-2507 \
    models/voxtral-small-4bit 4 64 dense-encoder --lm-head-bits 4
```

## Other options considered

Six rival ASR engines have been measured against the shipped build — Parakeet,
Qwen3-ASR, VibeVoice, Cohere Transcribe, transcribe.cpp's own Voxtral and the
streaming Realtime model — and none replaced it. That is a different question
from which Voxtral build to ship, with a different harness and a different
cadence, so it has its own file:
[other-asr-engines.md](other-asr-engines.md). The short version, because it
bears on everything above: on FLEURS the field is separated by tenths of a point
and this build is mid-table; on hand-corrected German conversation the same
field spreads over an order of magnitude and the order reverses.

Two moves that are about *this* build rather than another model stay here.

### Re-quantising with another tool changes nothing

**Re-quantising changes the file format and nothing else.** MLX's affine
quantisation is data-free — scale and bias come from each group's own min and
max, there is no calibration set and no randomness. Quantising the same tensor
twice returns bit-identical results, and so does quantising a fresh copy of it.
mlx-audio's quantisation predicate is `not p.startswith("audio_tower")`, which
selects exactly the set these builds already carry: **213 modules (the 212
`language_model` linears plus `lm_head`), all at 8 bit, group size 64, affine,
encoder left dense**. Same weights in, same layer set,
same parameters, therefore the same tensors out. Only the keys change, from
`language_model.layers.0…` to `language_model.model.layers.0…`.

So a re-quantised build would be *the same build*, and the tables above would
still describe it. **Quality and speed would not move because of the
re-quantisation** — quantised matmuls are the same kernels either way. What
could move them is everything around it, and that is the migration's problem,
not this document's: `docs/migration-mlx-audio.md` carries the risks, the steps
and the acceptance criteria.

### The aligner and the Viterbi moved, and neither is a build question

The forced aligner now runs its emissions on the GPU and its Viterbi DP in numpy
(`noScribe/ctc_align.py`). Both changed timings this document once quoted, and
both are written up where they belong, with one set of numbers each:
[viterbi-numpy-brief.md](viterbi-numpy-brief.md). Nothing about either changes
which build to ship.

## What is still open

- **Which encoder precision is better on the 3B model — closed as not worth
  answering.** Measured: the bit width changes what is heard, at ~2-3 acoustic
  differences per 1000 words, scaling with the reduction. The direction is
  unknown and will stay that way; the only route to it is listening to the
  disagreements one by one, and the answer would not change the build. bf16
  stays.
- **Why is the 24B model's encoder so much more sensitive?** Its sweep held the
  language model at 4 bit while the 3B sweep ran at 8. A coarsely quantised body
  may amplify encoder noise instead of averaging it out. **Not answerable on a
  32 GB machine:** the clean counter-test is the 24B encoder sweep repeated with
  an 8-bit body, and small-8bit is 25 GB of weights, which `_auto_chunk_sec()`
  refuses outright. It needs a bigger machine, and it would only matter if mixed
  builds are ever laid out differently.
- **`MEM_MODEL` for the local-only 24B builds is stale.** `small` was
  recalibrated with the shipped build; `small6` and `small8` still hold
  one-shot-prefill numbers, which err on the safe side, and nobody ships them.
- Most of the hard-audio side of this document rests on one 422-word,
  2034-character passage. That is a deliberate decision, not an oversight — but
  it is why differences under ~0.15 CER points are treated as noise throughout.
  A second hand-corrected reference (859 words, five minutes, a video call)
  exists and is used for engine comparisons — the Voxtral build scores **1.98 %
  WER / 1.09 % CER** on it. That reference was made by correcting this build's
  own draft, so it flatters Voxtral by construction; an earlier 0.81 % from it
  was a self-comparison and should not be quoted (see
  `other-asr-engines.md`). The 2026-09 re-run scored the shipped builds and
  the 3B bit formats on it; the older sweeps were not repeated.
- Voxtral drops parentheticals. The *audio* side of that is measured and closed —
  `docs/voxtral-audio-preprocessing.md` finds no leveller that helps on real
  material; whether it is steerable at the decode is untested.
- The measurements are tied to this MLX version and macOS release; re-run the
  scripts after an upgrade before trusting the memory model.
