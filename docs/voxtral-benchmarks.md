# Voxtral in noScribe — Measurement Results

As of 19/20 July 2026, updated in August · M1 Max, 32 GB · test material:
"Mona Podcast 323" (20.4 min, German, 2 speakers)

**What lives in the code is no longer repeated here.** The loop-detection
thresholds, the repair ladder, the memory model, the lost-head repair and the
name correction carry their measurement in the comment next to the constant,
as this repository has it. This file keeps what has no place there:
measurements behind decisions that were *not* taken, evidence that a defect is
not ours, and measurement traps. Quantisation has its own document,
`voxtral-quantisation.md`.

The reference measure for readability is comma density per 100 words; on this
material Whisper sits at **9.58–10.62** (depending on the excerpt), which
counts as "readable".

---

## 1. The most important finding: `repetition_penalty`

`mlx_voxtral.generate()` defaults the value to **1.2** — a chat default we had
never overridden. Why that costs precisely the punctuation in spoken language
is explained in the comment above the call in
`voxtral_engine.transcribe_array`; the rungs of the penalty ladder are at
`RETRY_REPETITION_PENALTIES`. What remains here are the two measurement series
those comments draw their numbers from.

| `repetition_penalty` | commas/100 w (mini, 600 s) | doubled "nicht" |
|---|---|---|
| **1.0** | **10.87** | kept |
| 1.01 | 10.17 | kept |
| 1.05 | 10.01 | kept |
| 1.1 | 9.60 | **swallowed** |
| 1.15 | 9.21 | **swallowed** |
| 1.2 (old default) | 8.57 | swallowed |

On the full podcast through the real pipeline:

| | words | commas/100 w | sentence ends/100 w |
|---|---|---|---|
| Whisper (reference) | 4069 | 9.76 | 9.58 |
| mini **old** (1.2) | 3954 | 7.89 | 7.54 |
| mini **new** (1.0) | 4054 | **10.51** | 9.25 |

This second table is where the figure "27 % of all commas" comes from: 3954
words at 7.89 are 312 commas, 4054 at 10.51 are 426.

**There is no documented default.** Neither the Voxtral paper nor the model
card names one; Mistral's reference call is
`TranscriptionRequest(model, audio, language, temperature=0.0)` with no
penalty. The 1.1 in `mzbac/mlx.voxtral` sits in an illustrative example. The
literature explains the conflict: the penalty acts on *all* tokens alike and
trades repetition suppression for grammatical fluency.

**Other parameters:** at `temperature=0.0` (greedy), `top_p`/`top_k`/`min_p`
have no effect — which is why the repair ladder perturbs the decode through the
temperature and not through them. `logit_bias` works as genuine hotword
steering (token `'y'` +2 corrected "Mohnas" → "Mona's" with no measurable
collateral damage), but it is context-free and tokenisation-dependent —
`Markus` is a single token, `Mona` splits into `[' Mon','a']`. What was built
instead is the correction list, which matches whole phrases.

---

## 2. `max_new_tokens` — latent data loss

German speech produces **4.64 text tokens/s** (measured; that is where the
factor 20 in the token budget comes from). The fixed limit of 4096 would have
truncated the end of every pass over ~15 min:

| Pass | needed | old limit 4096 |
|---|---|---|
| 600 s | 2782 | ok |
| 1006 s | ~4664 | **truncates** |
| 1500 s | ~6955 | truncates |

The two long rows are historical: since `TRUSTED_CHUNK_SEC`, the automatic
chunking caps every pass at 600 s, because Voxtral is not measured beyond that.

The limit is purely a loop brake — it only bounds the generation loop; the
KVCache grows from the tokens *actually* produced. A high value reserves
**nothing**. Now `min(32768, duration*20 + 512)` = 4.3× headroom. (The greedy
path now runs through `mlx_lm.generate_step`, where the same argument applies;
only the penalty rung still goes through `mlx_voxtral.generate`.) The context
window of ~32k is not binding: audio costs 12.6 tokens/s, so a 600-s pass is
~7.6k + ~2.8k tokens.

---

## 3. Language

`language=None` (noScribe's "Auto") inserts **no** `lang:` token — genuine
auto-detection, no hidden English default.

**Caution:** a *wrong* language setting makes Voxtral **translate** instead of
transcribe — `language='en'` on German audio yields fluent English. That is why
the language is only passed on when it was chosen explicitly.

Auto and `de` are nevertheless not interchangeable even on clearly German
audio: §5 shows where the difference surfaces — there the `de` setting once
costs the first ~150 words, and once the auto-detection costs the lead-in.

*The residual risk that was open at the time* — a single pass flips and
translates itself — has since been guarded against, and it took neither a
Whisper load nor a new dependency: a stopword and script detector determines
the language from the *text*, two chunks have to agree, and a translated pass
becomes a reason for the repair ladder. The mechanism and its field
measurements are at `_detect_language` / `_file_language` / `want_lang` in
`voxtral_engine.py` and in the tests for them.

---

## 4. Repetition loops

Without a penalty, small tipped into a loop on a 410-s pass: **4099 identical
words**. That was the trigger; detection thresholds, calibration corpus and
the order of repair have changed several times since and live where they
apply — at `DEGENERATE_CYCLE_REPEATS` and in `_transcribe_guarded`.

What stays here, because it justifies a decision that was *not* taken: a
penalty cannot tell a loop from meaning. 1.1 turned "wir können es dir **nicht
nicht** erzählen" ("we can't *not* tell you") into "… nicht erzählen" — the
sentence flips into its opposite. A fluently readable but wrong sentence is
more dangerous than visibly broken text. That is why the penalty sits at the
*end* of the ladder and not at the start, and why splitting comes first: as
2×205 s the 410-s pass produced clean text, including the doubled "nicht" and
the genuine fourfold "Jetzt. Jetzt. Jetzt. Jetzt."

---

## 5. The lost start of a pass

A long pass sometimes comes back **without its first seconds of speech** — no
loop, no wrong language, no trace in the log. The defect, its frequency and
the repair (`_recover_lost_head`) are described in the code. Here are the three
things that have no place there.

**First: it depends on the end of the window, not the start.** Fine sweep on
Auto over a 20-minute video whose first 2.4 s hold a remark before the take;
the start of the audio is bit-identical in all seven runs, only the end moves:

| Window | 1180 | 1195 | 1205 | 1215 | 1220 | 1223 | 1226 |
|---|---|---|---|---|---|---|---|
| Start | there | **gone** | there | there | there | there | **gone** |

Not monotonic, no threshold — the same signature as the language flip.

**Second: it is not ours.** That refutes the obvious remedies: shorter windows
(1195 loses, 1223 does not), pinning the language (flips in both directions —
once `lang:de` cost the first ~150 words of a 1426-s window), prepending
silence (0 of 6, and in one test it even triggered the loss). Reference-exact
audio features change nothing either, and our prompt tokens are bit-identical
with `mistral_common`.

**Third, the measurement trap:** FLEURS is no use for this question. There the
start is clean at 120/300/450/600 s — but also at 1200 s, where it should not
be. The positive control fails, so the clean rows say nothing. Anyone
re-measuring this needs material that demonstrably shows the defect; a raw
Zoom recording (4.8 h) does so reliably — on it, 5 of 64 window runs (32
windows of 300 s and 600 s, each once on Auto and once on `de`) lose their
start, in the worst case 18 words of continuous speech. That is the figure
`VOXTRAL.md` cites; the breakdown is in the comment on `_recover_lost_head`.

---

## 6. What rules out 8-bit small on 32 GB (checked, rejected)

The base is the problem: **26.4 GB for the weights alone**. On top of that a
second, independent knock-out: **0.80× real time**, i.e. slower than the
recording. The ways out are measured through and ticked off in
`voxtral-quantisation.md`; two calculations from here still stand because
they explain *why* they came out that way.

**The KV cache is much smaller than the slope.** Computed from the architecture
(`layers × 2 × KV heads × head_dim × tokens`, audio costs 12.6 tokens/s):

| | KV cache (bf16) | measured total slope |
|---|---|---|
| small, 300 s | 0.62 GB | 4.05 GB |
| small, 1500 s | 3.10 GB | 20.3 GB |
| mini, 600 s | 0.93 GB | 6.2 GB |

The KV cache accounts for only **~15 %** of the slope; the rest goes to the
audio encoder. KV quantisation can therefore hardly lower the peak — the later
direct measurement even shows that it *raises* it.

**Do longer passes bring any quality at all?** Measured: single 600-s pass
10.87 commas/100 w vs. 2×300 s 10.66 — practically equal. The literature
supports this: there is an *optimum* chunk length that depends on the training
(Whisper 30 s, Distil-Whisper 15 s), not "the longer the better". The real
problem is the **boundaries** (cut-off sentence references, speaker changes at
the edge) — pause-aligned cuts plus overlap already work against that. That is
the evidence that the 600-s cap costs nothing.

---

## 7. What became of the open points

All five are decided; the list stays so that nobody opens them a second time.

- **mini 8-bit as default** — built and registered as `voxtral-mini-8bit`, now
  the recommendation.
- **small 8-bit** — built and registered as `voxtral-small-8bit`. The question
  "does it run on 32 GB?" is answered: no, it needs ~34 GB for its shortest
  pass and is rejected before starting below that. Since replaced by
  `voxtral-small-4bit`, which does (`voxtral-quantisation.md`).
- **Language pinning across several passes** — built, see §3. Remaining gap:
  chunks that were already written before the file language was settled only
  get a warning with their number.
- **Voxtral-Mini-4B-Realtime** — **rejected**, on Mistral's own FLEURS figures:
  German 6.19 % WER at 480 ms, 4.15 % even at 2.4 s delay, against 3.54 % for
  the offline Mini 3B — a smaller model beats it. Streaming trades lookahead
  for latency, and latency is worthless in a file transcription.
- **`voxtral-mini-2602` ("Transcribe 2")** — still **API-only** and therefore
  unsuitable for confidential interviews.

---

## 8. Reproducing

The scripts are in `docs/scripts/`: `bitmatrix.py` (speed/word fidelity),
`decisive.py` (punctuation per penalty), `rep_matrix.py` (penalty rungs against
the doubled "nicht"), `split_retry.py` (loop repair), `cli_check.py` (format
check). The conversion is in the repository as `tools/quantize_voxtral.py`.
