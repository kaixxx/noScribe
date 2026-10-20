# The Clamp Floor of the log-Mel

As of 24 August 2026 · M1 Max · build `voxtral-mini-8bit`, mlx-voxtral 0.0.6.
All figures measured, none estimated.

The log-Mel spectrogram is clamped from below before the model sees it. Where
this floor lies, the library decided from the **maximum of the whole input** —
so that a single loud cell sets it for the entire pass. On real material that
measurably costs recognition quality, and since 23 August 2026 the engine
takes a **percentile** instead. This file is the measurement record for that:
what the old floor costs, what the new one costs, what argues against it, and
where its limit lies.

The mechanism itself is in the code, at the constant block above
`MEL_FLOOR_PERCENTILE` in `noScribe/voxtral_engine.py`; the implementation
details are in the docstrings of `clamp_log_mel` and `_PercentileFloorFeatures`
and in `tests/test_mel_floor.py`. Here is what has no place there.

---

## 1. Why the floor moves

The floor lies at `log_max − 8`, where `log_max` is the maximum over the whole
input. A 0.1-s slam raises it by **9.6 dB** and changes **100 %** of the frames
in the untouched audio before it.

**Which real noises produce which dose** (same peak amplitude, quiet
recording): clapping 10.3 dB, noise burst 17.3 dB, **low-frequency door slam
27.1 dB**, square block 31.5 dB. The most expensive case is the most everyday
one.

**The outlier is not a lab artefact.** Distance between the loudest Mel cell
and the 99.9th percentile of the same spectrogram, measured without a model
(`mel_outlier_gap.py`): raw interview 9.4 and 9.1 dB, raw podcast 13.2 dB,
mastered podcast 9.3 dB, unprocessed raw track **21.9 dB**, `zoom` 14.4 dB,
VoxPopuli streams 13.0 dB at the median. In **every** recording the floor is
set 7–22 dB higher than a robust measure would set it, most strongly on raw
material — i.e. exactly what users bring.

## 2. What the max floor costs

**The solid figure stands on streams.** Ten VoxPopuli streams of 300 s each,
each once clean and once with an inserted transient, paired per floor
(`voxpopuli_spike.py`):

| Floor rise (median) | max floor | percentile 99.9 |
|---|---|---|
| 5.2 dB | −0.07 [−0.29, +0.15] | +0.04 [+0.00, +0.09] |
| 16.2 dB | **+0.35 [+0.03, +0.66]** \* | +0.03 [+0.00, +0.07] |
| 28.2 dB | **+1.52 [+0.78, +2.26]** \* | +0.04 [+0.00, +0.10] |

\* = interval excludes zero. Monotonic, demonstrable at the two upper doses,
the percentile floor flat at all three. **+1.52 at 28 dB is the figure that
carries.**

The passage measurements before that said +1.66 to **+5.21** (`hart`,
attenuated, slam constant over +2.6 / +14.7 / +27.1 dB) and once +11.99 — they
**overestimate by four- to eightfold**. Under greedy decoding one token flips
on a single passage and the rest follows; the stream measurement averages that
out. For the same reason, the observation that the damage was "exactly the
same size" at three slam positions was no evidence of generality, only of
determinism — mechanically it is still correct, since the floor rises
independently of position.

## 3. The percentile floor: what it costs and which percentile

**On clean material: nothing demonstrable.** Measured on **VoxPopuli de** —
spontaneous parliamentary speech with a gold transcript, the corpus we lacked
next to the read-aloud FLEURS, and at 11.4 % WER markedly harder — 10 streams
of 300 s, 50.6 min, paired on bit-identical audio (`voxpopuli_floor.py`). Sweep
over the candidates, all against the same max baseline: 99 → **−0.12** [−0.85,
+0.49], 99.9 → −0.10 [−0.83, +0.47], 99.99 → −0.25 [−0.99, +0.31]. All three
indistinguishable from the maximum and from each other.

**For robustness the percentile very much is a choice.** The slam's cells all
sit at the top of the distribution and displace the order statistic by their
count — the more so, the shorter the pass and the more broadband the slam.
Floor shift in dB on `tests/data/interview.mp3` (levelled at −20 dB, transient
at full scale, without a model):

| Pass | Transient | max | 99 | 99.9 | 99.99 |
|---|---|---|---|---|---|
| 60 s | square block 100 ms | 29.6 | 0.15 | 0.50 | 4.38 |
| 60 s | door slam 60 Hz 100 ms | 25.5 | 0.05 | 0.22 | 1.81 |
| 60 s | **noise burst 100 ms** | 9.7 | **0.75** | **7.10** | 9.20 |
| 60 s | clapping 5 ms | 4.2 | 0.13 | 0.45 | 2.03 |
| 120 s | square block 100 ms | 29.6 | 0.07 | 0.27 | 1.45 |
| 120 s | noise burst 100 ms | 9.9 | 0.33 | 2.90 | 8.33 |
| 300 s | square block 100 ms | 30.5 | 0.03 | 0.12 | 0.54 |
| 300 s | noise burst 100 ms | 10.0 | 0.14 | 0.93 | 7.57 |

The noise burst occupies around 1300 cells above the threshold; the top
per-mille of a 60-s pass is 768 cells, the top percent 7680. **99.99 is thus
out entirely, and 99 beats 99.9 by an order of magnitude** — the two
statements "uncritical" and "very much a choice" do not contradict each other,
they apply to different questions: the cost on clean material is not, the
robustness is.

**The protection is bounded in quantity.** Broadband noise that fills more
than the top percent of a pass — roughly 0.6 s in 60 s — counts as signal and
raises the floor again as before.

## 4. What argues against it

**The input leaves the trained value range.** The status quo delivers by
construction **exactly 2.000** units of range. With the percentile floor:
2.182 (`hart`), 2.235 (interview), 2.360 (`zoom`), 2.547 (raw track) — up to
**27 % wider**, downwards. In
[openai/whisper#269](https://github.com/openai/whisper/discussions/269) the
Whisper author warns of exactly this: without this normalisation the input is
*out-of-distribution*. At the widths real material produces it is measurably
without consequence — but it is the reason this could not be built in
unmeasured. With 99 the range becomes somewhat wider than with 99.9
(interview 2.39–2.45 against 2.22–2.27), but stays below the 2.547 the raw
track reaches with 99.9 without consequence.

**Architectural fidelity.** All four reference implementations take the bare
maximum — transformers, mlx-audio (delegates to transformers), transcribe.cpp
(`per_utterance`) and mlx-voxtral 0.0.6. That is what the model was trained
with. And a web search finds no report of this mechanism: there is no outside
experience to lean on.

**Why the intervention nevertheless sits here and not in the signal.** The
maximum is used *exclusively* for the floor — a fixed affine map follows — so
a robust statistic is precisely the right thing there. The alternatives, all
rejected:

* **Limiter on the audio** — attacks the same physics, but pays in the signal
  and is irreversible. `voxtral-audio-preprocessing.md` §4 measured three
  level tools on real material: neutral to harmful.
* **Loudness normalisation** — does not help at all. It multiplies the
  waveform by a constant, in log space a constant addend on *every* cell, the
  outlier included. The slam-to-speech distance remains, the floor still
  hangs on the slam.
* **Fixed floor** — Voxtral Realtime does this (`global_log_mel_max` in
  transcribe.cpp) and would also be chunk-independent, but then forces exactly
  the loudness normalisation just rejected.
* **The 8.0** stays untouched: it corresponds to librosa's `top_db=80` and is
  the range the model was trained on.

## 5. The limit: silent passes, and the rejected cap

All streams above are dense speech; there the 99th percentile lies 19.5–26.8 dB
below the maximum (median 22.0 — on the edited interview only 15.6–17.8). In a
pass that consists mainly of pauses — the 60-s head probe on a recording that
starts with silence — the percentile slides down within the speech cells and
the floor lies lower than ever measured. `voxpopuli_sparse.py` measures that:
the same clips, with white noise at −60 dBFS as room tone in between, until
the speech share is right.

| Speech share | Distance max → p99 | max | percentile 99 | paired |
|---|---|---|---|---|
| ~100 % | 22.0 dB (19.5–26.8) | 11.39 % / 7.32 % | 11.27 % / 7.14 % | −0.12 [−0.85, +0.49] · CER −0.18 [−0.79, +0.28] |
| 50 % | 23.6 dB (20.7–28.9) | 10.14 % / 6.35 % | 10.38 % / 6.67 % | **+0.24 [−0.03, +0.55]** · CER +0.32 [−0.01, +0.77] |
| 20 % | 28.5 dB (26.1–30.0) | 12.30 % / 8.25 % | 12.36 % / 8.54 % | +0.07 [−0.72, +1.10] · CER +0.28 [−0.25, +1.02] |

No interval excludes zero, but at 50 % it misses doing so by a hundredth, and
both CER intervals sit just short of it. The lower floor exposes noise
structure in the pauses that the dense streams never had — the asymmetry at
their limit. The room tone here is synthetic and white, real room tone would
be more low-frequency; and the 20 % run carries only 10 min of speech.

**The cap, measured and rejected (2026-08-24).** The obvious way out is a
floor that never lies more than 20 or 25 dB below the maximum:
`max(p99, log_max − cap) − 8`. `clamp_log_mel` carries a `cap` parameter for
this (default `MEL_FLOOR_CAP = None`), the scripts the arm shorthand `99c20`.
All four regimes, same streams:

| Regime | max | p99 | p99 cap 20 dB | p99 cap 25 dB |
|---|---|---|---|---|
| slam 28 dB, cost clean→slam | **+1.52 [+0.78, +2.26]** \* | +0.01 [−0.03, +0.06] | **+0.81 [+0.18, +1.57]** \* | **+0.75 [+0.10, +1.53]** \* |
| dense clean, against max | — | −0.12 [−0.85, +0.49] | −0.07 [−0.81, +0.54] | −0.10 [−0.83, +0.51] |
| 50 % silent, against max | — | +0.24 [−0.03, +0.55] | **+0.05 [−0.17, +0.37]** | +0.24 [−0.03, +0.55] |
| 20 % silent, against max | — | +0.07 [−0.72, +1.10] | +0.07 [−0.67, +1.07] | +0.07 [−0.67, +1.07] |

The cap does what it should on the silent streams (+0.24 → +0.05 at 50 %), but
under the slam it demonstrably loses: the slam raises the maximum, the cap
hangs on the maximum, so the floor rises again with it. About three quarters
of the +0.81 consists of a gain handed back — on the slam corpus attenuated by
24 dB the deep percentile floor leads the max floor by around 0.6 points
(11.36 against 10.76 clean, repeated in all four arms), because quietly
levelled material is exactly the case where the speech peak alone already
sets the floor too high. 25 dB is dominated on both sides. Trade-off: the cap
buys +0.19 in the 50 % regime, where no interval excludes zero, and pays +0.8
in the slam regime, where one does — the use case the floor was built for.
**The uncapped percentile 99 stays.**

## 6. The implementation (2026-08-23)

`_PercentileFloorFeatures` and `clamp_log_mel` in `noScribe/voxtral_engine.py`,
constant `MEL_FLOOR_PERCENTILE = 99`, tests in `tests/test_mel_floor.py`. The
engine swaps the feature extractor on the processor; the library itself is not
touched.

Two implementation traps, both documented in the code and only named here so
they are not rediscovered: the library's `global_max` lever is **no use** for
the bit-identity check (it only yields the affine spectrogram, which loses
mantissa bits on quiet material — 12 % of the inputs get a floor that is off
by one bit), which is why the unclamped spectrogram is recomputed from the
library's primitives; and the percentile runs **only over the real frames**,
because the zero padding to 30 s would otherwise drag it down by the padding
share (5-s clip: 18 dB). Both are set out in full in the docstrings of
`_PercentileFloorFeatures` and `clamp_log_mel`.

**Reproduction by the production code** (`voxpopuli_spike.py 10 300 24
99.9,99`, the same ten streams, floor rise median 28.2 dB):

| Floor | clean | with slam | cost of the slam, paired |
|---|---|---|---|
| max (library) | 11.36 % / 7.36 % | 12.88 % / 7.90 % | **+1.52 [+0.78, +2.26]** \* |
| percentile 99.9 | 10.82 % / 6.93 % | 10.86 % / 6.88 % | +0.04 [+0.00, +0.10] |
| percentile 99 | 10.76 % / 6.88 % | 10.77 % / 6.88 % | +0.01 [−0.03, +0.06] |

Max and 99.9 reproduce the figures above to the second decimal; 99 is the only
variant whose interval contains zero.

**On the transcript the effect is small and not zero.** On 120 s of interview,
−20 dB, 60-Hz slam at 45 %: under the max floor the slam changes 15 places
across the whole passage, under the percentile floor 3, at least two of them
far from the slam — the residual shift of 0.1 dB moves every clamped cell by
one bit, and greedy decoding lets a marginal token flip somewhere. The
bit-identity on `hart` was a passage finding, not a behaviour.

**The stack benchmark, re-recorded (2026-08-24).** A local benchmark script
(not in the repository) has the baseline `macos27-p99floor`. On the 60-s bench
file the floor changes **exactly one word of 167**. The cross-check sits
deeper: today's stack with the library's max floor reproduces the July text
**bit-identically** — so across the moves from mlx-voxtral 0.0.4 to 0.0.6, mlx
0.32.0 → 0.32.1 and the numpy Viterbi, the decode path was bit-stable, and the
floor is the only text change. Runtime and memory unchanged.

## 7. Side finding: the Whisper path is not affected

`faster_whisper/feature_extractor.py:227` carries the same line, and noScribe
hands Whisper the whole file — but the effect does not materialise there. On
24 streams (122.3 min, 28.7 dB floor rise) the transient costs **−0.20
[−1.85, +1.19]** with VAD and **−0.51 [−1.98, +0.83]** without. Both intervals
contain zero **and exclude Voxtral's +1.52**; an effect of that size is
therefore not merely unshown there but ruled out. The fix thus stays
Voxtral-only, lives in a file upstream does not have anyway, and needs no PR
to faster-whisper. Script: `whisper_spike_streams.py`.

## 8. Prehistory: per block versus whole file

Up to 0.0.5, mlx-voxtral computed the log-Mel **per 30-s block** and
normalised each block against its own maximum. Voxtral specifies one
spectrogram over the whole input, split only afterwards (arXiv:2507.13264
§2.1) — so at the clamp a block with its own maximum lands on a different
floor. On 90 s of speech the block maxima were 1.897 / 1.721 / 1.526 against a
file maximum of 1.897.

A genuine deviation, reported and with a patch
([mzbac/mlx.voxtral#3](https://github.com/mzbac/mlx.voxtral/issues/3),
[PR #5](https://github.com/mzbac/mlx.voxtral/pull/5)) — but on the transcript
it changes nothing. 18 FLEURS streams of 300 s, 92 min, both versions on
bit-identical audio, paired: dWER **+0.02** [−0.19, +0.24] · dCER **−0.03**
[−0.13, +0.04]. Tight intervals: an effect beyond ±0.24 WER points is ruled
out. Script: `mel_stream.py`.

**So it was not patched locally** — with no quality gain, a patch on a
third-party library weighs heavier than architectural fidelity. Upstream the
change was right all the same, and mzbac merged PR #5 on 2026-08-22 and
released it as **0.0.6**.

**Since the percentile floor was built in, the engine computes the spectrogram
itself anyway** — over the whole file, from the library's primitives. The
whole-file property therefore no longer hangs on the pin; what the pin on
0.0.6 still does is keep the bit-identity test meaningful, the one that checks
our own computation against `VoxtralFeatureExtractor`. A fallback to 0.0.5
showed up as a **failing test**, not as a changed transcript.

## 9. The 24B build, and both models on more material (2026-09)

Everything above was measured on the 3B build. When the 24B model came back as
`voxtral-small-4bit`, the floor was re-measured on both, max against
percentile 99, on four hand-corrected passages, FLEURS and the VoxPopuli
transient streams. Per-passage figures are in `voxtral-quantisation.md`
("The re-run on four passages"); here only what bears on the floor.

**The 3B build confirms the choice, and more strongly than before.** On the
VoxPopuli streams (attenuated as `voxpopuli_spike.py` does) the percentile floor
is better even without a transient, paired dWER −0.60 [−1.38, −0.01], and with
one −2.11 [−2.93, −1.40]; the transient figures reproduce section 6 to the
second decimal. On the hand-corrected passages no cell is demonstrably worse
under the percentile floor and one is better (podcast A, −0.82 [−1.62, −0.18]);
FLEURS moves by 0.08 points, inside its noise.

**On the 24B build the transient protection holds** — the bang costs +1.51
[+0.92, +2.13] under the max floor and exactly nothing under the percentile —
**but the comparison is dominated by loops, under either floor.** The 24B model
runs away on inputs the 3B model handles, and the floor only moves *which*
inputs: under the max floor a hand-corrected podcast passage came back with
122 inserted words, which the percentile floor cleared (dCER −7.67); under the
percentile floor one VoxPopuli stream of 300 s came back with 5468 words where
the max floor produced 724 and the 3B build 722. Leaving that stream out, the
two floors score alike on VoxPopuli (10.62 % against 10.39 %); on FLEURS the
percentile floor is ahead (2.78 % / 0.75 % against 2.99 % / 0.88 %), on the
video call too (dWER −1.16 [−2.79, −0.12]), on the hard passage it is behind
without the difference being demonstrable (dCER +1.62 [−0.25, +4.78]).

So **the floor stays the same for both models**; a per-model floor would trade
one set of loops for another. The engine's loop ladder catches the runaways in
production — the 5468-word stream came back with 721 words through
`transcribe()` — but it took twenty minutes for five of audio, splitting the
pass twice. Loops, not the floor, are the 24B build's weak point.

## 10. Tools and a citation trap

| Script | for |
|---|---|
| `voxpopuli_floor.py` | floors on clean material, paired; `load_clips` (VoxPopuli, `VOXPOPULI_PARQUET` for the local file) and the arm shorthand `floor_arm` (`max`, `99`, `99c20`) |
| `voxpopuli_spike.py` | transient damage over many streams; `pair()` guarantees the dose; `VOX_BUILD` picks the build, two arms are also paired against each other |
| `voxpopuli_sparse.py` | the same on mostly silent streams (speech share selectable) |
| `mel_outlier_gap.py` | distance from maximum to percentile in real material, without a model |
| `mel_stream.py` | per block against whole file, paired over long streams |
| `whisper_spike_streams.py` | the same transient test on the standard Whisper engine |

All in `docs/scripts/`, all started with `venv/bin/python3`. The percentile
arms come from `_PercentileFloorFeatures`, the max arm explicitly from
`VoxtralFeatureExtractor()` — since the implementation,
`vox.proc.feature_extractor` is the percentile path and no longer serves as
the stock arm.

**A trap when citing absolute VoxPopuli figures.** Around **8.7 %** of the
gold references write digits ("60 Jahre", "21. Februar"), but the speaker says
words, and `wer.py`'s `norm()` keeps digits. That costs **1.22 WER points**:
the same run gives 11.39 % with and **10.17 %** without these utterances
(`load_clips(..., skip_digits=True)`). Paired differences are unaffected,
because both arms see the same reference.
