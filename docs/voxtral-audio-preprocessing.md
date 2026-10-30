# Audio Preprocessing in the Voxtral Path

As of 24 August 2026 · M1 Max · PyAV 12.3.0, mlx-voxtral 0.0.6, build
`voxtral-mini-8bit`. Sections 1–5 measured on 28 July 2026. All figures
measured, none estimated.

**The result in one sentence: on the audio path — from the file to the float32
array — there is nothing to improve.** Every intervention checked there is
either ineffective or harmful. The only intervention checked that does bring
something sits one stage later, in the feature path: the clamp floor of the
log-Mel (section 6, built in). Whatever else moves recognition lies before our
pipeline — in the quality in which the recording reaches us.

---

## 1. What the path does

One line in `noScribe/audio/convert.py:49` does the entire reduction:

```python
self.container_output.add_stream("pcm_s16le", rate=16000, layout="mono")
```

PyAV automatically inserts a resampler ahead of `encode()`. So all three things
happen there at once:

| Step | from | to |
|---|---|---|
| Sample rate | 44,100 / 48,000 Hz | 16,000 Hz (libswresample, Kaiser, −3 dB at 7.6 kHz, stopband attenuation −74 dB) |
| Channels | 2 | 1, as `(L+R)/2` |
| Format | float32 / 24 bit | int16, rounded, **without** dither |

It is read back in `noScribe/voxtral_engine.py` as float32 — `sf.read(...,
dtype="float32")`, soundfile scales the int16 values itself.
This is **bit-for-bit OpenAI Whisper's reference path**. Mistral's own
`mistral_common` would sit on the other side of this fork: float32 plus `soxr`
HQ, with no int16 stage. Measured, that makes no difference (section 2).

The same temporary WAV also feeds pyannote (`_run_diarize_subprocess` in
`main.py`) and the VAD pause correction (`decode_audio` in the same file) — a
change to the intermediate format would not be local to Voxtral.

## 2. What was checked, and what it brought

Evaluation is always on the **transcript**, never on the spectrum. All
intervals are paired 95 % bootstrap intervals.

| Intervention | Result | Evidence |
|---|---|---|
| **Drop int16** (pass float through) | **nothing.** `soxr-s16` and `soxr-flt` yield *identical* values over 180 FLEURS recordings | `fleurs_resample.py` |
| **Resampler soxr instead of swr** | **nothing.** dCER +0.01 [+0.00, +0.03] | `fleurs_resample.py` |
| **3 dB headroom before int16** | **nothing.** dCER +0.01 [−0.07, +0.11] — even though in this test *every* recording was driven to full scale | `fleurs_gain.py` |
| **Loudness normalisation** | **nothing.** −12 dB over 180 recordings: dCER +0.02 [−0.18, +0.18] | `fleurs_gain.py` |
| **Dynamics leveller** | **neutral to harmful** on real material | section 4 |
| **Denoising** | **nothing**, `anlmdn` worsens the WER | `fleurs_noise.py` |
| **High-pass** (to gain headroom) | unsuitable: tends to raise the peak (−0.58 to **+1.21** dB) | — |
| **Align chunks to the 30-s window** | **nothing.** dWER +0.00 [−0.75, +0.52] | section 5 |
| **log-Mel over the whole file instead of per 30-s block** | **nothing.** dWER +0.02 [−0.19, +0.24] | section 6 |
| **Clamp floor from a percentile instead of the maximum** | **built in** — prevents +1.52 [+0.78, +2.26] on a door slam, costs −0.12 [−0.85, +0.49] on clean material | section 6 |
| **Transcribe channels separately** | moot: all test material is dual mono (correlation +0.98 to +1.0000) | `audio_audit.py` |

### Why bit depth does not matter

The log-Mel is clamped from below, and the int16 intermediate stage is only
visible when its noise floor lies above this clamp. In Mel space the
quantisation costs 1.2 · 10⁻⁴ at 0 dBFS, 2.0 · 10⁻³ at −24 dBFS and 1.9 · 10⁻²
at −36 dBFS — so the error grows the more quietly the material is levelled.

**Since the percentile floor, the margin is smaller than this calculation once
assumed.** It started from the maximum floor: 80 dB below the loudest Mel cell,
against a 16-bit noise floor at −96 dBFS, i.e. 16 dB of reserve, which gave a
threshold at roughly −24 dBFS peak level. The production path now clamps at
`percentile99 − 8`, and on real material that percentile lies 17.6 dB
(`speech_60s.wav`) to 18.0 dB (`bench_4min.wav`) below the maximum — so the
floor sits around **98 dB** below the loudest cell, not 80, and the 16 dB of
reserve are used up.

**The finding still stands, the derivation no longer does.** What was measured
was `soxr-s16` against `soxr-flt` over 180 FLEURS recordings — bit-identical
output — but under the old floor, which clamped the difference away by
construction. Under today's floor the measurement has not been repeated. Noted
as an open question; it would only affect very quietly levelled 16-bit
material, and §3 says what dominates there anyway.

### Why clipping is not an issue

Over the full test files, between **0 and 255 samples** land on the int16
rail — out of 72 to 239 million samples per file. The raw Zoom recording does
not clip **at all** (peak 0.9985, no sample above; Zoom limits itself). The
true peak is practically on the sample peak everywhere.

### Why level changes nothing

The argument survives the percentile floor unchanged, because the percentile
too is relative to the same spectrogram — a constant addend shifts the
statistic and the cells alike. (What does *not* scale along is the int16
quantisation noise; see the open question above.)

A level change is a **pure offset** in Mel space: after subtracting the best
constant, a shape error of 5 · 10⁻⁷ remains. Peak normalisation, loudness
normalisation and attenuation therefore demonstrably do no harm — but they do
nothing either, because the model is insensitive to this offset.

## 3. What actually counts: the source

Of one podcast episode, a raw version (26 kbps AAC) and the version from a
commercial mastering service (lossless) are available. The existing reference
can be located in both, so same words, one truth:

| Version | LUFS | WER | CER |
|---|---|---|---|
| mastering service, lossless | −16.0 | 6.40 % | **2.70 %** |
| raw, 26 kbps | −25.1 | 10.19 % | **5.06 %** |
| raw + `dynaudnorm` | −16.9 | 11.85 % | 6.93 % |

Almost a halving of the error rate — **the largest effect in this entire
document.** Of that, 0.54 points go to the bitrate (the same lossless file
brought down to 26 kbps: 2.70 → 3.24 %).

The rest is **not** reproducible by levelling: the same control applied to the
same raw file worsens it from 5.06 to 6.93 %. And the service's remaining
chain contains noise and reverb reduction — of which the literature says it
harms ASR (section 7). That leaves **source quality** as the most likely
explanation: the raw file is a 26-kbps export, the processed one comes from a
better chain.

> **The recommendation to users is therefore not "run it through a mastering
> service" but "export in decent quality".** How much of this is processing and
> how much bitrate cannot be separated further with the material at hand —
> that would need the same recording unprocessed in good quality.

## 4. Why no leveller is built in

The candidate that looked convincing the longest. Whoever proposes it again
will find the figures here.

**On constructed material it works strongly.** Built from FLEURS pairs —
`LOUD | pause | QUIET(−N dB)`, both halves scored separately — `dynaudnorm`
lifts the recall of the quiet half to the level of the loud one:

| Gap | Recall quiet, raw | with `dynaudnorm` |
|---|---|---|
| 10 dB | 96.3 % | 96.6 % |
| 20 dB | 94.8 % | 96.5 % |
| 25 dB | 90.5 % | 96.3 % |
| 35 dB | **62.4 %** | **95.2 %** |

The threshold of effect lies at roughly **20 dB of spread within one passage**.

**On real material it brings nothing.** Two reasons, both measured:

1. **No real passage reaches the threshold.** Spread p90−p10 over the speech
   seconds: raw podcast passage 4.9 dB, Zoom reference passage 13.8 dB,
   mastered version 2.9 dB. Over the whole 4.8-hour Zoom recording the median
   is 6 dB, the maximum 18.5 dB. Raw material is not more dynamic than
   mastered material — it is only **quieter**, and quiet alone is demonstrably
   irrelevant.
2. **The constructed test was too optimistic.** An attenuated *clean*
   recording keeps its signal-to-noise ratio; a genuinely quiet voice in the
   same room does not. With a realistic noise bed the gain halves (+2.86
   instead of +5.81 points), and the **loud** half starts to suffer — without
   noise it was unaffected in every condition.

On the transcript, against a fair baseline (`p2-soxr` — the same float path as
the levellers, just without the filter):

| | dWER | dCER |
|---|---|---|
| `dynaudnorm` | −0.47 [−1.17, +0.23] | +0.07 [−0.37, +0.59] |
| `loudnorm16` | **−1.28** [−2.56, −0.35] * | −0.37 [−0.81, +0.10] |
| `speechnorm` | **−1.63** [−3.60, −0.23] * | −0.57 [−1.65, +0.18] |

\* = interval excludes zero. `dynaudnorm` is neutral, the others are worse. No
benefit, partly harm.

## 5. Why the chunker stays unchanged

Individual short utterances react measurably to where they sit in the 30-s
encoder window. On **continuous speech** — and that is all our chunker cuts —
the effect disappears: 18 streams of ~300 s each from FLEURS recordings with
known transcript, 92 min of audio, each once with a full and once with a
partial last window, evaluated paired:

> dWER **+0.00** [−0.75, +0.52] · dCER **−0.32** [−1.16, +0.16]

A rework was built, measured and removed again. On a single passage the effect
looked large and clean (CER 0.89 against 3.04 %) and was reproducible twice —
on the second passage the sign reversed.

**Relevant to Voxtral only.** The Whisper path hands the whole file to
`faster_whisper.transcribe()` with `vad_filter=True`
(`noScribe/whisper_mp_worker.py:140`); segmentation and windowing happen
internally there, we have no lever.

## 6. The Mel path and its clamp floor

One stage behind the audio path, in the feature path, sits the **only
intervention checked in this project that brought something**: the floor the
log-Mel is clamped to came from the maximum of the whole input, so that a
single loud cell set it for the entire pass — a door slam thus costs **+1.52
WER points** [+0.78, +2.26]. Since 23 August 2026 the engine takes a
percentile, which removes the damage and costs nothing on clean material.

That is a measurement record of its own and lives in
[voxtral-mel-clamp-floor.md](voxtral-mel-clamp-floor.md): the damage, the
cost, the choice of percentile, what argues against it, the limit on mostly
silent passes together with the cap rejected there — and the prehistory of why
the log-Mel is computed over the whole file instead of per 30-s block.

## 7. What the literature confirms

Two papers, checked against the arXiv originals:

* **Chondhekar et al. 2025**, [arXiv:2512.17562](https://arxiv.org/abs/2512.17562):
  MetricGAN+ on 500 medical recordings, four ASR systems — enhancement
  worsens results in *all* conditions, by 1.1 to 46.6 %. Untreated noisy audio
  consistently beats the enhanced audio.
* **Islam, Nahar & Hamid 2026**, [arXiv:2603.04710](https://arxiv.org/abs/2603.04710):
  SAM-Audio raises the PSNR from 32.28 to 35.99 dB and in doing so worsens
  every configuration (Whisper large-v3 on Bengali 65.83 → 77.35 % WER).

Both find exactly the separation this document maintains throughout: a better
signal metric says nothing about what the model hears.

## 8. What to watch out for when measuring

Six traps, each stepped into once:

1. **Measure on the transcript, not on the spectrum.** The production path has
   the largest Mel shape error of all variants checked — and changes nothing
   on the transcript, because the error sits in the top 16 of 128 Mel bins.
2. **A passage of 422 to 859 words carries ±1.8 to ±2.5 CER points.** Anything
   below that is undecidable. Under greedy decoding a single flipped token
   makes a long divergence — two short passages that contradict each other
   are the normal case, not a puzzle.
3. **A reference that was produced by correcting a transcript leans towards
   that arm** — quantified here at **0.25 CER points**, measured by running
   the same path with a different resampler against the same reference.
   Comparisons therefore run against `p2-soxr`, not against `p0-ist`; the
   disputed spots are listed by `adjudicate.py` with a timestamp for listening
   back. What that means for the second hand reference is in
   `other-asr-engines.md`.
4. **Reproducibility is no substitute for a sample.** The measurement that
   justified the chunker rework was fully reproducible and still wrong. The
   same with the clamp floor: that the slam damage was "exactly the same size"
   at three positions proved determinism, not generality.
5. **An impulse scaled over the speech peak gets clipped.** Real material is
   already levelled up to 1.0; a requested +12 dB arrived as a 5.2 dB floor
   rise, and the result looked like a clean null effect. Correct: attenuate
   the audio, impulse at full scale. `voxpopuli_spike.py` has since aborted
   when the dose does not arrive.
6. **A point estimate without a solid interval points in the wrong
   direction.** The Whisper arm of the transient test changed sign between 8
   and 24 streams — the behaviour of noise, not of an effect. Figures in
   `voxtral-mel-clamp-floor.md`, section 7.

## 9. Tools

All in `docs/scripts/`. Origin, quality and alignment of the test material are
in the local material notes (not in the repository).

| Script | for |
|---|---|
| `audio_audit.py` · `audit_long.py` | level, peak, clipping, loudness, spread of a file |
| `find_passage.py` · `build_reference.py` | select and prepare a passage for hand correction |
| `locate_passage.py` | find the same passage again in another version |
| `preproc_variants.py` · `preproc_cer.py` · `preproc_mel.py` | build path variants, compare on the transcript and in Mel space |
| `pair_cer.py` | arms from different source files against one reference |
| `audio_filters.py` | named libavfilter chains (levellers, denoisers) |
| `adjudicate.py` | disputed spots of two arms with timestamp |
| `wer.py` · `bootstrap_cer.py` · `encoder_diff.py` | scoring, intervals, reference-free comparison |
| `fleurs_*.py` | the individual questions on FLEURS: level, resampler, quiet passages, noise, window position, long streams |
| `mel_stream.py` · `voxpopuli_*.py` · `mel_outlier_gap.py` · `whisper_spike_streams.py` | the clamp-floor tools; their table is in [voxtral-mel-clamp-floor.md](voxtral-mel-clamp-floor.md) |
