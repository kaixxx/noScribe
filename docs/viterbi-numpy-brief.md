# The forced aligner: emissions on the GPU, the Viterbi in numpy

## Outcome (2026-08-23)

**Done.** The DP lives in `noScribe/ctc_align.py`; `_Aligner._emission` returns a
float32 numpy array and is the torch→numpy boundary; `align_words` and `align_prefix`
no longer touch torch. It landed in five steps:

* this brief's round five (630 → 347 ms, torchaudio's tie order) and the benches in `docs/scripts/viterbi_bench*.py`;
* the blank-label bug: every verification case used `blank=0`, so `ext = np.zeros` passed 277 of 277 and mislabelled every blank frame for any other index;
* the migration itself, after a simplification and review round;
* the 43 reference emissions stored in the npz, so the acceptance test needs no torch;
* every degrading `_spread` in `align_words` now logs why (`_spread_loudly`).

**What it bought.** The crash class from pytorch/audio#4208 cannot occur in noScribe
any more (64-bit indexing: the worst case is slow, not dead). And on an exact tie
between the advance-one and advance-two candidates the path is now the optimum —
torchaudio takes the worse value, which became upstream issue
[#4221](https://github.com/pytorch/audio/issues/4221) and PR
[#4222](https://github.com/pytorch/audio/pull/4222) (one-clause fix, CPU and CUDA,
CLA green). On real audio that tie is latent: 737 s of interview, 740 M cells,
55 921 tie cells, **none on the winning path**, all 2 253 timestamps identical.

**What it cost.** ~1.35x the C++ kernel on a 300 s chunk (347 vs 258 ms), about one
second per hour of audio against ~50 s of emissions. Memory unchanged (one byte per
DP cell), so `FORCED_ALIGN_MAX_CELLS` and the splitting stayed — their comments now
give the memory and latency reasons that were always there beside the crash.

**What did not survive contact.** `torchaudio==2.11` is **still pinned**:
`pyannote.audio` and `torch-audiomentations` require torchaudio, so the package
stays installed; what went is our own call into its kernel. Claim 2 below was wrong
about that.

**What was learned, and is pinned in tests now.** The recorded reference had two
blind spots — every case used `blank=0`, and random emissions never tie exactly —
so `tests/test_ctc_align.py` adds two torch-free oracles: brute force over every
path with blanks 0/2/4 and integer log-probs, and a scalar Viterbi past the uint8
range of the backtrace cursor. Three regression guards (cap, density, prefix
salvage) carried a stale `importorskip("transformers")` and had never run on the
Linux CI; they do now. And the density test had promised "not silently" in its
docstring for a month without asserting it.

**Verified.** 381 tests; the 43 recorded torchaudio alignments bit-for-bit; end to
end with the real aligner on three recordings (the 737 s interview and the two
hand-corrected references in `Audiotest2/referenz/`, 3 524 words) — every word
timestamp identical, every probability equal to the last bit, three times over as
the code changed underneath; PyInstaller's own module graph follows the engine's
static `from noScribe import ctc_align`. A fourth confirmation came for free on
2026-08-24 from a local stack benchmark (not in the repository): today's aligner reproduces the July `stamps_digest`
on the July benchmark text exactly — the July baselines predate this
migration, so that digest crossed it bit-for-bit. (`align_sec` also dropped
4.8–6.8 s → 1.0 s on the 60 s file between those baselines and now; cause not
attributed, noted here only so nobody credits or blames the DP for it —
the DP itself is the measured 1.35x of the C++ kernel.)

## The other 97 %: emissions on the GPU (2026-08-22)

The DP is 3 % of alignment; the wav2vec2 forward that produces the emissions is
the rest, and it had been running on the **CPU**. `_Aligner.__init__` never moved
its model anywhere, in a module that only runs on Apple Silicon, while
`pyannote_mp_worker.py` selected MPS for the diarizer a few files away. A review
recorded this on 2026-07-27 as the largest single open win and it had been sitting
since. Measured on the 300 s reference, 20 s windows, identical `(14985, 38)`
output:

| | time | realtime | argmax vs CPU |
|---|---:|---:|---:|
| CPU fp32 — before | 14.76 s | 20.3x | — |
| **MPS fp32 — now** | **4.24 s** | **70.8x** | **100.0000 %** |
| MPS fp16 — declined | 3.52 s | 85.1x | 99.933 % |

End to end on both hand-corrected references, 1268 words: **every timestamp
bit-identical**, at 3.14x and 3.25x. Alignment is ~13 % of a job's runtime, so this
is roughly **130 seconds per hour of audio**, ten minutes on a four-hour file.

fp16 was measured and declined: faster again, but 0.067 % of frames pick a different
argmax and the worst log-prob deviation is 1.68 — enough to move a word boundary. A
change that should be free of effect ought to stay that way.

These are the numbers for this measurement. It has been quoted elsewhere with
slightly different figures (14.85 → 4.60 s in a code comment, 20.2x → 65.2x
earlier in this file); those are earlier runs of the same experiment, and this
table is the one to use.

The device choice mirrors `pyannote_mp_worker`'s, macOS floor included, and falls
back to the CPU once and for good if the forward raises — an unsupported op on a
backend must not cost the whole job its transcript. `clear_cache()` now runs
*before* alignment rather than after, so Voxtral's buffers are gone by the time the
aligner loads next to them; the reasoning is in `_free_decode_buffers`, which
both alignments call.

## Faster aligners that exist, and what they measure (surveyed 2026-08-21)

mlx-audio carries two candidates that would take the aligner off torch entirely.
Both were measured on the hard passage (418 words, 120 s) against the shipped
torch aligner.

**`qwen3_forced_aligner` — fast, close enough in the middle, and it collapses at
length.** 32x against the torch aligner's 13.9x, median agreement 32 ms, 86.5 %
of words within 100 ms and 96.1 % within 200 ms. But it emits on an 80 ms grid,
3 % of its spans have zero duration, and it degrades with input length:

| length | zero-duration spans | last word ends at |
|---|---:|---|
| 120 s | 6 % | 119.8 s |
| 180 s | 5 % | 179.9 s |
| 240 s | 9 % | 239.8 s |
| 270 s | 10 % | 269.5 s |
| **300 s** | **20 %** | **271.0 s** (29 s unaccounted) |

Its own model card caps it at five minutes; against the ten-minute passes chosen
here that is disqualifying on its own.

**`wav2vec` — the German aligner already runs there today, unchanged.**
mlx-audio's MMS wrapper is the same encoder plus `lm_head`, and the German
aligner model loads with no code change beyond `model_type: "mms"`: identical
argmax, maximum absolute difference 0.00068. Throughput on the same clip:

| | time | realtime factor |
|---|---:|---:|
| torch `_emission` (CPU, 2026-08-21) | 15.64 s | 19.2x |
| mlx-audio via MMS | **2.89 s** | **103.8x** |

That 15.64 s is the **CPU** baseline, measured the day before the aligner moved
to MPS. Against the 4.24 s the GPU path now takes, the real saving is about
1.3 s per 300 s — roughly **15 seconds per hour of audio, not 150**, which is
what the same paragraph used to claim. Two practical notes if it is ever picked
up: the checkpoint needs converting from `pytorch_model.bin`, and the only thing
worth offering upstream is a `MODEL_REMAPPING` entry.

What mlx-audio does **not** have is a Viterbi: there is no `forced_align` in the
tree, and MMS's `_ctc_decode` is greedy argmax. The DP would still be ours.

**None of this belongs in a migration.** `docs/migration-mlx-audio.md` says to
leave alignment alone, and that stands: swap the engine first, prove the
transcript is unchanged, and only then open any of this. It is recorded so it is
not rediscovered from scratch.

---

## What the migration bought

Three things, none of them dependency removal:

1. **It removes a crash class.** `torchaudio`'s CPU `forced_align` indexes its
   `frames × (2·tokens+1)` DP buffer with 32-bit integers and segfaults past that —
   observed in a real run. A numpy DP with 64-bit indexing cannot do that, so the
   worst case degrades from SIGSEGV to slow.
2. **It takes torchaudio's kernel out of our call path.** (Not the pin: `pyannote.audio`
   requires torchaudio, so the package stays installed and pinned as part of the tested
   stack.)
3. **It removes the dependency on an unmerged upstream fix.** Our
   [pytorch/audio#4209](https://github.com/pytorch/audio/pull/4209) fixes exactly
   this overflow, was approved by a maintainer on 2026-08-05, and is still unmerged.

**What it does *not* buy.** It is tempting to think the cap and the recursive
split existed purely to dodge the overflow and could go with it. That is wrong on
both counts, as `tests/test_forced_align_cap.py` explains in its docstring:

* **The cap stays, with a new justification.** Measured on the 300 s reference chunk,
  *both* implementations use **1.0 byte per DP cell** (141 MB vs 140 MB peak over
  146 M cells). At the measured density — 50 frames and 16.3 tokens per second — a
  full 1500 s window is 3.66 billion cells, i.e. a **3.4 GB backtrace table and 15.8 s
  of DP**. So a cell budget around 2^30 is still required; only its comment changes,
  from "or the process segfaults" to "or it eats a gigabyte". `FORCED_ALIGN_MAX_CELLS`
  and `SALVAGE_ALIGN_MAX_CELLS` do become two budgets of the same kind and could
  plausibly merge into one.
* **The recursive split stays untouched.** It has a second, fully independent
  trigger: `too_dense`, where a window holds more target tokens than audio frames.
  That is a property of CTC, not of torchaudio, and it will fire on dense speech and
  on Voxtral over-generation no matter what runs the DP. `_spread`,
  `_quietest_frame_near`, `MAX_SPLIT_DEPTH`, `MAX_DENSE_SPLIT_DEPTH`, the halving and
  the pause-snapped audio cut all survive.
* **`align_prefix`'s piecing stays** for the same two reasons — density and the
  memory budget — including `_prefix_piece` and its binary search.
* **The transcription chunking is not involved at all.** Pause-aware cut points and
  the pass length (capped at 600 s, `TRUSTED_CHUNK_SEC`) are driven by Voxtral's
  memory and measured range. Nothing there reads a forced-align constant.

So the honest ledger is roughly **+50 lines net** in `voxtral_engine.py`: about 30
lines go (the constant's overflow comment, `_dp_cells`, the `too_big` explanation and
its hard-failure branch, the cap half of `_prefix_piece`), and a ~55-line DP plus a
`merge_tokens` replacement and its tests come in. Take the task for the crash class
and the tie fix, not for a smaller file -- and not for the pin, which stays.

## Everything that was checked so it need not be built

Searched 2026-08-22. Nothing usable was found, and the reasons differ:

| candidate | verdict |
|---|---|
| `ctc-segmentation` | no torch, right domain — but **source distribution only**, so installing needs a C compiler, which a click-and-run app cannot ask of users; last release 2022-10-11; and it solves utterance segmentation robust to imperfect text, not exact-target frame alignment |
| `ctc-forced-aligner` | requires torch — trading torchaudio for another torch consumer |
| `speechbrain`, `whisperx`, `stable-ts`, `nemo-toolkit` | all require torch, all far heavier than the problem |
| `onnx-asr` | contains no alignment code at all (repository search: zero hits) |
| `kaldialign` | edit distance between token sequences, not Viterbi over emissions |
| `pyctcdecode` | beam-search decoding, not forced alignment |
| mlx-audio `qwen3_forced_aligner` | measured: collapses at its documented 5-minute cap, 20 % zero-duration spans, last 30 s of a 300 s clip on one timestamp |
| mlx-audio generally | no `forced_align` or Viterbi anywhere; MMS's `_ctc_decode` is greedy argmax |

**Do not re-run this survey.** One candidate did survive it, though, and it
deserves its own section.

## The one that nearly makes this unnecessary: `ctc-forced-aligner`

PyPI's `ctc-forced-aligner` 1.0.2 exposes

```python
forced_align(log_probs, targets, input_lengths=None, target_lengths=None, blank=0)
    -> (paths, scores)
```

— the same signature as `torchaudio.functional.forced_align`, over numpy arrays,
backed by a compiled C++ extension. It also ships `merge_repeats`, the counterpart
to `merge_tokens`. **Measured against torchaudio on 40 random cases** (varying T
and target length, including tight `T == L + repeats` fits): **40 identical, 0
divergent**. So the algorithm question is answered — if you adopt it, there is
nothing to write.

Four things to weigh before you do, none of them fatal on its own:

* **Source distribution only.** A 22 KB tarball that builds a C++ extension at
  install time. That is the same objection that ruled out `ctc-segmentation`: a
  click-and-run desktop app cannot ask users for a compiler, and PyInstaller has to
  be shown to bundle the built extension.
* **Custom licence.** "Deskpai Open Source License (DOSL) 1.0" — BSD-shaped, but
  clause 3 requires any distribution to carry a README that *clearly displays*
  "This software is distributed with permission from https://www.deskpai.com." in
  a prominent, visible location, and clause 6 requires the licence file in the
  distribution root. Attribution terms of that kind are plausibly permitted by
  GPL-3.0 §7(b), but noScribe is GPL-3.0 and this is exactly the class of question
  that cost this project a week already. Get it read properly before depending on
  it.
* **Name confusion.** The PyPI name belongs to `deskpai/ctc_forced_aligner` (11
  stars, one release burst in 2025-02, untouched since). It is **not**
  MahmoudAshraf's `ctc-forced-aligner` (552 stars, actively maintained), whose
  *model* this project uses as its multilingual aligner. Anyone reaching for "the
  ctc forced aligner package" will assume the wrong one.
* It was found sitting in the project venv, installed deliberately and listed in no
  requirements file, imported by nothing. It has since been removed.

**Whatever you decide, install it temporarily as a second oracle.** It is an
independent implementation that agrees with torchaudio, so checking a hand-written
DP against two references rather than one costs one `pip install` and no thought.

If you do write the DP instead, torchaudio's own "Forced Alignment with Wav2Vec2"
tutorial carries it in a few dozen lines of readable Python — adapt that rather
than deriving it from the paper.

## Writing it is easy. Making it fast took five rounds.

Both halves of that were measured, so plan around the numbers rather than the
intuition that 151 lines of C++ cannot be doing much.

**Correctness is a short afternoon.** A vectorised numpy DP — state axis
vectorised, time axis looped, ~30 lines — reproduced `torchaudio.functional.forced_align`
on **40 of 40 random cases** including tight `T == L + repeats` fits, first try.
The algorithm is not where the risk is.

**Speed is where the work is.** On a real 300 s chunk (T = 14 985 frames,
L = 4 886 tokens, N = 9 773 states, **146 million DP cells**):

| implementation | time | vs C++ |
|---|---:|---:|
| `torchaudio.functional.forced_align` (C++) | 264 ms | — |
| `ctc-forced-aligner` (C++) | 253 ms | 1.0x |
| naive vectorised numpy | 3 144 ms | 11.9x |
| optimised numpy, round four | 630 ms | 2.4x |
| **optimised numpy, round five** (below) | **347 ms** | **1.35x** |

All of them returned identical paths on the chunk.

The remaining gap is not C++ being clever. The recurrence is sequential in time, so
numpy runs ~15 000 iterations of a handful of array operations on 9 773-element
arrays and pays dispatch overhead on each; C++ runs one tight loop over the same
cells with none. Vectorising harder cannot help — the long axis is the one that
cannot be vectorised. What *did* help was cutting the work inside each iteration:

* **float32 throughout.** numpy defaults to float64, which doubles memory traffic
  for no precision that matters here.
* **Every buffer preallocated, every operation given `out=`,** so the loop allocates
  nothing at all.
* **The advance-by-two lane written only where a skip is legal** (at most L of N
  positions). Every other position stays at the floor for the whole run, so it never
  needs rewriting.
* **Emissions never gathered into a `[T, N]` array** — that alone is 586 MB on this
  chunk, and profiling put 170 ms in it. Even states are all blank and share one
  scalar per frame; odd states are the targets, a gather of L.

Together: 3 144 ms → 630 ms, a factor of 5.

**Round five (review, 2026-08-23) took 630 → 347 ms with *less* code,** by timing
every operation of the loop on its own at N = 9 773 and replacing the three that
were not bandwidth-bound:

* **The advance-by-two lane as one contiguous add.** `two[skip_idx] = alpha[src_idx]`
  was the single most expensive line, 8.7 µs per frame, because fancy indexing is
  what numpy is slow at. A penalty vector `pen` that is 0 where a skip is legal and
  `-inf` everywhere else turns it into `np.add(alpha[:-2], pen[2:], out=two[2:])` —
  1.3 µs, no index arrays, and the blank-between-repeats rule lives in one line.
  This supersedes round four's "written only where a skip is legal".
* **Backpointers by arithmetic, not by mask.** `np.copyto(row, 2, where=c2)` costs
  in proportion to how many entries of `c2` are set — 0.5 µs on an empty mask, 17 µs
  at 30 % — and on the chunk the skip wins at 12.6 % of the cells per frame on
  average. `row = max(c1, c2 + c2)` on the `uint8` views is two cheap byte
  operations regardless of density.
* **`-inf` as the floor, as in torchaudio.** Nothing in the loop subtracts, so no
  NaN can arise; the `-3.4e38` floor and its comment were guarding against an
  operation that does not occur. It also lets `-inf + -inf` stay `-inf` instead of
  overflowing with a warning.

Per-frame accounting after that: two `greater` (7.1 µs — comparisons are oddly
slow, 3x a `maximum` on the same views), two `maximum` (2.6), the penalty add (1.3),
the two byte ops (1.0), the two strided emission adds (4.7), the gather (3.7), and
about 10 µs of Python and dispatch for a dozen calls. There is nothing large left to
remove; the rest is the cost of driving 15 000 iterations from Python.

**Ideas measured and lost — do not re-try them.**

* Replacing the strided `nxt[0::2] += …` / `nxt[1::2] += …` pair with one
  contiguous `np.take` into a preallocated row and a single add: **twice as slow**
  (1 249 ms).
* Touching the advance-by-two lane only at its legal positions via fancy indexing
  instead of comparing across the full width: **19 % slower** (748 ms).
* `np.take(lp, targets, out=g)` to avoid the gather's allocation: **20 µs instead of
  3.7 µs** for plain `lp[targets]`. The `out=` path is the slow one.
* Any masked write for values — `np.copyto(..., where=)`, `a[m] = b[m]`, `putmask`:
  all scale with the mask density (10–56 µs at 30 %). A micro-benchmark with an empty
  mask makes them look free; that is how the first attempt at this round ended up
  slower than what it replaced.
* Separate contiguous blank and token lanes (`B[L+1]`, `K[L]`) so that every
  operation is half-width and unstrided: **slower**, because the number of calls
  doubles and dispatch eats what the bandwidth saved.
* `signbit(a - b)` instead of `greater(b, a)`: 7 % faster, but only with a finite
  floor, since `-inf - -inf` is NaN. Fragile; rejected.
* **Restricting the work to torchaudio's reachable band** (states `2·(L-(T-t)) ≤ s
  ≤ 2t+1`), which skips about a third of all cells: **299 ms, 1.16x C++**. Correct
  and measured, and still rejected: it is eight lines of index arithmetic in the
  hottest loop, the first attempt at it had a real bug (the first token state in the
  band lost its advance-by-two source `K[lo-1]`, which lies just outside the band),
  and what it buys is 0.6 s per hour of audio.

The lesson generalises: in this loop numpy is cheap on bandwidth and expensive on
indexing, masking and call count, so a full-width pass of plain ufuncs beats every
cleverer access pattern.

**One thing did help and is nearly free**: writing the advance-by-one lane straight
into the output buffer from overlapping slices (`np.maximum(alpha[1:], alpha[:-1],
out=nxt[1:])`) instead of shifting into a scratch buffer first — 646 → 630 ms.

**347 ms is affordable, and that settles the design.** Measured on the same chunk
with the aligner on MPS, the emission step costs **4.06 s** and the DP is the rest:

| | emissions | DP | total |
|---|---:|---:|---:|
| C++ | 4.06 s | 0.26 s | 4.32 s |
| numpy | 4.06 s | 0.35 s | 4.41 s |

**+90 ms per 300 s chunk, about 1.1 s per hour of audio, +2 % on the alignment
step.** Against the ~130 s per hour that moving the aligner to the GPU won, that is
noise. Short windows are dispatch-bound — at T = 1 000, L = 330 numpy takes 7 ms to
C++'s 1.1 ms — and equally irrelevant in absolute terms.

**So build one path, not two.** An earlier version of this brief proposed keeping
C++ for the common case and using numpy only where `FORCED_ALIGN_MAX_CELLS` would
otherwise force a split. Reject that: the fallback would run only on unusually dense
speech, which is precisely the condition under which a defect in it would go
unnoticed for a long time and then surface on a user's recording. A single path that
costs 4.6 s per hour is worth more than a fast path plus a rarely-exercised one.

**The reference implementation is `noScribe/ctc_align.py`** — this file carried a
listing of the round-five DP until it went into the product, and a second copy would
only drift. What the module adds over that listing is validation (blank inside the
vocabulary and outside the targets, an `adjacent_repeats` helper the engine shares,
an empty-path `merge_tokens`) and the `TokenSpan` named tuple the engine reads. It
was verified identical to `torchaudio.functional.forced_align` on 200 random cases
(including tight `T == L + repeats` fits), on the 43 recorded cases, and on the 300 s
chunk, all under `np.errstate(all="raise")`; on 34 constructed tie cases it follows
torchaudio's tie order except where torchaudio itself is wrong (see the *Ties* trap).
Re-deriving the optimisations from scratch is the expensive part, not the algorithm —
the benches that reproduce every number here are `docs/scripts/viterbi_bench*.py`.

The backtrace loop is Python over T, which is fine — it is ~15 000 scalar steps and
measures under 20 ms. The `ValueError`s mirror torchaudio's checks: the engine
pre-checks the frame budget, but its `except … warn … _spread` path is only
meaningful if the DP raises rather than returning garbage when it is ever called
without them.

## Memory: the standard fix works, and you still should not use it

Asked and measured 2026-08-23, because 3.4 GB for a full window looks like the thing
to attack. It is fixable, and fixing it buys less than it seems.

**The forward pass already streams.** Only two state vectors of N floats are ever
live. What cannot stream is the *backtrace*: Viterbi only knows which path won after
it has seen the last frame, so the per-frame decisions must be kept. That is the
`T × N` table, and it is inherent to the algorithm, not to this implementation.

**Checkpointing removes it, at 1.7x time.** Pass 1 runs the recurrence with no
backtrace at all, snapshotting alpha every `seg = 2·√T` frames. Pass 2 walks the
segments backwards: restore a snapshot, recompute that one segment's backtrace into a
`seg × N` buffer, follow it down, discard. Memory drops from `O(T·N)` to `O(N·√T)`;
every cell is computed twice. Implemented and verified **identical on 60 of 60 random
cases and on the 300 s chunk**:

| | time | memory |
|---|---:|---:|
| full table | 579 ms | 140 MB |
| checkpointed | 977 ms | **5 MB** |

On a projected 1500 s window that is **3.41 GB → 51 MB, a factor of 68.** (The cheaper
alternative, packing the 0/1/2 backtrace at 2 bits per cell instead of 8, gives a flat
4x for much less work — worth knowing, not measured.)

**Why it does not let you drop the splitting anyway.** Three reasons, and the first
was the surprise:

1. **Splitting is a compute optimisation, not a workaround.** Halving a window halves
   *both* axes, so total DP cells fall by half at every level. Measured on the 300 s
   chunk: whole 587 ms, two pieces 338 ms, four pieces **200 ms — 2.9x faster than
   not splitting.** Aligning a 1500 s window in one call is ~5x the cell count of
   aligning it in five, and with checkpointing's 1.7x on top that is roughly **9x the
   DP work** to avoid a split you also cannot avoid.
2. **`too_dense` is untouched by any of this.** More target tokens than audio frames
   is a CTC constraint; no memory technique makes such a window alignable.
3. **The DP never gets near 3.4 GB in production, and on a small machine it is
   tiny.** That number is `MAX_CHUNK_SEC`; auto-sizing never picks it. See below.

**Is the allocation a problem on a small machine?** It is genuinely additional —
only the decode pass's MLX buffers are released before alignment
(`vox._mx.clear_cache()`), the Voxtral weights stay resident — so this is not
memory the model has already paid for. But the exposure is bounded twice over, and
the second bound is the interesting one:

* **`RAM_RESERVE_GB = 7` explicitly holds the aligner back.** Its comment names "the
  forced aligner that runs alongside"; `MIN_HEADROOM_GB` budgets it at ~2 GB.
  Measured: the German wav2vec2 aligner is **0.50 GB resident**, so model plus a
  full-size DP fits that budget with room.
* **DP size scales with the *square* of the window, and the window is already sized
  to the machine's RAM.** Cells are `frames × (2·tokens+1)`, both linear in seconds.
  So the coupling runs the protective way: less RAM → shorter passes → quadratically
  smaller DP. For `mini8` at the reference density:

  | machine | pass chosen | DP peak |
  |---|---:|---:|
  | 8 GB | 60 s (warns; likely refused) | **5 MB** |
  | 16 GB | 416 s | **262 MB** |
  | 24 GB and up | 600 s (clamped by `TRUSTED_CHUNK_SEC`, not RAM) | **546 MB** |

  Above 24 GB the window stops growing, so the DP stops growing too. 3.4 GB needs a
  1500 s window, which only a config override can pin — and `FORCED_ALIGN_MAX_CELLS`
  then splits it, holding the peak at 1 GB. The cap is what makes that true, which is
  the second reason not to drop it.

**Where it would genuinely pay:** as a replacement for the `_spread` hard-failure
path, so a window that cannot be split far enough gets slow real timestamps instead
of evenly spread wrong ones. Check first whether that path is reachable — with
`MAX_SPLIT_DEPTH = 12` a too-big window is 1/16 000 000 of its original cell count by
then, so in practice the reachable hard failure is `too_dense`, which checkpointing
does not help. Probably dead code to optimise. Measure before building.

## Acceptance (all met)

* ✔ Frame-for-frame identical output to `torchaudio.functional.forced_align` on the
  recorded reference, and on two hand-corrected references (local material,
  not in the repository).
* ✔ `venv/bin/python3 -m pytest tests/ -q` green.
* ✔ Word timestamps unchanged end to end — compared against stored runs of the real
  aligner, 3 524 words, not eyeballed.
* ✔ No `torchaudio` left in `noScribe/`: `grep -rn torchaudio noScribe/` empty except
  comments that record the history.
* ✔ The DP cap and the recursive split **stay** (see *What the migration bought*);
  only the cap's comment changes, from segfault to memory. A pass that would
  previously have tripped the cap still splits and still aligns
  (`tests/test_forced_align_cap.py`, now also on CI).
* The acceptance chunk is real emissions, not uniform random ones: at chunk scale,
  random emissions push |alpha| to ~5·10⁴ where float32 sums collide constantly, and
  the resulting ties make *every* implementation — including torchaudio against
  itself with a different tie rule — disagree on thousands of frames. Measured: 7 918
  differing frames on a random chunk, 0 on a peaky one of the same size.

## Traps

* **Blank index.** The CTC blank is `config.pad_token_id`, not the token spelled
  `<pad>`. `_Aligner.__init__` documents why: the multilingual MMS aligner's vocab
  has `{"<blank>": 0, "<pad>": 1, …}` and emits blanks on 0, so reading `<pad>`
  picked an index the model never produces, and every word stretched to meet its
  neighbour. Keep that logic exactly.
* **`uint8` backtrace, Python `int` cursor.** Writing the backtrace walk as
  `s -= bt[t, s]` passes every small test and then raises `OverflowError` the first
  time N exceeds 255, because numpy types the in-place operation from the `uint8`
  right-hand side. It cost nothing to find here only because the real chunk was
  measured; on 60 random cases it was invisible. Write `s -= int(bt[t, s])`. This is
  the shape of bug the single-path decision above is meant to keep out of production.
* **Repeated tokens.** Two identical labels in a row require a blank between them —
  advance-by-two must not skip it. torchaudio enforces this; so must you.
* **Ties.** The recorded reference contains none, so a DP with the wrong tie order
  passes it and then disagrees with torchaudio on real audio. torchaudio's CPU
  kernel (`forced_align/cpu/compute.cpp`) decides, per cell with predecessors
  `x0` (stay), `x1` (advance one), `x2` (advance two):
  `if (x2 > x1 && x2 > x0) → 2; else if (x1 > x0 && x1 > x2) → 1; else → 0`,
  and ends on the final blank only if `alpha[S-1] > alpha[S-2]`, strictly — a tie
  goes to the last token. Round four had `>=` there and diverged on 24 of 34
  constructed tie cases. The reference above reproduces the first clause, the
  `x1 > x0` half of the second, and the final-state rule exactly.
  **What it deliberately does not reproduce:** when `x1 == x2 > x0`, torchaudio's
  chain falls through to `x0` — the strictly *worse* predecessor. That is a defect
  in torchaudio, not a tie convention: a scalar Python port of the C++ loop matches
  torchaudio on 234 of 234 cases, the same port without the `x1 > x2` clause matches
  the reference above on 234 of 234, and on 10 of the 13 cases where the two differ
  torchaudio's path has a strictly lower total score. Reproducing it would cost two
  extra full-width operations per frame (~15 %) to copy a bug. It is rare — 2 064 of
  146 million cells on the peaky chunk, none on the winning path — but it is the one
  known way the numpy path can differ from torchaudio, so if a word timestamp ever
  differs, check for an exact tie before suspecting the DP.
* **Scores.** `merge_tokens` returns a score per span. Check what the existing code
  does with it before changing its meaning; note that what the engine stores as
  `prob` is `exp()` of that score, so it matches `whisper_mp_worker`'s field.
* **Span ends are exclusive.** Verified against torchaudio: for a token occupying
  frames 0, 1, 2, `merge_tokens` returns `start=0, end=3`, so `end - start` is the
  duration and `sp.end / fps` is the time after the last frame. The engine relies on
  this. Do not add a `+1` — MahmoudAshraf's aligner needed one
  ([c344f5b](https://github.com/MahmoudAshraf97/ctc-forced-aligner/commit/c344f5bc900323aa434a7cb200b7c629d463bd02))
  because *its own* `merge_repeats` returns an inclusive end; the conventions differ.
* **The frame rate is derived, and that is fine.** `fps = n_frames / duration`
  rather than the model's `inputs_to_logits_ratio` (320, i.e. exactly 50 fps).
  Measured on a 300 s chunk the derived value is 49.95, and because it comes from
  the concatenated total the resulting timestamp error stays within **10-20 ms and
  oscillates rather than accumulating**. Not worth changing, and changing it would
  move every timestamp the tests pin.
* **Do not touch the surrounding logic** — the emission windowing, the tokenizer, the
  prefix-salvage path. Swap the DP, nothing else.
* **A generic HMM Viterbi is the wrong shape, and searching for "Viterbi in numpy"
  will hand you one.** Those implementations take a dense `N x N` transition matrix
  and an emission *table* indexed by a discrete observation alphabet, then loop over
  both time and states in Python. Every part of that fights this problem. CTC's
  transitions are a narrow band — from state `s` only to `s`, `s+1` or `s+2` — so
  for a ten-minute pass the dense matrix is 2401 x 2401 entries of which **0.12 %
  are not `-inf`**, rebuilt on every call because the target sequence changes per
  chunk. The emissions are already a `[T, vocab]` array from the network, so what
  you need is a gather `emissions[t, target[s]]`, not a table lookup. And the double
  loop is **72 million Python iterations** where vectorising over the state axis
  leaves 30 000. Worse, none of the actual difficulty — the blank interleaving, the
  advance-by-two rule, the backtrace, `merge_tokens` — appears in a generic
  implementation at all. Take the torchaudio tutorial instead.
* **numpy is not a new dependency.** `voxtral_engine.py` already imports it, and a
  dozen installed packages require it transitively. Nothing to weigh there.
* Read the docstrings in `voxtral_engine.py` before changing any constant.

## Rollback

Now that it is in: rollback means restoring the torchaudio call path in
`align_words` and `align_prefix` (the commit that introduced `ctc_align` is one
`git revert`); the `torchaudio==2.11` pin is untouched either way, since
pyannote.audio keeps the package installed.
