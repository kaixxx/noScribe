# Brief: move the Voxtral engine from mlx-voxtral to mlx-audio

This is a conditional work order, not a plan of record: nothing here is
scheduled, and it only becomes relevant if one of the triggers below fires.

Facts re-verified 2026-08-22 against `mlx-voxtral` 0.0.6, `mlx-audio` 0.5.0 and
`mlx` 0.32.1, the two upstream reports re-checked 2026-09-02 against `mlx-audio`
0.5.1 in a throwaway venv, and the whole brief audited 2026-09-03 against
`mlx-audio` main (`b809500`, the merge of #901) — that audit added the log-Mel
floor, which the brief had missed, and reversed the `mx.metal` argument. Each
fact states how it was checked, so you can re-check rather than trust.

## When this becomes relevant

Not now. `mlx-voxtral` is MIT, actively maintained again, and carries every fix
this project reported — which is the point: it shipped under a "Personal Use
License" (MIT plus a ban on commercial use) that noScribe's GPL-3.0 could not
carry, and that licence, not the code, was what once made this migration urgent.
The author relicensed within a day of being asked. Do the migration only if one of these happens:

* **mlx-voxtral goes quiet again** and a defect turns up that nobody upstream will
  fix.
* **An MLX or mlx-lm bump breaks it.** The exposure is `mlx-lm`, but not for the
  reason this brief used to give. `mlx-lm` has had no release since 0.31.3
  (2026-04-22) although `main` is active (checked 2026-09-03); the engine drives
  decoding through its `generate_step`, `KVCache` and `make_sampler`, so a
  release that changes those, or a stale release that stops installing against
  a newer `mlx`, breaks the decode loop. mlx-audio dropped `mlx-lm` as a core
  dependency on 2026-08-09 and vendors a snapshot of 0.31.3 under
  `mlx_audio/lm`, so the move would make the engine independent of `mlx-lm`
  releases. That is the argument; the `mx.metal` one does not hold — see fact 4.
* **A second engine is worth shipping.** `mlx-audio` carries Voxtral alongside a
  dozen other ASR models behind one API, so the move pays for itself in one go
  rather than one engine at a time. What those models are worth is measured in
  `docs/other-asr-engines.md`; what its aligner models are worth, in
  `docs/viterbi-numpy-brief.md`.

If none of those is true, close the task.

## Scope

In: `noScribe/voxtral_engine.py`, `tools/quantize_voxtral.py`,
`environments/requirements_voxtral_macOS_arm64.txt`, the affected tests,
re-publishing the two quantised builds, and **the log-Mel percentile floor**
(`_PercentileFloorFeatures`, `docs/voxtral-mel-clamp-floor.md`) — it is built on
the library's STFT helpers and has to be rebuilt on whatever computes the
features afterwards. Step 3 says how.

Out: chunking, loop detection, the temperature ladder, forced alignment, prefix
salvage. None of that touches the library. **Do not refactor them while you are in
there.**

## The coupling is six calls

Verified with `grep -rn "mlx_voxtral\|mlx_lm" noScribe/ tools/ tests/`. Line numbers
drift — grep rather than trust them:

| where | what |
|---|---|
| `_Voxtral.__init__` | `load_voxtral_model`, `VoxtralProcessor` |
| `_Voxtral.__init__` | `proc.feature_extractor = _PercentileFloorFeatures()` — swaps the log-Mel path |
| `_PercentileFloorFeatures.__call__` | `mlx_voxtral.audio_processing`: `stft_mlx`, `get_mel_filters`, `hanning`, `pad_to_multiple` and the frame constants |
| `_Voxtral.transcribe_array` | `proc.apply_transcrition_request(...)` (note the upstream typo) |
| `_Voxtral.transcribe_array` | `model.generate(...)` — the non-greedy fallback only |
| `_Voxtral.transcribe_array` | `proc.decode(...)` |
| `tools/quantize_voxtral.py` | `mlx_voxtral.quantization`, `utils.model_loading.download_model` |

Plus `_merged_embeddings` and `_LMAdapter`, both of which likely become unnecessary
— see below.

The floor rows were missing until 2026-09-03: the brief was re-verified on
2026-08-22 and the floor landed on 2026-08-23. It is the one piece of the engine
that reaches *into* the library's signal path rather than calling its API, and
it is why "the same transcript through old and new" (step 5) is only a valid
check once the floor sits on both sides. Under mlx-audio the features do not
come from an MLX STFT at all: its `generate()` takes `input_features` from
transformers' `VoxtralProcessor`, computed in torch on the CPU with the plain
maximum floor. Two ways to keep ours, pick one in step 3: subclass or wrap
transformers' extractor, or port the window, STFT and filter bank to numpy and
hand `input_features` straight to `Model.stream_generate`, which accepts them.
Either way `tests/test_mel_floor.py` changes its reference: it pins bit-identity
at pct=100 against mlx-voxtral's path today, and would pin it against the
transformers extractor instead. That the two are interchangeable was measured
on 2026-08-22, when mlx-voxtral 0.0.6 moved to the whole-file log-Mel: on 300 s
of podcast material its features differ from transformers' by at most 0.00017
(against 1.30 between 0.0.5 and 0.0.6). Recorded here because no other
document carries it.

## Verified facts you must not re-derive

**1. mlx-audio will not load the published builds, and it says nothing.**

Loading `models/voxtral-mini-8bit` through `mlx_audio.stt.utils.load_model` yields
**0 quantised modules and 405 dense `Linear`**, 211 of them in the language model,
with no exception. Mechanism, all in `mlx_audio/utils.py`:

* `apply_quantization`'s `get_class_predicate` looks up `p in quantization` and
  otherwise falls back to `f"{p}.scales" in weights`.
* Our config and weights use `language_model.layers.0…`; mlx-audio's module tree is
  `language_model.model.layers.0…` (its `LanguageModel.__init__` does
  `self.model = LlamaModel(config)`). Neither lookup matches.
* `sanitize()` only transposes conv weights — it does not remap the prefix.
* `load_model(..., strict=False)` is the **default**, so the mismatched keys are
  skipped silently and the language model stays at its initial values.

The audio tower does match and does load, which is why a naive smoke test looks
half-plausible. Reported as
[Blaizzy/mlx-audio#902](https://github.com/Blaizzy/mlx-audio/issues/902). **Re-tested
on 0.5.1 (2026-09-02): reproduces unchanged and still silently.** 0 quantised
modules, 211 dense `Linear` in the language model, and `q_proj.weight` comes back
float32 with std 0.0104 — random initialisation, not our 8-bit weights. Nothing on
stdout; the only thing on stderr is an unrelated transformers tokenizer notice.
Issue still open and unanswered. Check again before relying on a warning.

**2. Re-quantising is format-only. It cannot change quality or speed.**

MLX's affine quantisation is data-free — scale and bias come from each group's own
min and max, no calibration set, no randomness. Verified: `mx.quantize` on the same
tensor twice returns bit-identical results, and so does a fresh copy of it.
mlx-audio's predicate is `not p.startswith("audio_tower")`, which selects exactly
the set the current builds carry: **213 modules, all 8 bit, group size 64, affine,
encoder dense**. Same weights in, same tensors out; only the keys change.

So the measurement tables in `docs/voxtral-quantisation.md` still describe a
re-quantised build. **Do not re-run the bit sweeps.**

**3. What could change output, and therefore must be checked:**

* mlx-audio's `_merge_input_embeddings` scatters without promoting dtype, where
  `_merged_embeddings` promotes deliberately — and on the current build that
  promotion is needed. Measured on `voxtral-mini-8bit`: `embed_tokens` returns
  bfloat16, but the projector returns float32, because the log-Mel features are
  float32 (transformers' processor, which mlx-audio uses, returns them as float32
  too, and mlx-audio does not cast them) and MLX promotes the bf16 weights'
  output to the input's dtype. Scattering float32 into the bf16 array rounds
  every audio embedding to bf16 (1.0001 → 1.0). Keep the promotion in any
  migration; the failure is invisible in the text and shows up only as different
  logits.
* **Stop tokens: fixed upstream, keep resolving them anyway.** mlx-audio carried
  `_VOXTRAL_EOS_TOKEN_IDS = [2, 4, 32000]`, where 32000 is not a pad token but the
  ordinary text token `" Capital"`, so a transcript containing that word was
  truncated silently — clean prose that merely ended early. Our
  [Blaizzy/mlx-audio#901](https://github.com/Blaizzy/mlx-audio/pull/901) fixed it to
  `[2, 4, 11]`, merged 2026-09-02 and **first released in 0.5.2** (2026-09-07;
  0.5.1 of 2026-08-31 still carries 32000), so the `mlx-audio>=` floor is 0.5.2. **Do not rely on the library default even then**:
  resolve the ids from the processor, as `_resolve_stop_tokens` does, and pass them
  explicitly on both decode paths.
* mlx-audio vendors its own `generate_step` (`mlx_audio.lm.generate`) rather than
  using `mlx_lm`'s. Same chunked prefill (`prefill_step_size=2048`), so the ~18 %
  peak saving survives — but `MEM_MODEL` is calibrated against the current path and
  **must be re-measured**.
* **`AutoProcessor` and `AutoTokenizer` resolve different tokenizer backends for
  this build.** On transformers 5.16.1, `AutoTokenizer.from_pretrained` picks
  `MistralCommonBackend` and reads `tekken.json`; loading the same directory through
  mlx-audio's `AutoProcessor` took the fast-tokenizer path instead
  (`tokenization_auto.py` swaps the class when `_use_mistral_format` is false), which
  announced itself only as a regex warning. A different backend means differently
  encoded prompts, and this build ships no `tokenizer.json` for a fast backend to
  read. **Assert the tokenizer class after loading**, the same way step 5 asserts the
  quantised-module count. (The regex warning itself is a false positive — see the
  comment above `transformers>=5` in
  `environments/requirements_voxtral_macOS_arm64.txt`.)
* **The penalty rung has no parameter to land on.** mlx-audio's Voxtral
  `generate()` exposes temperature, top-p, top-k and min-p but no
  `repetition_penalty`, so the fallback `model.generate(repetition_penalty=...)`
  call has no counterpart. The vendored loop takes `logits_processors`, and
  `mlx_audio.lm.sample_utils.make_logits_processors(repetition_penalty=...,
  repetition_context_size=...)` builds the same processor `mlx_lm` does — drive
  the rung through `stream_generate`/`generate_step` with that, and pass the
  stop tokens yourself, because `stream_generate` only breaks on
  `tokenizer.eos_token_ids`.
* **Its `generate()` defaults are a smoke-test trap, not an engine concern:**
  `max_tokens=128` and `language="en"`. A quick "does it transcribe" call with
  the defaults truncates at 128 tokens and prompts for English. The engine
  never goes through `generate()`; it will drive `stream_generate` directly.
* Voxtral is one of three STT models in mlx-audio that use `AutoProcessor`
  (`mms` and `qwen2_audio` are the others, as of main 2026-09-03), and for
  Voxtral it resolves to transformers' `VoxtralProcessor`, which **requires
  torch — which `pip install mlx-audio` does not install.** Confirmed on 0.5.1: a clean install
  loads the module fine and then dies in `post_load_hook` with
  `ImportError: VoxtralProcessor requires the PyTorch library`. So torch is an
  undeclared dependency of this path and must go in the requirements explicitly.
  noScribe has torch for the diarizer and the aligner, so this is not a blocker, but
  it changes
  the worker's import graph — re-check `tests/test_worker_import_lightweight.py`
  and the PyInstaller specs. mlx-audio also hard-requires `sounddevice` and
  `miniaudio` (native, PortAudio); the STT import path does not pull them, but pip
  installs them, so the frozen build likely needs excludes. **Prove that with a
  throwaway PyInstaller build — never infer frozen behaviour from source.**

**4. The move does not reduce the `mx.metal` exposure; it enlarges it.**

Measured on `mlx` 0.32.1 with warnings forced on: `mx.metal.is_available()`
emits nothing and has no documented replacement; what is deprecated ("will be
removed in a future version") is `mx.metal.device_info()` and the
`mx.metal.get_*_memory()` family, replaced by the same names on `mx`. `mlx-lm`
0.31.3 calls `mx.metal.is_available()` at nine sites and none of the deprecated
ones — its `wired_limit` already reads `mx.device_info()`. mlx-audio calls
`is_available()` at seven sites and `mx.metal.device_info()` at two, one of
them `mlx_audio/stt/utils.py:wired_limit`, the context manager Voxtral's
`stream_generate` runs inside. So whichever package the engine sits on, a
removal of `mx.metal.is_available` breaks it, and only mlx-audio also breaks
on the removal that is actually announced. Do not cite `mx.metal` as a reason
to move. `mlx` itself is at 0.32.2 (2026-08-25); the pin stays at 0.32.1 until
bumped deliberately.

## The alignment path is not affected — but know this before you touch it

Out of scope here, and stated so you do not go looking: **the German forced-aligner
model already loads through mlx-audio**, via `mms/mms.py`, which wraps the same
`Wav2Vec2Model` encoder and adds the `lm_head` a `Wav2Vec2ForCTC` checkpoint
carries. It needs no code change, only `model_type: "mms"` in a converted config,
and it reproduces the torch emissions exactly — identical argmax on every frame,
max |Δ| 0.00068.

That is not a reason to bundle it in. It is independent of which package loads
Voxtral, the Viterbi half is numpy since 2026-08-23 (`noScribe/ctc_align.py`, which
mlx-audio neither has nor needs to provide), and the aligner already runs on the GPU
where the speed was. A migration that also rewrites alignment cannot be shown to have
changed nothing.

## Do this first: try to avoid re-publishing at all

Before re-quantising 6 GB and 25 GB and making every user re-download, evaluate a
key remap in mlx-audio's Voxtral `sanitize()` that accepts the mlx-voxtral layout
(`language_model.X` → `language_model.model.X`). If that works it is a small
upstream contribution, it fixes the same problem for every other published
mlx-voxtral build, and it removes the largest single cost from this migration. It
is already offered in #902.

Send it as a PR to mlx-audio and see. If it is rejected or takes too long, fall
back to re-quantising.

Checked 2026-09-03: nobody else has sent that PR, #902 has had no reply beyond
a tracker bot, and there is no ready-made build to fall back on either —
`mlx-community` publishes only a bf16 Voxtral-Mini for mlx-audio, and the 8-bit
Mini and Small builds other people uploaded are in the mlx-voxtral layout, so
they load exactly as badly as ours. The remap is the only route that avoids
re-publishing.

## Steps

1. Install `mlx-audio[stt]` **in a throwaway venv first** and confirm the facts
   above still hold against whatever version is current. Do not touch the project
   venv until the approach is settled — verify with `pip freeze` before and after
   that it comes back byte-identical.
2. Try the `sanitize()` remap route. If it works, the build story is solved.
3. Rewrite the six call sites. Expect `_LMAdapter` to become unnecessary —
   mlx-audio's `Model.__call__(input_ids, input_features, cache)` already returns
   logits — and `_merged_embeddings` likewise, since its merge is already a single
   scatter. Delete them only after the equality check in step 5 passes. **Rebuild
   the percentile floor on the new feature path** (the two options are under
   *The coupling is six calls*) and repoint `tests/test_mel_floor.py`'s
   bit-identity reference at the transformers extractor. Do this before step 5:
   without it the equality check measures the floor, not the migration.
4. If re-quantising is needed: `python -m mlx_audio.convert -q --q-bits 8` already
   carries what `tools/quantize_voxtral.py` exists for — the group-size guard
   (`weight.shape[-1] % 64`) and the `not audio_tower` predicate are built into
   its `build_quant_predicate` — so try it before rewriting the tool, and keep
   only what it cannot do. Rebuild `voxtral-mini-8bit` and `voxtral-small-4bit`,
   re-publish, update the model URLs.
5. **Equality check before anything else is believed:** the same audio through the
   old and new paths must produce the same transcript, **with the percentile
   floor active on both sides**. `tests/test_voxtral_smoke.py`
   asserts fast == library today; extend it to assert **the number of quantised
   modules is non-zero**, because the failure mode in fact 1 is silent.
6. Re-measure `MEM_MODEL` (see `docs/voxtral-quantisation.md`, *Decode path and
   memory*) and update the entries.
7. Re-run the two hand-corrected references and FLEURS with
   `docs/scripts/wer.py` and `docs/scripts/fleurs.py`. Expect the numbers to match
   the tables. **If they do not, something in fact 3 is biting — find it, do not
   update the tables.**
8. Update the dependency notes in `VOXTRAL.md` and
   `environments/requirements_voxtral_macOS_arm64.txt`, and the *Re-quantising
   with another tool changes nothing* section of `docs/voxtral-quantisation.md`
   if the build tables move.

## Acceptance

* Transcript identical to the current engine on the same audio, or the difference
  explained and measured.
* `venv/bin/python3 -m pytest tests/ -q` green, including the new
  quantised-module-count assertion.
* WER/CER on both references and FLEURS within noise of the published tables.
* A throwaway PyInstaller build starts and transcribes.
* The loaded processor's tokenizer class is asserted, not assumed.
* No `mlx_voxtral` left in the shipped code:
  `grep -rn mlx_voxtral noScribe/ tools/ tests/ --include="*.py"` empty. The five
  measurement scripts under `docs/scripts/` that import it reproduce numbers
  taken on the old path; they keep the import and get a one-line note saying
  so. Rewriting them would be re-measuring, which step 7 covers with the two
  scripts that matter.

## Traps

* **Never trust a silent load.** Assert quantised-module count, not just "it
  loaded".
* **Do not re-run the bit sweeps.** Fact 2 says they still hold; re-running them is
  days of compute for a known answer.
* **Do not touch the pinned venv** until the approach is settled.
* **Do not infer frozen behaviour.** Build it.
* **Do not delete `_merged_embeddings`' dtype guard reflexively.** It costs nothing
  and protects against a build whose dtypes disagree.
* Read the docstrings in `voxtral_engine.py` before changing a constant. Several
  encode a defect that was expensive to find.

## Two things mlx-audio has that are not this migration

Surveyed 2026-09-03 so they are not rediscovered. Neither is a reason to add
mlx-audio as a *second* dependency next to mlx-voxtral: both jobs are small
enough for the torch stack noScribe already carries, and the added package
brings `miniaudio`, `sounddevice` (native) and a `transformers>=5.14` floor
into the worker's import graph for nothing the GPU is needed for.

* **Language identification** (`mlx_audio.lid`): `facebook/mms-lid-256`
  (wav2vec2, ships safetensors, loads directly) and VoxLingua107-ECAPA
  (community conversions only). Where this would matter is not here but in the
  language-flip guard of `voxtral_engine.py`: its uncovered case is "first
  chunk translated, no loop, language on Auto", because `want_lang` is then
  unknown. Audio LID over the first minute would supply it without any text.
  It helps the *guard*, not the model — `lang:de` in the prompt was measured
  not to prevent the flip. transformers 5.16.1 already has
  `Wav2Vec2ForSequenceClassification` for the same checkpoint, on torch.
* **Silero VAD** (`vad/silero_vad`, `get_speech_timestamps`), as an
  alternative to the energy-based cut points in `_quiet_runs`. Unmeasured, and
  the lost start of a pass was shown to depend on where the window *ends*, not
  on where it is cut (`docs/voxtral-benchmarks.md`, section 5), so do not
  expect a better cut to buy that.

Everything else was already known: no Viterbi anywhere in the tree, the
Qwen3 aligner collapses at length (`docs/viterbi-numpy-brief.md`), sortformer
caps at four speakers, the eval normaliser is English
only, and Voxtral Realtime is fully ported there but was ruled out on Mistral's
own WER figures (`docs/other-asr-engines.md`).

## Rollback

The current state is committed. If the migration stalls,
the pins are stable and nothing is broken — revert. The measurement documents
describe the pre-migration state accurately.
