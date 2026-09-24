"""
Voxtral transcription engine for noScribe (Apple Silicon / MLX).

Provides an alternative to the faster-whisper backend:

- Transcription is done by Mistral's Voxtral (via `mlx-voxtral`), which produces
  clean text but *no* timestamps.
- When timestamps are required (subtitles, speaker assignment, pause marking),
  word-level timestamps are recovered with CTC forced alignment against a
  wav2vec2 model chosen per chunk from the transcribed text (`ALIGN_MODELS`,
  `_AlignerPool`), the same approach WhisperX uses.

The public entry point `transcribe()` streams segment dicts through its
`segment_cb` and returns `(segments, info)`. The dicts are shape-compatible
with what `whisper_mp_worker` streams to the main app:

    {"start": float_seconds, "end": float_seconds, "text": str,
     "words": [{"word": str, "start": float, "end": float, "prob": float}] | None}

Long files are processed in passes so memory and model context stay bounded.

Reading order: constants and pass sizing, chunk cutting, loop detection and
repair, language guards, head recovery, the guarded decode, the log-Mel floor,
the `_Voxtral` model wrapper, the aligner, segment building, `transcribe()`.

Author: Markus Kämmerer <https://markus-kaemmerer.de>
"""

import functools
import importlib.resources as impres
import logging
import math
import os
import re
import time

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Rough count of text tokens Voxtral generates per second of audio, used ONLY to
# drive the intra-pass liveness estimate. Measured German conversation runs at
# ~4.64 text tokens/s (see _transcribe_guarded); this is deliberately set below
# that, so the estimate runs AHEAD of the real position (by ~1.5x) and reaches
# its cap at ~60% of the pass. It is capped at 95% of the pass and snaps to the
# true position when a pass finishes, so running ahead only makes the bar wait
# at the cap for a while, never pass the real chunk boundary.
_EST_TOKENS_PER_SEC = 3.0


def is_available():
    """True if the Voxtral backend and its dependencies can be imported.

    Deliberately NOT cached: _register_voxtral_models re-checks this on every
    dropdown open so a backend pip-installed while the app runs appears without a
    restart. The check is three cheap importlib.util.find_spec lookups (no heavy
    import), so re-running it per dropdown open is negligible."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("mlx_voxtral", "transformers", "soundfile"))


def _log(cb, level, msg):
    if cb:
        try:
            cb(level, msg)
        except Exception:
            pass
    else:
        levels = {"error": logging.ERROR, "warn": logging.WARNING,
                  "info": logging.INFO}
        logger.log(levels.get(level, logging.DEBUG), msg)


# --- Long-audio chunking (reference approach: one pass up to the model's
# context, split only beyond that) -------------------------------------------
# Voxtral uses flash attention + a rolling KV cache, so a single generate()
# call's memory grows ~linearly with the audio length (NOT O(T^2)). We feed the
# whole file in one pass when it fits the memory budget, and only split longer
# files. The per-pass length is chosen from the machine's RAM by
# `_auto_chunk_sec()` -- see MEM_MODEL below -- unless the caller/config pins it.
PREF_MIN_CHUNK_SEC = 180     # preferred quality floor (below this we warn)
HARD_MIN_CHUNK_SEC = 60      # absolute floor; memory safety wins over this only with a warning
MAX_CHUNK_SEC = 1500         # 25 min: safely under Voxtral's ~30 min / 32k ctx
# 10 min: the longest pass Voxtral has been *measured* on, which is a different
# fact from what fits its context and so a different constant. The 30/40-minute
# figure everyone quotes is a capacity calculation -- 12.5 Hz frame rate against
# 32k tokens -- not a quality result. The paper's own long-form ASR protocol is
# shorter: "we take the one-hour long earnings calls from Earnings-21 and
# Earnings-22, and segment them into shorter, 10 minute variants"
# (arXiv:2507.13264). Beyond that there is no published measurement, and past it
# this project has recorded two distinct failures -- whole chunks coming back
# translated (windows around 1425 s) and passes returning without their opening
# (1195 s and 1226 s of a 1226 s file, while 15 shorter points came back clean).
# Neither is a threshold effect that a slightly shorter window would dodge, so
# this is not a fix for either; it is a refusal to run the model twice as far
# out as anyone has measured it. Raise it when a longer window has been measured.
# The cap costs no quality: a 600 s single pass and two 300 s passes punctuate
# the same material equally (10.87 against 10.66 commas per 100 words), and
# chunk length has an optimum set by training rather than a "longer is better"
# curve. What does cost quality is the *boundary*, which pause-aligned cuts and
# the overlap already address.
TRUSTED_CHUNK_SEC = 600
# The binding limit is the Voxtral *generate* working set: it must fit in
# physical RAM, because MLX compute on swapped-out buffers thrashes and never
# finishes. The one-off load transient (weights dict + model briefly duplicated)
# is NOT counted here -- it is allowed to swap since it frees before generate
# starts. So this reserve only holds back non-swappable kernel/wired memory, the
# forced aligner that runs alongside, and a margin; the OS/GUI may page out.
# The generate pass may use (total_RAM - this).
#
# Measured on a 32 GB M1 Max: a 21.8 GB pass ran with 2.7 GB still free, and a
# 26.4 GB pass completed as well, so 7 GB is defensible while leaving room for
# the aligner. Users who free up the machine can go further via the config key
# `voxtral_ram_reserve_gb`, which buys noticeably longer passes for the memory
# hungry builds (6-bit small: 229 s at 8 GB, 304 s at 7, 378 s at 6).
RAM_RESERVE_GB = 7
# Below this much free memory a run does not just get slow, it stops progressing:
# the forced aligner (~2 GB) and the OS still need room next to the model, and
# once MLX has to compute on swapped-out buffers nothing finishes. Used to refuse
# a model outright instead of starting a run that cannot succeed.
# Calibrated on a 32 GB machine: passes peaking at 25 GB (7 GB headroom) ran
# fine, while the 8-bit 24B build at 27.2 GB (4.8 GB headroom) exhausted the swap
# and stopped progressing once the aligner loaded next to it.
MIN_HEADROOM_GB = 6
# Measured peak unified-memory model per pass:  peak_GB ~= fixed + slope*seconds.
#   mini : anchored on a real measurement (600 s single pass = 17.4 GB peak on
#          an M1 Max; the full 1143 s pass exhausted 32 GB -> unsafe).
#   small: the shipped 24B build, voxtral-small-4bit (4-bit LM + lm_head, bf16
#          encoder, 15 GB of weights). Calibrated on the generate_step path:
#          fresh-process full-pass peaks on an M1 Max were 16.65 / 17.25 / 18.53
#          / 19.99 / 21.36 GB at 180 / 300 / 600 / 900 / 1200 s of a podcast and
#          17.35 / 18.99 / 22.54 GB at 300 / 600 / 1200 s of an interview, a line
#          peak ~= 15.71 + 0.0051*s. The entry over-predicts every point by
#          7-17% (7-12% up to the 600 s cap passes can reach), a thinner
#          cushion than mini8's because most of this peak is weights, which do
#          not vary. On 32 GB the 600 s cap binds; on 24 GB it allows short
#          passes (~70 s auto, up to ~210 s pinned).
#   mini8 : the 3B weights at 8 bit with the audio encoder left in bf16
#          (tools/quantize_voxtral.py, mode dense-encoder). RECALIBRATED for the
#          generate_step decode path (which chunks the prompt): fresh-process
#          full-pass peaks on an M1 Max were 6.82 / 7.41 / 8.72 / 10.00 / 11.18 GB
#          at 180 / 300 / 600 / 900 / 1200 s (two podcast files, ~1.6% apart),
#          a clean line peak ~= 6.11 + 0.00427*s. The old one-shot-prefill path
#          measured ~2.5x that slope (120 s = 7.7, 1100 s = 18.7 GB); switching to
#          generate_step genuinely lowered it. The entry below keeps a safety
#          margin over the fit (6.5 + 0.0060*s over-predicts every point by
#          11-23%): the slope is dominated by the audio prompt-token rate (~12.5
#          tok/s, fixed by the recording), so speech density moves it only ~6%,
#          but machine/version variance warrants the cushion. From 24 GB up,
#          mini8 is capped by TRUSTED_CHUNK_SEC anyway, so the recalibration now
#          only binds at 16 GB. small6/small8 below are still on old-path
#          numbers (local-only builds, not re-measured) -- safe, just
#          conservative.
#          Why the encoder stays dense: measured against a hand-corrected
#          German reference, quantising it costs real accuracy while costing
#          nothing to keep -- the encoder runs ONCE per pass, so its precision
#          does not affect speed (6.60x vs 6.68x for the fully-quantised
#          build), and this build reproduces the bf16 transcript word for word
#          at 4.5x the speed. lm_head, by contrast, runs once per generated
#          token: leaving it dense costs 27% throughput and measured no better,
#          so it stays quantised.
#   small8: the 8-bit build of the same layout, shipped until voxtral-small-4bit
#          replaced it so the 24B model fits 32 GB (a 6-bit body measured no
#          better than 4; 8 bit itself is unmeasured). Now only for a build of
#          your own passed as transcribe(voxtral_repo=...); the model menu offers
#          VOXTRAL_MODELS alone, and main.RENAMED_MODELS points the old name at
#          voxtral-small-4bit. Needs
#          ~34 GB minimum, so it is refused below that and wants 48 GB+. Even
#          when it fits it runs at ~0.80x realtime -- slower than the recording.
#          On a 32 GB machine the 25 GB of weights cannot stay resident: a
#          benchmark swapped continuously and did not finish a 25 min pass in
#          4.5 h. This entry exists so auto-sizing refuses it up front there.
#   small6: the same 24B weights quantised to 6 bit instead of 4 (converted
#          locally from the original mistralai release). Measured 150 s =
#          22.9 GB and 410 s = 26.4 GB, i.e. ~6 GB more fixed than the 4-bit
#          build, which is why it only allows much shorter passes on 32 GB.
MEM_MODEL = {
    "mini":   {"fixed": 7.9, "slope": 0.016},
    "mini8":  {"fixed": 6.5, "slope": 0.0060},   # generate_step path; see note above
    "small":  {"fixed": 16.5, "slope": 0.0070},  # generate_step path; see note above
    "small6": {"fixed": 20.9, "slope": 0.0135},
    "small8": {"fixed": 27.4, "slope": 0.0135},
}

# Window size for the wav2vec2 alignment forward pass. Kept small so the
# O(T^2) self-attention memory of the *aligner* stays bounded regardless of
# chunk length (this is the aligner, not Voxtral). 20 s is a judgement call, not
# a measured optimum: wav2vec2 is trained on utterances of that order, and the
# window only has to be long enough that a word never straddles two of them.
EMISSION_WINDOW_SEC = 20
# Hard cap on the forced-alignment DP size, frames * (2*tokens + 1). The DP
# (noScribe/ctc_align.py) keeps one byte per cell for its backtrace, so 2**30
# cells is a 1 GB table and ~5 s for one call -- and halving a window halves
# both axes of that table, so splitting is cheaper than not splitting
# (measured: 587 -> 200 ms on a 300 s chunk in four pieces). The constant began
# as a guard against torchaudio's 32-bit index overflow; that history and the
# figures are in docs/viterbi-numpy-brief.md, the splitting itself is pinned
# by tests/test_forced_align_cap.py.
FORCED_ALIGN_MAX_CELLS = 2**30
# `_Aligner.align_prefix` is free to choose its own piece sizes, so it uses a
# smaller budget than the hard cap: measured with the C++ kernel, one call
# peaks at ~1.0 GB and 3.9 s at 2**30 cells but ~0.3 GB and 0.9 s at 2**28
# (the numpy DP: same memory, ~1.35x the time). The prefix
# salvage runs right after a Voxtral pass, i.e. exactly when memory is
# tightest, and more (smaller) pieces cost no accuracy and slightly less total
# DP work. 2**29 keeps a 1500 s window at ~7 pieces of ~220 s of speech each.
SALVAGE_ALIGN_MAX_CELLS = 2**29
# When splitting a long file, snap each cut to the clear speaker pause nearest
# the target boundary within a *wide* radius of it (the longest quiet run only
# when there is no clear pause; see _chunk_boundaries). Because the passes are
# long we have plenty of slack to hunt far for a real pause, so a cut lands
# between utterances and never mid-word. 90 s is a judgement call: wide enough
# that a 10-minute pass always finds a real pause, narrow enough that consecutive
# passes stay roughly equal in length.
SILENCE_SEARCH_SEC = 90
# A frame counts as "silence" when its AMPLITUDE is at or below this fraction of
# the window's typical (75th-percentile) speech level -- about -16.5 dB. Callers
# compare in the power domain and must square it: _frame_energy returns mean
# squared amplitude, and applying the fraction unsquared puts the gate at
# -8.2 dB, which is quiet speech rather than a pause.
QUIET_LEVEL = 0.15
# Lead-in overlap (seconds) read from the previous chunk so the words right
# after a boundary have preceding audio context; the duplicated overlap is then
# dropped by timestamp. Combined with pause-aware cuts, seams are ~lossless.
# 15 s is a judgement call: a few sentences of context, short against a pass.
OVERLAP_SEC = 15

# Known model repositories (MLX).
VOXTRAL_MODELS = {
    # Two builds, both quantised on Apple Silicon and published so they download
    # on first use (the picker only shows them on arm64 Macs; elsewhere MLX
    # cannot run at all). Each keeps the audio encoder in bf16, because the
    # encoder runs once per pass, so its precision is nearly free, and
    # compressing it measurably costs accuracy on hard audio; the language
    # model and lm_head are quantised (docs/voxtral-quantisation.md).
    #
    # mini (3B): the everyday build, 8 bit. Reproduces the bf16 transcript word
    # for word on our hard-German reference at ~4.5x the speed, runs on any Mac
    # with 16 GB, and is the one to use for conversational audio.
    "voxtral-mini-8bit": "MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder",
    # small (24B): 4 bit, for clean, read-aloud audio and material heavy with
    # names (FLEURS de 2.78 % WER against mini's 4.89 %), ~2x realtime, fits a
    # 32 GB Mac. It replaced an 8-bit build of the same layout that needs
    # ~34 GB; a 6-bit body measured no better than 4, 8 bit is unmeasured
    # (docs/voxtral-quantisation.md). See MEM_MODEL / min_ram_gb.
    "voxtral-small-4bit": "MarkusKaemmerer/Voxtral-Small-24B-2507-4bit-dense-encoder",
}

# The raw unquantised source releases: handing one of these to transcribe()
# would download tens of GB (24B: 48 GB) and, because the basename carries no
# bit width, _model_kind would meter it with a quantised profile and wave a
# too-long pass through. No model here points at them, but a direct caller might
# copy one in, so transcribe() refuses them (see below).
SOURCE_REPOS = frozenset({
    "mistralai/Voxtral-Mini-3B-2507", "mistralai/Voxtral-Small-24B-2507",
})
# The same releases as bare directory names. A copy under the package `models/`
# dir (the documented place for local builds, see _local_copy) resolves to a
# filesystem path, so matching the full repo id alone let exactly the build this
# guard exists to stop walk straight through it.
SOURCE_REPO_NAMES = frozenset(r.rsplit("/", 1)[-1].lower() for r in SOURCE_REPOS)


def _local_copy(name):
    """Path of a converted build under the package `models/` dir, or None.

    The one place that decides "this build exists locally": resolve_model()
    prefers it and has_local_build() gates the picker on it, and both must
    agree on the same filesystem fact.
    """
    try:
        local = impres.files("models") / str(name)
        if (local / "config.json").is_file():
            return str(local)
    except Exception:
        pass
    return None


def has_local_build(name):
    """True if `name` is usable on this machine. Both shipped builds download on
    first use, so this is always true for known models; kept as the picker's
    gate in case a future entry is local-only again."""
    return name in VOXTRAL_MODELS or _local_copy(name) is not None

# Forced-alignment (word timestamps) is language dependent. Use a wav2vec2 CTC
# model that matches the transcription language when we have one (best accuracy,
# native alphabet incl. umlauts/accents), and fall back to a multilingual MMS
# aligner for "auto"/"multilingual" or unmapped languages (handles code-switched
# audio like German+English; characters outside its vocabulary are skipped and
# their timing interpolated).
#
# The language-specific table is not redundant with the multilingual model, which
# is the obvious simplification to reach for. Measured 2026-09-02: the MMS
# forced-aligner's vocabulary is 27 characters, a-z plus apostrophe, with NO
# umlauts at all -- it is romanised by design. The German model has a, o and u
# umlauts. Folding German onto MMS would therefore drop a character from every
# word containing one and interpolate its timing.
#
# Neither vocabulary contains the German sharp s. On the hand-corrected references
# that is 13 of 1268 words (1.03 %) and 13 of 6064 letters (0.21 %), and each such
# word still has every other letter aligned, so the word boundary barely moves.
# Not worth chasing a different checkpoint over -- but it is why "native alphabet"
# above means "nearly all of it", not "all".
ALIGN_MODELS = {
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
}
ALIGN_MODEL_MULTILINGUAL = "MahmoudAshraf/mms-300m-1130-forced-aligner"

# Letters an aligner vocabulary may not carry, spelled the way the models were
# actually trained. Applied ONLY where the letter is missing from that model's
# vocabulary, so a model that does carry it keeps the real one -- the German
# model has the umlauts and must not be handed "ae" for "ä".
#
# Measured 2026-09-02 on the 300 s reference. Both mappings are what the models
# themselves produce when decoded freely, which is the only evidence that settles
# the spelling:
#
#   sharp s -> "ss". The German aligner writes "weiss" and "heisst" of its own
#   accord, so its training text was normalised that way. Dropping the letter
#   instead left a word-final /s/ unaccounted for and ended the word too early:
#   the three such words moved +80, +100 and +160 ms later, and all 841 other
#   words in the chunk moved by exactly 0 ms. Word-internal occurrences do not
#   move, because the boundary comes from the letters around them -- confirmed on
#   the second reference (hart_780-900), whose four occurrences are all internal
#   and where nothing moved at all, in any of its 418 words. Across both
#   references word-final is 3 of 1268 words, so this is a small and very local
#   correction, not a broad one.
#
#   umlauts -> "ae"/"oe"/"ue", not "a"/"o"/"u". The multilingual MMS aligner
#   carries a-z and an apostrophe only, and is romanised in the German manner
#   rather than stripped -- it writes "naechste" for "nächste". With this table
#   its mean frame score on the same audio improves from -0.278 to -0.256 while
#   aligning 86 MORE tokens, which biases the mean the other way.
# German only, and deliberately so. Extending the table looks free -- an entry
# can only fire where the letter is missing anyway -- but the romanisation it
# encodes is language-specific: "ae" is the German rule, while uroman, which the
# multilingual model follows, gives plain "a" for Swedish. Languages without
# their own entry in ALIGN_MODELS fall back to that same multilingual model, so
# a table with no notion of language would apply the German rule to Swedish,
# Czech and Turkish alike. Those characters are no longer lost either way: since
# align_words gained its wildcard column they are placed on the audio rather
# than dropped, which is less precise than a correct respelling and much safer
# than a wrong one. Add a language here only after measuring it, the way these
# two were measured -- not on the reasoning that it cannot hurt.
_ALIGN_CHAR_FALLBACKS = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def min_ram_gb(repo):
    """Smallest amount of RAM in which this model can realistically run.

    Shown next to each model in the picker, so the choice is made before a run
    starts rather than after the machine has begun to swap.
    """
    m = MEM_MODEL[_model_kind(repo)]
    return m["fixed"] + m["slope"] * HARD_MIN_CHUNK_SEC + MIN_HEADROOM_GB


def max_safe_chunk_sec(repo):
    """Longest pass (seconds) whose estimated generate peak still fits physical
    RAM, i.e. the hard ceiling past which a run stops progressing (see
    MIN_HEADROOM_GB). This is THE memory-safety invariant: the picker hint
    (min_ram_gb), the automatic sizing and the pinned-value clamp must all
    agree with it, so it lives in exactly one place.

    Raises MemoryError when even the shortest pass (HARD_MIN_CHUNK_SEC) cannot
    fit -- proceeding would not fail, it would swap forever.
    """
    kind = _model_kind(repo)
    m = MEM_MODEL[kind]
    total = _total_ram_gb()
    need = min_ram_gb(repo)
    if total < need:
        # The whole requirement, headroom included: the peak alone ("17 GB")
        # read as if it fit an 18 GB machine that was just refused.
        raise MemoryError(
            f"Voxtral {kind} needs about {need:.0f} GB even for its shortest "
            f"chunk, which does not fit in {total:.0f} GB. Pick a smaller model "
            f"(see the memory hint next to each model) or use a machine with more RAM."
        )
    if not m["slope"]:
        return MAX_CHUNK_SEC
    return (total - MIN_HEADROOM_GB - m["fixed"]) / m["slope"]


def resolve_align_model(language):
    """Pick the alignment model for a language code (e.g. "de"); fall back to the
    multilingual aligner for None/"auto"/"multilingual"/unmapped languages."""
    code = (language or "").strip().lower()[:2]
    return ALIGN_MODELS.get(code, ALIGN_MODEL_MULTILINGUAL)


def _model_kind(repo):
    """Classify a repo id / local path into a MEM_MODEL entry.

    Only the final path component (model dir / repo name) is matched, so a
    parent directory that happens to contain "small" cannot misclassify the
    model and shrink its passes for no reason. The bit width matters as much as
    the parameter count: a 6-bit 24B build needs ~6 GB more than the 4-bit one,
    and mistaking one for the other would size passes too long and exhaust RAM.

    A name with no recognisable size token falls back to the most conservative
    (highest-RAM) profile, never the cheap `mini` one, so an unrecognised build
    is sized safely-short rather than metered too generously and swapped.
    """
    name = os.path.basename(os.path.normpath(str(repo))).lower()

    def _has(*tokens):
        return any(t in name for t in tokens)

    eight = _has("8bit", "8-bit")
    # The shipped builds keep the audio encoder in bf16 ("dense-encoder"); the
    # encoder is small, so this adds well under a GB and the bit-width profile
    # still fits (mini dense-encoder measured 6.35 GB fixed vs 6.0 for uniform
    # 8-bit -- within the mini8 entry). Classify by parameter count and bit
    # width; the MEM_MODEL entries already carry a safety margin.
    if "small" in name or "24b" in name:
        if eight:
            return "small8"
        if _has("6bit", "6-bit"):
            return "small6"
        if _has("4bit", "4-bit"):
            return "small"
        # A 24B build whose name carries no bit width at all. `small` is the
        # 4-bit profile and the CHEAPEST of the three 24B entries, so metering an
        # unquantised or 6-bit copy with it waves a several-hundred-second pass
        # through for weights that need 25-48 GB, and the run swaps forever
        # instead of being refused. Fall back to the hungriest 24B profile, the
        # same direction this function's docstring promises for a name with no
        # size token at all. (The mini branch below needs no such guard: without
        # a bit width it already lands on `mini`, the bf16 and more expensive of
        # the two mini entries.)
        logger.warning(
            "Voxtral build %r is a 24B model with no bit width in its name; "
            "sizing passes with the conservative small8 profile. Name a local "
            "build with its bit width (e.g. voxtral-small-8bit) to have it "
            "sized correctly.", name)
        return "small8"
    if "mini" in name or "3b" in name:
        return "mini8" if eight else "mini"
    # Unrecognised build: the name carries no reliable size signal. Under-sizing
    # a pass only runs slower, but over-sizing swaps forever -- so fall back to
    # the most memory-hungry profile rather than optimistically metering an
    # unknown model as the cheap `mini` one (which would wave a too-long pass
    # through on a large model). The old code defaulted anything without "small"
    # to mini; that is the direction that can swap.
    logger.warning(
        "Unrecognised Voxtral build %r; sizing passes with the conservative "
        "small8 profile. Name a local build with mini/small and its bit width "
        "(e.g. voxtral-mini-8bit) to have it sized correctly.", name)
    return "small8"


@functools.lru_cache(maxsize=None)
def _total_ram_gb():
    """Total physical RAM in GB (sysctl on macOS; psutil fallback; else 16).

    Cached: total RAM is a process constant, but this is queried once per model
    on every model-dropdown open (via App.model_label) and again during a job,
    so without the cache each call would spawn a `sysctl` subprocess for a value
    that never changes."""
    try:
        import subprocess
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 1024**3
    except Exception:
        try:
            import psutil
            return psutil.virtual_memory().total / 1024**3
        except Exception:
            return 16.0  # conservative default


def _auto_chunk_sec(repo, log_cb=None, ram_reserve_gb=None):
    """Choose the per-pass length (seconds) from the machine's RAM.

    Voxtral feeds the whole pass into one generate() call, whose peak
    unified-memory is ~`fixed + slope*seconds` (MEM_MODEL). We pick the longest
    pass whose estimated peak still fits `(total_RAM - RAM_RESERVE_GB)`, clamped
    to [HARD_MIN_CHUNK_SEC, TRUSTED_CHUNK_SEC] and to what `max_safe_chunk_sec`
    allows. So most files go through in a single pass on a roomy
    machine, while low-RAM machines (or the memory-hungry 24B `small` model)
    automatically get shorter, still-safe passes -- and a warning when RAM is the
    limiting factor.
    """
    kind = _model_kind(repo)
    m = MEM_MODEL[kind]
    total = _total_ram_gb()
    # Refuses outright when even the shortest pass cannot fit: proceeding would
    # not merely run slowly -- the working set no longer fits, MLX computes on
    # swapped-out buffers and the run stops making progress at all (observed
    # with the 8-bit 24B build, which sat at 25 GB while the aligner pushed the
    # machine into swap).
    ceiling = max_safe_chunk_sec(repo)
    # `is None`, not falsiness: an explicit 0 means "hold nothing back" and must
    # not quietly turn into the 7 GB default (the GUI already maps its 0 to None).
    reserve = RAM_RESERVE_GB if ram_reserve_gb is None else float(ram_reserve_gb)
    budget = total - reserve
    raw = (budget - m["fixed"]) / m["slope"] if m["slope"] else MAX_CHUNK_SEC
    # A reserve below MIN_HEADROOM_GB must not turn into a refusal (the model
    # fits -- the *reserve* is what doesn't); it just can't buy passes beyond
    # the hard ceiling.
    chunk = int(max(HARD_MIN_CHUNK_SEC,
                    min(TRUSTED_CHUNK_SEC, MAX_CHUNK_SEC, raw, ceiling)))
    est_peak = m["fixed"] + m["slope"] * chunk
    if raw < HARD_MIN_CHUNK_SEC:
        _log(log_cb, "warn",
             f"Low RAM for Voxtral {kind} ({total:.0f} GB total): chunks forced "
             f"to {chunk}s (~{est_peak:.0f} GB peak) and may still swap. Consider "
             f"the mini model or a machine with more RAM.")
    elif raw < PREF_MIN_CHUNK_SEC:
        _log(log_cb, "info",
             f"Voxtral {kind}: {total:.0f} GB RAM -> short {chunk}s chunks "
             f"(~{est_peak:.0f} GB peak). More RAM allows longer, higher-context chunks.")
    else:
        _log(log_cb, "info",
             f"Voxtral {kind}: {total:.0f} GB RAM -> ~{chunk // 60}m{chunk % 60:02d}s "
             f"chunks (~{est_peak:.0f} GB peak).")
    return chunk


# --------------------------------------------------------------------------- #
# Text-based language detection for the aligner choice
# --------------------------------------------------------------------------- #
# Voxtral auto-detects the spoken language internally but never reports it --
# yet the *aligner* choice decides timestamp quality: a char-native
# per-language model clearly beats the romanised multilingual fallback
# (measured German word boundaries @50ms: ~75% for romanised MMS-CTC,
# arXiv 2606.10675). The transcribed text of a chunk exists BEFORE that chunk
# is aligned, so the language can be read off the text itself: function words
# are so frequent that a handful per language decides within a few hundred
# words. Mixed speech (e.g. German with English phrases) keeps the majority
# language's function words dominant, so the majority model wins -- and it
# aligns the minority-language words fine too (same Latin alphabet, and the
# CTC emissions only anchor characters). Only when no language dominates does
# the romanised multilingual model take over.
_STOPWORDS = {
    "de": {"der", "die", "das", "und", "ich", "nicht", "ist", "wir", "ein",
           "eine", "mit", "auf", "für", "aber", "auch", "dann", "wenn",
           "noch", "dass", "sind", "habe", "schon", "mal", "jetzt"},
    "en": {"the", "and", "you", "that", "this", "not", "with", "for", "have",
           "are", "was", "but", "they", "what", "just", "like", "know",
           "then", "would", "been", "your", "about"},
    "es": {"que", "los", "del", "las", "por", "con", "una", "para", "está",
           "pero", "como", "más", "muy", "bien", "eso", "esta", "todo",
           "porque", "cuando", "también"},
    "fr": {"les", "des", "est", "que", "qui", "pas", "vous", "nous", "une",
           "dans", "pour", "mais", "avec", "sur", "c'est", "plus", "tout",
           "être", "fait", "comme"},
    "it": {"che", "per", "con", "una", "sono", "non", "come", "anche",
           "questo", "della", "più", "perché", "quindi", "cosa", "molto",
           "quando", "questa", "hanno", "fare", "essere"},
    "nl": {"het", "een", "van", "dat", "niet", "voor", "maar", "zijn", "ook",
           "dan", "dus", "nog", "naar", "wel", "hebben", "deze", "worden",
           "heeft", "kunnen", "moet"},
    "pt": {"que", "não", "uma", "para", "com", "mais", "como", "mas", "por",
           "isso", "você", "muito", "então", "tem", "está", "também", "vamos",
           "fazer", "quando", "porque"},
    "pl": {"nie", "jest", "się", "tak", "ale", "czy", "jak", "dla", "tego",
           "być", "przez", "tylko", "bardzo", "może", "już", "gdzie",
           "wszystko", "jeszcze", "przecież", "trzeba"},
}
# Non-Latin scripts identify their language directly (no stopwords needed).
_SCRIPT_RANGES = (
    ("ja", "぀", "ヿ"),   # hiragana + katakana (checked before han)
    ("zh", "一", "鿿"),   # han
    ("ru", "Ѐ", "ӿ"),   # cyrillic
    ("ar", "؀", "ۿ"),   # arabic
    ("el", "Ͱ", "Ͽ"),   # greek
)


# What _detect_language can answer; any other language it can only misname.
_DETECTABLE = frozenset(_STOPWORDS) | {code for code, _, _ in _SCRIPT_RANGES}


def _detect_language(text):
    """Best-effort language of `text` -> (code, dominance) or (None, 0.0).

    Script check first (CJK/Cyrillic/Arabic/Greek are unambiguous), then
    function-word counting for the Latin-script languages. Returns None when
    the evidence is thin or no language dominates clearly -- the caller then
    keeps its previous choice or the multilingual fallback.
    """
    if not text:
        return None, 0.0
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 40:
        return None, 0.0
    for code, lo, hi in _SCRIPT_RANGES:
        n = sum(1 for c in letters if lo <= c <= hi)
        if n / len(letters) > 0.3:
            return code, n / len(letters)
    tokens = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text.lower())
    if len(tokens) < 40:
        return None, 0.0
    hits = {code: sum(1 for t in tokens if t in words)
            for code, words in _STOPWORDS.items()}
    ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
    best_code, best = ranked[0]
    second = ranked[1][1]
    total = sum(hits.values())
    # Enough absolute evidence, clear lead over the runner-up, and function
    # words at a plausible density for connected speech.
    if best >= 8 and best >= 1.5 * max(second, 1) and best / len(tokens) >= 0.04:
        return best_code, best / max(total, 1)
    return None, 0.0


# A second alignment, level-matched, for a pass whose words leave diarized speech
# empty. Normalising each window (_Aligner._forward_windows) cannot lift a voice
# 15-20 dB below one that talks through the same window, and the DP then crowds
# the quiet voice's words into the loud one's time. Dividing the audio by its
# own smoothed envelope first fixes that, but levelling everything costs on
# ordinary material (CallHome +0.1 to +0.24 points of words under the wrong
# speaker after the voice check) and now and then breaks a file that aligned
# well. So it only replaces the first alignment where its words cover the
# diarized speech clearly better -- the one symptom of the failure that needs no
# ground truth. Chosen on 32 recordings of a second-voice corpus with AMI,
# CallHome and VoxConverse, then checked on 154 recordings of that corpus that
# took no part (paired 95 % intervals, words under the wrong speaker after the
# voice check): call centre -1.04 points [-2.21, -0.15], office -0.19
# [-1.21, +0.45]; on 105 AMI, CallHome and VoxConverse files not a single pass
# switches. Margins from +2 to +10 points lie on one plateau. The second
# alignment runs where the first leaves over LEVEL_COVERAGE_MARGIN of the speech
# uncovered: on 2 of 3 AMI passes (far-field meetings), 6-11 % of CallHome's and
# none of VoxConverse's -- a second forward pass and DP over the pass, some
# seconds against its decode. It runs after _free_decode_buffers, once the first
# alignment's emission is gone, and adds one levelled copy of the pass (96 MB at
# 1500 s, ~0.5 GB for a moment while its envelope is taken) -- well inside
# MIN_HEADROOM_GB, so the pass length needs no allowance for it.
LEVEL_ENVELOPE_SEC = 0.4      # smoothing of the envelope the audio is divided by
LEVEL_FLOOR_PERCENTILE = 30   # the envelope is never taken below this percentile:
#                               pauses and room tone stay quiet as long as they are
#                               under that share of the pass (a longer silence is
#                               lifted with the rest -- the coverage check is what
#                               keeps such an alignment out). 50 already loses most
#                               of the gain (32 % of the quiet voice's words wrong
#                               instead of 5 %)
LEVEL_MAX_GAIN_DB = 60        # and never more than this below the envelope's peak:
#                               a pass that is over 30 % digital silence has a
#                               percentile of 0, and near-zero dither beside it
#                               would otherwise be lifted without bound, or divided
#                               by zero (3 of 449 measured passes lie below it; with
#                               it the holdout result above is the same to the word)
LEVEL_COVERAGE_MARGIN = 0.05  # coverage the second alignment must add to be taken
# The coverage's resolution and reach are the values the measurement above used,
# not tuned; the margin and the holdout result rest on them.
COVERAGE_FRAME_SEC = 0.1
COVERAGE_REACH_SEC = 0.5      # a word counts for the speech this close to it


def _level_matched(audio):
    """`audio` divided by its moving RMS over LEVEL_ENVELOPE_SEC, never by less
    than that envelope's LEVEL_FLOOR_PERCENTILE percentile nor more than
    LEVEL_MAX_GAIN_DB below its peak; all-zero audio comes back unchanged. The
    moving mean is centred and mirrored at the edges as scipy's uniform_filter1d
    has it (the measurement used that; scipy is no dependency here). Buffers are
    reused: this runs right after a decode, and a 1500-s pass would otherwise hold
    ~0.9 GB of float64 temporaries at once."""
    import numpy as np
    n = max(1, int(LEVEL_ENVELOPE_SEC * SAMPLE_RATE))
    sq = np.pad(np.square(audio, dtype=np.float32), (n // 2, (n - 1) // 2), mode="symmetric")
    # A running sum of non-negative values never decreases, so the differences are
    # never negative and their root never NaN -- unlike uniform_filter1d's running
    # update, which dips a hair below zero after a loud stretch and so turned a
    # whole measured pass into NaN.
    c = np.empty(len(sq) + 1)
    c[0] = 0.0
    np.cumsum(sq, dtype=np.float64, out=c[1:])
    del sq
    env = c[n:] - c[:-n]
    del c
    env /= n
    np.sqrt(env, out=env)
    floor = max(np.percentile(env, LEVEL_FLOOR_PERCENTILE), env.max() * 10 ** (-LEVEL_MAX_GAIN_DB / 20))
    if floor <= 0:
        return audio
    np.maximum(env, floor, out=env)
    out = env.astype(np.float32)
    del env
    return np.divide(audio, out, out=out)


def _speech_coverage(stamps, turns, t_offset):
    """Share of diarized speech (COVERAGE_FRAME_SEC frames inside any turn) that
    has a word within COVERAGE_REACH_SEC. `turns` are window-relative
    (_clip_turns), `stamps` absolute, as align_words returns them."""
    import numpy as np
    frames = int(max(end for _, end, _ in turns) / COVERAGE_FRAME_SEC) + 1
    speech = np.zeros(frames, bool)
    near = np.zeros(frames, bool)
    for start, end, _ in turns:
        speech[int(start / COVERAGE_FRAME_SEC):int(end / COVERAGE_FRAME_SEC)] = True
    for w in stamps:
        near[max(0, int((w["start"] - t_offset - COVERAGE_REACH_SEC) / COVERAGE_FRAME_SEC)):
             int((w["end"] - t_offset + COVERAGE_REACH_SEC) / COVERAGE_FRAME_SEC) + 1] = True
    return (speech & near).sum() / max(1, speech.sum())


class _AlignerPool:
    """Chooses and caches the alignment model chunk by chunk.

    With an explicit language the model stays fixed (as before), but each
    chunk's text is still checked so a wrong menu choice produces a warning
    instead of silently degraded timestamps. With language=None
    (Auto/Multilingual) the model follows the *detected* language of each
    chunk -- Auto gets the char-native per-language quality automatically,
    the language may change mid-file, and only text without a dominant
    language falls back to the romanised multilingual model. Loaded aligners
    are cached (bounded -- they are ~1.2 GB each) because a load costs
    seconds and languages may alternate."""

    MAX_CACHED = 2

    def __init__(self, language, log_cb):
        self._explicit = bool((language or "").strip())
        self._language = (language or "").strip().lower()[:2]
        self._log_cb = log_cb
        self._cache = {}          # model name -> _Aligner, insertion-ordered
        self._last_model = None
        self._warned = False

    def align_for_salvage(self, words, window):
        """Alignment for the loop ladder: maps a clean prefix onto its audio
        time so that only the remainder has to be decoded again. It lives here
        rather than as a closure in transcribe() because a closure binds to a
        variable name -- and exactly that broke silently once, when the
        alignment architecture was moved onto this pool.

        `remember=False` because this is an *internal* alignment of a fragment
        of a pass that may yet be discarded: were it allowed to move the pool's
        state, the NEXT chunk would silently inherit the fragment's language,
        and the one-off language warning would be spent on text nobody ever
        sees. `align_prefix` rather than `align_words` because the prefix covers
        only the start of the window: there the audio is not shared out by the
        words' character counts; instead the words are split into pieces and
        each piece is aligned against all of the remaining audio."""
        return self.aligner_for(" ".join(words), remember=False).align_prefix(
            words, window)

    def align_words(self, words, window, t_offset, turns=None):
        """Word timestamps for a whole pass, with the aligner its text calls
        for. Through here no caller ever holds an aligner: a reference kept
        across chunks outlived its eviction, and _release_gpu_memory then
        freed nothing (measured in its docstring).

        With `turns` (window-relative, from _clip_turns) naming two speakers or
        more, a level-matched second alignment replaces the first where its words
        cover the diarized speech by LEVEL_COVERAGE_MARGIN more (see there). It
        is not even run where the first already covers 1 - LEVEL_COVERAGE_MARGIN
        or more: coverage cannot exceed 1, so the second could never be taken."""
        aligner = self.aligner_for(" ".join(words))
        stamps = aligner.align_words(words, window, t_offset=t_offset)
        if not turns or len({label for _, _, label in turns}) < 2:
            return stamps
        covered = _speech_coverage(stamps, turns, t_offset)
        if covered >= 1 - LEVEL_COVERAGE_MARGIN:
            return stamps
        levelled = aligner.align_words(words, _level_matched(window), t_offset=t_offset)
        covered_levelled = _speech_coverage(levelled, turns, t_offset)
        # Words spread evenly because a window could not be aligned (prob 0.0)
        # sit near every diarized frame and would win on coverage alone.
        guessed = lambda ws: sum(not w.get("prob") for w in ws)
        if covered_levelled > covered + LEVEL_COVERAGE_MARGIN and guessed(levelled) <= guessed(stamps):
            _log(self._log_cb, "info",
                 f"Word timestamps level-matched: they cover {covered_levelled:.0%} "
                 f"of the diarized speech instead of {covered:.0%} (a quieter voice).")
            return levelled
        _log(self._log_cb, "debug",
             f"Level-matched word timestamps not taken: {covered_levelled:.0%} of the "
             f"diarized speech covered, against {covered:.0%}.")
        return stamps

    def aligner_for(self, text, remember=True):
        """The aligner for `text`. `remember=False` only reads: the choice for
        the next chunk and the one-off language warning stay untouched."""
        detected, _ = _detect_language(text)
        if self._explicit:
            model = resolve_align_model(self._language)
            if (remember and _reads_as_other(detected, self._language)
                    and not self._warned):
                self._warned = True
                _log(self._log_cb, "warn",
                     f"Language is set to '{self._language}' but the transcript "
                     f"looks like '{detected}'. Word timestamps use the "
                     f"'{self._language}' alignment model; check the language "
                     f"setting if that was not intended.")
        elif detected:
            model = ALIGN_MODELS.get(detected, ALIGN_MODEL_MULTILINGUAL)
            if remember and model != self._last_model:
                _log(self._log_cb, "info",
                     f"Detected language '{detected}' -> alignment model: {model}")
        else:
            # Thin or mixed evidence: keep what worked for the previous chunk,
            # multilingual on the very first one.
            model = self._last_model or ALIGN_MODEL_MULTILINGUAL
            if remember and model != self._last_model:
                _log(self._log_cb, "info",
                     f"No dominant language detected -> alignment model: {model}")
        return self._load(model, remember=remember)

    @staticmethod
    def _release_gpu_memory():
        """Return an evicted aligner's GPU memory to the system.

        Dropping the reference frees torch's *allocator* but not the driver
        reservation: measured, `driver_allocated_memory` stayed at 2.04 GB and
        only fell to zero on an explicit `empty_cache()`. On Auto a file with
        three alignment languages would otherwise hold that for the rest of the
        job, against a MIN_HEADROOM_GB budget MLX is also drawing on.

        **Call this only after the last reference to the evicted aligner is
        gone**, and note that dropping the dict entry is not enough on its own.
        Measured on one evicted aligner, driver memory went 2.04 GB -> 2.04 GB
        while a local still held it, -> 1.01 GB once that reference went away,
        and only -> 0.00 GB after a `gc.collect()`: a `Wav2Vec2ForCTC` sits in
        reference cycles, so refcounting alone leaves half of it resident until
        the collector runs.
        """
        try:
            import gc

            import torch
            gc.collect()
            torch.mps.empty_cache()
        except Exception:
            pass

    def _load(self, model, remember=True):
        aligner = self._cache.pop(model, None)
        if aligner is None:
            # Evict before loading, not after: an aligner is ~1.2 GB, and this
            # runs right after a Voxtral pass, when headroom is tightest. Loading
            # first would hold MAX_CACHED + 1 of them at once.
            while len(self._cache) >= self.MAX_CACHED:
                oldest = next(iter(self._cache))
                on_gpu = getattr(self._cache[oldest], "device", "cpu") != "cpu"
                del self._cache[oldest]      # last reference to the evicted one
                if on_gpu:
                    self._release_gpu_memory()
            if self._cache and _unfetchable(model):
                # Offline, a chunk in a language whose aligner was never
                # downloaded would otherwise end the whole job after its pass
                # was decoded. The aligner already in use anchors another
                # language's words too (see _STOPWORDS), only less precisely.
                fallback = next(reversed(self._cache))
                _log(self._log_cb, "warn",
                     f"The alignment model {model} is not downloaded and cannot be "
                     f"fetched now; aligning this chunk with {fallback} instead.")
                aligner = self._cache.pop(fallback)
                self._cache[fallback] = aligner
                return aligner
            _log(self._log_cb, "info", f"Loading alignment model: {model}")
            aligner = _load_from_hub(lambda: _Aligner(model), "alignment model",
                                     model, _ALIGNER_DOWNLOAD_GB, self._log_cb)
        self._cache[model] = aligner   # re-insert = mark most recently used
        if remember:
            self._last_model = model
        return aligner


def _frame_energy(window, frame):
    """Per-frame mean squared amplitude of `window` in `frame`-sample blocks --
    the energy curve used to find quiet spots. Trailing samples that don't fill
    a whole frame are dropped."""
    import numpy as np
    nf = len(window) // frame
    return (window[:nf * frame].reshape(nf, frame).astype(np.float32) ** 2).mean(axis=1)


def _quietest_frame_near(audio, target, radius_sec):
    """Sample index of the quietest 100 ms frame within +/- `radius_sec` of
    `target`, so a cut lands in a pause rather than mid-word. Returns `target`
    unchanged when the search window is too small to snap."""
    frame = max(1, int(0.1 * SAMPLE_RATE))
    lo = max(frame, target - int(radius_sec * SAMPLE_RATE))
    hi = min(len(audio) - frame, target + int(radius_sec * SAMPLE_RATE))
    if hi - lo <= frame:
        return target
    window = audio[lo:hi]
    if len(window) // frame < 2:
        return target
    return lo + int(_frame_energy(window, frame).argmin()) * frame + frame // 2


def _quiet_runs(energy, nf, thresh, good_run, target_f):
    """Scan the frame energies for runs at or below `thresh`.

    Returns (nearest_clear_pause, longest_run, longest_run_len) in frame units:
    the midpoint of the clear pause (>= `good_run` frames) closest to
    `target_f`, plus the longest quiet run as a fallback.
    """
    silent = energy <= thresh
    best_close, best_close_d = None, None       # clear pause nearest target
    best_long, best_long_len = None, 0          # longest quiet run (fallback)
    i = 0
    while i < nf:
        if silent[i]:
            j = i
            while j < nf and silent[j]:
                j += 1
            run, mid = j - i, (i + j) // 2
            if run > best_long_len:
                best_long_len, best_long = run, mid
            if run >= good_run:
                d = abs(mid - target_f)
                if best_close_d is None or d < best_close_d:
                    best_close_d, best_close = d, mid
            i = j
        else:
            i += 1
    return best_close, best_long, best_long_len


def _chunk_boundaries(audio, chunk_len, back_len, fwd_len, max_len=None):
    """Split `audio` into ~`chunk_len`-sample passes, snapping every cut to a real
    speaker pause found in the window `[target - back_len, target + fwd_len]`.

    The search leans *backward*: a shorter pass is always memory-safe, so we hunt
    far back (large `back_len`) for a genuine pause and allow only a small forward
    reach (`fwd_len`) to avoid growing the pass. Among clear pauses we take the one
    *closest to the target* (keeps passes near the intended length); if none is
    clearly a pause we fall back to the longest quiet run, then to the target
    itself. Cutting between utterances means a pass never splits a word, and with
    the lead-in overlap the seam is effectively lossless.

    `max_len` is a hard per-pass cap (the RAM-safe length): no pass — including
    the final one after tail-merging and backward snaps — ever exceeds it. And
    no pass comes out shorter than about a quarter of `chunk_len`: when backward
    snaps leave a remainder just over `max_len` but too short for a full pass
    plus a real tail, it is halved instead (see below).
    """
    import numpy as np
    n = len(audio)
    max_len = max_len or chunk_len * 2
    if n <= min(chunk_len, max_len):
        return [0, n]
    frame = max(1, int(0.05 * SAMPLE_RATE))  # 50 ms resolution
    good_run = 8   # >= ~400 ms silence counts as a clear speaker pause
    min_run = 3    # >= ~150 ms silence is an acceptable fallback cut
    long_tail = chunk_len + chunk_len // 2
    bounds = [0]
    while True:
        remaining = n - bounds[-1]
        # Done when the tail fits one RAM-safe pass AND is not much more than
        # the balanced length (small tails merge into the final pass instead of
        # becoming a tiny extra chunk) — but a tail beyond max_len always gets
        # another cut, so backward snaps can't inflate the final pass past the
        # memory budget.
        if remaining <= min(max_len, long_tail):
            break
        # A full pass here would leave less than half a pass behind it. That
        # happens when earlier cuts snapped backward and the remainder ended up
        # just over max_len: aiming at chunk_len then cut 1 s before the end
        # (measured on a crafted 1190 s file at 600 s: 589.5 + 599.9 + 0.55 s).
        # Halve the remainder instead; both halves stay within max_len. Either
        # way the target lies before the end of the audio.
        step = min(chunk_len if remaining >= long_tail else remaining // 2, max_len)
        target = bounds[-1] + step
        # never let a pass shrink below half the step, even hunting backward
        lo = max(bounds[-1] + max(frame, step // 2), target - back_len)
        # ...never let a snap or tail-merge push this pass past max_len, and
        # never leave less than half a step after the cut
        hi = min(target + fwd_len, bounds[-1] + max_len, n - step // 2)
        if hi - lo <= 2 * frame:
            cut = target
        else:
            window = audio[lo:hi]
            nf = len(window) // frame
            energy = _frame_energy(window, frame)
            speech = float(np.percentile(energy, 75))
            target_f = (min(target, hi) - lo) / frame  # target position in frames
            # _frame_energy returns mean SQUARED amplitude, so an amplitude
            # fraction has to be squared before it can be compared against it.
            # Applying QUIET_LEVEL unsquared put the gate at -8.2 dB below the
            # typical speech level -- that is ordinary quiet speech, not a pause
            # (a real pause sits 20-40 dB down). Any 400 ms of unstressed
            # syllables then counted as a "clear speaker pause" and, being
            # nearer the target, beat the genuine pause: measured, the cut moved
            # off a real 1.5 s pause at 589.8 s to 598.5 s, mid-word. The
            # preceding pass has no overlap to fall back on, so its last word
            # was decoded cut in half.
            best_close, best_long, best_long_len = _quiet_runs(
                energy, nf, speech * QUIET_LEVEL ** 2 + 1e-9, good_run, target_f)
            if best_close is None and best_long_len < min_run:
                # Nothing that quiet anywhere in the window: a recording with a
                # high noise floor (room tone, hum, constant background). Rather
                # than give up and cut at the raw target, fall back to the
                # looser gate -- its relative minimum is still the best pause on
                # offer here, which is what the old threshold always used.
                best_close, best_long, best_long_len = _quiet_runs(
                    energy, nf, speech * QUIET_LEVEL + 1e-9, good_run, target_f)
            if best_close is not None:
                cut = lo + best_close * frame + frame // 2
            elif best_long_len >= min_run:
                cut = lo + best_long * frame + frame // 2
            else:
                cut = target
        bounds.append(min(max(cut, bounds[-1] + frame), n))
    bounds.append(n)
    return bounds


def _pass_bounds(audio, chunk_sec):
    """Sample offsets of the passes for a RAM-safe pass length of `chunk_sec`.

    Balance the passes: take the fewest passes that keep each within the
    RAM-safe length, then split evenly. Even passes share the memory headroom
    and avoid a tiny leftover tail (e.g. 1143s @1006 -> 2x571s, not 1006+137),
    which also lowers the per-pass peak.
    """
    max_len = max(1, int(chunk_sec * SAMPLE_RATE))
    n_passes = max(1, math.ceil(len(audio) / max_len))
    chunk_len = math.ceil(len(audio) / n_passes)
    # Wide backward hunt for a real pause (a shorter pass is always memory-safe),
    # small forward reach so a cut can't grow the pass much past the RAM budget.
    back_len = min(int(SILENCE_SEARCH_SEC * SAMPLE_RATE), chunk_len // 2)
    fwd_len = int(min(SILENCE_SEARCH_SEC, 20) * SAMPLE_RATE)
    return _chunk_boundaries(audio, chunk_len, back_len, fwd_len, max_len)


def resolve_model(name, default_repo=None):
    """Prefer a persistent local copy under the package `models/` dir (e.g.
    `models/voxtral-mini`) so models don't have to live in the HF cache and
    won't be re-fetched after an aborted run. Falls back to the Hugging Face
    repo id (download on first use) when no local copy is present.
    """
    default_repo = default_repo or VOXTRAL_MODELS.get(name, name)
    return _local_copy(name) or default_repo


# Friendlier names for whole-component reporting in _quant_summary.
_COMPONENT_LABELS = {
    "audio_tower": "audio encoder",
    "multi_modal_projector": "projector",
}
_DTYPE_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def _weight_names(repo):
    """All tensor names of a local build, from the safetensors index (sharded)
    or the single-file safetensors header. Empty list when unreadable."""
    import json
    import struct
    try:
        with open(os.path.join(str(repo), "model.safetensors.index.json")) as f:
            return list(json.load(f)["weight_map"])
    except (OSError, ValueError, KeyError):
        pass
    try:
        with open(os.path.join(str(repo), "model.safetensors"), "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        return [k for k in header if k != "__metadata__"]
    except (OSError, ValueError, struct.error):
        return []


def _quant_summary(repo):
    """Exact quantisation of a build: config.json plus the weights themselves.

    The config's quantization dict only lists what *was* quantised -- a
    deliberately dense component (e.g. this project's bf16 audio encoder) does
    not appear there at all and would go unreported. So the weight names are
    checked too: any top-level component without quantisation scales is
    reported dense. Result e.g.
    "8-bit, group 64, affine; audio encoder & projector bf16".
    Empty string when no local config is readable (a not-yet-downloaded repo).
    """
    import json
    try:
        with open(os.path.join(str(repo), "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return ""
    dtype = _DTYPE_SHORT.get(str(cfg.get("torch_dtype")), str(cfg.get("torch_dtype", "")))
    q = cfg.get("quantization") or cfg.get("quantization_config")
    if not isinstance(q, dict) or "bits" not in q:
        return dtype  # fully unquantised build
    parts = [f"{q['bits']}-bit"]
    if q.get("group_size"):
        parts.append(f"group {q['group_size']}")
    if q.get("mode"):
        parts.append(str(q["mode"]))
    base = ", ".join(parts)
    deviations = []
    # Whole components with no quantised tensors at all stay at the model
    # dtype -- the weights are the authority here, not the quant dict.
    names = _weight_names(repo)
    if names:
        tops = sorted({n.split(".")[0] for n in names})
        quantised_tops = {n.split(".")[0] for n in names if n.endswith(".scales")}
        dense = [t for t in tops if t not in quantised_tops]
        if dense:
            labels = [_COMPONENT_LABELS.get(t, t) for t in dense]
            deviations.append(f"{' & '.join(labels)} {dtype}".strip())
    # Components quantised differently from the base width (or explicitly
    # skipped, mlx marks those `false`): report each distinct one once.
    per_layer = set()
    for key, val in q.items():
        leaf = key.rsplit(".", 1)[-1]
        if isinstance(val, dict) and val.get("bits") not in (None, q["bits"]):
            per_layer.add(f"{leaf} {val['bits']}-bit")
        elif val is False:
            per_layer.add(f"{leaf} {dtype}".strip())
    deviations.extend(sorted(per_layer))
    if deviations:
        base += "; " + ", ".join(deviations)
    return base


# A pass that collapses into a repetition loop repeats a short cycle of words
# far more often than any real utterance. The cycle is not always one word:
# besides "Jetzt. Jetzt. Jetzt. ..." the observed loops ran "dass das, dass
# das, ..." and "ja, ich, ja, ich, ..." -- two words, so no two *adjacent*
# words are ever equal and counting identical neighbours (which is what this
# used to do) sees nothing at all.
#
# Counting repeats of a k-word cycle instead. Measured over 278 real units
# (122 Voxtral chunks plus 139 finished Whisper transcripts, German and
# English): everything that reads as speech stays at or below 11, and the
# observed loops run 12, 19, 32, 39, 51, 65, 68 and 102. There is no wide gap
# -- the two populations very nearly touch, so this threshold is a judgement
# call, not a valley.
#
# It sits at 12 because a false positive costs time, not correctness: a
# wrongly flagged pass keeps its clean prefix and re-decodes the rest, which
# yields the same transcript by a longer route. A missed loop, by contrast,
# ships. An earlier version used 20 to leave room for rhetorical repetition
# that the corpus did not actually contain, and it would have let the 19-repeat
# loop through.
#
# If false positives do show up in practice, the answer is not another nudge:
# add a second signal (how much of the pass the cycle actually displaces), so
# a twelve-word stutter and a three-thousand-word loop stop being scored the
# same.
DEGENERATE_CYCLE_REPEATS = 12
# k up to 8 covers a repeated half-sentence; beyond that a verbatim repeat is
# long enough to be deliberate.
DEGENERATE_CYCLE_MAX_WORDS = 8
# Compression ratio as a secondary net, calibrated on the same files: real
# transcripts measure 2.59-2.63, the loop measured 5.75. Whisper uses 2.4 for
# this, which would flag good German prose here, so the threshold sits in the
# wide gap between the two populations.
DEGENERATE_COMPRESSION_RATIO = 4.0
# Last-resort penalties for a pass that still loops after being split, tried in
# order so the gentlest effective one wins. A penalty cannot tell a degenerate
# loop from a meaningful repetition, and a strong one deletes doubled words: 1.1
# turned "wir können es dir nicht nicht erzählen" into "... nicht erzählen",
# inverting the statement. A sentence that reads fluently but says the opposite
# is worse than an obviously broken one. Measured on the pass that looped:
#
#   1.01 -> loop gone, "nicht nicht" kept, 10.17 commas/100 words
#   1.05 -> loop gone, "nicht nicht" kept, 10.01
#   1.10 -> loop gone, "nicht nicht" LOST,  9.55
#
# A barely-there nudge is enough to tip a self-reinforcing loop but too weak to
# override a repetition the model is confident about.
RETRY_REPETITION_PENALTIES = (1.01, 1.1)
# Splitting a looping pass is tried first, because the loop is a long-generation
# effect: the pass that produced 4099 identical words at 410 s transcribed
# cleanly as 2x205 s, with the doubled negation intact. Depth 3 halves a
# 600 s pass down to 75 s; a piece below 45 s would carry too little context
# to be worth a further split. Both are judgement calls, not measured optima.
LOOP_SPLIT_MAX_DEPTH = 3
LOOP_SPLIT_MIN_SEC = 45
# Gentle sampling before any penalty, Whisper's own fallback strategy: a bit
# of sampling noise keeps a self-reinforcing loop from re-forming -- without
# deleting repeated words the way a penalty does. Seeded, so retries stay
# reproducible. The rungs are calibrated on a real pathological 346 s window
# (a group session where one speaker repeats the same question to many
# people): greedy LOOP, T=0.2 LOOP, T=0.5 LOOP, T=0.8 CLEAN (fluent text,
# zero breaker kicks) -- low temperatures
# barely move a confident argmax, so the second rung must be high enough to
# actually break the attractor (Whisper's own ladder goes to 0.8/1.0 too).
# T=0.2 stays as the near-free first nudge before the split.
RETRY_TEMPERATURES = ((0.2, 0), (0.8, 1))  # (temperature, seed)

# --------------------------------------------------------------------------- #
# In-place loop breaking during generation
# --------------------------------------------------------------------------- #
# A degenerate loop repeats one token cycle for thousands of tokens. The
# _LoopBreaker logits processor detects that periodicity while the tokens are
# being generated and bans exactly the one token that would continue the
# cycle -- everywhere else the logits pass through untouched, so a clean pass
# stays bit-identical to plain greedy. Thresholds are deliberately strict:
# the *entire* window must be one exact token cycle, which for a 3-token word
# means ~64 verbatim repeats (a real mantra passage differs long before that,
# and the transcript-level DEGENERATE_CYCLE_REPEATS=12 net stays in place --
# it has to, because a single stray token ends the exact periodicity this
# processor requires while the loop itself carries on).
LOOP_BREAK_WINDOW = 192      # tail tokens that must be fully periodic
LOOP_BREAK_MAX_PERIOD = 32   # longest token cycle we detect (~20 words)
LOOP_BREAK_MIN_CYCLES = 6    # window/period must fit this many full cycles
LOOP_BREAK_CHECK_EVERY = 16  # steps between periodicity checks (latency/cost)
# A model that re-enters a loop right after being kicked out this many times
# is beyond in-place repair: give up so the retry ladder takes over cheaply
# (~3 windows of wasted tokens instead of a full max_tokens run).
LOOP_BREAK_MAX_KICKS = 3


class _LoopBreaker:
    """Logits processor that kills degenerate repetition loops in place.

    Backend-agnostic on purpose: ``tokens`` only needs ``len``/indexing and
    ``logits`` item assignment (mlx arrays and numpy both qualify), so the
    logic is unit-testable without weights or mlx.
    """

    def __init__(self):
        self._tail = []       # our incremental copy of the generated tokens
        self._seen = 0        # tokens already folded into _tail (NOT len(_tail))
        self._step = 0
        self._period = 0      # >0 while a ban episode is active
        self.kicks = 0        # completed interventions (episodes)
        self.gave_up = False

    def _find_period(self):
        tail = self._tail
        if len(tail) < LOOP_BREAK_WINDOW:
            return 0
        t = tail[-LOOP_BREAK_WINDOW:]
        for p in range(1, LOOP_BREAK_MAX_PERIOD + 1):
            if LOOP_BREAK_WINDOW // p < LOOP_BREAK_MIN_CYCLES:
                break
            if all(t[i] == t[i - p] for i in range(p, LOOP_BREAK_WINDOW)):
                return p
        return 0

    def __call__(self, tokens, logits):
        if self.gave_up:
            return logits
        # Incremental sync (one new token per step); resync fully if the
        # bookkeeping ever drifts (e.g. a first call with a non-empty prompt).
        # The count of already-seen tokens is tracked separately and must NOT be
        # inferred from len(self._tail): the tail is trimmed to a fixed bound
        # below, so once it saturates it stops growing and every later step would
        # look like a drift and take the full resync branch -- which converts
        # ~220 device scalars per generated token and dominates the decode.
        n = len(tokens)
        if n == self._seen + 1:
            self._tail.append(int(tokens[-1]))
        elif n != self._seen:
            keep = LOOP_BREAK_WINDOW + LOOP_BREAK_MAX_PERIOD
            self._tail = [int(x) for x in tokens[-keep:]]
        self._seen = n
        del self._tail[:-(LOOP_BREAK_WINDOW + LOOP_BREAK_MAX_PERIOD)]
        self._step += 1

        if self._period:
            # Active episode: keep banning until the model actually diverges
            # (the divergent token breaks the window's periodicity).
            if self._find_period() == self._period:
                logits[:, self._tail[-self._period]] = float("-inf")
            else:
                self._period = 0
            return logits

        if self._step % LOOP_BREAK_CHECK_EVERY:
            return logits
        p = self._find_period()
        if p:
            self.kicks += 1
            if self.kicks > LOOP_BREAK_MAX_KICKS:
                self.gave_up = True
                return logits
            self._period = p
            # Ban the token that would continue the cycle (the one p back).
            logits[:, self._tail[-p]] = float("-inf")
        return logits


def _longest_cycle_repeats(words, max_k=DEGENERATE_CYCLE_MAX_WORDS):
    """How often the most-repeated short word cycle repeats back-to-back.

    A cycle of length k shows up as a run of positions where ``w[i] == w[i-k]``;
    a run of n such positions means the k-word block was written (n + k) / k
    times. Returns the largest count over all k, so "a a a" and "a b a b a b"
    are both measured on the same scale.
    """
    best = 0
    for k in range(1, max_k + 1):
        run = 0
        for i in range(k, len(words)):
            run = run + 1 if words[i] == words[i - k] else 0
            best = max(best, (run + k) // k)
    return best


def _looks_degenerate(text):
    """True if a pass collapsed into a repetition loop.

    Voxtral is run without a repetition penalty because that is what Mistral's
    reference transcription does and because a penalty strips punctuation from
    verbatim speech. Rarely -- seen with the 24B model -- a pass still
    degenerates into repeating the same few words thousands of times, which
    must not reach the transcript.

    The cycle count is the load-bearing signal. The compression ratio stays as
    a net for degeneracies that are not a clean cycle, but it cannot be relied
    on for local loops: it is computed over the whole pass, so a loop filling
    4% of a 3300-word chunk moved the ratio from 2.6 to only 2.8 -- and even a
    half-loop/half-prose pass measured below the threshold.
    """
    import zlib
    if not text:
        return False
    words = text.split()
    if len(words) < 30:
        return False
    if _longest_cycle_repeats(words) >= DEGENERATE_CYCLE_REPEATS:
        return True
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(zlib.compress(raw))) > DEGENERATE_COMPRESSION_RATIO


# A looping pass is not wrong from the start: the model transcribes normally
# and only then falls into the attractor, so everything before the loop is its
# best greedy output. Re-running the whole window throws that away and pays to
# regenerate it. Keeping the clean part instead needs one thing the ladder did
# not have -- knowing *where* in the audio that part ends -- which forced
# alignment can answer for a fraction of a decode.
#
# Below this much salvaged audio it is not worth it: a full retry costs about
# the same and keeps the whole window's context, which a resumed pass loses.
SALVAGE_MIN_PREFIX_SEC = 60
# Word endings that mark a sentence, ignoring trailing quotes and brackets.
_SENTENCE_END = ('.', '!', '?', '…', ':')
_SENTENCE_TRAIL = '"\'»)]'


def _degenerate_span(words, max_k=DEGENERATE_CYCLE_MAX_WORDS,
                     min_repeats=DEGENERATE_CYCLE_REPEATS):
    """Word indices [start, end) of the earliest run repeating a cycle at least
    `min_repeats` times, or None if the text has no such run.

    _looks_degenerate answers *whether* a pass looped; this answers *where*,
    which is what makes keeping the clean part possible.
    """
    best = None
    for k in range(1, max_k + 1):
        run = 0
        for i in range(k, len(words)):
            if words[i] != words[i - k]:
                run = 0
                continue
            run += 1
            if (run + k) // k < min_repeats:
                continue
            start = i - run - k + 1
            end = i + 1
            while end < len(words) and words[end] == words[end - k]:
                end += 1
            if best is None or start < best[0]:
                best = (start, end)
            break                       # earliest qualifying run for this k
    return best


def _prefix_end_index(words, loop_start):
    """How many words to keep: everything up to the last sentence that ends
    before the loop begins, or 0 when there is no such sentence.

    Cutting at a sentence boundary keeps the seam clean and drops the handful
    of words a model tends to produce while sliding into the attractor,
    together with the sentence they belong to.
    """
    for i in range(loop_start - 1, -1, -1):
        if words[i].rstrip(_SENTENCE_TRAIL).endswith(_SENTENCE_END):
            return i + 1
    return 0


def _salvage_prefix(text, audio, align_cb):
    """Split a looping pass into its clean prefix and the sample index where a
    fresh pass should resume.

    Returns ``((prefix, cut), None)``, or ``(None, reason)`` when there is
    nothing worth keeping -- the reason is logged, because "it fell back to a
    full retry" is otherwise indistinguishable from "the rung never ran".

    Mapping the prefix onto its audio time is `_Aligner.align_prefix`'s job and
    is harder than it looks: plain forced alignment has no way to stop early,
    so it stretches the prefix's last words across the rest of the window --
    measured, an 85 s prefix came back ending at 299.98 s of a 300 s window,
    with flawless scores. See that method for how it is done instead; here it
    is enough to know that a prefix which could not be timed for real comes
    back evenly spread (prob 0.0) and is refused below. The cut is then snapped
    to the locally quietest frame, exactly as the split rung does, so the
    resumed pass never starts mid-word.
    """
    words = text.split()
    span = _degenerate_span(words)
    if span is None:
        return None, "the degeneration is not a clean cycle, so it cannot be located"
    keep = _prefix_end_index(words, span[0])
    if keep == 0:
        return None, f"the loop starts at word {span[0]}, before any sentence ends"
    prefix = words[:keep]
    stamps = align_cb(prefix, audio)
    if not stamps:
        return None, "the prefix could not be aligned"
    # _Aligner falls back to spreading words evenly when real alignment is
    # impossible, marking those with prob 0.0. Those times are far too rough to
    # cut audio on -- a wrong cut duplicates or drops speech.
    if not any(w.get("prob", 0) for w in stamps):
        # Nothing in the prefix was really aligned. `align_prefix` reports the
        # reason it gave up to the log; here it is enough to know that these
        # times are evenly spread, i.e. worthless for cutting audio.
        return None, ("the prefix could not be aligned to the audio, so every "
                      "timestamp is an even guess")
    # The check that matters is on the LAST stamp: it is the only one the cut
    # below is taken from. `any(...)` over the whole prefix passed as soon as a
    # single word carried a real score, so a prefix whose head aligned and whose
    # tail was spread or interpolated went straight through and the resume point
    # came from an invented, length-proportional timestamp (measured 27 s past
    # the truth, worst word 33 s). `prob` is a probability in (0, 1]; 0.0 is the
    # sentinel for a word that was spread or interpolated rather than aligned,
    # so truthiness is exactly the right test.
    if not stamps[-1].get("prob", 0):
        return None, ("the end of the clean part was not really aligned, so the "
                      "resume point would be guesswork")
    end_sample = min(int(stamps[-1]["end"] * SAMPLE_RATE), len(audio))
    if end_sample < SALVAGE_MIN_PREFIX_SEC * SAMPLE_RATE:
        return None, (f"only {end_sample / SAMPLE_RATE:.0f}s would be kept, "
                      f"below the {SALVAGE_MIN_PREFIX_SEC}s minimum")
    cut = _quietest_frame_near(audio, end_sample, 1.0)
    if cut <= 0 or cut > len(audio):
        return None, "the resume point fell outside the window"
    return (" ".join(prefix), cut), None


# --------------------------------------------------------------------------- #
# A pass that came back in the wrong language
# --------------------------------------------------------------------------- #
# Voxtral is not a pure ASR model -- speech translation is an advertised
# capability, and the `[TRANSCRIBE]` token fixes only that text comes out, not
# which language it is in. So a whole pass can come back translated, with no
# loop and nothing else wrong with it. Measured over six runs of one 5h54m
# German recording: four shipped 30 minutes of English from a chunk that had no
# loop at all. Which language wins turns out to hang on the exact window
# length, not on the content -- of ten windows between 1420 s and 1430 s, four
# came back English -- so this cannot be prevented at the prompt, only caught.
# Measured on the two windows that translate reproducibly: writing `lang:xx`
# into the prompt (the documented instrument, and ours is token-identical to
# Mistral's own) fixed 0 of 2, while a retry at the ladder's first temperature
# fixed both. Perturbing the decode beats telling the model what to do, which
# is why this is a reason for the ladder to run rather than a repair of its own.


def _other_language(text, want):
    """The language `text` reads as when that is decidedly not `want`, else None.

    None whenever there is nothing to compare (no expectation, or too little
    text to judge), so an undecidable pass is never called wrong.
    """
    got, _ = _detect_language(text)
    return got if _reads_as_other(got, want) else None


def _reads_as_other(got, want):
    """Whether a text the detector reads as `got` is decidedly not in `want`.

    For a language the detector can name, any other reading counts. For one it
    cannot, only English does -- the language Voxtral translates into, and the
    one the detector names reliably: any other reading may be the detector
    misnaming `want` itself. The script check reads all Cyrillic as "ru" and
    all Arabic script as "ar", so a correctly pinned Ukrainian or Persian file
    was "translated" on every pass; Czech reads as Polish by its function
    words.
    """
    if not (got and want) or got == want:
        return False
    return want in _DETECTABLE or got == "en"


def _halves_disagree(a, b, log_cb, label, name_a, name_b):
    """True when two halves of one window read as different languages.

    Both rungs that repair a window stitch two independently produced halves --
    the kept prefix plus its remainder, and the two sides of a split. Either
    half can come back translated, and the seam is the one place where that is
    visible without knowing the file's language at all: this test is
    *relative*. Speakers who mix German and English run through this material
    constantly, so a detector that leans the same way on both halves has to
    stay silent; only a genuine change between the halves may fire.
    """
    la, _ = _detect_language(a)
    lb, _ = _detect_language(b)
    if la and lb and la != lb:
        _log(log_cb, "warn",
             f"{label}: {name_a} reads as '{la}' but {name_b} as '{lb}' -- that "
             f"is a translated pass, not a transcript; dropping it and retrying "
             f"the whole window.")
        return True
    return False


def _file_language(tally):
    """The file's language so far, or None while the evidence is too thin.

    One chunk is never enough: if the *first* chunk is the translated one,
    taking its language as the truth would send every later chunk to be
    "repaired" into the wrong language -- the guard would cause exactly the
    damage it exists to prevent. Two have to agree.
    """
    if not tally:
        return None
    best, n = max(tally.items(), key=lambda kv: kv[1])
    runner_up = max([v for k, v in tally.items() if k != best], default=0)
    return best if n >= 2 and n > runner_up else None


# --------------------------------------------------------------------------- #
# Recovering a head a long pass dropped
# --------------------------------------------------------------------------- #
# A long window sometimes comes back missing its first seconds of speech: no
# loop, no wrong language, nothing else wrong with it -- the opening is simply
# not in the text. Measured on a 1226 s recording whose first 2.4 s are a remark
# before the take: on Auto the pass returned 3620 words without it, with `de`
# 3631 words with it, and the two agreed everywhere else. The reverse has been
# seen too (a pinned language costing the first ~150 words of a 1426 s window),
# so no prompt setting is the safe side, and it depends on the recording as much
# as on the length -- which is why this is a check rather than a length rule. On
# that recording 20 s through 1180 s all kept the opening and only 1195 s and
# 1226 s lost it; FLEURS material never shows it at any length; but on a raw
# 4.8 h Zoom recording it happens well inside the passes we actually run: of 32
# windows cut at 300 s and 600 s, three lost their opening on Auto and two with
# `de`, the worst dropping 18 words of fluent speech from a 600 s window and one
# dropping a whole quoted sentence from a 300 s one in both language settings.
#
# That is the repair: decode a short head of the same audio and splice back
# whatever the long pass is missing. The seam is found in the text rather than
# assumed, so a pass that lost nothing (the normal case) is left untouched.
# 60 s is a judgement call: short windows were never seen to lose their
# opening in the sweep, and the probe costs ~10 % of a pass at this length.
HEAD_PROBE_SEC = 60.0
# Grown only when the seam is not in the probe at all, i.e. the pass lost more
# than the probe covers. Past this the loss is no longer a dropped opening.
HEAD_PROBE_MAX_SEC = 240.0
# The run of words from the pass that is looked for in the probe. Long enough
# that common words cannot line up by chance, short enough to survive the small
# wording differences between two independent decodes.
HEAD_ANCHOR_WORDS = 8
HEAD_ANCHOR_MIN_HITS = 6


def _norm_word(word):
    """A word reduced to what two decodes of the same speech agree on."""
    return re.sub(r"[^\w]", "", word.lower())


def _find_anchor(probe, anchor, min_hits=HEAD_ANCHOR_MIN_HITS):
    """Where in `probe` the `anchor` run begins, or None. Both normalised.

    Two independent decodes of the same speech differ in small ways ("Ja,
    herzlich" against "Ja. Herzlich"), so the match is scored rather than exact.
    Ties go to the earliest position: that is the one that recovers the least
    text, and recovering too little only leaves the status quo while recovering
    too much would duplicate words the pass already has.
    """
    best_at, best_hits = None, 0
    width = len(anchor)
    for i in range(len(probe) - width + 1):
        hits = sum(1 for a, b in zip(anchor, probe[i:i + width]) if a == b)
        if hits > best_hits:
            best_at, best_hits = i, hits
    return best_at if best_hits >= min_hits else None


def _recover_lost_head(vox, audio, language, text, log_cb, label):
    """Put back the opening a long pass dropped. Returns the text either way.

    Left alone when there is no long-window defect to repair (a short pass), no
    seam to find (too little text), or nothing missing at the seam. The caller
    decides which passes are worth checking at all.
    """
    words = text.split()
    duration = len(audio) / SAMPLE_RATE
    if len(words) < HEAD_ANCHOR_WORDS or duration <= HEAD_PROBE_SEC * 1.5:
        return text
    anchor = [_norm_word(w) for w in words[:HEAD_ANCHOR_WORDS]]

    # Never probe more than half the pass: past that the probe is a second
    # transcription rather than a check, and the defect is no longer a lost
    # opening. That holds for the first probe too (a 100 s pass gets a 50 s
    # one). Growing doubles and stops at the last doubling within `longest`
    # rather than stretching to it: each probe decodes the same opening again,
    # and on a real interview a 226 s probe after the 120 s one still did not
    # find it -- the anchor is missed on wording, which more audio does not fix.
    longest = min(HEAD_PROBE_MAX_SEC, duration / 2)
    sec = min(HEAD_PROBE_SEC, longest)
    while True:
        probe = vox.transcribe_array(audio[:int(sec * SAMPLE_RATE)], language,
                                     max_new_tokens=int(sec * 20) + 512)
        if not probe or _looks_degenerate(probe):
            # A looping probe is a property of this speech, not of the window
            # length, and a degenerate decode is the one that runs to its full
            # token budget. Growing it would buy a longer loop, not an answer.
            _log(log_cb, "warn",
                 f"{label}: could not check the opening -- the {sec:.0f}s head "
                 f"probe came back unusable.")
            return text
        probe_words = probe.split()
        at = _find_anchor([_norm_word(w) for w in probe_words], anchor)
        if at is not None:
            break
        # The pass' opening is not in this probe at all, so it lost more than the
        # probe covers. Reach further, up to `longest`.
        grown = sec * 2
        if grown > longest:
            _log(log_cb, "warn",
                 f"{label}: could not check the opening -- a {sec:.0f}s head "
                 f"probe does not contain the first words of the pass.")
            return text
        sec = grown

    if at == 0:
        return text  # the pass starts where the audio does: nothing was lost
    missing = " ".join(probe_words[:at]).strip()
    _log(log_cb, "warn",
         f"{label}: the pass dropped its opening ({at} words); recovered from a "
         f"{sec:.0f}s head probe.")
    return f"{missing} {text.lstrip()}"


def _quietest_split(audio):
    """Sample index nearest the middle that sits in the quietest 100 ms frame,
    so a pass is never split in the middle of a word."""
    return _quietest_frame_near(audio, len(audio) // 2, 5.0)


def _clip_turns(turns, w0, w1):
    """Speaker turns intersected with the window [w0, w1) (seconds), shifted to
    window-relative times. None when nothing meaningful remains."""
    out = []
    for s, e, lbl in turns or []:
        s2, e2 = max(s, w0), min(e, w1)
        if e2 - s2 > 0.05:
            out.append((s2 - w0, e2 - w0, lbl))
    return out or None


def _turn_gap_split(audio, turns):
    """Sample index of the best speaker-change boundary near the middle, or
    None when the window offers no usable one.

    Cutting where the speaker changes is the cleanest split there is: the
    context *within* a turn stays intact, and across a turn boundary it
    matters least. Candidates are boundaries between consecutive turns of
    *different* speakers inside the middle half of the window (keeps the
    halves reasonably balanced); overlapping-speech boundaries are skipped
    (cutting inside an overlap clips someone mid-word). The winner is the
    boundary closest to the middle, snapped to the locally quietest 100 ms.
    """
    if not turns:
        return None
    lo, hi = len(audio) // 4, 3 * len(audio) // 4
    mid = len(audio) // 2
    ordered = sorted(turns, key=lambda t: t[0])
    best = None
    # Track the furthest point reached so far and who is still speaking there,
    # not merely the immediately preceding turn. Sorted by start time, a short
    # turn nested inside a longer one -- pyannote's standard backchannel
    # pattern, an "mhm" during someone's sentence -- makes the *next* turn look
    # like it follows the backchannel. Comparing against that one's end put the
    # cut in the middle of the enclosing speaker's utterance, and made a
    # speaker resuming after their own backchannel look like a change.
    prev_end, prev_label = ordered[0][1], ordered[0][2]
    for s1, e1, l1 in ordered[1:]:
        # Same speaker resuming after a pause is not a boundary; neither is a
        # boundary buried in overlapping speech (cutting there clips a word).
        if l1 != prev_label and s1 >= prev_end - 0.2:
            p = int((prev_end + max(s1, prev_end)) / 2 * SAMPLE_RATE)
            if lo <= p <= hi and (best is None or abs(p - mid) < abs(best - mid)):
                best = p
        if e1 >= prev_end:
            prev_end, prev_label = e1, l1
    if best is None:
        return None
    return _quietest_frame_near(audio, best, 1.0)


def _turn_profile(turns, dur):
    """One-line diarization profile of a window, for the escalation log."""
    ordered = sorted(turns, key=lambda t: t[0])
    speakers = {lbl for _, _, lbl in ordered}
    changes = sum(1 for a, b in zip(ordered, ordered[1:]) if a[2] != b[2])
    speech = sum(e - s for s, e, _ in ordered)
    dur = max(dur, 1e-9)
    return (f"{len(speakers)} speaker(s), {changes} turn change(s) "
            f"({60.0 * changes / dur:.1f}/min), ~{min(100.0, 100.0 * speech / dur):.0f}% speech")


def _free_decode_buffers(vox):
    """Release a finished decode's MLX buffers before an alignment. They are
    dead by then, and the aligner's forward runs on the same GPU -- leaving
    them resident is exactly the co-residency MIN_HEADROOM_GB is meant to
    prevent."""
    try:
        vox._mx.clear_cache()
    except Exception:
        pass


def _transcribe_guarded(vox, audio, language, log_cb, label, depth=0, token_cb=None,
                        turns=None, align_cb=None, want_lang=None):
    """Transcribe one pass and repair it if it comes back unusable.

    Two things make a pass unusable, and both take the same repairs. A
    repetition loop is the common one. The other is a *translated* pass:
    Voxtral is not a pure ASR model -- speech translation is an advertised
    capability, and `[TRANSCRIBE]` fixes only that text comes out, not which
    language it is in. Measured over six runs of one German recording, four of
    them shipped 30 minutes of English from a chunk that had no loop at all, so
    a bad language has to make a pass bad here, where the rungs are, rather
    than in a second repair path beside them. `want_lang` is the language the
    caller expects, or None while it does not know yet. The two defects share
    the rungs but not all of them: a pass bad for its language alone skips the
    prefix rung (there is no loop to cut off) and never reaches the penalty
    rung; after the gentle retry and one split (or, for a window too short to
    split, the other temperatures), what still reads as another language is
    taken to be spoken in it.

    Repair order, gentlest first:

    1. Greedy with the _LoopBreaker armed. Clean passes stay bit-identical;
       a loop is usually repaired in place at zero extra cost.
    2. Keeping the clean prefix and resuming after it (needs `align_cb`): the
       text before the loop is the model's best greedy output, so this is the
       only rung that repairs a pass without regenerating what it got right.
       It is also the cheapest -- one alignment plus a decode of the remainder,
       instead of a decode of the whole window.
    3. Low-temperature sampling (Whisper's own fallback): barely deviates
       from greedy, breaks the loop attractor, keeps repeated words.
    4. Splitting -- the loop is a long-generation effect, shorter pieces
       often come out clean at plain greedy. The cut prefers a speaker-turn
       boundary from the diarization (`turns`, window-relative seconds):
       context within a turn stays intact, across a change it matters least.
       Without a usable boundary the quietest frame near the middle is used,
       as before.
    5. Stronger sampling, then the windowed repetition penalty as the very
       last resort, because a penalty silently deletes meaningful repeated
       words (a doubled negation flips the meaning of the sentence).

    Failed attempts are cheap: the breaker aborts a hopeless generation after
    ~3 loop windows instead of running to max_new_tokens.
    """
    dur = len(audio) / SAMPLE_RATE
    # 20 tokens/s is ~4.3x reserve: German runs 4.64 text tokens/s measured
    # (docs/voxtral-benchmarks.md, section 2). The budget bounds only the
    # generation loop -- a high value reserves no memory.
    max_new = min(32768, int(dur * 20) + 512)
    loop = "repetition loop"

    def attempt(temperature=0.0, seed=None, penalty=1.0):
        """(text, defect, info): the defect is None for a usable pass, else
        what the log calls it -- `loop`, or the translated-pass description."""
        info = {}
        text = vox.transcribe_array(audio, language, max_new_tokens=max_new,
                                    repetition_penalty=penalty, token_cb=token_cb,
                                    temperature=temperature, seed=seed, info=info)
        # A gave-up attempt is truncated mid-loop; the full-text detector may
        # miss that (a late loop gets diluted), so flag it explicitly.
        if info.get("loop_gave_up", False) or _looks_degenerate(text):
            return text, loop, info
        got = _other_language(text, want_lang)
        defect = f"translated pass ('{got}' instead of '{want_lang}')" if got else None
        return text, defect, info

    # Fallback candidates for the case where no rung resolves the loop.
    # "Shorter is less degenerate" only holds between attempts that actually
    # cover the whole window. When the loop breaker gives up, _consume_tokens
    # stops pulling tokens mid-stream, so that attempt is the shortest by
    # construction and used to win every comparison -- measured, a 148-word
    # stump shipped in place of a 2733-word near-complete transcript, and the
    # rest of the chunk's speech was gone with only a log line to show for it.
    # Partial results are therefore only used when nothing else was produced.
    candidates = []                     # (text, covers_only_part_of_the_window)

    def keep(candidate, partial=False):
        if candidate:
            candidates.append((candidate, partial))

    def best():
        whole = [c for c, part in candidates if not part]
        return min(whole or [c for c, _ in candidates], key=len) if candidates else ""

    text, defect, info = attempt()
    if info.get("loop_kicks") and not defect:
        _log(log_cb, "info", f"{label}: repetition loop broken in place "
                             f"({info['loop_kicks']} intervention(s)).")
    if not defect:
        return text
    keep(text, bool(info.get("loop_gave_up")))

    # Keep what the pass got right. This runs before any full re-decode: it is
    # cheaper (one alignment plus the remainder) and it is the only rung that
    # preserves the greedy output for the clean part instead of re-rolling it.
    # A pass that is bad only for its language has no loop to cut off.
    if align_cb is not None and depth < LOOP_SPLIT_MAX_DEPTH and defect == loop:
        # The looping decode is typically the longest of the job.
        _free_decode_buffers(vox)
        salvaged, why_not = _salvage_prefix(text, audio, align_cb)
        if salvaged is None:
            _log(log_cb, "info", f"{label}: cannot keep the clean part "
                                 f"({why_not}); retrying the whole window.")
        else:
            prefix, cut = salvaged
            cut_sec = cut / SAMPLE_RATE
            rest_sec = dur - cut_sec
            if rest_sec < 1.0:
                # Every other rung is gated on its result not looking
                # degenerate; this exit returned unchecked, so a prefix that is
                # itself degenerate shipped as a success and never escalated.
                # (_looks_degenerate's compression arm is not bounded by the
                # cycle length _degenerate_span searches for, so it can fire on
                # a prefix the span finder considers clean.)
                if not _looks_degenerate(prefix):
                    _log(log_cb, "info",
                         f"{label}: repetition loop at the very end; "
                         f"keeping the clean first {cut_sec:.0f}s.")
                    return prefix
                _log(log_cb, "info",
                     f"{label}: the clean part still looks degenerate; "
                     f"retrying the whole window.")
                keep(prefix, partial=True)   # covers only up to the cut
            else:
                _log(log_cb, "warn", f"{label}: repetition loop after {cut_sec:.0f}s; "
                                     f"keeping that part and re-transcribing the "
                                     f"remaining {rest_sec:.0f}s.")
                rest = _transcribe_guarded(vox, audio[cut:], language, log_cb, label,
                                           depth + 1, token_cb,
                                           _clip_turns(turns, cut_sec, dur), align_cb,
                                           want_lang)
                joined = f"{prefix} {rest}".strip()
                if not _halves_disagree(prefix, rest, log_cb, label,
                                        "the kept part", "the remainder") \
                        and not _looks_degenerate(joined):
                    return joined
                keep(joined)

    temps = list(RETRY_TEMPERATURES)
    temperature, seed = temps.pop(0)
    _log(log_cb, "warn", f"{label}: {defect}; retrying with gentle "
                         f"sampling (temperature={temperature}).")
    retry, rdefect, rinfo = attempt(temperature=temperature, seed=seed)
    if not rdefect:
        return retry
    keep(retry, bool(rinfo.get("loop_gave_up")))

    # Neither decode looped; both are bad only because they read as another
    # language. Perturbing the decode is what fixed both reproducible cases
    # (this rung; measured above `_other_language`), and which language wins
    # also hangs on the exact window length, which the split changes. So a
    # window gets the split once, at the top level, and its halves stop here;
    # one too short to split gets the remaining temperatures instead. What
    # still reads as another language after that is taken to be spoken in it --
    # an English passage on Auto, or a wrong pinned language -- where climbing
    # on to the penalty rung cost 60 decodes of a 600 s window and then deleted
    # real repeated words. transcribe() reports the mismatch.
    splittable = depth < LOOP_SPLIT_MAX_DEPTH and dur >= 2 * LOOP_SPLIT_MIN_SEC
    language_only = loop not in (defect, rdefect)
    if language_only and depth > 0:
        _log(log_cb, "info",
             f"{label}: {rdefect} again at temperature={temperature}, without a "
             f"loop; keeping the first decode rather than repairing a language "
             f"the audio may really be in.")
        return text

    if splittable:
        cut = _turn_gap_split(audio, turns)
        how = "at a speaker change" if cut is not None else "at the quietest pause"
        if cut is None:
            cut = _quietest_split(audio)
        cut_sec = cut / SAMPLE_RATE
        _log(log_cb, "warn", f"{label}: unresolved, splitting the chunk {how} "
                             f"and retrying ({dur:.0f}s -> {cut_sec:.0f}s + {dur - cut_sec:.0f}s).")
        left = _transcribe_guarded(vox, audio[:cut], language, log_cb, label,
                                   depth + 1, token_cb,
                                   _clip_turns(turns, 0, cut_sec), align_cb, want_lang)
        right = _transcribe_guarded(vox, audio[cut:], language, log_cb, label,
                                    depth + 1, token_cb,
                                    _clip_turns(turns, cut_sec, dur), align_cb, want_lang)
        joined = f"{left} {right}".strip()
        if language_only:
            # Each half has had its own two decodes at a new window length and
            # kept what it read as; a half that came back in the wanted
            # language is the repair, and one that did not is taken as spoken.
            return text if _looks_degenerate(joined) else joined
        if not _halves_disagree(left, right, log_cb, label,
                                "the first half", "the second half") \
                and not _looks_degenerate(joined):
            return joined
        keep(joined)

    # Escalating past the gentle stages: log what the diarization saw in this
    # window -- it tells a music/no-speech hallucination apart from a
    # repetitive many-speaker exchange without digging through the audio.
    if turns:
        _log(log_cb, "info", f"{label}: diarization profile of this window: "
                             f"{_turn_profile(turns, dur)}.")

    for temperature, seed in temps:
        _log(log_cb, "warn", f"{label}: unresolved; retrying with "
                             f"temperature={temperature}.")
        retry, rdefect, rinfo = attempt(temperature=temperature, seed=seed)
        if not rdefect:
            return retry
        keep(retry, bool(rinfo.get("loop_gave_up")))

    if language_only:
        _log(log_cb, "info",
             f"{label}: still reads as another language at every temperature; "
             f"keeping the first decode rather than repairing a language the "
             f"audio may really be in.")
        return text

    # Everything gentler failed. Fall back to the gentlest penalty that works,
    # because a stronger one starts deleting meaningful repeated words.
    for penalty in RETRY_REPETITION_PENALTIES:
        _log(log_cb, "warn", f"{label}: still looping; retrying with "
                             f"repetition_penalty={penalty}. Note that a penalty can drop "
                             f"meaningful repeated words.")
        retry, rdefect, rinfo = attempt(penalty=penalty)
        if not rdefect:
            return retry
        keep(retry, bool(rinfo.get("loop_gave_up")))
    _log(log_cb, "warn", f"{label}: could not resolve the loop; keeping the "
                         f"shortest attempt that covers the whole window.")
    return best()


# A sentence is a run up to and including terminal punctuation; the final
# alternative captures a trailing run that has no terminal punctuation (the last
# pass of a file, or text Voxtral emits unpunctuated) as ONE fragment. An earlier
# `\S+$` fallback matched only the last whitespace token, silently dropping every
# word between the last period and the end of the text.
_SENT_SPLIT = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$", re.UNICODE)


# --------------------------------------------------------------------------- #
# The log-Mel clamp floor: a percentile instead of the maximum
# --------------------------------------------------------------------------- #
# Every reference log-Mel path (transformers, mlx-audio, transcribe.cpp and
# mlx-voxtral alike) clamps the spectrogram at `log_max - 8`, with log_max the
# maximum over the *whole* input, so one loud cell sets the floor for the
# entire pass: a 0.1 s transient 28 dB over the speech peak costs a measured
# +1.52 WER points [+0.78, +2.26] over 50 min of VoxPopuli streams. A
# percentile in place of the maximum removes that (+0.01 [-0.03, +0.06]) and
# costs nothing on clean material (-0.12 [-0.85, +0.49], same streams,
# paired); it can only lower the floor, never raise it. The case against --
# the model was trained on the maximum-clamped input -- and the measurements
# are in docs/voxtral-mel-clamp-floor.md.
#
# On clean material the percentile is not a critical value: 99, 99.9 and
# 99.99 are indistinguishable from the maximum and from each other. Under a
# transient it is, because the transient's own cells sit at the top of the
# distribution and displace the statistic by their count: the floor follows a
# 100 ms full-scale noise burst (~1300 cells) by 7 dB at 99.9 on a 60 s pass,
# whose top 0.1 % is only 768 cells, and by 0.75 dB at 99. 99.99 is unusable
# for the same reason. The defence is count-limited either way: broadband
# noise that fills more than the top percent of a pass (roughly 0.6 s on a
# 60 s pass) is treated as signal and raises the floor like it always did.
# The one regime with a thin margin is a pass that is mostly silence: the
# percentile then sits deeper in the speech cells (24-29 dB below the maximum
# against 20-27 on dense streams), and 50 % room tone between the
# utterances measured
# +0.24 [-0.03, +0.55] -- not demonstrable, but close, and capping the
# floor to fix it was measured and rejected (see MEL_FLOOR_CAP). The 8.0 is
# librosa's `top_db=80` and is the range the model was trained on -- it
# stays.
MEL_FLOOR_PERCENTILE = 99.0
MEL_FLOOR_RANGE = 8.0
# Optional lower bound on the floor, in log10 units below the maximum (2.0 =
# 20 dB): `max(percentile, log_max - cap) - 8`. Measured at 20 and 25 dB over
# all four regimes and REJECTED: a cap hangs off the maximum, so a knock
# lifts it along -- +0.81 [+0.18, +1.57] at 20 dB where the uncapped floor
# costs +0.01 -- while buying only +0.24 -> +0.05 on half-silent streams,
# where no interval excludes zero (docs/voxtral-mel-clamp-floor.md,
# section 5). None stays the
# production value; the parameter exists so the measurement scripts can
# drive the arm (spec `99c20` in docs/scripts).
MEL_FLOOR_CAP = None


def clamp_log_mel(log_spec, pct=MEL_FLOOR_PERCENTILE, real_frames=None,
                  cap=MEL_FLOOR_CAP):
    """Clamp an unclamped [frames, mels] log-Mel spectrogram at
    `percentile(pct) - 8` and apply the reference affinity `(x + 4) / 4`;
    with `cap`, the statistic never sits more than `cap` below the maximum.

    The percentile is taken over the first `real_frames` frames only (all of
    them by default): the block is zero-padded to a 30 s multiple, and padding
    cells sit at the -10 minimum, so a percentile over the whole block would
    drift downwards with the amount of padding -- 18 dB lower on a 5 s clip
    than on the same clip unpadded. The clamp itself covers every frame. The
    maximum of a padded block is never in a padding-only frame, so this
    changes nothing at pct=100.

    float32 throughout, with numpy scalars so the dtype holds under both the
    legacy and the NEP 50 promotion rules: with pct=100 the result must be
    bit-identical to mlx-voxtral's own `maximum(x, max(x) - 8)`, which is how
    tests/test_mel_floor.py proves that only the source of the floor changed.
    """
    import numpy as np
    x = np.asarray(log_spec, dtype=np.float32)
    stat = x if real_frames is None else x[:real_frames]
    top = np.float32(np.percentile(stat, pct))
    if cap is not None:
        top = max(top, stat.max() - np.float32(cap))
    floor = top - np.float32(MEL_FLOOR_RANGE)
    return (np.maximum(x, floor) + np.float32(4.0)) / np.float32(4.0)


class _PercentileFloorFeatures:
    """Drop-in for mlx-voxtral's VoxtralFeatureExtractor on the one input the
    engine hands it -- a 16 kHz mono float32 array -- with the clamp floor from
    `clamp_log_mel` instead of the maximum. Installed on the processor by
    _Voxtral.__init__; the library itself is not patched.

    The unclamped spectrogram is built from the library's own primitives (the
    same window, STFT, filter bank and log as its `log_mel_spectrogram`), not
    from that function's `global_max` lever. The lever only hands out the
    affine spectrogram `(x + 4) / 4`, which loses mantissa bits wherever
    x > -2, and from those the reference floor cannot be reproduced exactly on
    quiet material: with log_max in [-2, 0) one input in eight gets a floor
    one bit off, and with it every clamped cell (measured on 300 synthetic
    signals: 12 %). Bit-identity at pct=100 is the proof that only the floor
    changed, so the spectrogram has to be exact.
    """

    def __init__(self, pct=MEL_FLOOR_PERCENTILE, cap=MEL_FLOOR_CAP):
        self.pct = pct
        self.cap = cap

    def __call__(self, raw_speech, sampling_rate=SAMPLE_RATE, **kwargs):
        import numpy as np
        import mlx.core as mx
        from mlx_voxtral.audio_processing import (
            N_FFT, N_FRAMES, N_MELS, N_SAMPLES, HOP_LENGTH,
            get_mel_filters, hanning, pad_to_multiple, stft_mlx)
        # The library would resample or downmix here; the engine never hands
        # it anything but 16 kHz mono (None means "already 16 kHz" there too).
        if sampling_rate not in (None, SAMPLE_RATE):
            raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sampling_rate}")
        audio = np.asarray(raw_speech, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"expected mono audio, got shape {audio.shape}")
        # One spectrogram over the whole (30 s-padded) input, split afterwards,
        # as the reference specifies (arXiv:2507.13264 section 2.1).
        padded = pad_to_multiple(mx.array(audio), N_SAMPLES)
        n_chunks = padded.shape[0] // N_SAMPLES
        freqs = stft_mlx(padded, hanning(N_FFT), nperseg=N_FFT,
                         noverlap=N_FFT - HOP_LENGTH)[:-1]
        mel = (mx.abs(freqs) ** 2) @ get_mel_filters(N_MELS).T
        log_spec = np.array(mx.log10(mx.maximum(mel, 1e-10)))  # [frames, mels]
        # Frames that contain any input sample (the STFT is centred, so the
        # window reaches N_FFT // 2 samples past the last one).
        real_frames = -(-(len(audio) + N_FFT // 2) // HOP_LENGTH)
        feats = clamp_log_mel(log_spec, self.pct, real_frames, self.cap).T
        feats = feats.reshape(N_MELS, n_chunks, N_FRAMES).transpose(1, 0, 2)
        return {"input_features": mx.array(feats)}


# --------------------------------------------------------------------------- #
# First-use downloads
# --------------------------------------------------------------------------- #
# What a first-use download weighs, for the one log line before it starts: a
# silent 15 GB download is indistinguishable from a hang. Sizes of the
# published snapshots; every wav2vec2-large aligner is ~1.3 GB of weights.
_DOWNLOAD_GB = {
    VOXTRAL_MODELS["voxtral-mini-8bit"]: 5.6,
    VOXTRAL_MODELS["voxtral-small-4bit"]: 15,
}
_ALIGNER_DOWNLOAD_GB = 1.3


def _needs_download(repo):
    """True when `repo` is a hub id with nothing of it in the local Hugging
    Face cache. False whenever that cannot be told, so the log never announces
    a download that is not happening. Only the config is looked for, so a
    download interrupted after its first files resumes unannounced: which
    weight files a snapshot has differs from model to model."""
    if os.path.isdir(str(repo)):
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
        return not isinstance(try_to_load_from_cache(str(repo), "config.json"), str)
    except Exception:
        return False


def _hub_offline():
    try:
        from huggingface_hub import constants
        return bool(constants.HF_HUB_OFFLINE)
    except Exception:
        return False


def _hub_reachable():
    """Whether Hugging Face answers at all -- any HTTP answer counts."""
    try:
        import httpx
        from huggingface_hub import constants
        httpx.head(constants.ENDPOINT, timeout=5)
        return True
    except Exception:
        return False


def _unfetchable(repo):
    """True when `repo` is not downloaded and cannot be fetched right now."""
    return _needs_download(repo) and (_hub_offline() or not _hub_reachable())


def _not_fetchable_error(what, repo, size_gb):
    size = f" (~{size_gb:g} GB)" if size_gb else ""
    if _hub_offline():
        how = ("downloads are switched off (HF_HUB_OFFLINE); allow them once so "
               "it can be downloaded")
    else:
        how = ("could not be fetched from Hugging Face (no internet connection, "
               "or the site is not reachable); connect once so it can be "
               "downloaded")
    return RuntimeError(f"The {what} {repo} is not downloaded yet, and {how}{size}.")


def _load_from_hub(load, what, repo, size_gb, log_cb):
    """`load()`, announced when it will download first. A model that is not
    downloaded and cannot be fetched fails with a sentence the user can act on
    instead of the library's text; the original stays chained.

    Whether it could be fetched is asked of the hub itself, not read off the
    exception: the libraries wrap their errors too differently for that.
    transformers turns an aligner's missing files, a 401 for a stale token and
    an unwritable cache alike into a bare OSError; a connection that drops
    during the download ends in an httpx error, which is no OSError at all;
    hf_xet reports a full disk without an errno. So when a missing model fails
    to load and the hub does not answer, the model could not be fetched; when
    it does answer, the library's own error says what else went wrong.
    """
    size = f" (~{size_gb:g} GB)" if size_gb else ""
    missing = _needs_download(repo)
    if missing and not _hub_offline():
        _log(log_cb, "info", f"Downloading the {what} {repo}{size}; this happens "
                             f"only once.")
    try:
        from httpx import HTTPError as _TransportFailure
    except Exception:
        _TransportFailure = OSError
    try:
        return load()
    except (OSError, _TransportFailure) as exc:   # what fetching can raise
        if not missing or not (_hub_offline() or not _hub_reachable()):
            raise
        raise _not_fetchable_error(what, repo, size_gb) from exc


# --------------------------------------------------------------------------- #
# Voxtral transcription
# --------------------------------------------------------------------------- #
def _cap_mlx_memory():
    """Nudge MLX to release its reusable buffer cache once usage nears physical
    RAM, so cached (not active) buffers don't push the working set into swap.
    MLX's limit is *soft* (it never hard-fails an allocation), so the real memory
    control is the per-pass length picked by _auto_chunk_sec; this only trims
    cache retention near the ceiling. Set just below total RAM so it doesn't
    throttle a normal generate pass (which is sized to stay under RAM anyway)."""
    try:
        import mlx.core as mx
        total = int(_total_ram_gb() * 1024**3)
        mx.set_memory_limit(max(8 * 1024**3, total - 4 * 1024**3))
    except Exception:
        pass


class _Voxtral:
    """The loaded mlx-voxtral model, driven through `mlx_lm.generate_step`.

    `transcribe_array` is the only entry point; everything else prepares the
    prompt, resolves this build's stop tokens or consumes the token stream."""

    def __init__(self, repo):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_voxtral import load_voxtral_model, VoxtralProcessor
        self._mx = mx
        _cap_mlx_memory()
        self.model, _ = load_voxtral_model(repo, dtype=mx.bfloat16)
        self.proc = VoxtralProcessor.from_pretrained(repo)
        self.proc.feature_extractor = _PercentileFloorFeatures()
        self._STOP_TOKENS = self._resolve_stop_tokens()

        # Adapter so mlx_lm.generate_step can drive our LM: generate_step calls
        # model(tokens, cache=, input_embeddings=) and expects logits, while our
        # language_model takes `inputs_embeds` and returns hidden states (the
        # lm_head lives on the parent). This bridges both.
        class _LMAdapter(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.language_model = parent.language_model
                self.lm_head = parent.lm_head

            def __call__(self, inputs, cache=None, input_embeddings=None):
                h = self.language_model(inputs, cache=cache,
                                        inputs_embeds=input_embeddings)
                return self.lm_head(h)

        self._lm_adapter = _LMAdapter(self.model)

    # Control-token ids that end a generation: </s>, [/INST] and <pad>.
    #
    # mlx_voxtral.generate_stream defaulted to (2, 4, 32000) up to 0.0.4 and
    # called 32000 "a potential padding token" (it defaults to (2, 4, 11) since
    # 0.0.5). It is not one: Voxtral's Tekken vocabulary has 131072 entries of
    # which only the first 1000 are control tokens, <pad> is id 11, and id 32000
    # is the ordinary text token " Capital". Treating it as a stop truncated
    # every pass that transcribed that word ("Venture Capital") mid-sentence and
    # threw away the rest of the chunk -- silently, because the
    # truncated text is clean prose, so neither _looks_degenerate nor the loop
    # breaker flags it and no retry rung fires. These values are only the
    # fallback; __init__ reads the real ids off the processor.
    _STOP_TOKENS = (2, 4, 11)

    def _resolve_stop_tokens(self):
        """The build's own </s> / [/INST] / <pad> ids, or the class fallback."""
        ids = getattr(self.proc, "_special_token_ids", None) or {}
        stops = []
        for key in ("eos", "inst_end", "pad"):
            value = ids.get(key) if hasattr(ids, "get") else None
            if isinstance(value, int) and value not in stops:
                stops.append(value)
        return tuple(stops) or _Voxtral._STOP_TOKENS

    def _consume_tokens(self, token_stream, token_cb=None, breaker=None):
        """Collect ids from a greedy token stream up to the first stop token
        (which is dropped, matching decode(skip_special_tokens=True)). The stream
        is already length-bounded by generate_step's max_tokens.

        This deliberately does NOT replicate generate_stream's 10-identical-token
        backstop. That backstop cuts a single-token repetition loop off after only
        10 tokens, which lands *under* _looks_degenerate's thresholds (a 12-word
        run, or a >4 compression ratio) -- so the degenerate pass would slip
        through un-flagged, keeping truncated garbage and losing everything after
        the loop. Letting the loop run instead lets _looks_degenerate catch it
        (the compression-ratio net) and fire the split/penalty retry, which
        recovers clean text. In practice the backstop almost never fired anyway:
        real Voxtral loops repeat a *word* ("Jetzt. Jetzt.") whose tokens cycle,
        so consecutive-identical-token never reached 10. Output is therefore
        identical to model.generate() on every normal pass and better (retried
        rather than truncated) on the rare single-token loop."""
        stops = self._STOP_TOKENS
        out = []
        last_beat = 0.0
        for t in token_stream:
            t = int(t)
            if t in stops:
                break
            # The in-generation loop breaker has given up on this attempt:
            # stop wasting tokens, the caller's retry ladder takes over.
            if breaker is not None and breaker.gave_up:
                break
            out.append(t)
            # Liveness heartbeat: Voxtral emits a whole pass at once, so without
            # this the log/progress sits silent for the entire (possibly minutes-
            # long) decode. Throttled to ~1.5 s so it never floods the queue.
            if token_cb is not None:
                now = time.monotonic()
                if now - last_beat >= 1.5:
                    last_beat = now
                    try:
                        token_cb(len(out))
                    except Exception:
                        pass
        return out

    def _merged_embeddings(self, mi):
        """Audio embeddings written into the prompt at the [AUDIO] placeholders.

        Kept rather than calling the library's `_merge_input_embeddings`, which
        since 0.0.6 does the same thing (bit-identical on the batch of 1 that
        production uses, same peak memory, same wall clock -- measured on a
        300 s prompt): it is private API, and this copy is testable without the
        model, so CI covers it where the model-gated smoke test cannot.

        The dtype promotion is load-bearing. `embed_tokens` returns bf16 while
        the projector returns float32 -- its weights are bf16, but the log-Mel
        features are float32 and MLX promotes the output to the input's dtype
        (measured on voxtral-mini-8bit). Scattering into the bf16 array would
        silently round every audio embedding away -- a change that looks like
        nothing and shows up only as different logits.
        """
        mx = self._mx
        import numpy as np
        model = self.model
        input_ids = mi["input_ids"]
        features = mi.get("input_features")
        if features is None:
            return model.embed_tokens(input_ids)
        audio_embeds = model.get_audio_embeds(features)
        embeds = model.embed_tokens(input_ids).astype(audio_embeds.dtype)
        ids = np.array(input_ids)
        # get_audio_embeds returns one stream ([1, n, hidden]) however many 30 s
        # chunks went in, so a multi-row prompt shares it. Any other count is
        # refused here rather than indexed: reading past the end of an mlx array
        # yields *zeros* instead of raising, which would replace the audio with
        # silence -- invisible in the text, visible only as wrong logits.
        n_audio = audio_embeds.shape[0]
        if n_audio not in (1, ids.shape[0]):
            raise ValueError(
                f"{n_audio} audio streams for a prompt of {ids.shape[0]} rows; "
                f"expected one shared stream or one per row.")
        for i in range(ids.shape[0]):
            pos = np.where(ids[i] == model.config.audio_token_id)[0]
            if pos.size == 0:
                continue
            if pos.size != audio_embeds.shape[1]:
                raise ValueError(
                    f"Batch {i}: {audio_embeds.shape[1]} audio embeddings for "
                    f"{pos.size} [AUDIO] placeholders in the prompt.")
            embeds[i, mx.array(pos)] = audio_embeds[i if n_audio > 1 else 0]
        return embeds

    def _fast_generate(self, mi, max_new_tokens, token_cb=None,
                       temperature=0.0, seed=None, breaker=None):
        """Decode through the maintained mlx_lm.generate_step, returning the
        generated token ids ([1, n]).

        At temperature 0.0 the output matches model.generate() (same greedy
        argmax) on every normal pass; it is faster and -- crucially -- lower peak
        memory on long passes, because generate_step processes the (large audio)
        prompt in prefill_step_size chunks instead of one forward. Measured on
        mini-8bit at a 600s prompt: ~7% faster and ~18% lower peak; near
        break-even on short prompts, where memory is not the constraint anyway.
        (The one intentional divergence is on a single-token repetition loop --
        see _consume_tokens.)

        temperature > 0 samples through mlx_lm's make_sampler (the retry
        ladder's gentle-sampling stages); a fixed `seed` keeps those retries
        reproducible. `breaker` is an optional _LoopBreaker armed as a logits
        processor. The rare penalty path (repetition_penalty > 1) stays on the
        library implementation.
        """
        mx = self._mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import KVCache
        model = self.model

        sampler = None  # greedy argmax, matching temperature 0.0
        if temperature > 0:
            from mlx_lm.sample_utils import make_sampler
            if seed is not None:
                mx.random.seed(seed)
            sampler = make_sampler(temp=temperature)

        # [seq, hidden] merged audio+text embeddings; generate_step adds the batch.
        embeds = self._merged_embeddings(mi)[0]
        cache = [KVCache() for _ in range(len(model.language_model.layers))]
        stream = generate_step(prompt=mx.array([], dtype=mx.int32),
                               input_embeddings=embeds, model=self._lm_adapter,
                               max_tokens=max_new_tokens, sampler=sampler,
                               logits_processors=[breaker] if breaker else None,
                               prompt_cache=cache)
        toks = self._consume_tokens((t for t, _ in stream), token_cb=token_cb,
                                    breaker=breaker)
        return mx.array([toks], dtype=mx.uint32)

    def transcribe_array(self, audio, language, max_new_tokens=4096,
                         repetition_penalty=1.0, token_cb=None,
                         temperature=0.0, seed=None, info=None):
        """Transcribe one audio array to text.

        `info`, if given, is a dict that receives loop-breaker telemetry for
        the retry ladder: {"loop_kicks": int, "loop_gave_up": bool}.
        """
        inp = self.proc.apply_transcrition_request(audio=audio, language=language,
                                                   sampling_rate=SAMPLE_RATE)
        mi = {"input_ids": inp.input_ids, "input_features": inp.input_features}
        if getattr(inp, "attention_mask", None) is not None:
            mi["attention_mask"] = inp.attention_mask
        # temperature=0.0 and NO repetition penalty, matching Mistral's reference
        # transcription request. mlx_voxtral defaults repetition_penalty to 1.2,
        # which is a chat default: it divides the logit of every token seen in
        # the last 20 tokens, and in verbatim speech the most-repeated tokens are
        # punctuation and function words. Measured on a 10 min German podcast,
        # the default cost 27% of all commas (426 -> 312 over a 20 min podcast,
        # 10.51 -> 7.89 per 100 words) and swallowed real repetitions
        # ("sehr, sehr" -> "sehr"), which is what made
        # transcripts read worse than Whisper's. Generation stays bounded by
        # max_new_tokens.
        if repetition_penalty == 1.0 and "attention_mask" not in mi:
            # Fast path: identical tokens at temperature 0, faster and lower
            # peak memory on long passes. See _fast_generate. Skipped when a
            # padding mask is present (the fast path assumes a single unpadded
            # sequence and the model's internal causal mask). The loop breaker
            # rides along and repairs a repetition loop in place; if it gives
            # up, the truncated attempt is reported degenerate via `info` so
            # the retry ladder never mistakes it for a clean short pass.
            breaker = _LoopBreaker()
            gen = self._fast_generate(mi, max_new_tokens, token_cb=token_cb,
                                      temperature=temperature, seed=seed,
                                      breaker=breaker)
            if info is not None:
                info["loop_kicks"] = breaker.kicks
                info["loop_gave_up"] = breaker.gave_up
        else:
            # Pass the resolved stop ids explicitly so both branches stop on
            # the ids this build's processor reports (see `_STOP_TOKENS` for
            # the history); the library's default is hard-coded.
            out = self.model.generate(**mi, max_new_tokens=max_new_tokens,
                                      temperature=temperature,
                                      repetition_penalty=repetition_penalty,
                                      stop_tokens=list(self._STOP_TOKENS))
            gen = out[:, inp.input_ids.shape[1]:]  # drop the prompt
        return self.proc.decode(gen[0], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# CTC forced alignment (text + audio -> word timestamps)
# --------------------------------------------------------------------------- #
def _align_device():
    """Torch device for the alignment forward pass.

    The aligner used to be the last piece of this module on the CPU, in a
    module that only runs on Apple Silicon. On MPS the emissions are the same
    -- argmax identical on every frame of a 300 s reference, worst log-prob
    deviation ~5e-3 -- and the pass is measurably faster: 14.85 s to 4.60 s for
    that clip, which is most of the aligner's runtime since the forward is ~97%
    of it. fp16 would be faster again and is deliberately not used: it moves
    0.067% of argmaxes and up to 1.68 in log-prob, enough to shift a word
    boundary, and this change is meant to be free of effect.

    Repeats `pyannote_mp_worker.pyannote_proc_entrypoint`'s probe rather than
    sharing it, macOS floor included -- MPS before 12.3 is not something to
    rely on. There is no shared helper to call: `utils.py` is deliberately
    torch-free. Callers that need to pin a device pass `_Aligner(device=...)`.
    """
    try:
        import platform
        import torch
        if platform.system() != "Darwin":
            return "cpu"
        if platform.mac_ver()[0] < "12.3":
            return "cpu"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


class _Aligner:
    # Set from the model's vocabulary size once one is loaded. None means the
    # emission has no room for a wildcard column, so `_tokenize` falls back to
    # dropping what it cannot spell -- which is also what a hand-built _Aligner
    # in the tests gets.
    wild = None
    # Recursion cap for the oversized-window split. Deep enough that halving
    # brings a genuinely too-big window under FORCED_ALIGN_MAX_CELLS (the cell
    # count shrinks ~4x per level), while still terminating.
    MAX_SPLIT_DEPTH = 12
    # A *density* problem is not fixed by halving: the audio is cut at the
    # words' character share, so tokens-per-frame comes out the same in both
    # halves and the test re-fires at every level. Measured, that ran the tree
    # to the full depth and pushed 10-13x the chunk's audio through wav2vec2
    # (~17-21 min of CPU for a 1500 s chunk) to reach leaves that end in
    # _spread anyway. Two levels still catch the case where the pause snap
    # shifts the character proportion enough to help.
    MAX_DENSE_SPLIT_DEPTH = 2
    # Pieces `align_prefix` may chain. The cell cap allows ~440 s of German
    # speech per piece against a 1500 s window (and more as the remaining audio
    # shrinks), so a prefix filling even the largest chunk we produce needs a
    # handful. The cap only stops a pathological chain that advances by a word
    # or two per piece from paying for a full-window DP each time.
    MAX_PREFIX_PIECES = 16
    # How far the next piece starts before the previous one ended, so its first
    # word is whole even when CTC let the previous word run long.
    PREFIX_PIECE_BACKOFF_SEC = 0.5

    def __init__(self, model_name=ALIGN_MODEL_MULTILINGUAL, device=None):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self._torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
        # Left in its loaded dtype on purpose -- see _align_device on fp16.
        self.device = device or _align_device()
        if self.device != "cpu":
            try:
                self.model = self.model.to(self.device)
            except Exception as exc:
                logger.warning("Alignment model will not move to %s (%s); "
                               "staying on CPU.", self.device, exc)
                self.device = "cpu"
        self.vocab = self.proc.tokenizer.get_vocab()
        # The CTC blank is the index the network actually emits between labels,
        # which is NOT reliably the token spelled "<pad>": the multilingual MMS
        # aligner's vocab is {"<blank>": 0, "<pad>": 1, ...} and it emits blanks
        # on 0, so reading "<pad>" picked an index that model never produces --
        # forced_align could then never park on blank during a pause and
        # stretched every word to meet its neighbour (measured: inter-word gaps
        # all collapsed to zero, words up to 0.8 s too long, scores -3.9 instead
        # of -0.09). config.pad_token_id carries the trained blank for both
        # families (0 for the per-language jonatasgrosman models and for MMS).
        self.blank = getattr(self.model.config, "pad_token_id", None)
        if self.blank is None:
            self.blank = self.vocab.get("<blank>", self.vocab.get("<pad>", 0))
        self.delim = self.vocab.get("|", None)
        # One column past the model's own vocabulary; see the class attribute.
        self.wild = int(getattr(self.model.config, "vocab_size", 0) or 0) or None

    def _emission(self, audio):
        """Windowed wav2vec2 log-prob emissions [T, vocab] (bounded memory),
        as a float32 numpy array -- the DP downstream is numpy."""
        import numpy as np
        torch = self._torch
        # Zero-copy view of the (already float32) audio buffer instead of a
        # duplicate allocation per alignment call.
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        win = int(EMISSION_WINDOW_SEC * SAMPLE_RATE)
        # wav2vec2's conv feature extractor raises on inputs shorter than its
        # receptive field (~400 samples / 25 ms). A trailing remainder that small
        # carries negligible alignment signal, so skip it rather than crash the
        # whole job -- the guard is `< min_win`, not just the empty case.
        min_win = int(0.025 * SAMPLE_RATE)
        try:
            parts = self._forward_windows(wav, win, min_win)
        except Exception as exc:
            # An unsupported op on this backend must not cost the whole job a
            # transcript. Fall back once, for good, and say so. The work done
            # before the failure is discarded, which is bounded by one chunk
            # and happens at most once per aligner.
            if self.device == "cpu":
                raise
            logger.warning("Alignment on %s failed (%s); falling back to CPU.",
                           self.device, exc)
            self.device = "cpu"
            self.model = self.model.to("cpu")
            parts = self._forward_windows(wav, win, min_win)
        if not parts:
            return None
        out = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        # A view for the float32 models this ships with (inference-mode CPU
        # tensors); .float() only does work for a half-precision checkpoint,
        # whose bfloat16 .numpy() would otherwise raise outside every try.
        return out.float().numpy()

    def _forward_windows(self, wav, win, min_win):
        """Log-prob emissions per window, always returned on the CPU: the
        DP that consumes them is numpy and reads host memory."""
        torch = self._torch
        parts = []
        with torch.inference_mode():
            for i in range(0, len(wav), win):
                seg = wav[i:i + win]
                if seg.numel() < min_win:
                    continue
                # Zero mean, unit variance per window: what the aligners were
                # trained on (do_normalize in the preprocessor config of all four
                # checked -- MMS and the English, German and Spanish xlsr-53 -- done
                # by Wav2Vec2FeatureExtractor, which we bypass). On the raw signal the
                # model scales with the recording level, and a quiet voice gets
                # emissions too weak to hold its own words -- the DP, which must
                # place every character, then crowds them into the louder
                # speaker's time. Measured by realigning the same Voxtral text over
                # 23 AMI meetings and 64 CallHome calls: words more than 0.5 s off
                # AMI's manual stamps 7.1 % -> 3.1 %, words under the wrong
                # speaker after the voice check 2.65 % -> 2.24 % (70 of 82 files
                # better); on 23 VoxConverse broadcasts, which took no part and are
                # mastered, no effect (+0.05 points, 95 % [-0.06, +0.16]). It
                # does not reach a voice 15-20 dB below one that talks
                # through the whole window: the window's variance is the loud
                # voice's, whatever its gain (the level-matched second alignment in
                # _AlignerPool.align_words is for that). Population variance and 1e-7, as the
                # extractor has them. A window shorter than `win` -- the rest at
                # the end of a pass -- takes its statistics from the `win` samples
                # that end with it: on its own, a few hundred ms of room tone
                # after the last word would be lifted to speech level.
                ref = wav[max(0, i + seg.numel() - win):i + seg.numel()]
                seg = (seg - ref.mean()) / torch.sqrt(ref.var(correction=0) + 1e-7)
                if self.device != "cpu":
                    seg = seg.to(self.device)
                lg = self.model(seg.unsqueeze(0)).logits[0]
                parts.append(torch.log_softmax(lg, dim=-1).cpu())
        return parts

    def _tokenize(self, words, wildcard=False):
        """Characters -> target ids. With `wildcard`, a character the model
        cannot spell becomes `self.wild` instead of vanishing; the caller must
        then have extended the emission with that column."""
        tokens, tok_word = [], []
        for wi, w in enumerate(words):
            for ch in w.lower():
                if ch in self.vocab:
                    tokens.append(self.vocab[ch])
                    tok_word.append(wi)
                    continue
                # A mapping that exists but cannot be spelled in THIS vocabulary
                # must fall through to the wildcard rather than be treated as
                # handled -- otherwise the character vanishes silently and the
                # word loses an anchor with nothing in the log to say so.
                spelled = [self.vocab[c]
                           for c in _ALIGN_CHAR_FALLBACKS.get(ch, "")
                           if c in self.vocab]
                if spelled:
                    tokens.extend(spelled)
                    tok_word.extend([wi] * len(spelled))
                elif wildcard and self.wild is not None and ch.isalnum():
                    tokens.append(self.wild)
                    tok_word.append(wi)
            if self.delim is not None:
                tokens.append(self.delim)
                tok_word.append(-1)
        return tokens, tok_word

    @staticmethod
    def _adjacent_repeats(tokens):
        """The DP's own rule for the extra frames it needs; one definition."""
        from noScribe import ctc_align
        return ctc_align.adjacent_repeats(tokens)

    @staticmethod
    def _dp_cells(n_targets, frames):
        """Size of the forced-alignment DP: one byte per cell for the backtrace
        table, and the time to fill it. Every caller checks it against a cap
        before handing the work over -- see FORCED_ALIGN_MAX_CELLS."""
        return frames * (2 * n_targets + 1)

    def _predict_frames(self, n_samples):
        """Frames `_emission` would return for `n_samples`, without running it.

        _emission feeds the audio through wav2vec2 in EMISSION_WINDOW_SEC
        windows and skips a trailing remainder below the conv receptive field,
        so the count follows the feature extractor's conv geometry exactly
        (verified bit-exact against the real model from 1 s to 45 s). This only
        routes the split decision -- the real emission still decides at the
        leaf, so an off-by-a-frame here cannot produce a wrong alignment.
        """
        cfg = getattr(getattr(self, "model", None), "config", None)
        kernels = tuple(getattr(cfg, "conv_kernel", ()) or ())
        strides = tuple(getattr(cfg, "conv_stride", ()) or ())
        win = int(EMISSION_WINDOW_SEC * SAMPLE_RATE)
        min_win = int(0.025 * SAMPLE_RATE)
        total = 0
        for i in range(0, n_samples, win):
            seg = min(win, n_samples - i)
            if seg < min_win:
                continue
            if kernels and strides:
                out = seg
                for k, s in zip(kernels, strides):
                    out = (out - k) // s + 1
            else:
                out = seg // 320        # geometry unavailable: ~20 ms frames
            total += max(0, out)
        return total

    def _spread_loudly(self, why, words, audio, t_offset, tokens, frames,
                       exc_info=False):
        """_spread with the log line that makes it visible. Never silent: a
        spread window has plausible-looking times that are wrong by up to
        minutes, and without this line it is indistinguishable in the log
        from a good alignment. The two silent callers of _spread are the
        benign ones -- a window under 0.1 s, or one with no alignable
        characters at all."""
        logger.warning(
            "Forced alignment %s for a %.0fs window (%d words, %d tokens, "
            "%d frames); falling back to evenly spread timestamps.",
            why, len(audio) / SAMPLE_RATE, len(words), len(tokens), frames,
            exc_info=exc_info)
        return self._spread(words, audio, t_offset)

    def _spread(self, words, audio, t_offset):
        """Fallback: distribute words evenly (by length) over the audio when
        real alignment isn't possible (empty/too-dense text)."""
        if not words:
            return []
        dur = len(audio) / SAMPLE_RATE
        total = sum(len(w) for w in words) or len(words)
        out, pos = [], 0.0
        for w in words:
            s = t_offset + dur * pos
            pos += (len(w) or 1) / total
            e = t_offset + dur * pos
            out.append({"word": w, "start": s, "end": e, "prob": 0.0})
        return out

    def _stamps_from_spans(self, spans, words, tok_word, fps, t_start, t_end):
        """Merged CTC token spans -> [{word,start,end,prob}] for `words`.

        Frames are counted from `t_start` (the time the aligned emission
        begins) and `t_end` bounds the trailing fill. None when not a single
        word could be placed -- the caller decides what to fall back to.
        """
        out, ti = {}, 0
        for sp in spans:
            if sp.token == self.blank:
                continue
            if ti < len(tok_word):
                wi = tok_word[ti]
                if wi >= 0:
                    s = t_start + sp.start / fps
                    e = t_start + sp.end / fps
                    if wi not in out:
                        # merge_tokens scores are log probabilities (<= 0);
                        # whisper_mp_worker puts a plain probability in this
                        # field, and the module docstring promises the same
                        # shape. Convert here, at the one place a real score
                        # enters, so 0.0 keeps its meaning downstream: "not
                        # actually aligned, spread or interpolated". A genuine
                        # score of exactly 0.0 now maps to 1.0 rather than
                        # colliding with that sentinel.
                        out[wi] = [s, e, math.exp(float(sp.score))]
                    else:
                        out[wi][1] = e
                ti += 1

        # Fill words whose characters were all out-of-vocabulary (numbers,
        # symbols) by interpolating between aligned neighbours.
        n = len(words)
        idx = sorted(out)
        if not idx:
            return None
        full = [None] * n
        for wi in idx:
            full[wi] = out[wi]
        # Leading/trailing OOV words: spread them over the audio before the
        # first / after the last aligned word (capped at ~0.6 s per word)
        # instead of collapsing them to zero-width spans, which would turn
        # into zero-duration cues that subtitle players skip or reject.
        # A tiny per-word floor so a lead/tail OOV word never collapses to a
        # zero-duration cue when the first/last aligned word sits exactly on
        # the chunk boundary (step would otherwise be 0). The resulting
        # sub-frame overlap into neighbouring audio is harmless for cues.
        min_step = 0.02
        if idx[0] > 0:
            s1 = out[idx[0]][0]
            lead = max(t_start, s1 - 0.6 * idx[0])
            step = max((s1 - lead) / idx[0], min_step)
            for i in range(idx[0]):
                full[i] = [lead + step * i, lead + step * (i + 1), 0.0]
        n_tail = n - (idx[-1] + 1)
        if n_tail > 0:
            e0 = out[idx[-1]][1]
            tail_end = min(t_end, e0 + 0.6 * n_tail)
            step = max((tail_end - e0) / n_tail, min_step)
            for k, i in enumerate(range(idx[-1] + 1, n)):
                full[i] = [e0 + step * k, e0 + step * (k + 1), 0.0]
        # Interior OOV words: tile the gap between the two aligned
        # neighbours, with the same min_step floor the lead/tail runs use.
        # They used to collapse to a zero-width [t, t] span -- the very
        # thing min_step was introduced to prevent -- and every digit is
        # out of vocabulary in these aligners, so a German year or number
        # lands here routinely. When such a word forms a cue of its own
        # (it ends in '.', so the cue is flushed around it) the result was
        # a zero-duration VTT cue, which the spec forbids and players skip;
        # nothing downstream guards against it.
        for a, b in zip(idx, idx[1:]):
            if b - a > 1:
                s0, s1 = out[a][1], out[b][0]
                step = max((s1 - s0) / (b - a - 1), min_step)
                for k in range(a + 1, b):
                    t0 = s0 + step * (k - a - 1)
                    full[k] = [t0, t0 + step, 0.0]
        return [{"word": words[i], "start": full[i][0],
                 "end": full[i][1], "prob": full[i][2]} for i in range(n)]

    def align_words(self, words, audio, t_offset=0.0, depth=0):
        """Return [{word,start,end,prob}] for `words` against `audio`.

        The words must span the audio. Text that covers only the front of its
        window -- what the loop ladder's prefix salvage produces -- belongs in
        `align_prefix`; the split below would cut the audio at the words'
        character share and place the later words on audio that never contained
        them.

        CTC forced alignment requires at least as many audio frames as target
        tokens. Dense speech (or a Voxtral over-generation) in a long chunk can
        have more character tokens than frames, which makes forced_align raise.
        When that happens we split the words in half, cut the audio at a nearby
        pause, and align each half recursively — so we always get a transcript
        instead of a crash. As a last resort the words are spread evenly.
        """
        from noScribe import ctc_align
        if not words:
            return []
        if len(audio) < int(0.1 * SAMPLE_RATE):
            return self._spread(words, audio, t_offset)

        tokens, tok_word = self._tokenize(words, wildcard=True)
        if not tokens:
            return self._spread(words, audio, t_offset)

        # Route on a PREDICTED frame count. The emission is only needed in order
        # to align, not in order to decide whether to split, and computing it up
        # front meant every window that then split paid for a wav2vec2 forward
        # pass that was thrown away while each half recomputed its own --
        # measured at 2.00x the necessary forward-pass work (~97 s of CPU per
        # 1500 s chunk) on every file past ~13.5 min, where too_big always
        # fires. The real frame count still governs at the leaf.
        n_frames = self._predict_frames(len(audio))
        # forced_align needs one frame per target token PLUS one for each pair
        # of adjacent identical tokens (a CTC path has to put a blank between
        # them); the DP enforces exactly that and names the repeat count in
        # its error. The old `len(tokens) > n_frames * 0.95` left only ~5.3%
        # slack, while German text runs 3.9-6.1% adjacent repeats -- 8.9% with
        # the multilingual aligner, and up to 29.7% for number-heavy text, where
        # every all-OOV word contributes just a delimiter and those land back to
        # back. Windows inside that band passed the check, forced_align raised,
        # and the whole window silently degraded to evenly-spread timestamps.
        repeats = self._adjacent_repeats(tokens)
        too_dense = len(tokens) + repeats > n_frames
        # Above FORCED_ALIGN_MAX_CELLS (see its comment): split. Smaller
        # windows are also much faster to align.
        too_big = self._dp_cells(len(tokens), n_frames) > FORCED_ALIGN_MAX_CELLS
        splittable = len(words) > 1 and depth < self.MAX_SPLIT_DEPTH
        if too_dense and not too_big and depth >= self.MAX_DENSE_SPLIT_DEPTH:
            splittable = False          # halving does not reduce density
        if too_big and not splittable:
            # Cannot split further -- do not run a DP that size.
            return self._spread_loudly("skipped, window too big to split further",
                                       words, audio, t_offset, tokens, n_frames)
        if not too_big and (not too_dense or not splittable):
            emission = self._emission(audio)
            if emission is None:
                return self._spread_loudly("produced no emission", words, audio,
                                           t_offset, tokens, 0)
            if self.wild is not None and self.wild in tokens:
                import numpy as np
                # A character the model has no letter for -- a digit, a symbol,
                # a foreign script. Dropping it left the whole word without an
                # anchor, so `_stamps_from_spans` interpolated its time across
                # the gap between its neighbours and the word came out too wide.
                # One extra column, as good as the best real token at every
                # frame, lets the DP place it on the audio instead. Measured on
                # the 300 s reference, where "10", "20" and "50" are the only
                # such words: identical to the nearest alternative (spelling the
                # number out in German) to the millisecond on all three, 80 ms
                # from the interpolation on average and 160 ms at worst, and not
                # one of the other 847 words moved. The wildcard wins on cost:
                # spelling out needs a number speller per language, this needs
                # none. `align_prefix` deliberately does NOT use it -- its own
                # trailing star is a sentinel it searches the spans for, and a
                # second star-like column would make that search ambiguous.
                #
                # The column scores at least as high as any real token at every
                # frame, so it does take some frames from its neighbours, and
                # where it takes them from is a guess -- the aligner does not
                # know what the word sounds like. Measured adversarially by
                # replacing every 20th word of the reference with digits, then
                # comparing the untouched words against their true times:
                #
                #   distance to the nearest wildcard word   mean      worst
                #     1 (immediately beside)              31.1 ms   1061 ms
                #     2                                    4.2 ms    300 ms
                #     3-4                                  0.4 ms     40 ms
                #     5 and beyond                         0.0 ms      0 ms
                #
                # So it is a seam, not a drift: it decays to exactly nothing
                # within five words and never accumulates. Dropping the word
                # instead keeps that one neighbour closer (worst 480 ms) but is
                # worse on average at every density tested (4.6 vs 3.8 ms here,
                # 22.3 vs 13.5 at 20 %, 48.1 vs 30.2 at 50 %). Penalising the
                # column was measured as the obvious cure and is not one: at
                # -0.25 through -2.0 nats nothing changes at all, and at -4.0
                # the number words break (160 ms off) while the worst neighbour
                # stays at 1061 ms. There is nothing here to tune.
                nb = [i for i in range(emission.shape[1]) if i != self.blank]
                emission = np.concatenate(
                    [emission, emission[:, nb].max(1, keepdims=True)], axis=1)
            n_real = emission.shape[0]
            fps = n_real / (len(audio) / SAMPLE_RATE)
            # The prediction only routed us here; the real frame count decides
            # whether the DP can run at all, and within budget. Spreading
            # beats a raise.
            if len(tokens) + repeats > n_real:
                return self._spread_loudly(
                    f"skipped, {len(tokens)} tokens + {repeats} repeats need "
                    f"more frames than the window has",
                    words, audio, t_offset, tokens, n_real)
            if self._dp_cells(len(tokens), n_real) > FORCED_ALIGN_MAX_CELLS:
                return self._spread_loudly(
                    "skipped, over the DP cell budget at the real frame count",
                    words, audio, t_offset, tokens, n_real)
            try:
                aligned, scores = ctc_align.forced_align(
                    emission, tokens, blank=self.blank)
                spans = ctc_align.merge_tokens(aligned, scores, blank=self.blank)
            except Exception:
                return self._spread_loudly("failed", words, audio, t_offset,
                                           tokens, n_real, exc_info=True)

            stamps = self._stamps_from_spans(spans, words, tok_word, fps,
                                             t_offset,
                                             t_offset + len(audio) / SAMPLE_RATE)
            if stamps is None:
                return self._spread_loudly("placed no word", words, audio,
                                           t_offset, tokens, n_real)
            return stamps

        # Too dense or too big: split words in half and audio at their
        # character share, cut at a nearby pause, recurse. The share is why this
        # method needs the words to span the audio -- on text covering only the
        # front of the window it put the later words on audio that never held
        # them, and they came back with ordinary scores (measured: the resume
        # point landed 152 s past the prefix's true end, and that speech was
        # dropped from the transcript). `align_prefix` exists for that case.
        mid = max(1, len(words) // 2)
        first_chars = sum(len(w) for w in words[:mid]) + mid
        total_chars = sum(len(w) for w in words) + len(words)
        cut = int(len(audio) * first_chars / max(1, total_chars))
        # snap the cut to the quietest 100 ms frame within +/- 3 s, then keep it
        # a valid interior split point
        fr = max(1, int(0.1 * SAMPLE_RATE))
        cut = _quietest_frame_near(audio, cut, 3.0)
        cut = min(max(cut, fr), len(audio) - fr)
        left = self.align_words(words[:mid], audio[:cut], t_offset, depth + 1)
        right = self.align_words(words[mid:], audio[cut:],
                                 t_offset + cut / SAMPLE_RATE, depth + 1)
        return left + right

    def _prefix_piece(self, words, start, frames):
        """How many words from `start` one forced_align call can take against
        `frames` frames -- the largest count that stays under the cell cap and
        inside the density limit, 0 when not even one word fits.

        Both limits grow monotonically with the word count, so a binary search
        over the exact predicate is sound (and cheap: ~log2(n) tokenisations).
        """
        # The salvage budget is the smaller of the two in production; the min
        # is there so that lowering the hard cap still binds.
        cap = min(FORCED_ALIGN_MAX_CELLS, SALVAGE_ALIGN_MAX_CELLS)

        def fits(k):
            tokens, _ = self._tokenize(words[start:start + k])
            if not tokens:
                return False
            n = len(tokens) + 1                 # + the trailing star token
            if n + self._adjacent_repeats(tokens) > frames:
                return False
            return self._dp_cells(n, frames) <= cap

        if not fits(1):                     # also covers no frames / no words
            return 0
        lo, hi = 1, len(words) - start
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def align_prefix(self, words, audio, t_offset=0.0):
        """Timestamps for words that cover only the FRONT of `audio`.

        This is what the loop ladder's prefix salvage needs: a clean prefix of
        a looping pass, mapped onto the audio time where it ends. Two things
        make that different from `align_words`, and both were measured on real
        emissions (300 s of German, ground truth from the aligner's own greedy
        decode, so every word's true frame is known):

        1. **Plain forced alignment cannot end early.** A CTC path has to
           account for every frame, and parking on blank across 215 s of real
           speech costs far more (-21905) than smearing the prefix's last words
           over it, so the maximum-likelihood path does exactly that: the last
           word of a 85 s prefix came back ending at 299.98 s of a 300 s
           window, and it came back with a *perfect* score (-0.000), which is
           why no guard downstream could ever have caught it. It is not a
           tie-break artefact -- the drifted path wins by 802 nats. The fix is
           the star token from the MMS/torchaudio partial-transcript recipe: an
           extra vocabulary column, as good as the best real token at every
           frame, appended once after the prefix's tokens, so the aligner can
           say "the rest of this audio is not in my text". With it the same
           prefix ends 0.02 s (one frame) from the truth, at every prefix
           length from 5% to 100% of the window.

        2. **A long window blows past FORCED_ALIGN_MAX_CELLS.** `align_words`
           answers that by halving the words and cutting the audio at their
           character share, which is unsound for a prefix: the share assumes
           the words fill the window, so the later words were put on audio that
           never contained them (measured: resume point 152 s late, that speech
           dropped from the transcript). Here the words are cut instead of the
           audio -- each piece is small enough for one forced_align call and is
           aligned, star and all, against ALL the audio left after the previous
           piece. No audio boundary is ever estimated from text length. The
           window's emission is computed once and sliced, so the wav2vec2 cost
           is that of a single alignment.

        Any piece that cannot be aligned for real fails the whole call to
        evenly spread times (prob 0.0), because a chain is only as trustworthy
        as its weakest link: a guessed piece boundary would move every later
        piece, and those would come back carrying perfectly real scores.
        """
        import numpy as np
        from noScribe import ctc_align
        if not words:
            return []
        if len(audio) < int(0.1 * SAMPLE_RATE):
            return self._spread(words, audio, t_offset)

        emission = self._emission(audio)
        if emission is None:
            return self._spread(words, audio, t_offset)
        n_frames = emission.shape[0]
        dur = len(audio) / SAMPLE_RATE
        fps = n_frames / dur
        back = int(self.PREFIX_PIECE_BACKOFF_SEC * fps)
        # The star column is a per-frame maximum, so slicing rows first would
        # give the same values: build it once for the window and let each piece
        # take a row view.
        star = emission.shape[1]
        emission = np.concatenate(
            [emission, emission.max(axis=-1, keepdims=True)], axis=-1)

        def extend(new):
            # The next piece is given a little audio the previous one already
            # used (see the back-off below), so its first word can start just
            # before the previous word ended. Keep the returned stamps in order
            # -- a non-monotonic list would break cue building if these ever
            # reach it.
            for st in new:
                if stamps and st["start"] < stamps[-1]["end"]:
                    st["start"] = stamps[-1]["end"]
                    st["end"] = max(st["end"], st["start"])
                stamps.append(st)

        stamps, f0, i, piece_no, why = [], 0, 0, 0, None
        while i < len(words):
            piece_no += 1
            if piece_no > self.MAX_PREFIX_PIECES:
                why = f"more than {self.MAX_PREFIX_PIECES} pieces"
                break
            k = self._prefix_piece(words, i, n_frames - f0)
            if k <= 0:
                why = (f"{len(words) - i} words left do not fit the "
                       f"{n_frames - f0} frames left")
                break
            piece = words[i:i + k]
            tokens, tok_word = self._tokenize(piece)
            try:
                # One extra vocabulary column, as good as the best real token at
                # every frame, and one extra target token at the very end: that
                # is how the aligner is told "after these words the audio is not
                # mine". Without it the words are stretched over the rest (see
                # the docstring), with it the last word lands frame-exact.
                aligned, scores = ctc_align.forced_align(
                    emission[f0:], tokens + [star], blank=self.blank)
                spans = ctc_align.merge_tokens(aligned, scores, blank=self.blank)
            except Exception:
                logger.warning("Prefix alignment failed on piece %d (%d words, "
                               "%d tokens, %d frames).", piece_no, len(piece),
                               len(tokens), n_frames - f0, exc_info=True)
                why = "forced alignment raised"
                break
            t_start = t_offset + f0 / fps
            # Where the star begins is where this piece's speech ends, so it is
            # also the ceiling for interpolating any trailing out-of-vocabulary
            # word -- the window's end would be nonsense here.
            t_end = t_offset + dur
            # The star is the last target token, so its span is at the end.
            last = next((sp for sp in reversed(spans) if sp.token == star), None)
            if last is not None:
                t_end = min(t_end, t_start + last.start / fps)
            got = self._stamps_from_spans(spans, piece, tok_word + [-1], fps,
                                          t_start, t_end)
            if got is None:
                why = f"piece {piece_no} placed no word at all"
                break
            if i + k >= len(words):
                extend(got)                     # last piece: keep every word
                break
            # Chain on the last word that was really aligned. A piece can end
            # in out-of-vocabulary words (numbers, symbols) whose times are
            # interpolated -- those are no basis for slicing the audio, so they
            # are handed back to the next piece instead, which aligns them
            # against audio that starts before them.
            j = next((x for x in range(len(got) - 1, -1, -1)
                      if got[x].get("prob")), None)
            if j is None:
                why = f"piece {piece_no} was not really aligned"
                break
            extend(got[:j + 1])
            i += j + 1
            # Step back a little so the next piece's first word is whole even
            # if this piece's last word ended slightly long.
            f_end = int(round((got[j]["end"] - t_offset) * fps))
            f0 = min(max(f0 + 1, f_end - back), n_frames)
            if f0 >= n_frames:
                why = "the prefix ran past the end of the window"
                break
        if why:
            logger.warning("Prefix alignment degraded to evenly spread times "
                           "(%s); the caller must not cut audio on these.", why)
            return self._spread(words, audio, t_offset)
        return stamps


# --------------------------------------------------------------------------- #
# Segment building
# --------------------------------------------------------------------------- #
def _split_sentences(text):
    return [m.group(0).strip() for m in _SENT_SPLIT.finditer(text) if m.group(0).strip()]


# Subtitle cue sizing: aim for short, readable phrases (a few words), not one
# word per cue and not 30-second blocks -- comparable to what noScribe produces
# from Whisper's phrase-level segments. The limits follow common subtitle
# practice (about one line of 42-45 characters, a few seconds on screen); they
# are conventions, not measured optima.
SUB_MAX_CHARS = 45
SUB_MAX_WORDS = 12
SUB_MAX_SEC = 6.0
SUB_MIN_WORDS = 4


def _segments_from_words(word_stamps):
    """Group aligned words into subtitle-sized cues.

    A cue is ended at sentence punctuation, at a clause boundary (comma/colon)
    once it is long enough, or when it hits a length cap (characters, words or
    duration). This keeps cues to a readable phrase rather than a whole
    sentence or a fixed time window.
    """
    segments = []
    cur = []

    def flush():
        if not cur:
            return
        txt = " ".join(w["word"] for w in cur).strip()
        if txt:
            segments.append({
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "text": " " + txt,
                "words": list(cur),
            })
        cur.clear()

    for w in word_stamps:
        # Close the open cue BEFORE adding a word that would push it past the
        # duration cap. `dur` below is measured with the word already appended
        # and flush() can only end a cue *including* it, so the cap did not
        # actually bind: any non-speech gap longer than SUB_MAX_SEC produced a
        # cue as long as the gap (a 30 s interlude gave one 31 s two-word cue,
        # an ordinary 9 s thinking pause gave 10.4 s), and the merge pass below
        # only ever concatenates, so nothing downstream repaired it.
        if cur and (w["end"] - cur[0]["start"]) > SUB_MAX_SEC:
            flush()
        cur.append(w)
        tok = w["word"]
        chars = sum(len(x["word"]) + 1 for x in cur)
        dur = cur[-1]["end"] - cur[0]["start"]
        if tok.endswith((".", "!", "?", "…")):
            flush()
        elif (len(cur) >= SUB_MIN_WORDS and tok.endswith((",", ";", ":"))
              and (chars >= SUB_MAX_CHARS * 0.55 or dur >= SUB_MAX_SEC * 0.55)):
            flush()
        elif len(cur) >= SUB_MAX_WORDS or chars >= SUB_MAX_CHARS or dur >= SUB_MAX_SEC:
            flush()
    flush()

    # Merge a tiny leftover cue (e.g. a lone "habe.") back into the previous one
    # when the result still fits, so subtitles don't get one-word fragments.
    #
    # A lone word is a fragment however long it is drawn out, so only the
    # two-word case still needs the duration guard. With the guard applied to
    # both, a stretched final word survived on its own: "angesagt." after a cue
    # closed by SUB_MAX_CHARS ran ~1.2 s into the pause before the next cue and
    # stayed a one-word segment. A segment that short takes its speaker from
    # whatever the diarization happens to put under it -- 0.4 s of a spurious
    # backchannel cluster was enough to hand one word its own speaker and its own
    # paragraph, mid-sentence. The combined-length checks below still refuse the
    # merge when the result would be an oversized cue, so a lone word after a
    # long pause is left alone as before.
    merged = []
    for seg in segments:
        short = (len(seg["words"]) == 1
                 or (len(seg["words"]) == 2 and (seg["end"] - seg["start"]) < 1.2))
        if merged and short:
            prev = merged[-1]
            combined = prev["words"] + seg["words"]
            if len(combined) <= SUB_MAX_WORDS + 3 and (seg["end"] - prev["start"]) <= SUB_MAX_SEC + 2:
                prev["words"] = combined
                prev["end"] = seg["end"]
                prev["text"] = " " + " ".join(w["word"] for w in combined).strip()
                continue
        merged.append(seg)
    return merged


def _segments_text_only(text, duration):
    """Short path: sentence segments with proportional (approximate) times."""
    sentences = _split_sentences(text)
    total = sum(len(s) for s in sentences) or 1
    segments, pos = [], 0
    for s in sentences:
        start = duration * pos / total
        pos += len(s)
        end = duration * pos / total
        segments.append({"start": start, "end": end, "text": " " + s, "words": None})
    return segments


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def transcribe(audio_path, language="de", need_timestamps=True,
               voxtral_repo=None, chunk_sec=None,
               corrections_path=None, speaker_names=None, ram_reserve_gb=None,
               speaker_turns=None,
               log_cb=None, progress_cb=None, segment_cb=None):
    """
    Transcribe `audio_path` with Voxtral and return noScribe-compatible segments.

    need_timestamps=True  -> also run forced alignment for word-level times
                             (needed for VTT, speaker assignment, pauses).
    need_timestamps=False -> fast path, plain text only (approx. segment times).

    corrections_path -> optional YAML word-correction list (brand/product names)
                        applied to the transcribed text.
    chunk_sec        -> per-pass length in seconds; None/0 = pick automatically
                        from RAM (see _auto_chunk_sec). A short file that fits one
                        pass is never split regardless of this value.
    segment_cb       -> optional callable(segment_dict); called for each finished
                        segment as soon as its pass completes, so callers can
                        stream/autosave partial transcripts instead of waiting
                        for the whole file.
    speaker_turns    -> optional [[start_s, end_s, label], ...] from a prior
                        diarization on the SAME audio timeline. Used to cut
                        looping chunks at speaker-turn boundaries (the cleanest
                        split) and to log a diarization profile when a loop
                        resists the gentle repairs. Without it, splits fall
                        back to the quietest pause, exactly as before.
    """
    import soundfile as sf

    # "auto"/"multilingual" are how the menu spells "no language". Taken
    # literally they became the pinned language "au"/"mu" -- in the prompt, in
    # the translated-pass guard and in the aligner choice.
    if (language or "").strip().lower() in ("", "auto", "multilingual"):
        language = None

    from noScribe import transcript_corrections
    corrections = transcript_corrections.load_corrections(corrections_path)
    if corrections:
        _log(log_cb, "info", f"Applying {len(corrections)} word correction(s).")
    # The speaker names the user entered are the correct spelling of words that
    # are very likely to be spoken; a spelling of one of them that sounds the
    # same and is no word is set to theirs (apply_name_corrections).
    speaker_names = [n for n in (speaker_names or []) if n]
    if speaker_names:
        _log(log_cb, "info", f"Correcting the spelling of the speaker names: "
                             f"{', '.join(speaker_names)}")

    # A missing file must be diagnosed as a missing file, not as whatever the
    # memory sizing below happens to find wrong with the machine.
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    # An empty recording is an empty transcript, as with Whisper -- not a
    # model load followed by "[as_strided] Negative dimensions" from the
    # log-Mel features, which cannot be computed for zero samples.
    if sf.info(audio_path).frames == 0:
        _log(log_cb, "warn", "The audio contains no samples; nothing to transcribe.")
        return [], {"duration": 0.0, "language": language}

    # The 8-bit mini build as the bare-API default: always fetchable from the
    # hub and its repo name classifies correctly in MEM_MODEL.
    repo = voxtral_repo or VOXTRAL_MODELS["voxtral-mini-8bit"]
    # The unquantised source releases carry no bit width in their name, so
    # _model_kind would meter them with a quantised profile and wave a too-long
    # pass through -- and the 24B one is a 48 GB download. The GUI never gets
    # here (it only offers the published builds), but a direct caller must be
    # stopped just as early.
    if (str(repo) in SOURCE_REPOS
            or os.path.basename(os.path.normpath(str(repo))).lower()
            in SOURCE_REPO_NAMES):
        raise ValueError(
            f"{repo} is an unquantised source release, not a runnable build. "
            f"Use one of the published builds ({', '.join(VOXTRAL_MODELS)}) or "
            f"convert it first with tools/quantize_voxtral.py.")

    # Size the passes BEFORE loading anything: the MemoryError for models that
    # cannot fit this machine is only worth something if it comes before 20+ GB
    # of weights have already pushed the OS into swap.
    # The config value may arrive as a string (YAML round-trip); coerce before
    # any arithmetic and fall back to auto-sizing on junk.
    try:
        chunk_sec = float(chunk_sec) if chunk_sec else None
    except (TypeError, ValueError):
        chunk_sec = None
    if not chunk_sec or chunk_sec <= 0:
        chunk_sec = _auto_chunk_sec(repo, log_cb, ram_reserve_gb)
    else:
        # A pinned voxtral_chunk_sec must not bypass the safety nets (it may
        # well predate a switch to a hungrier model). It may exceed the *auto*
        # length -- that one holds back the configurable reserve, and pinning
        # is the documented way to trade that reserve for context -- but
        # neither the hard memory ceiling past which the working set no longer
        # fits and the run stops progressing, nor the model-context cap
        # (MAX_CHUNK_SEC). A model whose shortest pass cannot fit is refused
        # outright, exactly as in the automatic path.
        ceiling = max_safe_chunk_sec(repo)  # raises MemoryError if unfit
        pinned = chunk_sec
        chunk_sec = int(min(pinned, ceiling, MAX_CHUNK_SEC, TRUSTED_CHUNK_SEC))
        if chunk_sec < HARD_MIN_CHUNK_SEC:
            # The clamp above only ever shortens, so a value this small came
            # from the config -- and int() turns anything under 1 s into 0,
            # which makes max_len one sample below and splits the file into one
            # decode per 50 ms (a 2-minute file became 2401 passes: the job
            # never finishes). The automatic path floors at HARD_MIN_CHUNK_SEC;
            # a pinned value must not be able to duck under it either.
            _log(log_cb, "warn",
                 f"voxtral_chunk_sec={pinned:g}s is below the "
                 f"{HARD_MIN_CHUNK_SEC}s minimum; using {HARD_MIN_CHUNK_SEC}s.")
            chunk_sec = HARD_MIN_CHUNK_SEC
        elif chunk_sec < pinned:
            # Three different reasons a pin gets shortened, and saying the wrong
            # one is worse than saying nothing: the machine, the model's context,
            # or our own refusal to run further out than Voxtral was measured.
            if ceiling < min(MAX_CHUNK_SEC, TRUSTED_CHUNK_SEC):
                what = "more memory than this machine has"
            elif MAX_CHUNK_SEC <= TRUSTED_CHUNK_SEC:
                # Unreachable while TRUSTED_CHUNK_SEC stays below the context
                # cap; kept for the day it is raised to it, as its comment invites.
                what = "more context than the model has"
            else:
                what = ("a longer window than Voxtral has been measured on "
                        f"({TRUSTED_CHUNK_SEC}s)")
            _log(log_cb, "warn",
                 f"voxtral_chunk_sec={pinned:.0f}s would need {what}; "
                 f"using {chunk_sec}s.")

    quant = _quant_summary(repo)
    _log(log_cb, "info", f"Loading Voxtral model: {repo}"
                         + (f" ({quant})" if quant else ""))
    vox = _load_from_hub(lambda: _Voxtral(repo), "Voxtral model", repo,
                         _DOWNLOAD_GB.get(repo), log_cb)

    # The aligner is chosen per chunk from the *transcribed text* (see
    # _AlignerPool): the text exists before its chunk is aligned, so Auto can
    # use the char-native per-language model instead of the weaker romanised
    # multilingual fallback, and mid-file language changes are followed.
    aligner_pool = _AlignerPool(language, log_cb) if need_timestamps else None
    # With the language set, its aligner is known now: fail before the first
    # pass is decoded rather than after it, if it can be neither found nor
    # fetched. (Loading it here would keep it resident through the decode.)
    if aligner_pool is not None and language:
        align_model = resolve_align_model(language)
        if _unfetchable(align_model):
            raise _not_fetchable_error("alignment model", align_model,
                                       _ALIGNER_DOWNLOAD_GB)

    # Language bookkeeping for the translated-chunk guard. With an explicit
    # language there is a target from the first chunk on; on Auto the file has
    # to say what it is first (see _file_language), which means the earliest
    # chunks can only be flagged after the fact -- they are already streamed to
    # the caller by then.
    pinned_lang = (language or "").strip().lower()[:2] or None
    lang_tally = {}
    chunk_langs = []          # (chunk, language) for chunks decoded before `want`

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sr}")
    duration = len(audio) / SAMPLE_RATE
    # Normalise the diarization turns defensively (they cross a process
    # boundary as plain lists): sorted (start_s, end_s, label) tuples, junk
    # dropped. None disables the turn-aware splitting.
    turns = None
    if speaker_turns:
        try:
            turns = sorted((float(s), float(e), str(lbl))
                           for s, e, lbl in speaker_turns if float(e) > float(s))
        except (TypeError, ValueError):
            turns = None
        turns = turns or None
    bounds = _pass_bounds(audio, chunk_sec)
    n_chunks = len(bounds) - 1
    if n_chunks > 1:
        _log(log_cb, "info",
             f"Audio {duration / 60:.1f} min exceeds the longest RAM-safe "
             f"decode window ({chunk_sec:.0f}s) -> transcribing in {n_chunks} "
             f"chunks (cut at pauses, {OVERLAP_SEC}s overlap).")
    # Overlap gives the model lead-in context at a seam; only used on the long
    # path where the duplicate can be dropped cleanly by timestamp.
    overlap = int(OVERLAP_SEC * SAMPLE_RATE) if aligner_pool is not None else 0
    all_segments = []
    # Progress is measured against the WHOLE audio (each pass contributes its own
    # share of the total duration, so a short final pass moves the bar only a
    # little), and never runs backward. _prog_max keeps it monotonic across the
    # intra-pass estimate, the pass-complete snap, and any split-retry re-decode.
    n_samples = max(1, len(audio))
    _prog_max = [0]

    def _emit_progress(pct):
        pct = int(pct)
        if pct > _prog_max[0]:
            _prog_max[0] = pct
        else:
            pct = _prog_max[0]
        if progress_cb:
            try:
                progress_cb(pct)
            except Exception:
                pass

    for ci in range(n_chunks):
        a0 = bounds[ci]
        a1 = bounds[ci + 1]
        a_read = max(0, a0 - overlap) if ci > 0 else a0
        chunk = audio[a_read:a1]
        t_offset = a_read / SAMPLE_RATE
        _log(log_cb, "info", f"Transcribing chunk {ci + 1}/{n_chunks} "
                             f"({a0 / SAMPLE_RATE:.0f}-{a1 / SAMPLE_RATE:.0f}s)")
        # Intra-pass liveness: Voxtral returns the whole pass at once, so estimate
        # how far the decode is from the token count and map it onto this pass's
        # slice of the overall bar (capped below the pass boundary; the real
        # position is set when the pass finishes below).
        _base = a0 / n_samples
        _span = max(0.0, (a1 - a0) / n_samples)
        _exp_tokens = max(1.0, (a1 - a0) / SAMPLE_RATE * _EST_TOKENS_PER_SEC)

        def _heartbeat(ntok, _base=_base, _span=_span, _exp=_exp_tokens):
            _emit_progress((_base + min(0.95, ntok / _exp) * _span) * 100)

        label = f"Chunk {ci + 1}/{n_chunks}"
        # The diarization's turns in this pass, window-relative: for the loop
        # ladder's cuts and for the alignment's coverage check alike.
        pass_turns = _clip_turns(turns, t_offset, a1 / SAMPLE_RATE)
        # What language should this chunk come back in? With an explicit
        # setting there is an answer from the first chunk on; on Auto the file
        # has to say what it is first (see _file_language), so the earliest
        # chunks go unchecked and can only be named afterwards.
        want = pinned_lang or _file_language(lang_tally)
        text = _transcribe_guarded(vox, chunk, language, log_cb, label,
                                   token_cb=_heartbeat,
                                   turns=pass_turns,
                                   align_cb=(aligner_pool.align_for_salvage
                                             if aligner_pool is not None else None),
                                   want_lang=want)
        if not text:
            continue
        # Before anything reads the text: a pass that silently dropped its
        # opening is not otherwise detectable, and the words are simply gone
        # from the transcript.
        #
        # Only the first pass needs this. Every later one starts OVERLAP_SEC
        # early and its overlap words are dropped again below by timestamp, so a
        # recovered opening would land before `a0` and be discarded -- which is
        # exactly what happened the one time the probe fired on a later chunk.
        # The previous pass already transcribed that audio. A loss LONGER than
        # the overlap would reach past `a0` and stays uncovered; no such loss has
        # been observed, and probing every pass for it cost ~10% of each decode.
        if a_read == 0:
            text = _recover_lost_head(vox, chunk, language, text, log_cb, label)
        chunk_lang, _ = _detect_language(text)
        if _other_language(text, want):
            # The ladder may have kept this on purpose (the audio can really be
            # in that language), so say both: only the listener can tell.
            _log(log_cb, "warn",
                 f"{label}: reads as '{chunk_lang}' rather than '{want}'. Either "
                 f"it is spoken in that language, or these "
                 f"{len(text.split())} words are a translation, not a transcript.")
        if chunk_lang:
            lang_tally[chunk_lang] = lang_tally.get(chunk_lang, 0) + 1
            if not want:
                chunk_langs.append((ci + 1, chunk_lang))
        if corrections:
            text = transcript_corrections.apply_corrections(text, corrections)
        if speaker_names:
            # Every language the text may be in: the one it reads as, and the
            # one expected (see apply_name_corrections).
            text = transcript_corrections.apply_name_corrections(
                text, speaker_names, (chunk_lang, want))
        # Unconditional, so the text-only path frees them too.
        _free_decode_buffers(vox)
        if aligner_pool is not None:
            words = re.findall(r"\S+", text)
            _log(log_cb, "info", f"{label}: aligning word timestamps "
                                 f"({len(words)} words)...")
            stamps = aligner_pool.align_words(words, chunk, t_offset, turns=pass_turns)
            if a_read < a0:
                # Drop the *words* already covered by the previous pass (the
                # overlap region). Filtering at word rather than cue level
                # means a cue straddling the seam can neither duplicate its
                # pre-seam words nor lose its post-seam ones.
                b = a0 / SAMPLE_RATE
                stamps = [w for w in stamps if (w["start"] + w["end"]) / 2 >= b]
            segs = _segments_from_words(stamps)
        else:
            segs = _segments_text_only(text, (a1 - a_read) / SAMPLE_RATE)
            for seg in segs:
                seg["start"] += t_offset
                seg["end"] += t_offset
        all_segments.extend(segs)
        if segment_cb:
            # Stream this pass's segments right away so the caller can show
            # and autosave a partial transcript during long files.
            for seg in segs:
                segment_cb(seg)
        # Snap the bar to this pass's true end position in the whole audio
        # (duration-weighted; the final pass lands on 100%).
        _emit_progress(a1 / n_samples * 100)

    # On Auto the file's language is only established once a couple of chunks
    # agree, so a chunk translated before that point was streamed to the caller
    # unchallenged. It cannot be taken back -- but leaving it unmentioned would
    # be worse: the transcript changes language part way through and nothing
    # says why.
    settled = pinned_lang or _file_language(lang_tally)
    odd = [ci for ci, lg in chunk_langs if settled and lg != settled]
    if odd:
        _log(log_cb, "warn",
             f"Chunk(s) {', '.join(str(c) for c in odd)} of {n_chunks} read as "
             f"a different language than the rest of the file ('{settled}') and "
             f"were already written out before that was known. Voxtral most "
             f"likely translated them; they are worth re-running on their own.")

    # Report the language the file settled on, so "auto" comes back as what
    # was actually transcribed rather than as the request.
    return all_segments, {"duration": duration, "language": settled or language}
