"""Build a quantised Voxtral model for the noScribe Voxtral engine.

Quantised weights are much faster and need far less memory than the bf16
release, which in turn allows longer passes. The 3B model at 8 bit is
word-for-word identical to bf16 on our test material at roughly five times the
speed, so it is the recommended build.

mlx_voxtral ships its own `scripts/quantize_voxtral.py`, but its argparse
restricts --bits to 2/4/8. MLX supports 4/5/6/8 and mlx_lm's quantize_model
accepts any of them, so this is the same pipeline with the restriction lifted.

Both Voxtral releases are Apache-2.0, so the resulting weights may be shared.

A Voxtral build has three parts that can carry different precision, and they
behave very differently:

  * the audio encoder (audio_tower + projector, ~0.6B params in BOTH model
    sizes) runs ONCE per pass, so its precision barely affects speed;
  * lm_head (vocab x hidden, 0.4B in the 3B model / 0.7B in the 24B) runs once
    per GENERATED TOKEN, so its precision costs real time;
  * the language model body dominates size, and its bit width is the main
    quality/memory trade.

They are therefore separate options rather than one "mode".

Usage:
    python tools/quantize_voxtral.py <source-repo-or-dir> <output-dir> \
        [bits] [group] [mode] [--lm-head-bits N] [--encoder-bits N]

    group: 32, 64 (default) or 128 -- the only sizes MLX supports, and it must
           divide the last dimension of every tensor being quantised. Both
           Voxtral releases satisfy all three; a checkpoint that does not is
           refused before quantising, because mlx_lm would otherwise skip the
           offending tensors silently and leave them dense.

    mode: uniform    -- every quantizable tensor at [bits] (default)
          mixed      -- audio tower, projector and lm_head two bits higher
                        (only meaningful below 8 bit)
          dense-audio -- audio tower, projector and lm_head stay bf16, the
                        language model is quantised to [bits]
          dense-encoder -- only the audio tower and projector stay bf16;
                        lm_head is quantised too, so the acoustic path is the
                        only thing that varies

    --lm-head-bits / --encoder-bits override the width the mode would pick for
    that part ("dense" = leave in bf16). Use them to vary ONE part at a time:
    a build that raises the encoder and lowers lm_head at the same time cannot
    tell you which change did what.

Examples (from the repository root, with the venv active):
    # recommended: 3B at 8 bit, ~5 GB, needs ~13 GB RAM to run
    python tools/quantize_voxtral.py mistralai/Voxtral-Mini-3B-2507 \
        models/voxtral-mini-8bit 8

    # 24B at 6 bit, ~20 GB, needs ~28 GB RAM to run (32 GB machine: short passes)
    python tools/quantize_voxtral.py mistralai/Voxtral-Small-24B-2507 \
        models/voxtral-small-6bit 6

    # 24B as shipped (voxtral-small-4bit): 4-bit body and lm_head, bf16 ear
    python tools/quantize_voxtral.py mistralai/Voxtral-Small-24B-2507 \
        models/voxtral-small-4bit 4 64 dense-encoder --lm-head-bits 4

The source weights are downloaded to the Hugging Face cache on first use
(3B: ~9 GB, 24B: ~48 GB). Conversion itself is quick and memory-light --
loading and quantising stay lazy, only saving evaluates the graph.
"""
import sys, time, json, shutil
from pathlib import Path

# MLX and mlx_voxtral are imported *after* the argument checks below, not here.
# They exist only on Apple Silicon, so importing them first turned every
# misspelled argument into a ModuleNotFoundError traceback on Windows and Linux
# -- and made the argument checks untestable anywhere else (they run in CI).

SUPPORTED_BITS = (4, 5, 6, 8)   # MLX affine quantisation
SUPPORTED_GROUPS = (32, 64, 128)  # mx.quantize accepts no others
MODES = ("uniform", "mixed", "dense-audio", "dense-encoder")

# Validate EVERYTHING before touching the network: a typo discovered after the
# multi-GB source download is a typo that cost an hour.
argv, overrides = [], {}
_it = iter(sys.argv[1:])
for a in _it:
    if a in ("--lm-head-bits", "--encoder-bits"):
        try:
            overrides[a] = next(_it)
        except StopIteration:
            sys.exit(f"{a} needs a value (a bit width or 'dense')")
    elif a in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0)
    elif a.startswith("--"):
        sys.exit(f"unknown option {a}")
    else:
        argv.append(a)

if len(argv) < 2:
    sys.exit(__doc__.split("Usage:")[1].split("Examples")[0].strip())
src = argv[0]
out = Path(argv[1])
bits = int(argv[2]) if len(argv) > 2 else 6
group = int(argv[3]) if len(argv) > 3 else 64
mode = argv[4] if len(argv) > 4 else "uniform"


def _width(flag):
    """Parse an override: a supported bit width, or 'dense' (stay bf16)."""
    v = overrides.get(flag)
    if v is None:
        return None
    if v.lower() == "dense":
        return False        # False = do not quantize (bf16)
    try:
        w = int(v)
    except ValueError:
        sys.exit(f"{flag} must be a number or 'dense', got {v!r}")
    if w not in SUPPORTED_BITS:
        sys.exit(f"{flag} must be one of {SUPPORTED_BITS} or 'dense', got {w}")
    return w


if bits not in SUPPORTED_BITS:
    sys.exit(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
if group not in SUPPORTED_GROUPS:
    # Checked here, before the download, because MLX rejects everything else
    # and the rejection would otherwise arrive an hour into a 48 GB fetch.
    sys.exit(f"group must be one of {SUPPORTED_GROUPS}, got {group}")
if mode not in MODES:
    sys.exit(f"mode must be one of {MODES}, got {mode!r}")
if mode == "mixed" and bits >= 8 and "--lm-head-bits" not in overrides \
        and "--encoder-bits" not in overrides:
    sys.exit("mixed with 8 bits is identical to uniform (the boost caps at 8) "
             "-- use uniform, or dense-audio to go beyond 8 bit")
# The mixed boost snaps to the next supported width (5+2=7 is not).
boost_bits = bits + 2 if bits + 2 in SUPPORTED_BITS else 8
lm_head_bits = _width("--lm-head-bits")
encoder_bits = _width("--encoder-bits")

if out.exists():
    sys.exit(f"output dir {out} already exists")

# Only now the Apple-Silicon stack, so that every check above works (and is
# testable) on a machine that does not have it, and reports a missing install
# as a sentence rather than a traceback.
try:
    import mlx.core as mx
    from mlx_voxtral import load_voxtral_model
    from mlx_voxtral.quantization import (
        quantize_model, save_model, save_config, compute_bits_per_weight,
    )
except ImportError as exc:
    sys.exit(f"this tool needs the Apple-Silicon Voxtral stack (pip install -r "
             f"environments/requirements_voxtral_macOS_arm64.txt): {exc}")
# Build into a scratch dir and rename at the very end, so an interrupted or
# failed run can never leave a half-written build that the model picker
# (which keys on config.json) would offer as usable.
tmp_out = out.parent / (out.name + ".partial")
if tmp_out.exists():
    shutil.rmtree(tmp_out)
tmp_out.mkdir(parents=True)

def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


t0 = time.time()
print(f"loading {src} (lazy, bfloat16) ...", flush=True)
model, config = load_voxtral_model(src, dtype=mx.bfloat16, lazy=True)
if "quantization" in config:
    sys.exit("source model is already quantized - use the original weights")
print(f"  loaded, RSS {rss_gb():.1f} GB, MLX peak {mx.get_peak_memory()/1e9:.1f} GB", flush=True)

SENSITIVE = ("audio_tower.", "multi_modal_projector.", "lm_head")


def _quantizable(path, module):
    if "embed_positions" in path or "pos_emb" in path:
        return False
    return hasattr(module, "to_quantized")


def uniform(path, module, *rest):
    return _quantizable(path, module)


def mixed(path, module, *rest):
    """Give the parts that carry the acoustic evidence extra bits: the audio
    encoder, the projector that feeds its output to the language model, and
    the output layer.

    This is a *simplified* variant of mlx_voxtral's own
    voxtral_mixed_quantization_predicate: the library additionally boosts the
    MLP projections of the first and last two language-model layers and adapts
    group sizes per tensor. That boost was not carried over because the part
    that is worth bits is the encoder, and the sweeps that establish it are in
    docs/voxtral-quantisation.md. The per-tensor group adaptation was not
    carried over for a different reason: it does not work. mlx_lm's
    quantize_model filters on the *global* group size before any custom
    predicate is consulted, so a
    tensor the group does not divide is skipped whatever a predicate returns
    for it. The group check before the quantize_model call below covers that
    case instead, by refusing.
    """
    if not _quantizable(path, module):
        return False
    if any(x in path for x in SENSITIVE):
        return {"group_size": group, "bits": boost_bits}
    return {"group_size": group, "bits": bits}


def dense_encoder(path, module, *rest):
    """Keep only the audio encoder and its projector in bf16; quantise the
    whole language model including lm_head to [bits].

    The isolating variant of dense-audio: it varies precision in the acoustic
    path *alone*, so a quality difference cannot be credited to a dense output
    layer (which for the 24B model is 1.3 GB on its own). Cheap, because the
    encoder is a small fraction of a large model -- the point of the test is
    whether a 24B language model is held back by a lossy ear.
    """
    if not _quantizable(path, module):
        return False
    if "audio_tower." in path or "multi_modal_projector." in path:
        return False
    return {"group_size": group, "bits": bits}


def dense_audio(path, module, *rest):
    """Quantise the language model, keep the acoustic path in bf16 -- the
    audio encoder AND the projector and lm_head (everything in SENSITIVE).

    The only way to go *beyond* 8 bit for these parts, since affine
    quantisation cannot express more.
    """
    if not _quantizable(path, module):
        return False
    if any(x in path for x in SENSITIVE):
        return False
    return {"group_size": group, "bits": bits}


_base = {"uniform": uniform, "mixed": mixed, "dense-audio": dense_audio,
         "dense-encoder": dense_encoder}[mode]

ENCODER = ("audio_tower.", "multi_modal_projector.")


def predicate(path, module, *rest):
    """The mode's choice, with per-part overrides applied last.

    Kept as one wrapper rather than four edited predicates so that "vary one
    part, hold the others" is the same operation whatever the base mode is.
    """
    result = _base(path, module, *rest)
    if not _quantizable(path, module):
        return result
    if lm_head_bits is not None and "lm_head" in path:
        return False if lm_head_bits is False else {"group_size": group, "bits": lm_head_bits}
    if encoder_bits is not None and any(x in path for x in ENCODER):
        return False if encoder_bits is False else {"group_size": group, "bits": encoder_bits}
    return result


# Refuse a group size that does not divide every tensor we mean to quantise.
#
# This is not defensive noise, and it cannot be delegated. mlx_lm's
# quantize_model wraps our predicate in one of its own that starts with
#
#     if module.weight.shape[-1] % group_size != 0: return False
#
# against the *global* group size -- so a tensor it does not divide is skipped
# silently, before our predicate is ever asked, and stays bf16 with no message.
# The build then looks finished: it loads, it transcribes, it is simply bigger
# and slower than the tables in docs/voxtral-quantisation.md say, and nothing
# names the layer that was left out. mlx_voxtral's own predicate has a fallback
# to group_size 32 for exactly this case; driven through quantize_model it is
# dead code, because the wrapper has already skipped the tensor (verified in
# tests/test_quantize_group_guard.py). Both Voxtral releases divide by all three
# supported group sizes, so this never fires today -- it exists so that a
# checkpoint with a different hidden dimension stops here, loudly, instead of
# producing a quietly mixed build.
skipped = []
for _path, _module in model.named_modules():
    if not _path or not hasattr(_module, "to_quantized"):
        continue
    if not predicate(_path, _module):
        continue                      # meant to stay dense; not our problem
    _last = _module.weight.shape[-1]
    if _last % group:
        skipped.append((_path, _last))
if skipped:
    shutil.rmtree(tmp_out)
    _lines = "\n".join(f"  {p} (last dimension {n})" for p, n in skipped[:10])
    _more = f"\n  ... and {len(skipped) - 10} more" if len(skipped) > 10 else ""
    _works = [g for g in SUPPORTED_GROUPS if all(n % g == 0 for _, n in skipped)]
    sys.exit(
        f"group size {group} does not divide {len(skipped)} tensor(s) that this "
        f"mode quantises:\n{_lines}{_more}\n"
        + (f"Use group {' or '.join(map(str, _works))} instead."
           if _works else
           "No supported group size divides all of them; this checkpoint needs "
           "a per-tensor group size, which mlx_lm's quantize_model cannot "
           "express.")
        + "\nRefusing rather than letting those tensors be skipped silently.")

print(f"quantizing to {bits} bit (group size {group}, mode {mode}) ...", flush=True)
qmodel, qconfig = quantize_model(
    model, config, group_size=group, bits=bits, quant_predicate=predicate,
)
print(f"  quantized, RSS {rss_gb():.1f} GB, MLX peak {mx.get_peak_memory()/1e9:.1f} GB", flush=True)
try:
    print(f"  average bits per weight: {compute_bits_per_weight(qmodel):.2f}", flush=True)
except Exception:
    pass

print(f"saving to {out} ...", flush=True)
save_model(tmp_out, qmodel, donate_model=True)
save_config(qconfig, tmp_out / "config.json")

# Tokenizer / processor files the loader needs alongside the weights. For a
# hub id the snapshot is already in the HF cache (load_voxtral_model fetched
# it above), so download_model only resolves the path.
if Path(src).exists():
    srcdir = Path(src)
else:
    from mlx_voxtral.utils.model_loading import download_model
    srcdir = Path(download_model(src))
copied = []
for name in ("tekken.json", "params.json", "preprocessor_config.json",
             "tokenizer_config.json", "tokenizer.json", "chat_template.json"):
    f = srcdir / name
    if f.is_file():
        shutil.copy2(f, tmp_out / name)
        copied.append(name)
# Without a tokenizer the build looks complete (config.json exists, so the
# model picker offers it) but fails cryptically at load -- refuse to produce it.
if not any(n in copied for n in ("tekken.json", "tokenizer.json")):
    shutil.rmtree(tmp_out)
    sys.exit(f"no tokenizer file found in {srcdir} -- refusing to write a "
             f"build that cannot be loaded")

tmp_out.rename(out)
size = sum(p.stat().st_size for p in out.glob("*.safetensors")) / 1e9
print(f"done in {time.time()-t0:.0f}s | {size:.1f} GB | RSS peak {rss_gb():.1f} GB | "
      f"MLX peak {mx.get_peak_memory()/1e9:.1f} GB", flush=True)
