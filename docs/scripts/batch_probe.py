"""Batched greedy decode: B equal-length Voxtral passes decoded together.

Equal-length audio gives equal-length prompts, so no padding and no mask
change is needed. Compares against production (_fast_generate, one pass at a
time) on wall time and tokens/s, and checks that every batched row is
token-identical. Result: docs/voxtral-benchmarks.md §7 -- 1.16x at B=2, 1.11x
at B=4, every row identical; decode is not bandwidth-bound on an M1 Max.

    python docs/scripts/batch_probe.py <16 kHz wav> <seconds per pass> <B> [B ...]

The wav must hold at least max(B) passes.
"""
import sys, time
import pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]  # docs/scripts/x.py -> repo root
sys.path.insert(0, str(REPO))

import numpy as np
import soundfile as sf
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from noScribe.voxtral_engine import _Voxtral

wav, sec, Bs = sys.argv[1], float(sys.argv[2]), [int(b) for b in sys.argv[3:]]
audio, sr = sf.read(wav, dtype="float32")
n = int(sec * sr)
vox = _Voxtral(str(REPO / "models/voxtral-mini-8bit"))
lm = vox._lm_adapter
stops = set(vox._STOP_TOKENS)
PREFILL = 2048  # generate_step's default prefill_step_size
MAX_NEW = 4096


def inputs(i):
    chunk = audio[i * n:(i + 1) * n]
    inp = vox.proc.apply_transcrition_request(audio=chunk, language=None, sampling_rate=sr)
    return {"input_ids": inp.input_ids, "input_features": inp.input_features}


def batched_greedy(mis):
    """Prefill all rows in PREFILL-sized steps, then decode one token per row per
    step until every row has hit a stop token."""
    embeds = mx.concatenate([vox._merged_embeddings(mi) for mi in mis], axis=0)
    mx.eval(embeds)
    B, L, _ = embeds.shape
    cache = [KVCache() for _ in range(len(vox.model.language_model.layers))]
    t0 = time.perf_counter()
    pos = 0
    while L - pos > 1:  # like generate_step: all but the last position
        step = min(PREFILL, L - 1 - pos)
        lm(None, cache=cache, input_embeddings=embeds[:, pos:pos + step])
        mx.eval([c.state for c in cache])
        pos += step
    logits = lm(None, cache=cache, input_embeddings=embeds[:, pos:])
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    t1 = time.perf_counter()
    out = [[] for _ in range(B)]
    done = [False] * B
    steps = 0
    while not all(done) and steps < MAX_NEW:
        nxt = mx.argmax(lm(y[:, None], cache=cache)[:, -1, :], axis=-1)
        mx.async_eval(nxt)
        for b, t in enumerate(y.tolist()):
            if not done[b]:
                if t in stops:
                    done[b] = True
                else:
                    out[b].append(t)
        y = nxt
        steps += 1
    t2 = time.perf_counter()
    return out, t1 - t0, t2 - t1, steps


mx.eval(vox._merged_embeddings(inputs(0)))  # warm-up (kernels, weights)
maxB = max(Bs)
mis = [inputs(i) for i in range(maxB)]
print(f"passes of {sec:.0f}s, prompt lengths {[mi['input_ids'].shape[1] for mi in mis]}", flush=True)

# Production baseline: one pass at a time through _fast_generate.
prod, prod_t = [], []
for mi in mis:
    t0 = time.perf_counter()
    toks = np.array(vox._fast_generate(mi, MAX_NEW))[0].tolist()
    prod_t.append(time.perf_counter() - t0)
    prod.append(toks)
print(f"production B=1: {sum(prod_t):.1f}s for {maxB} passes, "
      f"{sum(map(len, prod))} tokens", flush=True)

# Own loop at B=1: separates the loop's own effect from batching.
own1 = []
for i, mi in enumerate(mis[:1]):
    out, tp, td, _ = batched_greedy([mi])
    own1.append(out[0])
    print(f"own loop B=1 pass0: prefill {tp:.1f}s decode {td:.1f}s "
          f"{len(out[0]) / td:.1f} tok/s, identical to production: {out[0] == prod[0]}", flush=True)

for B in Bs:
    mx.clear_cache(); mx.reset_peak_memory()
    t0 = time.perf_counter()
    out, tp, td, steps = batched_greedy(mis[:B])
    wall = time.perf_counter() - t0
    same = [out[b] == prod[b] for b in range(B)]
    first_diff = []
    for b in range(B):
        k = next((j for j, (a, c) in enumerate(zip(out[b], prod[b])) if a != c), None)
        first_diff.append(k if k is not None else (None if len(out[b]) == len(prod[b]) else min(len(out[b]), len(prod[b]))))
    ntok = sum(map(len, out))
    print(f"B={B}: wall {wall:.1f}s (prefill {tp:.1f}, decode {td:.1f}, {steps} steps) vs "
          f"production {sum(prod_t[:B]):.1f}s -> {sum(prod_t[:B]) / wall:.2f}x; "
          f"{ntok / td:.1f} tok/s total; peak {mx.get_peak_memory() / 1e9:.1f} GB; "
          f"identical rows {same}; first differing token {first_diff}", flush=True)
    for b in range(B):
        if not same[b]:
            a = vox.proc.decode(out[b], skip_special_tokens=True).split()
            c = vox.proc.decode(prod[b], skip_special_tokens=True).split()
            print(f"  row {b}: words batched {len(a)} vs production {len(c)}", flush=True)
