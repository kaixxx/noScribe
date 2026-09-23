"""Punctuation vs loop resistance for repetition_penalty candidates.

Two grids over the penalty rungs: the mini build on the first 600 s
(/tmp/first600.wav) for commas per 100 words against the Whisper reference of
10.62, and the small build on the 410-s pass that looped in production
(/tmp/loop_410.wav) for the longest run of identical words and the verdict of
_looks_degenerate. Both inputs are scratch cuts the user provides. Feeds the
rungs of RETRY_REPETITION_PENALTIES in noScribe/voxtral_engine.py and the
first table in docs/voxtral-benchmarks.md §1.
"""
import sys, re, zlib, soundfile as sf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from noScribe.voxtral_engine import _Voxtral, _looks_degenerate

def stats(t):
    w=t.split(); best=run=1
    for a,b in zip(w,w[1:]): run=run+1 if a==b else 1; best=max(best,run)
    return len(w), t.count(',')/max(len(w),1)*100, best, _looks_degenerate(t)

# 1) punctuation quality: mini on 600s (where Whisper=10.62 is the reference)
audio600,_ = sf.read('/tmp/first600.wav', dtype='float32')
mini = _Voxtral('models/voxtral-mini')
print("=== mini 600s: punctuation (Whisper reference: 10.62 commas/100w) ===", flush=True)
for rep in (1.0, 1.1, 1.15, 1.2):
    t = mini.transcribe_array(audio600, 'de', max_new_tokens=12512, repetition_penalty=rep)
    w,c,run,deg = stats(t)
    print(f"  rep={rep:<5}  {w:5d} w | commas {c:5.2f}/100w | maxRun {run:3d} | {'DEGENERATE' if deg else 'ok'}", flush=True)
del mini
import mlx.core as mx; mx.clear_cache()

# 2) loop resistance: small on the critical 410s spot
audio410,_ = sf.read('/tmp/loop_410.wav', dtype='float32')
small = _Voxtral('models/voxtral-small')
print("\n=== small 410s: loop resistance at the critical spot ===", flush=True)
for rep in (1.0, 1.1, 1.15):
    t = small.transcribe_array(audio410, None, max_new_tokens=8704, repetition_penalty=rep)
    w,c,run,deg = stats(t)
    print(f"  rep={rep:<5}  {w:5d} w | commas {c:5.2f}/100w | maxRun {run:3d} | {'DEGENERATE' if deg else 'ok'}", flush=True)
