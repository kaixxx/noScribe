"""Does a repetition penalty break the 410-s loop, and at what price?

Runs the small build over the pass that looped in production (/tmp/loop_410.wav,
a scratch cut the user provides) once without penalty and once per rung of
RETRY_REPETITION_PENALTIES, and reports the longest run of identical words and
the verdict of _looks_degenerate. Feeds the penalty ladder in
noScribe/voxtral_engine.py (RETRY_REPETITION_PENALTIES); docs/voxtral-benchmarks.md §1.
"""
import sys, soundfile as sf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from noScribe.voxtral_engine import _Voxtral, _looks_degenerate, RETRY_REPETITION_PENALTIES
a,_ = sf.read('/tmp/loop_410.wav', dtype='float32')
vox=_Voxtral('models/voxtral-small')
for rep in (1.0, *RETRY_REPETITION_PENALTIES):
    t = vox.transcribe_array(a, None, max_new_tokens=8704, repetition_penalty=rep)
    w=t.split(); best=run=1
    for x,y in zip(w,w[1:]): run=run+1 if x==y else 1; best=max(best,run)
    print(f"\n=== small 410s, repetition_penalty={rep} ===", flush=True)
    print(f"  {len(w)} words | longest run: {best} | detector: {'DEGENERATE' if _looks_degenerate(t) else 'ok'}", flush=True)
    print(f"  end: ...{t[-150:]}", flush=True)
