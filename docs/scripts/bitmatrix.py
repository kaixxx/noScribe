"""Speed + fidelity across bit widths of the mini build.

Transcribes one 150-s cut (/tmp/ab_150.wav, a scratch file the user provides)
with the bf16 release and locally quantised 8/6/4-bit builds and reports
realtime factor, peak memory and word agreement with bf16. The hypothesis
under test, raised when the builds were made: MLX's 6-bit matmul kernels may
be slower than the 4- and 8-bit ones, so 6 bit could lose in speed what it
saves in size. Feeds the bit-width tables in docs/voxtral-quantisation.md;
docs/voxtral-benchmarks.md §8 lists it.
"""
import sys, time, difflib, soundfile as sf, mlx.core as mx
import pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]  # docs/scripts/x.py -> repo root
sys.path.insert(0, str(REPO))
from noScribe.voxtral_engine import _Voxtral

a,_ = sf.read('/tmp/ab_150.wav', dtype='float32')
ref=None
for name, path in (("bf16","models/voxtral-mini"), ("8-bit","/tmp/mini8"),
                   ("6-bit","/tmp/mini6"), ("4-bit","/tmp/mini4")):
    vox=_Voxtral(path); mx.reset_peak_memory()
    t0=time.time(); t=vox.transcribe_array(a,'de',max_new_tokens=3512); el=time.time()-t0
    if ref is None: ref=t; sim=100.0
    else: sim=difflib.SequenceMatcher(None, t.split(), ref.split()).ratio()*100
    print(f"### {name:5s} {len(t.split()):4d} W | {150/el:5.2f}x | Peak {mx.get_peak_memory()/1e9:5.1f} GB | "
          f"word-identical to bf16 {sim:6.2f}%", flush=True)
    del vox; mx.clear_cache()
