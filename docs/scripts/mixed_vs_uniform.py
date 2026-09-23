"""Mixed-precision builds against their uniform counterparts.

Builds are made with tools/quantize_voxtral.py:
small: uniform 6-bit vs mixed 6/8-bit ("mixed" mode -- audio tower, projector
and lm_head get 8 bit; NOT the library predicate, which would additionally
boost the outer LM MLP layers).
mini:  8-bit vs "dense-audio" (audio tower, projector and lm_head stay bf16,
the LM is 8-bit), the only way to go *above* 8 bit for the acoustic part.
"""
import sys, time, difflib, soundfile as sf, mlx.core as mx

import pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]  # docs/scripts/x.py -> repo root
sys.path.insert(0, str(REPO))
from noScribe.voxtral_engine import _Voxtral, _looks_degenerate


WAV = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ab_150.wav'
BUILDS = sys.argv[2:] or ['models/voxtral-small-6bit', 'models/voxtral-small-6bit-mixed']

audio, sr = sf.read(WAV, dtype='float32')
dur = len(audio) / sr
print(f"# {WAV}  {dur:.0f}s\n", flush=True)

ref = None
for path in BUILDS:
    vox = _Voxtral(path)
    mx.reset_peak_memory()
    t0 = time.time()
    text = vox.transcribe_array(audio, 'de', max_new_tokens=int(dur * 20) + 512)
    el = time.time() - t0
    words = text.split()
    commas = text.count(',') / max(1, len(words)) * 100
    if ref is None:
        ref, sim = text, 100.0
    else:
        sim = difflib.SequenceMatcher(None, text.split(), ref.split()).ratio() * 100
    print(f"### {path.split('/')[-1]:26s} {len(words):5d} W | {dur/el:5.2f}x | "
          f"Peak {mx.get_peak_memory()/1e9:5.1f} GB | Kommas/100W {commas:5.2f} | "
          f"gleich zu #1 {sim:6.2f}% | degeneriert {_looks_degenerate(text)}", flush=True)
    open(f"/tmp/mx_{path.split('/')[-1]}.txt", 'w').write(text)
    del vox
    mx.clear_cache()
