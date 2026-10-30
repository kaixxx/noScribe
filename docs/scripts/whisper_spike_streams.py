"""Does the transient defect also hit the default engine? On many streams.

`faster_whisper/feature_extractor.py:227` carries the same line as the Voxtral
path -- `np.maximum(log_spec, log_spec.max() - 8.0)` over the whole waveform
handed in -- and `noScribe/whisper_mp_worker.py` hands Whisper the whole file.
If the mechanism takes effect there, it would affect every user of the default
setting, i.e. far more than the Voxtral engine.

A first attempt on **one** passage was inconclusive, for an understandable
reason: Whisper's own scatter was larger than the effect sought (on
`hart_780-900` 9.00 % WER with VAD against 21.09 % without). The way out is
the same as with the chunker and the Mel: many streams instead of one passage,
paired on bit-identical audio, with a bootstrap interval.

Whisper has a second lever here that Voxtral lacks: the VAD filter can remove
the transient before it sets the Mel maximum. So both states run -- if
`vad_filter=True` is already the protection, there is nothing to do.

    python docs/scripts/whisper_spike_streams.py [n_streams] [seconds] [dB] [model]
"""
import pathlib
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer                                    # noqa: E402
from fleurs_stream import streams, paired, SR                # noqa: E402
from voxpopuli_floor import load_clips                       # noqa: E402
from voxpopuli_spike import pair, floor_rise_db              # noqa: E402


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    db_over = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    model_name = sys.argv[4] if len(sys.argv) > 4 else "precise"

    from faster_whisper import WhisperModel
    clips, refs = load_clips(int(n_streams * sec / 9) + 60)
    st = streams(clips, refs, n_streams, sec)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} streams from VoxPopuli de (gold), ~{sec:.0f}s each, "
          f"{total/60:.1f} min total")
    print(f"# Transient: {db_over:+.0f} dB above the speech peak, 100 ms, at 45 %")
    print(f"# Model: models/{model_name}, CPU int8, beam 5\n")

    rises = [floor_rise_db(*pair(x, db_over)) for x, _ in st]
    print(f"# Control: floor rise median {np.median(rises):.1f} dB "
          f"(min {min(rises):.1f}, max {max(rises):.1f})\n")
    assert np.median(rises) > 0.7 * db_over, "Dose does not arrive"

    model = WhisperModel(f"models/{model_name}", device="cpu", compute_type="int8")

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    def score(spike, vad):
        rows, t0 = [], time.time()
        for x, words in st:
            a = pair(x, db_over)[1 if spike else 0]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                sf.write(fh.name, a, SR)
                path = fh.name
            segs, _ = model.transcribe(path, language="de", beam_size=5,
                                       vad_filter=vad)
            hyp = norm(" ".join(s.text for s in segs))
            pathlib.Path(path).unlink(missing_ok=True)
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        return rows, time.time() - t0

    print(f"{'VAD':>6s} {'audio':10s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")
    store = {}
    for vad in (True, False):
        for au, spike in (("clean", False), ("with bang", True)):
            rows, el = score(spike, vad)
            store[(vad, au)] = rows
            print(f"{str(vad):>6s} {au:10s} {rate(rows,'w','ref_w'):6.2f}% "
                  f"{rate(rows,'c','ref_c'):6.2f}% {total/el:6.2f}x", flush=True)

    print("\nCost of the transient, paired per VAD state "
          "(95 %; if it contains 0, not demonstrable):")
    for vad in (True, False):
        a, b = store[(vad, "clean")], store[(vad, "with bang")]
        lo_w, hi_w = paired(a, b, "w", "ref_w")
        lo_c, hi_c = paired(a, b, "c", "ref_c")
        dW = rate(b, "w", "ref_w") - rate(a, "w", "ref_w")
        dC = rate(b, "c", "ref_c") - rate(a, "c", "ref_c")
        star = " *" if (lo_w > 0) == (hi_w > 0) else ""
        print(f"  vad_filter={str(vad):5s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star}"
              f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]")
    print("  (positive = the bang hurts)")


if __name__ == "__main__":
    main()
