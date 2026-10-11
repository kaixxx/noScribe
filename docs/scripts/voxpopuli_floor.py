"""Does a robust clamp floor cost anything on clean material? On VoxPopuli.

Two questions in one run.

**The corpus.** FLEURS is read aloud, clean, one speaker -- the gap that
`docs/voxtral-audio-preprocessing.md` concedes in several places.
`facebook/voxpopuli` de is **spontaneous** parliamentary speech with gold
transcripts and thus the closest public material to our use case. Only rows
with `is_gold_transcript` are taken.

**The question.** The clamp floor sits at `log_max - 8`, and `log_max` is the
maximum over the whole input. A single loud transient therefore lifts it for
the entire pass: measured, a 0.1-s bang lifts the floor by 9.6 dB and costs
1.66 WER points on `hart_780-900`, while a percentile floor ignores it
completely (dWER +0.00, text bit-identical with and without the bang).
Whether the percentile floor in turn costs anything on *clean* material
decides whether it is an option at all -- and that is exactly what this run
measures, paired on bit-identical audio.

Setup and bootstrap from fleurs_stream.py; two feature paths are compared on
the same streams.

    python docs/scripts/voxpopuli_floor.py [n_streams] [seconds] [percentiles]

`percentiles` may be a list (`99.0,99.9,99.99`). The max baseline is then
computed only once and all arms are paired against the same baseline -- that
is the sweep with which the percentile choice is decided.

`VOXPOPULI_PARQUET=/path/to/test-00000-of-00001.parquet` reads the locally
downloaded Hub file instead of streaming (see `load_clips`).
"""
import os
import pathlib
import re
import sys
import time

import numpy as np
import soundfile as sf
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer                                    # noqa: E402
from fleurs_stream import streams, paired, SR                # noqa: E402
from mlx_voxtral.audio_processing import VoxtralFeatureExtractor  # noqa: E402
from noScribe.voxtral_engine import _Voxtral, _PercentileFloorFeatures  # noqa: E402


def floor_arm(spec):
    """A floor arm from a short form: 'max' is the library path, '99' a
    percentile floor, '99c20' a percentile with a cap in dB (the floor never
    lies more than 20 dB below the maximum). The percentile and cap arms come
    from the production code."""
    if spec == "max":
        return "max (library)", VoxtralFeatureExtractor()
    if "c" in spec:
        pct, cap = spec.split("c")
        return (f"P{float(pct):g} cap {float(cap):g}dB",
                _PercentileFloorFeatures(float(pct), cap=float(cap) / 10.0))
    return f"percentile {float(spec):g}", _PercentileFloorFeatures(float(spec))


DIGIT = re.compile(r"\d")


def load_clips(n_rows, skip_digits=False):
    r"""Gold-transcribed VoxPopuli de rows as (audio, reference text).

    `skip_digits=True` discards utterances whose reference contains digits --
    roughly **9 %** of the corpus (measured over 200 rows). Reason: the
    reference writes "60 Jahre" and "21. Februar", the speaker says "sechzig
    Jahre" and "einundzwanzigsten Februar", and `wer.py`'s `norm()` keeps
    digits (`\w` includes them). Every such utterance therefore counts errors
    that were heard correctly. That is the known weakness of the VoxPopuli
    gold transcripts; a cleaned version exists only for English
    (`ArtificialAnalysis/VoxPopuli-Cleaned-AA`, checked).

    **Discarded rather than converted**, and that is deliberate. Cardinal
    numbers would be unambiguous, ordinals are not: "21." becomes
    "einundzwanzigsten", "einundzwanzigste" or "einundzwanzigster" depending
    on the sentence, and years before 2000 are spoken "neunzehnhundert..."
    rather than "eintausendneunhundert...". A conversion would therefore
    introduce new errors where it removes old ones.

    **The default stays False** so that this function does not retroactively
    shift the comparability of all previous runs. For *absolute* numbers that
    get published it belongs on True; **paired differences are not affected**,
    because both arms see the same reference and the error cancels out.
    """
    from datasets import Audio, load_dataset
    local = os.environ.get("VOXPOPULI_PARQUET")
    if local:
        # The same file as on the Hub (`de/test-00000-of-00001.parquet`,
        # 936 MB), fetched once with curl: streaming from the Hub aborts after
        # five timeouts on a bad connection, the download does not.
        ds = load_dataset("parquet", data_files=local, split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    else:
        ds = load_dataset("facebook/voxpopuli", "de", split="test", streaming=True)
    clips, refs, dropped = [], [], 0
    for row in ds:
        if not row.get("is_gold_transcript"):
            continue
        text = (row.get("raw_text") or "").strip()
        if len(text.split()) < 4:
            continue
        if skip_digits and DIGIT.search(text):
            dropped += 1
            continue
        a = row["audio"]
        wav = np.asarray(a["array"] if isinstance(a, dict) else a.get_all_samples().data,
                         dtype=np.float32).squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        clips.append(np.ascontiguousarray(wav))
        refs.append(text)
        if len(clips) >= n_rows:
            break
    if skip_digits:
        print(f"# {dropped} utterances with digits discarded "
              f"({dropped/max(1, dropped+len(clips))*100:.1f} %)")
    return clips, refs


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    specs = sys.argv[3].split(",") if len(sys.argv) > 3 else ["99.9"]

    clips, refs = load_clips(int(n_streams * sec / 9) + 60)
    st = streams(clips, refs, n_streams, sec)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} streams from VoxPopuli de (gold), ~{sec:.0f}s each, "
          f"{total/60:.1f} min total")

    vox = _Voxtral(str(REPO / "models" / "voxtral-mini-8bit"))
    # The percentile floor has been the production path since it went in, so
    # the max arm comes explicitly from the library, not from vox.proc.
    stock = VoxtralFeatureExtractor()
    arms = [floor_arm(s) for s in specs if s != "max"]

    # Does the swap take effect? Otherwise the run measures the same thing twice.
    probe = st[0][0]
    base_feat = np.array(stock(probe, sampling_rate=SR,
                               return_tensors="mlx")["input_features"])
    for name, ex in arms:
        d = np.abs(np.array(ex(probe)["input_features"]) - base_feat).max()
        print(f"# Control: feature max|diff| {name} vs max = {d:.4f}")
        assert d > 1e-3, f"{name} barely differs from the built-in path"

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    print(f"\n{'floor':16s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")
    store = {}
    for name, ex in [("max (library)", stock)] + arms:
        vox.proc.feature_extractor = ex
        rows, t0 = [], time.time()
        for x, words in st:
            a = np.ascontiguousarray(x, dtype=np.float32)
            hyp = norm(vox.transcribe_array(
                a, "de", max_new_tokens=int(len(a) / SR * 20) + 512))
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        print(f"{name:16s} {rate(rows,'w','ref_w'):6.2f}% {rate(rows,'c','ref_c'):6.2f}% "
              f"{total/el:6.2f}x", flush=True)
        store[name] = rows
        mx.clear_cache()

    a = store["max (library)"]
    star = lambda lo, hi: " *" if (lo > 0) == (hi > 0) else ""
    print("\nPaired against max (95%; if it contains 0, not demonstrable):")
    for name, _ in arms:
        b = store[name]
        lo_w, hi_w = paired(a, b, "w", "ref_w")
        lo_c, hi_c = paired(a, b, "c", "ref_c")
        dW = rate(b, "w", "ref_w") - rate(a, "w", "ref_w")
        dC = rate(b, "c", "ref_c") - rate(a, "c", "ref_c")
        print(f"  {name:16s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star(lo_w,hi_w)}"
              f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]{star(lo_c,hi_c)}")
    print("  (positive = percentile worse)")
    if len(st) < 8:
        print(f"  WARNING: only {len(st)} streams -- the interval does not hold.")


if __name__ == "__main__":
    main()
