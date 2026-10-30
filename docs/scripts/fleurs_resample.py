"""Is the resampler worth changing? FLEURS, with enough utterances to tell.

The podcast reference passage is too small to resolve a difference this size
(see docs/voxtral-audio-preprocessing.md, section 8), and FLEURS is natively
16 kHz, so nothing gets resampled there either. The trick: upsample every clip
to 44.1 kHz first with one high-quality resampler, then let each candidate path
bring it back down. The upsampling is common to all arms, so what is left is
exactly the difference between the paths -- measured over hundreds of
independent utterances.

Arms:
  swr-s16    what noScribe/audio/convert.py does: libswresample -> pcm_s16le
  soxr-s16   soxr VHQ -> the same int16 stage        (isolates the resampler)
  soxr-flt   soxr VHQ, no integer stage at all       (isolates the bit depth)

    python docs/scripts/fleurs_resample.py <n_samples> <build> [arm ...]
"""
import json
import pathlib
import sys
import time

import numpy as np
import soxr
import av

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from wer import norm, wer
from fleurs_gain import load_clips, paired_bootstrap

UP = 44100
SR = 16000
ARMS = ("swr-s16", "soxr-s16", "soxr-flt")


def swr_down(x44):
    """libswresample 44.1k -> 16k + pcm_s16le, the production chain for mono input.

    Mono in means no downmix, so PyAV's resampler and the one convert.py gets
    implicitly from encode() do the same thing here.
    """
    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(x44[np.newaxis, :], dtype=np.float32),
        format="fltp", layout="mono")
    frame.sample_rate = UP
    res = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SR)
    out = [f.to_ndarray().ravel() for f in res.resample(frame)]
    out += [f.to_ndarray().ravel() for f in res.resample(None)]
    return np.concatenate(out).astype(np.float32) / 32768


def to_s16(x):
    return (np.rint(x * 32768).clip(-32768, 32767)
            .astype(np.int16).astype(np.float32) / 32768)


def render(clip, arm):
    up = soxr.resample(clip.astype(np.float64), SR, UP, quality="VHQ")
    if arm == "swr-s16":
        return swr_down(up)
    down = soxr.resample(up, UP, SR, quality="VHQ").astype(np.float32)
    return to_s16(down) if arm == "soxr-s16" else down


def main():
    n, build = int(sys.argv[1]), sys.argv[2]
    arms = sys.argv[3:] or list(ARMS)

    import fleurs_gain
    fleurs_gain.N = n
    clips, refs = load_clips()
    sec = sum(len(c) for c in clips) / SR
    print(f"# FLEURS de_de test, {len(clips)} recordings, {sec/60:.1f} min")
    print(f"# 16k -> {UP} (soxr VHQ, common to all arms) -> 16k along each arm's path")
    print(f"# Build: {build}\n")
    print(f"{'Arm':10s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")

    import mlx.core as mx
    from noScribe.voxtral_engine import _Voxtral
    vox = _Voxtral(build)

    per_arm = {}
    for arm in arms:
        rows = []
        t0 = time.time()
        for clip, ref_text in zip(clips, refs):
            a = np.ascontiguousarray(render(clip, arm), dtype=np.float32)
            hyp = vox.transcribe_array(a, "de",
                                       max_new_tokens=int(len(a) / SR * 20) + 512)
            r, h = norm(ref_text), norm(hyp)
            rc, hc = "".join(r), "".join(h)
            rows.append(dict(w=wer(r, h)[0], ref_w=len(r),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        W = sum(r["w"] for r in rows) / max(1, sum(r["ref_w"] for r in rows)) * 100
        C = sum(r["c"] for r in rows) / max(1, sum(r["ref_c"] for r in rows)) * 100
        print(f"{arm:10s} {W:6.2f}% {C:6.2f}% {sec/el:6.2f}x", flush=True)
        per_arm[arm] = rows
        json.dump(per_arm, open("/tmp/fleurs_resample.json", "w"))
        mx.clear_cache()

    if len(arms) > 1:
        print("\nPaired differences over recordings "
              "(95% interval; if it contains 0, the difference is not demonstrable):")
        base = arms[0]
        for arm in arms[1:]:
            a, b = per_arm[base], per_arm[arm]
            lo_w, hi_w = paired_bootstrap(a, b, "w")
            lo_c, hi_c = paired_bootstrap(a, b, "c")
            dW = (sum(x['w'] for x in a) - sum(x['w'] for x in b)) / sum(x['ref_w'] for x in a) * 100
            dC = (sum(x['c'] for x in a) - sum(x['c'] for x in b)) / sum(x['ref_c'] for x in a) * 100
            print(f"  {base} - {arm:10s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]"
                  f"{'' if lo_w <= 0 <= hi_w else '  *'}   "
                  f"dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]"
                  f"{'' if lo_c <= 0 <= hi_c else '  *'}")
        print("\n* = interval excludes 0")


if __name__ == "__main__":
    main()
