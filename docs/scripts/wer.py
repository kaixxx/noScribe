"""Word error rate of every build against a hand-corrected reference.

Reference conventions (see Audiotest2/referenz/):
  //text//  passage spoken simultaneously by the second speaker while the main
            speaker continued. A transcript that drops it is not wrong -- the
            model is asked for the dominant voice -- so these words are scored
            separately, as a bonus rather than as errors.

    python docs/scripts/wer.py <reference.txt> <audio.wav> <build> [build ...]
"""
import re, sys, time, unicodedata
import soundfile as sf
import mlx.core as mx

import pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]  # docs/scripts/x.py -> repo root
sys.path.insert(0, str(REPO))
from noScribe.voxtral_engine import _Voxtral


OVERLAP = re.compile(r"//(.*?)//", re.S)


def norm(text):
    """Words, comparable: lowercase, no punctuation, umlauts kept, ß -> ss.

    Hyphens and slashes become spaces so "Hocus-Pocus" and "Hocus Pocus" count
    as the same two words -- the difference is typography, not recognition.
    """
    text = unicodedata.normalize("NFC", text.lower()).replace("ß", "ss")
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^\w\säöü]", " ", text)
    return text.split()


def wer(ref, hyp):
    """(errors, len(ref), substitutions, deletions, insertions) by edit distance."""
    n, m = len(ref), len(hyp)
    # d[i][j] = (cost, sub, del, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
            else:
                c_sub = prev[j - 1][0]
                c_del = prev[j][0]
                c_ins = cur[j - 1][0]
                best = min(c_sub, c_del, c_ins)
                if best == c_sub:
                    p = prev[j - 1]; cur[j] = (p[0] + 1, p[1] + 1, p[2], p[3])
                elif best == c_del:
                    p = prev[j]; cur[j] = (p[0] + 1, p[1], p[2] + 1, p[3])
                else:
                    p = cur[j - 1]; cur[j] = (p[0] + 1, p[1], p[2], p[3] + 1)
        prev = cur
    return prev[m]


# Everything below is the command line. It lives in main() so that other
# scripts can `from wer import norm, wer` without this running -- the
# normalisation must stay identical across every script that scores a build.
def main():
    raw = open(sys.argv[1], encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    # German lets the writer choose between "VitaFlor" and "Vita Flor", or
    # "Hokuspokus" and "Hocus Pocus" -- identical speech, different typing. The
    # word metric charges those as errors, so a character metric on the SPACE-FREE
    # text runs alongside it: it can only move when the sounds were heard
    # differently. Where WER and CER disagree, the difference was orthographic.
    ref_chars = "".join(ref)
    audio, sr = sf.read(sys.argv[2], dtype="float32")
    dur = len(audio) / sr
    print(f"# Reference {len(ref)} words / {len(ref_chars)} chars "
          f"(+{len(overlap_words)} overlapping), {dur:.0f}s\n")
    print(f"{'Build':30s} {'WER':>7s} {'CER':>7s} {'Sub':>5s} {'Del':>5s} {'Ins':>5s} "
          f"{'Speed':>7s} {'Peak':>7s}  Overlap")

    for path in sys.argv[3:]:
        vox = _Voxtral(path)
        mx.reset_peak_memory()
        t0 = time.time()
        text = vox.transcribe_array(audio, "de", max_new_tokens=int(dur * 20) + 512)
        el = time.time() - t0
        hyp = norm(text)
        err, sub, dele, ins = wer(ref, hyp)
        cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
        # how much of the overlapping utterance did it pick up?
        got = sum(1 for w in overlap_words if w in hyp) if overlap_words else 0
        name = path.rstrip('/').split('/')[-1]
        print(f"{name:30s} {err/len(ref)*100:6.2f}% {cer*100:6.2f}% {sub:5d} {dele:5d} {ins:5d} "
              f"{dur/el:6.2f}x {mx.get_peak_memory()/1e9:6.1f}G  {got}/{len(overlap_words)}",
              flush=True)
        open(f"/tmp/wer_{name}.txt", "w").write(text)
        del vox
        mx.clear_cache()


if __name__ == "__main__":
    main()
