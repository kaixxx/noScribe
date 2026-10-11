"""Prepare a passage for hand correction.

Produces three files following the convention of Audiotest2/referenz/:

  <name>.wav           the passage, through the real production path
                       (noScribe/audio/convert.py), i.e. exactly what the
                       engine will later get to hear
  <name>_ENTWURF.txt   raw transcript of the current build, one line, as the
                       starting point for the correction (ENTWURF = draft)
  <name>_PEGEL.txt     per-second level and the quiet stretches derived from
                       it -- so that while correcting it is clear where
                       listening closely pays off, and so that those stretches
                       can later be scored separately (PEGEL = level)

What the script CANNOT do: correct. That takes ears. The draft is the start of
the work, not the result.

Convention of the reference (see docs/scripts/wer.py):
  //text//   spoken simultaneously by the second voice; a transcript that
             leaves it out is not wrong and is counted separately

    python docs/scripts/build_reference.py <source> <from_s> <to_s> <target-base> [build]
"""
import pathlib
import sys

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "scripts"))
from find_passage import production_wav, SR

QUIET_REL_DB = 10.0     # this far below the loud level counts as a "quiet stretch"
MIN_QUIET_SEC = 2       # shorter dips are breaths, not passages


def level_profile(seg):
    n = len(seg) // SR
    r = seg[:n * SR].reshape(n, SR).astype(np.float64)
    return 20 * np.log10(np.sqrt((r ** 2).mean(axis=1)) + 1e-12)


def quiet_runs(db, floor, loud):
    """Contiguous seconds that are speech but markedly quieter."""
    speech = db > floor + 12
    quiet = speech & (db < loud - QUIET_REL_DB)
    runs, start = [], None
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if i - start >= MIN_QUIET_SEC:
                runs.append((start, i))
            start = None
    return runs


def main():
    src, t0, t1, base = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    build = sys.argv[5] if len(sys.argv) > 5 else "models/voxtral-mini-8bit"

    wav = production_wav(src)
    x, sr = sf.read(wav, dtype="float32")
    assert sr == SR, sr
    seg = np.ascontiguousarray(x[t0 * SR:t1 * SR])

    out = pathlib.Path(base)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out.with_suffix(".wav"), seg, SR, subtype="PCM_16")

    db = level_profile(seg)
    # The noise floor has to come from the WHOLE recording, not from the
    # passage: a speech-dense passage has hardly any silence, so its own 10th
    # percentile sits in the middle of the quiet speech. With the passage
    # percentile as the floor, exactly what matters here would drop out of the
    # speech mask -- the spread would then look much smaller than it is.
    floor = float(np.percentile(level_profile(x), 10))
    speech = db > floor + 12
    loud = float(np.percentile(db[speech], 90)) if speech.any() else floor
    runs = quiet_runs(db, floor, loud)
    spread = (loud - float(np.percentile(db[speech], 10))) if speech.any() else 0.0

    from noScribe.voxtral_engine import _Voxtral
    vox = _Voxtral(build)
    text = vox.transcribe_array(seg, "de",
                                max_new_tokens=int((t1 - t0) * 20) + 512).strip()
    open(f"{base}_ENTWURF.txt", "w", encoding="utf-8").write(text)

    with open(f"{base}_PEGEL.txt", "w", encoding="utf-8") as f:
        f.write(f"# {pathlib.Path(src).name} {t0}-{t1}s, {t1-t0}s, "
                f"draft {len(text.split())} words, build {build}\n")
        f.write(f"# noise floor {floor:.1f} dB, loud speech level (p90) {loud:.1f} dB, "
                f"spread p90-p10 {spread:.1f} dB, speech share {speech.mean()*100:.0f}%\n")
        f.write(f"# Quiet stretches: speech more than {QUIET_REL_DB:.0f} dB below p90, "
                f"at least {MIN_QUIET_SEC} s in a row. "
                f"Seconds relative to the start of the passage.\n#\n")
        f.write(f"# {len(runs)} quiet stretches, "
                f"{sum(b-a for a, b in runs)} s in total "
                f"({sum(b-a for a, b in runs)/max(1,len(db))*100:.0f}% of the passage):\n")
        for a, b in runs:
            f.write(f"#   {a//60:02d}:{a%60:02d} - {b//60:02d}:{b%60:02d}  "
                    f"({b-a:3d}s, {db[a:b].mean():.1f} dB, "
                    f"{loud-db[a:b].mean():.1f} dB below loud)\n")
        f.write("#\n# second\tdB\tspeech\tquiet\n")
        for i, v in enumerate(db):
            q = any(a <= i < b for a, b in runs)
            f.write(f"{i}\t{v:.1f}\t{int(speech[i])}\t{int(q)}\n")

    print(f"{out.with_suffix('.wav')}  {t1-t0}s")
    print(f"{base}_ENTWURF.txt  {len(text.split())} words")
    print(f"{base}_PEGEL.txt  spread {spread:.1f} dB, "
          f"{len(runs)} quiet stretches / {sum(b-a for a, b in runs)}s")


if __name__ == "__main__":
    main()
